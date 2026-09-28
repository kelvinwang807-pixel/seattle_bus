"""Load GTFS/OneBusAway data and build leakage-safe model tensors.

This module owns data ingestion and feature preparation. Model code lives under
``bus_delay``. The legacy ``to_tensor`` API remains available for old checkpoints.
"""

import json
import math
import random
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch


FEATURE_COLUMNS = [
    "arrival_min",
    "shape_dist_traveled",
    "delay",
    "stop_sequence",
    "route_idx",
    "direction_id",
]

DEFAULT_SNAPSHOT_MAX_AGE_SECONDS = 120.0
DEFAULT_SEQUENCE_MAX_GAP_SECONDS = 120.0


def _read_gtfs_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={
            "trip_id": "string",
            "route_id": "string",
            "stop_id": "string",
        },
    )


def _time_to_minutes(value: str) -> int:
    """GTFS times may exceed 24:00:00 for after-midnight service."""
    hours, minutes, _seconds = map(int, str(value).split(":"))
    return hours * 60 + minutes


def _trip_update_rows(entities: Iterable[dict]) -> list[dict]:
    rows: list[dict] = []

    for entity in entities:
        trip_update = entity.get("trip_update", {})
        trip = trip_update.get("trip", {})
        trip_id = trip.get("trip_id")
        if trip_id is None:
            continue

        for stop in trip_update.get("stop_time_update", []):
            if stop.get("schedule_relationship") == "SKIPPED":
                continue

            timing = stop.get("arrival") or stop.get("departure") or {}
            if "delay" not in timing:
                continue

            rows.append(
                {
                    "trip_id": str(trip_id),
                    "stop_sequence": int(stop["stop_sequence"]),
                    "stop_id": str(stop.get("stop_id", "")),
                    "delay": float(timing["delay"]),
                    "timestamp": timing.get("time"),
                }
            )

    return rows


def load_schedule(data_dir: str | Path = "data") -> pd.DataFrame:
    data_path = Path(data_dir)
    stop_times = _read_gtfs_csv(data_path / "stop_times.txt")
    trips = _read_gtfs_csv(data_path / "trips.txt")

    stop_times["stop_sequence"] = stop_times["stop_sequence"].astype(int)
    stop_times["shape_dist_traveled"] = pd.to_numeric(
        stop_times.get("shape_dist_traveled", 0.0), errors="coerce"
    ).fillna(0.0)
    trips["direction_id"] = pd.to_numeric(trips["direction_id"], errors="coerce").fillna(0).astype(int)

    schedule = stop_times.merge(
        trips[["trip_id", "route_id", "direction_id"]],
        on="trip_id",
        how="inner",
    )
    schedule["arrival_min"] = schedule["arrival_time"].apply(_time_to_minutes)
    return schedule


def load_realtime_updates(data_dir: str | Path = "data") -> pd.DataFrame:
    with (Path(data_dir) / "tripupdates_pb.json").open(encoding="utf-8") as f:
        data = json.load(f)

    rows = _trip_update_rows(data.get("entity", []))
    if not rows:
        return pd.DataFrame(columns=["trip_id", "stop_sequence", "stop_id", "delay", "timestamp"])

    return pd.DataFrame(rows)


def build_route_index(route_ids: Iterable[str]) -> dict[str, int]:
    return {str(route_id): idx for idx, route_id in enumerate(sorted({str(r) for r in route_ids}))}


def _strip_oba_agency_prefix(value: object) -> str:
    """Convert IDs such as ``1_635447420`` to their static-GTFS form."""
    text = str(value or "")
    prefix, separator, remainder = text.partition("_")
    return remainder if separator and prefix.isdigit() else text


def _number(value: object, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _snapshot_paths(data_dir: str | Path) -> list[Path]:
    return sorted((Path(data_dir) / "oba_snapshots").glob("**/vehicles-*.json"))


def process_snapshot_features(
    data_dir: str | Path = "data",
    max_age_seconds: float = DEFAULT_SNAPSHOT_MAX_AGE_SECONDS,
) -> pd.DataFrame:
    """Convert saved OneBusAway vehicle snapshots into chronological observations.

    Only fresh, predicted, in-progress trips are retained. OneBusAway IDs are
    joined to the static GTFS files so each observation has route, direction,
    and next-stop sequence values compatible with the existing LSTM inputs.
    """
    paths = _snapshot_paths(data_dir)
    if not paths:
        raise ValueError(f"No OneBusAway snapshots were found under {Path(data_dir) / 'oba_snapshots'}.")

    schedule = load_schedule(data_dir)
    schedule = schedule.copy()
    schedule["trip_key"] = schedule["trip_id"].map(_strip_oba_agency_prefix)
    schedule["stop_key"] = schedule["stop_id"].map(_strip_oba_agency_prefix)

    trip_info = (
        schedule.sort_values("stop_sequence")
        .drop_duplicates("trip_key")
        .set_index("trip_key")[["route_id", "direction_id"]]
        .to_dict("index")
    )
    stop_sequences = (
        schedule.sort_values("stop_sequence")
        .drop_duplicates(["trip_key", "stop_key"])
        .set_index(["trip_key", "stop_key"])["stop_sequence"]
        .to_dict()
    )

    rows: list[dict] = []
    for path in paths:
        try:
            with path.open(encoding="utf-8") as file:
                snapshot = json.load(file)
        except json.JSONDecodeError:
            # A collector may still be writing its newest snapshot.
            warnings.warn(f"Skipping incomplete or invalid snapshot: {path}", RuntimeWarning)
            continue

        payload = snapshot.get("payload", snapshot)
        if payload.get("code", 200) != 200:
            continue

        observation_ms = int(_number(payload.get("currentTime"), 0.0))
        if observation_ms <= 0:
            collected_at = pd.to_datetime(snapshot.get("collected_at"), utc=True, errors="coerce")
            if pd.isna(collected_at):
                continue
            observation_ms = int(collected_at.timestamp() * 1000)

        references = payload.get("data", {}).get("references", {}) or {}
        reference_trips = {
            _strip_oba_agency_prefix(trip.get("id")): trip
            for trip in references.get("trips", [])
            if trip.get("id")
        }

        for vehicle in payload.get("data", {}).get("list", []):
            status = vehicle.get("tripStatus")
            if not isinstance(status, dict) or status.get("predicted") is not True:
                continue

            trip_id = vehicle.get("tripId") or status.get("activeTripId")
            vehicle_id = vehicle.get("vehicleId") or status.get("vehicleId")
            if not trip_id or not vehicle_id:
                continue

            last_update_ms = int(_number(status.get("lastUpdateTime", vehicle.get("lastUpdateTime")), 0.0))
            update_age_seconds = (observation_ms - last_update_ms) / 1000.0
            if last_update_ms <= 0 or update_age_seconds < 0 or update_age_seconds > max_age_seconds:
                continue

            trip_key = _strip_oba_agency_prefix(trip_id)
            next_stop_key = _strip_oba_agency_prefix(status.get("nextStop"))
            static_trip = trip_info.get(trip_key, {})
            reference_trip = reference_trips.get(trip_key, {})
            route_id = static_trip.get("route_id") or _strip_oba_agency_prefix(reference_trip.get("routeId"))
            direction_id = static_trip.get("direction_id", reference_trip.get("directionId"))
            stop_sequence = stop_sequences.get((trip_key, next_stop_key))
            delay = _number(status.get("scheduleDeviation"))
            distance = _number(status.get("distanceAlongTrip"))
            total_distance = _number(status.get("totalDistanceAlongTrip"))

            if not route_id or direction_id is None or stop_sequence is None:
                continue
            if not math.isfinite(delay) or not math.isfinite(distance):
                continue

            service_date_ms = int(_number(status.get("serviceDate"), 0.0))
            if service_date_ms > 0:
                arrival_min = (observation_ms - service_date_ms) / 60_000.0
            else:
                arrival_min = (observation_ms / 60_000.0) % (24 * 60)

            position = status.get("position") or vehicle.get("location") or {}
            rows.append(
                {
                    "observation_time": observation_ms / 1000.0,
                    "service_date": str(service_date_ms) if service_date_ms else str(
                        pd.Timestamp(observation_ms, unit="ms", tz="UTC").tz_convert("America/Los_Angeles").date()),
                    "last_update_time": last_update_ms / 1000.0,
                    "vehicle_id": str(vehicle_id),
                    "trip_id": trip_key,
                    "route_id": _strip_oba_agency_prefix(route_id),
                    "direction_id": int(direction_id),
                    "arrival_min": float(arrival_min),
                    "shape_dist_traveled": float(distance),
                    "delay": float(delay),
                    "stop_sequence": int(stop_sequence),
                    "next_stop_id": next_stop_key,
                    "next_stop_time_offset": _number(status.get("nextStopTimeOffset")),
                    "closest_stop_time_offset": _number(status.get("closestStopTimeOffset")),
                    "progress_fraction": distance / total_distance if total_distance > 0 else float("nan"),
                    "latitude": _number(position.get("lat")),
                    "longitude": _number(position.get("lon")),
                    "orientation": _number(status.get("orientation")),
                    "update_age_seconds": float(update_age_seconds),
                    "snapshot_file": str(path),
                }
            )

    if not rows:
        raise ValueError("Snapshots contained no fresh predicted trips that matched the static GTFS schedule.")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(["observation_time", "vehicle_id", "trip_id"], keep="last")
    df = df.sort_values(["vehicle_id", "trip_id", "observation_time"]).reset_index(drop=True)
    route_to_idx = build_route_index(df["route_id"])
    df["route_idx"] = df["route_id"].map(route_to_idx).astype(int)
    df = df.dropna(subset=FEATURE_COLUMNS)
    df.attrs["route_to_idx"] = route_to_idx
    df.attrs["source"] = "onebusaway_snapshots"
    df.attrs["snapshot_count"] = len(paths)
    return df


def process_features(data_dir: str | Path = "data") -> pd.DataFrame:
    if _snapshot_paths(data_dir):
        return process_snapshot_features(data_dir)

    schedule = load_schedule(data_dir)
    realtime = load_realtime_updates(data_dir)

    if realtime.empty:
        raise ValueError("No usable realtime stop delay records were found.")

    df = schedule.merge(realtime, on=["trip_id", "stop_sequence"], how="inner", suffixes=("", "_rt"))
    if df.empty:
        raise ValueError("Realtime updates did not match any scheduled trip stop records.")

    route_to_idx = build_route_index(df["route_id"])
    df["route_idx"] = df["route_id"].map(route_to_idx).astype(int)
    df["delay"] = pd.to_numeric(df["delay"], errors="coerce")
    df = df.dropna(subset=FEATURE_COLUMNS)
    df = df.sort_values(["trip_id", "stop_sequence"]).reset_index(drop=True)
    df.attrs["route_to_idx"] = route_to_idx
    return df


def to_tensor(
    df: pd.DataFrame,
    window: int = 6,
    test_fraction: float = 0.2,
    seed: int = 42,
    max_sequence_gap_seconds: float = DEFAULT_SEQUENCE_MAX_GAP_SECONDS,
    validation_fraction: float = 0.0,
) -> tuple[torch.Tensor, ...]:
    """Build train/test windows without putting one trip in both partitions.

    Splitting at the trip level prevents overlapping windows from a single trip
    leaking into both training and evaluation data.  ``seed`` makes the split
    reproducible so that test metrics can be compared across runs.

    With validation_fraction > 0 returns X_train, y_train, X_val, y_val,
    X_test, y_test. Snapshot partitions are then chronological, with no
    observations shared across partitions; static data splits whole trips.
    """
    if validation_fraction:
        frames, _ = split_frames(df, validation_fraction, test_fraction, seed=seed)
        return tuple(t for frame in frames for t in sequence_tensors(
            frame, FEATURE_COLUMNS, window, max_sequence_gap_seconds
        ))
    if window < 1:
        raise ValueError("window must be at least 1")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")

    windows_by_trip: dict[str, list[tuple[list[list[float]], float]]] = {}
    is_snapshot_data = "observation_time" in df.columns and "vehicle_id" in df.columns
    group_columns = ["vehicle_id", "trip_id"] if is_snapshot_data else ["trip_id"]
    sort_columns = group_columns + (["observation_time"] if is_snapshot_data else ["stop_sequence"])

    for group_id, trip in df.sort_values(sort_columns).groupby(group_columns, sort=False):
        segments = [trip]
        if is_snapshot_data:
            trip = trip.drop_duplicates("observation_time", keep="last")
            segment_ids = trip["observation_time"].diff().gt(max_sequence_gap_seconds).cumsum()
            segments = [segment for _segment_id, segment in trip.groupby(segment_ids, sort=False)]

        trip_windows: list[tuple[list[list[float]], float]] = []
        for segment in segments:
            values = segment[FEATURE_COLUMNS].to_numpy(dtype="float32")
            if len(values) <= window:
                continue
            for i in range(len(values) - window):
                trip_windows.append(
                    (values[i : i + window].tolist(), float(values[i + window][FEATURE_COLUMNS.index("delay")]))
                )

        if trip_windows:
            windows_by_trip[str(group_id)] = trip_windows

    if len(windows_by_trip) < 2:
        if is_snapshot_data:
            raise ValueError(
                f"At least two vehicle trips need {window + 1} consecutive snapshots "
                "for a train/test split. Keep the collector running longer."
            )
        raise ValueError("At least two trips with enough stop records are required for a train/test split.")

    trip_ids = list(windows_by_trip)
    random.Random(seed).shuffle(trip_ids)
    split = min(max(1, int(len(trip_ids) * (1.0 - test_fraction))), len(trip_ids) - 1)
    train_trip_ids = trip_ids[:split]
    test_trip_ids = trip_ids[split:]

    train_samples = [sample for trip_id in train_trip_ids for sample in windows_by_trip[trip_id]]
    test_samples = [sample for trip_id in test_trip_ids for sample in windows_by_trip[trip_id]]
    if not train_samples or not test_samples:
        raise ValueError(f"Not enough stop records to create LSTM windows of size {window}.")

    xs_train, ys_train = zip(*train_samples)
    xs_test, ys_test = zip(*test_samples)

    X = torch.tensor(xs_train, dtype=torch.float32)
    y = torch.tensor(ys_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(xs_test, dtype=torch.float32)
    y_test = torch.tensor(ys_test, dtype=torch.float32).unsqueeze(1)
    return X, y, X_test, y_test


# The legacy six-column interface remains available for old checkpoints.
NUMERIC_COLUMNS = [
    "arrival_min", "shape_dist_traveled", "delay", "stop_sequence",
    "time_sin", "time_cos", "elapsed_seconds", "speed", "delay_change",
    "update_age_seconds",
]
CATEGORY_COLUMNS = ["route_id", "direction_id", "next_stop_id"]
EMBEDDING_COLUMNS = NUMERIC_COLUMNS + [f"{c}_encoded" for c in CATEGORY_COLUMNS]


def sequence_groups(df: pd.DataFrame) -> list[str]:
    return [c for c in ("vehicle_id", "trip_id", "service_date") if c in df]


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal features; no future observations are used."""
    df = df.copy()
    snapshot = "observation_time" in df
    order = "observation_time" if snapshot else "stop_sequence"
    groups = sequence_groups(df)
    df = df.sort_values(groups + [order])
    if "next_stop_id" not in df:
        df["next_stop_id"] = df.get("stop_id", "<unknown>")
    if snapshot:
        local = pd.to_datetime(df.observation_time, unit="s", utc=True).dt.tz_convert("America/Los_Angeles")
        minutes = local.dt.hour * 60 + local.dt.minute + local.dt.second / 60
        elapsed = df.groupby(groups)["observation_time"].diff()
        valid = elapsed.gt(0) & elapsed.le(DEFAULT_SEQUENCE_MAX_GAP_SECONDS)
        df["elapsed_seconds"] = elapsed.where(valid, 0)
        df["speed"] = (df.groupby(groups).shape_dist_traveled.diff() / elapsed).where(valid, 0)
        df["delay_change"] = df.groupby(groups).delay.diff().where(valid, 0)
    else:
        minutes = df.arrival_min
        df["elapsed_seconds"] = 0.0
        df["speed"] = 0.0
        df["delay_change"] = df.groupby(groups).delay.diff().fillna(0)
    angle = minutes * (2 * math.pi / 1440)
    df["time_sin"] = angle.map(math.sin)
    df["time_cos"] = angle.map(math.cos)
    if "update_age_seconds" not in df:
        df["update_age_seconds"] = 0.0
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].replace([float("inf"), -float("inf")], float("nan")).fillna(0)
    return df


def split_frames(df, validation_fraction=0.15, test_fraction=0.15, seed=42, boundaries=None):
    """Chronological snapshot partitions with windows built separately in each.

    Saved boundaries freeze the test interval when new snapshots are collected.
    Static GTFS fallback splits whole trips using a reproducible shuffle.
    """
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1 and validation_fraction + test_fraction < 1):
        raise ValueError("Validation and test fractions must be positive and sum to less than one.")
    if "observation_time" in df:
        times = sorted(df.observation_time.unique())
        if boundaries is None:
            if len(times) < 3:
                raise ValueError("At least three observation timestamps are required.")
            a = min(max(1, int(len(times) * (1-validation_fraction-test_fraction))), len(times)-2)
            b = min(max(a+1, int(len(times) * (1-test_fraction))), len(times)-1)
            boundaries = {"kind": "time", "validation_start": float(times[a]),
                          "test_start": float(times[b]), "test_end": float(times[-1])}
        t = df.observation_time
        frames = [df[t < boundaries["validation_start"]],
                  df[(t >= boundaries["validation_start"]) & (t < boundaries["test_start"])],
                  df[(t >= boundaries["test_start"]) & (t <= boundaries["test_end"])]]
    else:
        if boundaries is None:
            trips = sorted(df.trip_id.unique().tolist())
            if len(trips) < 3:
                raise ValueError("At least three trips are required for train/validation/test.")
            random.Random(seed).shuffle(trips)
            a = min(max(1, int(len(trips)*(1-validation_fraction-test_fraction))), len(trips)-2)
            b = min(max(a+1, int(len(trips)*(1-test_fraction))), len(trips)-1)
            boundaries = {"kind": "trip", "trips": [trips[:a], trips[a:b], trips[b:]]}
        frames = [df[df.trip_id.isin(ids)] for ids in boundaries["trips"]]
    return [f.copy() for f in frames], boundaries


def sequence_tensors(df, columns, window=6, max_gap=DEFAULT_SEQUENCE_MAX_GAP_SECONDS):
    if window < 1:
        raise ValueError("window must be at least 1")
    xs, ys = [], []
    snapshot = "observation_time" in df
    for _, trip in df.groupby(sequence_groups(df), sort=False):
        trip = trip.sort_values("observation_time" if snapshot else "stop_sequence")
        segments = [trip]
        if snapshot:
            trip = trip.drop_duplicates("observation_time")
            segments = [s for _, s in trip.groupby(trip.observation_time.diff().gt(max_gap).cumsum())]
        for segment in segments:
            values = torch.tensor(segment[columns].to_numpy(dtype="float32"))
            targets = torch.tensor(segment.delay.to_numpy(dtype="float32"))
            for i in range(len(values)-window):
                xs.append(values[i:i+window])
                ys.append(targets[i+window])
    if not xs:
        raise ValueError(f"A partition has no sequences of {window+1} observations; collect more data or reduce window.")
    return torch.stack(xs), torch.stack(ys).unsqueeze(1)


def encode_categories(df, vocabularies):
    df = df.copy()
    for column in CATEGORY_COLUMNS:
        df[f"{column}_encoded"] = df[column].astype(str).map(vocabularies[column]).fillna(0).astype(int)
    return df


def prepare_embedding_data(df, window=6, validation_fraction=0.15, test_fraction=0.15,
                           seed=42, metadata=None):
    frames, boundaries = split_frames(df, validation_fraction, test_fraction, seed,
                                     None if metadata is None else metadata["split"])
    # Compute motion features independently to prevent history crossing partitions.
    frames = [enrich_features(f) for f in frames]
    vocabularies = (metadata["vocabularies"] if metadata is not None else {
        c: {value: i+1 for i, value in enumerate(sorted(frames[0][c].astype(str).unique()))}
        for c in CATEGORY_COLUMNS
    })
    tensors = [sequence_tensors(encode_categories(f, vocabularies), EMBEDDING_COLUMNS, window) for f in frames]
    return tensors, {"split": boundaries, "vocabularies": vocabularies}
