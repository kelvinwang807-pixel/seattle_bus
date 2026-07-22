import json
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


def process_features(data_dir: str | Path = "data") -> pd.DataFrame:
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


def to_tensor(df: pd.DataFrame, window: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    if window < 1:
        raise ValueError("window must be at least 1")

    xs: list[list[list[float]]] = []
    ys: list[float] = []

    for _trip_id, trip in df.sort_values(["trip_id", "stop_sequence"]).groupby("trip_id", sort=False):
        values = trip[FEATURE_COLUMNS].to_numpy(dtype="float32")
        if len(values) <= window:
            continue

        for i in range(len(values) - window):
            xs.append(values[i : i + window].tolist())
            ys.append(float(values[i + window][FEATURE_COLUMNS.index("delay")]))

    if not xs:
        raise ValueError(f"Not enough stop records to create LSTM windows of size {window}.")

    X = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)
    return X, y
