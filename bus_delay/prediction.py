"""Inference helpers for snapshot forecasts and legacy stop-by-stop forecasts."""

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn

import process
from .trainer import DEFAULT_WINDOW, MODEL_PATH, load_checkpoint, standardize, train_and_save


def load_or_train(model_path: str | Path = MODEL_PATH, data_dir: str | Path = "data",
                  window: int = DEFAULT_WINDOW) -> tuple[nn.Module, dict[str, Any]]:
    """Load a checkpoint, training one with defaults only when none exists."""
    if not Path(model_path).is_file():
        train_and_save(data_dir=data_dir, model_path=model_path, window=window)
    return load_checkpoint(model_path)


def representative_route_rows(route_id: str, stop_sequence: int,
                              route_to_idx: dict[str, int], data_dir: str | Path,
                              direction_id: int | None) -> pd.DataFrame:
    """Choose a scheduled trip pattern containing the requested route stop."""
    route_id = str(route_id)
    if route_id not in route_to_idx:
        known = ", ".join(list(route_to_idx)[:10])
        raise ValueError(f"Unknown route_id {route_id!r}. Known examples: {known}")
    schedule = process.load_schedule(data_dir)
    route_rows = schedule[schedule["route_id"].astype(str) == route_id].copy()
    if direction_id is not None:
        route_rows = route_rows[route_rows["direction_id"] == direction_id]
    if route_rows.empty:
        raise ValueError(f"No scheduled stops found for route_id {route_id!r}.")
    target_rows = route_rows[route_rows["stop_sequence"] == int(stop_sequence)]
    if target_rows.empty:
        raise ValueError(f"No stop_sequence {stop_sequence} found for route_id {route_id!r}.")
    trip_id = target_rows.sort_values("arrival_min").iloc[0]["trip_id"]
    trip_rows = route_rows[route_rows["trip_id"] == trip_id].sort_values("stop_sequence").copy()
    trip_rows["route_idx"] = route_to_idx[route_id]
    return trip_rows


def build_prediction_window_from_history(history: pd.DataFrame,
                                         checkpoint: dict[str, Any]) -> torch.Tensor:
    """Convert recent processed history into one model input window."""
    window = int(checkpoint.get("window", DEFAULT_WINDOW))
    if history.empty:
        raise ValueError("Prediction history is empty")
    if checkpoint.get("format_version") == 2:
        if checkpoint.get("target") == "next_observation_delay":
            required = {"observation_time", "vehicle_id", "trip_id"}
            if not required.issubset(history.columns):
                raise ValueError("Snapshot models require timestamped vehicle history")
            if len(history[process.sequence_groups(history)].drop_duplicates()) != 1:
                raise ValueError("History must contain exactly one vehicle trip and service date")
            history = history.sort_values("observation_time").drop_duplicates("observation_time")
            recent = history.tail(window)
            if len(history) < window or recent.observation_time.diff().gt(
                process.DEFAULT_SEQUENCE_MAX_GAP_SECONDS
            ).any():
                raise ValueError("Not enough consecutive recent snapshots for prediction")
        history = process.encode_categories(
            process.enrich_features(history),
            checkpoint["preprocessing"]["vocabularies"],
        )
    while len(history) < window:
        history = pd.concat([history.head(1), history], ignore_index=True)
    columns = checkpoint.get("feature_columns", process.FEATURE_COLUMNS)
    values = history.tail(window)[columns].to_numpy(dtype="float32")
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


def build_future_stop_window(route_id: str, stop_sequence: int, current_delay: float,
                             checkpoint: dict[str, Any], data_dir: str | Path = "data",
                             direction_id: int | None = None) -> torch.Tensor:
    """Build a legacy next-stop input from static GTFS schedule rows."""
    if checkpoint.get("target") == "next_observation_delay":
        raise ValueError(
            "This model predicts the next snapshot, not the next stop. "
            "Use predict-snapshot with real history."
        )
    window = int(checkpoint.get("window", DEFAULT_WINDOW))
    route_to_idx = {str(key): int(value) for key, value in checkpoint.get("route_to_idx", {}).items()}
    route_rows = representative_route_rows(
        route_id, stop_sequence, route_to_idx, data_dir, direction_id
    )
    history = route_rows[route_rows["stop_sequence"] < int(stop_sequence)].tail(window).copy()
    if history.empty:
        history = route_rows[route_rows["stop_sequence"] <= int(stop_sequence)].head(1).copy()
    while len(history) < window:
        history = pd.concat([history.head(1), history], ignore_index=True)
    history = history.tail(window).copy()
    history["delay"] = float(current_delay)
    return build_prediction_window_from_history(history, checkpoint)


def predict(model: nn.Module, x: torch.Tensor,
            checkpoint: dict[str, Any] | None = None) -> torch.Tensor:
    """Run inference, applying checkpoint preprocessing when provided."""
    if checkpoint is not None:
        x = standardize(x, checkpoint["feature_mean"], checkpoint["feature_std"])
    with torch.no_grad():
        return model(x)


def predict_future_delay(route_id: str, stop_sequence: int, current_delay: float,
                         model_path: str | Path = MODEL_PATH,
                         data_dir: str | Path = "data",
                         direction_id: int | None = None) -> float:
    """Predict one future stop with a compatible legacy/stop-target model."""
    model, checkpoint = load_or_train(model_path=model_path, data_dir=data_dir)
    x = build_future_stop_window(
        route_id, stop_sequence, current_delay, checkpoint, data_dir, direction_id
    )
    return float(predict(model, x, checkpoint).item())


def predict_future_delay_range(route_id: str, current_stop_sequence: int,
                               end_stop_sequence: int, current_delay: float,
                               model_path: str | Path = MODEL_PATH,
                               data_dir: str | Path = "data",
                               direction_id: int | None = None) -> dict[int, float]:
    """Recursively predict each stop through an inclusive end stop."""
    if end_stop_sequence <= current_stop_sequence:
        raise ValueError("end_stop_sequence must be greater than current_stop_sequence")
    model, checkpoint = load_or_train(model_path=model_path, data_dir=data_dir)
    if checkpoint.get("target") == "next_observation_delay":
        raise ValueError("This model predicts the next snapshot, not future stops. Use predict-snapshot.")
    route_to_idx = {str(key): int(value) for key, value in checkpoint.get("route_to_idx", {}).items()}
    route_rows = representative_route_rows(
        route_id, current_stop_sequence, route_to_idx, data_dir, direction_id
    )
    history = route_rows[route_rows["stop_sequence"] <= int(current_stop_sequence)].copy()
    if history.empty:
        raise ValueError(f"No stop_sequence {current_stop_sequence} found for route_id {route_id!r}.")
    if "delay" not in history.columns:
        history["delay"] = float(current_delay)
    else:
        history["delay"] = pd.to_numeric(history["delay"], errors="coerce").fillna(float(current_delay))
    history.loc[history.index[-1], "delay"] = float(current_delay)
    predictions: dict[int, float] = {}
    for next_stop in range(current_stop_sequence + 1, end_stop_sequence + 1):
        stop_rows = route_rows[route_rows["stop_sequence"] == next_stop]
        if stop_rows.empty:
            raise ValueError(f"No stop_sequence {next_stop} found for route_id {route_id!r}.")
        x = build_prediction_window_from_history(history, checkpoint)
        predicted_delay = float(predict(model, x, checkpoint).item())
        predictions[next_stop] = predicted_delay
        next_row = stop_rows.iloc[[0]].copy()
        next_row["delay"] = predicted_delay
        history = pd.concat([history, next_row], ignore_index=True)
    return predictions


def predict_latest_snapshot(vehicle_id: str, trip_id: str,
                            model_path: str | Path = MODEL_PATH,
                            data_dir: str | Path = "data") -> float:
    """Predict after the latest saved observation for one vehicle trip."""
    model, checkpoint = load_checkpoint(model_path)
    if checkpoint.get("target") != "next_observation_delay":
        raise ValueError("predict-snapshot requires a model trained on snapshots")
    frame = process.process_snapshot_features(data_dir)
    history = frame[(frame.vehicle_id == vehicle_id) & (frame.trip_id == trip_id)].sort_values(
        "observation_time"
    )
    if not history.empty and "service_date" in history:
        history = history[history.service_date == history.iloc[-1].service_date]
    x = build_prediction_window_from_history(history, checkpoint)
    return float(predict(model, x, checkpoint).item())
