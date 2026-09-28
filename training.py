"""Backward-compatible entry point for the refactored ``bus_delay`` package.

New code may import from ``bus_delay.models``, ``bus_delay.trainer``, and
``bus_delay.prediction`` directly. Re-exports here keep existing scripts,
checkpoints, tests, and ``python training.py ...`` commands working.
"""

from bus_delay.cli import main, parse_args as _parse_args
from bus_delay.models import EmbeddedDelayModel, LSTM
from bus_delay.prediction import (
    build_future_stop_window,
    build_prediction_window_from_history,
    load_or_train,
    predict,
    predict_future_delay,
    predict_future_delay_range,
    predict_latest_snapshot,
    representative_route_rows as _representative_route_rows,
)
from bus_delay.trainer import (
    DEFAULT_WINDOW,
    MODEL_DIR,
    MODEL_PATH,
    evaluate_model,
    load_checkpoint,
    standardize as _standardize,
    test_accuracy,
    train_and_save,
    train_epochs as train,
)

__all__ = [
    "DEFAULT_WINDOW", "MODEL_DIR", "MODEL_PATH", "EmbeddedDelayModel", "LSTM",
    "build_future_stop_window", "build_prediction_window_from_history",
    "evaluate_model", "load_checkpoint", "load_or_train", "predict",
    "predict_future_delay", "predict_future_delay_range", "predict_latest_snapshot",
    "test_accuracy", "train", "train_and_save",
]


if __name__ == "__main__":
    main()
