"""Command-line interface for training, evaluation, and prediction."""

import argparse

from .prediction import (
    predict_future_delay,
    predict_future_delay_range,
    predict_latest_snapshot,
)
from .trainer import MODEL_PATH, test_accuracy, train_and_save


def parse_args() -> argparse.Namespace:
    """Parse the public ``training.py`` command-line interface."""
    parser = argparse.ArgumentParser(
        description="Train or run Seattle bus delay LSTM and MLP candidates."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Train and save a model checkpoint.")
    train_parser.add_argument("--data-dir", default="data")
    train_parser.add_argument("--model-path", default=str(MODEL_PATH))
    train_parser.add_argument("--window", type=int, default=6)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.add_argument("--architecture", choices=["lstm", "mlp"], default="lstm")
    train_parser.add_argument("--validation-fraction", type=float, default=0.15)
    train_parser.add_argument("--test-fraction", type=float, default=0.15)
    train_parser.add_argument("--patience", type=int, default=8)
    train_parser.add_argument("--seed", type=int, default=42)

    test_parser = commands.add_parser("test", help="Evaluate a saved model.")
    test_parser.add_argument("--data-dir", default="data")
    test_parser.add_argument("--model-path", default=str(MODEL_PATH))

    snapshot_parser = commands.add_parser(
        "predict-snapshot", help="Predict from the latest saved vehicle history."
    )
    snapshot_parser.add_argument("--vehicle-id", required=True)
    snapshot_parser.add_argument("--trip-id", required=True)
    snapshot_parser.add_argument("--data-dir", default="data")
    snapshot_parser.add_argument("--model-path", default=str(MODEL_PATH))

    predict_parser = commands.add_parser("predict", help="Predict delay at future stops.")
    predict_parser.add_argument("--route-id", required=True)
    predict_parser.add_argument("--stop-sequence", type=int, required=True)
    predict_parser.add_argument("--end-stop-sequence", type=int)
    predict_parser.add_argument("--current-delay", type=float, required=True)
    predict_parser.add_argument("--direction-id", type=int, choices=[0, 1])
    predict_parser.add_argument("--data-dir", default="data")
    predict_parser.add_argument("--model-path", default=str(MODEL_PATH))
    return parser.parse_args()


def main() -> None:
    """Dispatch the selected command."""
    args = parse_args()
    if args.command == "train":
        checkpoint = train_and_save(
            data_dir=args.data_dir,
            model_path=args.model_path,
            window=args.window,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            architecture=args.architecture,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            patience=args.patience,
            seed=args.seed,
        )
        final_mae = checkpoint["losses"][-1] if checkpoint["losses"] else float("nan")
        print(
            f"Saved {args.model_path}. Final training MAE: {final_mae:.2f} seconds; "
            f"test MAE: {checkpoint['test_metrics']['mae_seconds']:.2f} seconds"
        )
    elif args.command == "test":
        test_accuracy(args.model_path, args.data_dir)
    elif args.command == "predict-snapshot":
        delay = predict_latest_snapshot(
            args.vehicle_id, args.trip_id, args.model_path, args.data_dir
        )
        print(f"Predicted next-observation delay: {delay:.2f} seconds")
    elif args.end_stop_sequence is not None:
        predictions = predict_future_delay_range(
            args.route_id,
            args.stop_sequence,
            args.end_stop_sequence,
            args.current_delay,
            args.model_path,
            args.data_dir,
            args.direction_id,
        )
        for stop_sequence, delay in predictions.items():
            print(f"Predicted delay at stop_sequence {stop_sequence}: {delay:.2f} seconds")
    else:
        delay = predict_future_delay(
            args.route_id,
            args.stop_sequence,
            args.current_delay,
            args.model_path,
            args.data_dir,
            args.direction_id,
        )
        print(f"Predicted delay at stop_sequence {args.stop_sequence}: {delay:.2f} seconds")
