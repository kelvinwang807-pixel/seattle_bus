import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import process


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "bus_model_checkpoint.pt"
DEFAULT_WINDOW = 6


class LSTM(nn.Module):
    def __init__(self, input_size: int = 6, hidden_size: int = 64, num_layers: int = 2, output_size: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = x.new_zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = x.new_zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def _standardize(X: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (X - mean) / std.clamp_min(1e-6)


def train(
    model: nn.Module,
    optimizer: Adam,
    train_loader: DataLoader,
    epochs: int = 50,
    target_mae: float | None = None,
) -> list[float]:
    losses: list[float] = []
    criterion = nn.SmoothL1Loss()
    mae = nn.L1Loss(reduction="sum")
    model.train()

    for _epoch in range(epochs):
        total_abs_error = 0.0
        total = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            batch_size = y.size(0)
            total_abs_error += mae(y_pred.detach(), y).item()
            total += batch_size

        epoch_mae = total_abs_error / max(total, 1)
        losses.append(epoch_mae)
        if target_mae is not None and epoch_mae <= target_mae:
            break

    return losses


def train_and_save(
    data_dir: str | Path = "data",
    model_path: str | Path = MODEL_PATH,
    window: int = DEFAULT_WINDOW,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
) -> dict[str, Any]:
    df = process.process_features(data_dir)
    X, y = process.to_tensor(df, window=window)

    feature_mean = X.mean(dim=(0, 1), keepdim=True)
    feature_std = X.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    X = _standardize(X, feature_mean, feature_std)

    model = LSTM(input_size=len(process.FEATURE_COLUMNS))
    optimizer = Adam(model.parameters(), lr=learning_rate)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    losses = train(model=model, optimizer=optimizer, train_loader=loader, epochs=epochs)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feature_columns": process.FEATURE_COLUMNS,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "route_to_idx": df.attrs.get("route_to_idx", process.build_route_index(df["route_id"])),
        "window": window,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "losses": losses,
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_path)
    return checkpoint


def load_checkpoint(model_path: str | Path = MODEL_PATH) -> tuple[LSTM, dict[str, Any]]:
    checkpoint = torch.load(Path(model_path), map_location="cpu", weights_only=False)

    if "model_state_dict" in checkpoint:
        model = LSTM(
            input_size=len(checkpoint.get("feature_columns", process.FEATURE_COLUMNS)),
            hidden_size=int(checkpoint.get("hidden_size", 64)),
            num_layers=int(checkpoint.get("num_layers", 2)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, checkpoint

    # Backward compatibility with older files that saved only state_dict().
    model = LSTM()
    model.load_state_dict(checkpoint)
    model.eval()
    return model, {
        "feature_columns": process.FEATURE_COLUMNS,
        "feature_mean": torch.zeros(1, 1, len(process.FEATURE_COLUMNS)),
        "feature_std": torch.ones(1, 1, len(process.FEATURE_COLUMNS)),
        "route_to_idx": {},
        "window": DEFAULT_WINDOW,
    }


def load_or_train(
    model_path: str | Path = MODEL_PATH,
    data_dir: str | Path = "data",
    window: int = DEFAULT_WINDOW,
) -> tuple[LSTM, dict[str, Any]]:
    if not Path(model_path).is_file():
        train_and_save(data_dir=data_dir, model_path=model_path, window=window)
    return load_checkpoint(model_path)


def _representative_route_rows(
    route_id: str,
    stop_sequence: int,
    route_to_idx: dict[str, int],
    data_dir: str | Path,
    direction_id: int | None,
) -> pd.DataFrame:
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


def build_future_stop_window(
    route_id: str,
    stop_sequence: int,
    current_delay: float,
    checkpoint: dict[str, Any],
    data_dir: str | Path = "data",
    direction_id: int | None = None,
) -> torch.Tensor:
    window = int(checkpoint.get("window", DEFAULT_WINDOW))
    route_to_idx = {str(k): int(v) for k, v in checkpoint.get("route_to_idx", {}).items()}
    route_rows = _representative_route_rows(route_id, stop_sequence, route_to_idx, data_dir, direction_id)

    history = route_rows[route_rows["stop_sequence"] < int(stop_sequence)].tail(window).copy()
    if history.empty:
        history = route_rows[route_rows["stop_sequence"] <= int(stop_sequence)].head(1).copy()

    while len(history) < window:
        history = pd.concat([history.head(1), history], ignore_index=True)

    history = history.tail(window).copy()
    history["delay"] = float(current_delay)
    values = history[process.FEATURE_COLUMNS].to_numpy(dtype="float32")
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


def predict(model: nn.Module, x: torch.Tensor, checkpoint: dict[str, Any] | None = None) -> torch.Tensor:
    if checkpoint is not None:
        x = _standardize(x, checkpoint["feature_mean"], checkpoint["feature_std"])
    with torch.no_grad():
        return model(x)


def predict_future_delay(
    route_id: str,
    stop_sequence: int,
    current_delay: float,
    model_path: str | Path = MODEL_PATH,
    data_dir: str | Path = "data",
    direction_id: int | None = None,
) -> float:
    model, checkpoint = load_or_train(model_path=model_path, data_dir=data_dir)
    x = build_future_stop_window(
        route_id=route_id,
        stop_sequence=stop_sequence,
        current_delay=current_delay,
        checkpoint=checkpoint,
        data_dir=data_dir,
        direction_id=direction_id,
    )
    return float(predict(model, x, checkpoint).item())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or run the Seattle bus delay LSTM model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and save a model checkpoint.")
    train_parser.add_argument("--data-dir", default="data")
    train_parser.add_argument("--model-path", default=str(MODEL_PATH))
    train_parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)

    predict_parser = subparsers.add_parser("predict", help="Predict delay at a future stop.")
    predict_parser.add_argument("--route-id", required=True)
    predict_parser.add_argument("--stop-sequence", type=int, required=True)
    predict_parser.add_argument("--current-delay", type=float, required=True, help="Current delay in seconds.")
    predict_parser.add_argument("--direction-id", type=int, choices=[0, 1])
    predict_parser.add_argument("--data-dir", default="data")
    predict_parser.add_argument("--model-path", default=str(MODEL_PATH))

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "train":
        checkpoint = train_and_save(
            data_dir=args.data_dir,
            model_path=args.model_path,
            window=args.window,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        final_mae = checkpoint["losses"][-1] if checkpoint["losses"] else float("nan")
        print(f"Saved {args.model_path}. Final training MAE: {final_mae:.2f} seconds")
    elif args.command == "predict":
        delay = predict_future_delay(
            route_id=args.route_id,
            stop_sequence=args.stop_sequence,
            current_delay=args.current_delay,
            model_path=args.model_path,
            data_dir=args.data_dir,
            direction_id=args.direction_id,
        )
        print(f"Predicted delay at stop_sequence {args.stop_sequence}: {delay:.2f} seconds")


if __name__ == "__main__":
    main()
