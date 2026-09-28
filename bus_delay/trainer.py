"""Training, evaluation, and checkpoint persistence for delay models."""

import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import process
from .models import EmbeddedDelayModel, LSTM


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "bus_model_checkpoint.pt"
DEFAULT_WINDOW = 6


def standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Apply saved training statistics while protecting constant features."""
    return (x - mean) / std.clamp_min(1e-6)


def train_epochs(
    model: nn.Module,
    optimizer: Adam,
    train_loader: DataLoader,
    epochs: int = 50,
    target_mae: float | None = None,
    validation_loader: DataLoader | None = None,
    patience: int = 8,
) -> list[float]:
    """Train a model and restore the epoch with the best validation MAE."""
    losses: list[float] = []
    criterion = nn.SmoothL1Loss()
    mae = nn.L1Loss(reduction="sum")
    model.train()
    best_mae, best_state, stale_epochs = float("inf"), None, 0
    model.validation_losses = []

    for epoch in range(epochs):
        total_absolute_error = 0.0
        total = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_absolute_error += mae(prediction.detach(), y).item()
            total += y.size(0)

        epoch_mae = total_absolute_error / max(total, 1)
        losses.append(epoch_mae)
        if validation_loader is not None:
            model.eval()
            with torch.no_grad():
                validation_error = sum(
                    (model(x) - y).abs().sum().item()
                    for x, y in validation_loader
                )
            validation_mae = validation_error / len(validation_loader.dataset)
            model.validation_losses.append(validation_mae)
            print(
                f"Epoch {epoch + 1}: train MAE={epoch_mae:.3f}s; "
                f"validation MAE={validation_mae:.3f}s",
                flush=True,
            )
            model.train()
            if validation_mae < best_mae:
                best_mae = validation_mae
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= patience:
                break
        if target_mae is not None and epoch_mae <= target_mae:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return losses


def evaluate_model(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int = 256,
    label: str = "Test",
) -> dict[str, float]:
    """Return MAE, RMSE, and Smooth L1 in seconds for one partition."""
    if len(x_test) == 0:
        raise ValueError("The evaluation set is empty.")
    loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size)
    was_training = model.training
    model.eval()
    absolute_error = squared_error = smooth_l1 = 0.0
    total = 0
    criterion = nn.SmoothL1Loss(reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            prediction = model(x)
            error = prediction - y
            absolute_error += error.abs().sum().item()
            squared_error += error.square().sum().item()
            smooth_l1 += criterion(prediction, y).item()
            total += y.numel()
    if was_training:
        model.train()
    metrics = {
        "count": float(total),
        "mae_seconds": absolute_error / total,
        "rmse_seconds": (squared_error / total) ** 0.5,
        "smooth_l1_loss": smooth_l1 / total,
    }
    print(
        f"{label} samples: {total}; MAE: {metrics['mae_seconds']:.2f} seconds; "
        f"RMSE: {metrics['rmse_seconds']:.2f} seconds; "
        f"Smooth L1: {metrics['smooth_l1_loss']:.2f}"
    )
    return metrics


def train_and_save(
    data_dir: str | Path = "data",
    model_path: str | Path = MODEL_PATH,
    window: int = DEFAULT_WINDOW,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    architecture: str = "lstm",
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    patience: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    """Build partitions, train one candidate, evaluate it, and save a checkpoint."""
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")
    torch.manual_seed(seed)
    frame = process.process_features(data_dir)
    partitions, metadata = process.prepare_embedding_data(
        frame, window, validation_fraction, test_fraction, seed
    )
    (x_train, y_train), (x_validation, y_validation), (x_test, y_test) = partitions
    delay_index = process.NUMERIC_COLUMNS.index("delay")
    baseline_error = x_test[:, -1, delay_index].unsqueeze(1) - y_test
    baseline_metrics = {
        "mae_seconds": baseline_error.abs().mean().item(),
        "rmse_seconds": baseline_error.square().mean().sqrt().item(),
    }

    feature_mean = x_train.mean(dim=(0, 1), keepdim=True)
    feature_std = x_train.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(1e-6)
    # Category IDs are embedding indices and must never be standardized.
    feature_mean[..., len(process.NUMERIC_COLUMNS):] = 0
    feature_std[..., len(process.NUMERIC_COLUMNS):] = 1
    x_train = standardize(x_train, feature_mean, feature_std)
    x_validation = standardize(x_validation, feature_mean, feature_std)
    x_test = standardize(x_test, feature_mean, feature_std)

    model_config = {
        "category_sizes": [
            len(metadata["vocabularies"][column]) + 1
            for column in process.CATEGORY_COLUMNS
        ],
        "window": window,
        "architecture": architecture,
        "delay_mean": feature_mean[0, 0, delay_index].item(),
        "delay_std": feature_std[0, 0, delay_index].item(),
    }
    model = EmbeddedDelayModel(**model_config)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        TensorDataset(x_validation, y_validation), batch_size=batch_size
    )
    losses = train_epochs(
        model, optimizer, train_loader, epochs,
        validation_loader=validation_loader, patience=patience,
    )
    validation_metrics = evaluate_model(
        model, x_validation, y_validation, label="Validation"
    )
    test_metrics = evaluate_model(model, x_test, y_test)
    checkpoint = {
        "format_version": 2,
        "model_config": model_config,
        "preprocessing": metadata,
        "source": frame.attrs.get("source", "gtfs_tripupdates"),
        "target": "next_observation_delay" if "observation_time" in frame else "next_stop_delay",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "validation_losses": model.validation_losses,
        "validation_metrics": validation_metrics,
        "persistence_test_metrics": baseline_metrics,
        "model_state_dict": model.state_dict(),
        "feature_columns": process.EMBEDDING_COLUMNS,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "route_to_idx": metadata["vocabularies"]["route_id"],
        "window": window,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "losses": losses,
        "test_metrics": test_metrics,
    }
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    print(f"Persistence baseline: {baseline_metrics}")
    return checkpoint


def load_checkpoint(model_path: str | Path = MODEL_PATH) -> tuple[nn.Module, dict[str, Any]]:
    """Load current embedded checkpoints and both legacy checkpoint formats."""
    checkpoint = torch.load(Path(model_path), map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        if checkpoint.get("format_version") == 2:
            model = EmbeddedDelayModel(**checkpoint["model_config"])
        else:
            model = LSTM(
                input_size=len(checkpoint.get("feature_columns", process.FEATURE_COLUMNS)),
                hidden_size=int(checkpoint.get("hidden_size", 64)),
                num_layers=int(checkpoint.get("num_layers", 2)),
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, checkpoint

    model = LSTM()
    model.load_state_dict(checkpoint)
    model.eval()
    metadata = {
        "feature_columns": process.FEATURE_COLUMNS,
        "feature_mean": torch.zeros(1, 1, len(process.FEATURE_COLUMNS)),
        "feature_std": torch.ones(1, 1, len(process.FEATURE_COLUMNS)),
        "route_to_idx": {},
        "window": DEFAULT_WINDOW,
    }
    return model, metadata


def test_accuracy(model_path: str | Path = MODEL_PATH,
                  data_dir: str | Path = "data") -> dict[str, float]:
    """Recreate the saved test partition and evaluate a checkpoint."""
    model, checkpoint = load_checkpoint(model_path)
    frame = process.process_features(data_dir)
    if checkpoint.get("format_version") == 2:
        partitions, _ = process.prepare_embedding_data(
            frame,
            checkpoint["window"],
            checkpoint["validation_fraction"],
            checkpoint["test_fraction"],
            metadata=checkpoint["preprocessing"],
        )
        x_test, y_test = partitions[2]
    else:
        _, _, x_test, y_test = process.to_tensor(
            frame, window=int(checkpoint.get("window", DEFAULT_WINDOW))
        )
    return evaluate_model(
        model,
        standardize(x_test, checkpoint["feature_mean"], checkpoint["feature_std"]),
        y_test,
    )
