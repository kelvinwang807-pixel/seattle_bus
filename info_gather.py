from pathlib import Path

import training


DEFAULT_DATA_DIR = "data"
DEFAULT_MODEL_PATH = str(training.MODEL_PATH)


def _prompt_text(label: str, default: str | None = None) -> str:
    prompt = f"{label}"
    if default is not None:
        prompt += f" [{default}]"
    prompt += ": "

    value = input(prompt).strip()
    if value:
        return value
    if default is not None:
        return default
    raise ValueError(f"{label} is required")


def _prompt_int(label: str, default: int | None = None, choices: set[int] | None = None) -> int:
    while True:
        raw_default = str(default) if default is not None else None
        raw = _prompt_text(label, raw_default)
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if choices is not None and value not in choices:
            allowed = ", ".join(str(choice) for choice in sorted(choices))
            print(f"Please enter one of: {allowed}.")
            continue
        return value


def _prompt_float(label: str, default: float | None = None) -> float:
    while True:
        raw_default = str(default) if default is not None else None
        raw = _prompt_text(label, raw_default)
        try:
            return float(raw)
        except ValueError:
            print("Please enter a number.")


def _prompt_optional_direction() -> int | None:
    while True:
        raw = input("Direction ID, 0 or 1 (leave blank if unknown): ").strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            print("Please enter 0, 1, or leave it blank.")
            continue
        if value in {0, 1}:
            return value
        print("Please enter 0, 1, or leave it blank.")


def _prompt_yes_no(label: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{label} [{suffix}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please enter y or n.")


def _ensure_model(model_path: Path, data_dir: str) -> bool:
    if model_path.is_file():
        return True

    print(f"No model checkpoint found at {model_path}.")
    if not _prompt_yes_no("Train a model now?", default=True):
        print("Prediction cancelled because no model checkpoint is available.")
        return False

    epochs = _prompt_int("Training epochs", default=50)
    batch_size = _prompt_int("Batch size", default=64)
    learning_rate = _prompt_float("Learning rate", default=0.001)

    checkpoint = training.train_and_save(
        data_dir=data_dir,
        model_path=model_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    final_mae = checkpoint["losses"][-1] if checkpoint["losses"] else float("nan")
    print(f"Saved {model_path}. Final training MAE: {final_mae:.2f} seconds")
    return True


def main() -> None:
    print("Seattle bus delay predictor")
    print("Enter current trip information. Delay values are in seconds.")

    data_dir = _prompt_text("Data directory", DEFAULT_DATA_DIR)
    model_path = Path(_prompt_text("Model checkpoint path", DEFAULT_MODEL_PATH))

    if not _ensure_model(model_path, data_dir):
        return

    route_id = _prompt_text("Route ID")
    current_stop_sequence = _prompt_int("Current stop sequence")
    end_stop_sequence = _prompt_int("End stop sequence")
    current_delay = _prompt_float("Current delay in seconds")
    direction_id = _prompt_optional_direction()

    try:
        predictions = training.predict_future_delay_range(
            route_id=route_id,
            current_stop_sequence=current_stop_sequence,
            end_stop_sequence=end_stop_sequence,
            current_delay=current_delay,
            model_path=model_path,
            data_dir=data_dir,
            direction_id=direction_id,
        )
    except ValueError as exc:
        print(f"Could not predict delay: {exc}")
        return

    for stop_sequence, predicted_delay in predictions.items():
        print(f"Predicted delay at stop_sequence {stop_sequence}: {predicted_delay:.2f} seconds")


if __name__ == "__main__":
    main()
