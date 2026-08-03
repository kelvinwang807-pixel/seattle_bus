# Multi-Stop Delay Prediction Design

## Goal

Add recursive multi-stop prediction to the existing LSTM bus delay predictor. Given a route, the current stop sequence, the current measured delay, and a future end stop sequence, the system returns predicted delays for every stop after the current stop through the end stop.

Example: `current_stop_sequence=3`, `current_delay=120`, and `end_stop_sequence=9` returns predictions for stops 4, 5, 6, 7, 8, and 9.

## Approach

Use the existing one-step LSTM recursively. The model already predicts the next delay from a window of scheduled stop features plus delay history. For multi-stop prediction, each forecasted stop becomes part of the next input window:

1. Build the initial LSTM input window from scheduled rows up to the current stop.
2. Predict the delay for `current_stop_sequence + 1`.
3. Add that predicted delay to the scheduled features for that stop.
4. Repeat until `end_stop_sequence` is predicted.

This keeps the saved checkpoint format and training flow compatible with the current project.

## API

Add a public function in `training.py`:

```python
def predict_future_delay_range(
    route_id: str,
    current_stop_sequence: int,
    end_stop_sequence: int,
    current_delay: float,
    model_path: str | Path = MODEL_PATH,
    data_dir: str | Path = "data",
    direction_id: int | None = None,
) -> dict[int, float]:
    ...
```

The returned dictionary is ordered by insertion and maps each predicted stop sequence to the predicted delay in seconds.

The existing `predict_future_delay(...)` single-stop API remains available and unchanged for compatibility.

## CLI

Extend the `training.py predict` command with an optional `--end-stop-sequence` argument. When omitted, the command behaves exactly as it does today. When provided, the command treats `--stop-sequence` as the current stop sequence and prints one predicted delay per stop through `--end-stop-sequence`.

Update `info_gather.py` to ask for the current stop sequence and ending stop sequence, then print the full range.

## Validation

Reject invalid ranges with clear `ValueError`s:

- `end_stop_sequence` must be greater than `current_stop_sequence`.
- The route must exist in the model's `route_to_idx`.
- The current stop and all predicted stop sequences must exist in the representative scheduled trip.

## Testing

Add tests that verify:

- A range from stop 3 to stop 5 returns predictions for stops 4 and 5.
- The prediction for stop 4 is fed into the input window used for stop 5.
- Invalid ranges are rejected.
- Existing tensor generation behavior still works.

The tests should use real code paths and a deterministic small model or fake predictor where needed to make recursive behavior observable.
