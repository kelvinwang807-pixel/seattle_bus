# Multi-Stop Delay Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add recursive multi-stop bus delay prediction using the existing one-step LSTM.

**Architecture:** Keep the current model and checkpoint format unchanged. Add a range prediction helper in `training.py` that loads the same representative trip rows, rolls a feature window forward one stop at a time, and feeds each predicted delay into the next step. Keep single-stop prediction compatible.

**Tech Stack:** Python, pandas, PyTorch, argparse, pytest-compatible test functions.

## Global Constraints

- Use approach 1: recursive multi-stop inference with the existing one-step LSTM.
- Example input `current_stop_sequence=3`, `current_delay=120`, `end_stop_sequence=9` returns predictions for stops 4, 5, 6, 7, 8, and 9.
- Do not change the saved checkpoint architecture or require retraining.
- Preserve existing `predict_future_delay(...)` behavior.

---

### Task 1: Recursive Range Prediction Core

**Files:**
- Modify: `training.py`
- Create: `tests/test_training_multi_stop.py`

**Interfaces:**
- Consumes: existing `LSTM`, `_representative_route_rows(...)`, `_standardize(...)`, and `predict(...)`.
- Produces:
  - `build_prediction_window_from_history(history: pd.DataFrame, checkpoint: dict[str, Any]) -> torch.Tensor`
  - `predict_future_delay_range(route_id: str, current_stop_sequence: int, end_stop_sequence: int, current_delay: float, model_path: str | Path = MODEL_PATH, data_dir: str | Path = "data", direction_id: int | None = None) -> dict[int, float]`

- [ ] **Step 1: Write the failing recursive behavior test**

```python
import pandas as pd
import pytest
import torch

import process
import training


class LastDelayPlusOneModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1, process.FEATURE_COLUMNS.index("delay")].unsqueeze(1) + 1.0


def checkpoint() -> dict:
    return {
        "window": 2,
        "feature_mean": torch.zeros(1, 1, len(process.FEATURE_COLUMNS)),
        "feature_std": torch.ones(1, 1, len(process.FEATURE_COLUMNS)),
        "route_to_idx": {"10": 0},
    }


def route_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"trip_id": "t1", "route_id": "10", "arrival_min": 1, "shape_dist_traveled": 0.0, "delay": 0.0, "stop_sequence": 1, "route_idx": 0, "direction_id": 0},
            {"trip_id": "t1", "route_id": "10", "arrival_min": 2, "shape_dist_traveled": 1.0, "delay": 0.0, "stop_sequence": 2, "route_idx": 0, "direction_id": 0},
            {"trip_id": "t1", "route_id": "10", "arrival_min": 3, "shape_dist_traveled": 2.0, "delay": 0.0, "stop_sequence": 3, "route_idx": 0, "direction_id": 0},
            {"trip_id": "t1", "route_id": "10", "arrival_min": 4, "shape_dist_traveled": 3.0, "delay": 0.0, "stop_sequence": 4, "route_idx": 0, "direction_id": 0},
            {"trip_id": "t1", "route_id": "10", "arrival_min": 5, "shape_dist_traveled": 4.0, "delay": 0.0, "stop_sequence": 5, "route_idx": 0, "direction_id": 0},
        ]
    )


def test_predict_future_delay_range_feeds_each_prediction_into_the_next_window(monkeypatch):
    monkeypatch.setattr(training, "load_or_train", lambda model_path, data_dir: (LastDelayPlusOneModel(), checkpoint()))
    monkeypatch.setattr(training, "_representative_route_rows", lambda route_id, stop_sequence, route_to_idx, data_dir, direction_id: route_rows())

    result = training.predict_future_delay_range(
        route_id="10",
        current_stop_sequence=3,
        end_stop_sequence=5,
        current_delay=10.0,
    )

    assert result == {4: 11.0, 5: 12.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_training_multi_stop.py::test_predict_future_delay_range_feeds_each_prediction_into_the_next_window -q`

Expected: FAIL because `training.predict_future_delay_range` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def build_prediction_window_from_history(history: pd.DataFrame, checkpoint: dict[str, Any]) -> torch.Tensor:
    window = int(checkpoint.get("window", DEFAULT_WINDOW))
    while len(history) < window:
        history = pd.concat([history.head(1), history], ignore_index=True)
    values = history.tail(window)[process.FEATURE_COLUMNS].to_numpy(dtype="float32")
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


def predict_future_delay_range(...):
    if end_stop_sequence <= current_stop_sequence:
        raise ValueError("end_stop_sequence must be greater than current_stop_sequence")
    model, checkpoint = load_or_train(model_path=model_path, data_dir=data_dir)
    route_to_idx = {str(k): int(v) for k, v in checkpoint.get("route_to_idx", {}).items()}
    route_rows = _representative_route_rows(route_id, current_stop_sequence, route_to_idx, data_dir, direction_id)
    history = route_rows[route_rows["stop_sequence"] <= int(current_stop_sequence)].copy()
    if history.empty:
        raise ValueError(...)
    history.loc[history.index[-1], "delay"] = float(current_delay)
    predictions = {}
    for next_stop in range(current_stop_sequence + 1, end_stop_sequence + 1):
        stop_rows = route_rows[route_rows["stop_sequence"] == next_stop]
        if stop_rows.empty:
            raise ValueError(...)
        x = build_prediction_window_from_history(history, checkpoint)
        predicted_delay = float(predict(model, x, checkpoint).item())
        predictions[next_stop] = predicted_delay
        next_row = stop_rows.iloc[[0]].copy()
        next_row["delay"] = predicted_delay
        history = pd.concat([history, next_row], ignore_index=True)
    return predictions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_training_multi_stop.py::test_predict_future_delay_range_feeds_each_prediction_into_the_next_window -q`

Expected: PASS.

- [ ] **Step 5: Add invalid range test**

```python
def test_predict_future_delay_range_rejects_end_stop_before_next_stop():
    with pytest.raises(ValueError, match="end_stop_sequence must be greater"):
        training.predict_future_delay_range(
            route_id="10",
            current_stop_sequence=3,
            end_stop_sequence=3,
            current_delay=10.0,
        )
```

- [ ] **Step 6: Run task tests**

Run: `python -m pytest tests/test_training_multi_stop.py -q`

Expected: PASS.

---

### Task 2: CLI and Interactive Prompt Integration

**Files:**
- Modify: `training.py`
- Modify: `info_gather.py`
- Test: `tests/test_training_multi_stop.py`

**Interfaces:**
- Consumes: `predict_future_delay_range(...) -> dict[int, float]`
- Produces: CLI `training.py predict --end-stop-sequence N` and interactive range prompt output.

- [ ] **Step 1: Write failing CLI parser test**

```python
def test_predict_parser_accepts_end_stop_sequence(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "training.py",
            "predict",
            "--route-id",
            "10",
            "--stop-sequence",
            "3",
            "--end-stop-sequence",
            "5",
            "--current-delay",
            "10",
        ],
    )

    args = training._parse_args()

    assert args.end_stop_sequence == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_training_multi_stop.py::test_predict_parser_accepts_end_stop_sequence -q`

Expected: FAIL because `_parse_args()` does not define `end_stop_sequence`.

- [ ] **Step 3: Implement parser and print behavior**

Add:

```python
predict_parser.add_argument("--end-stop-sequence", type=int)
```

In `main()`, when `args.end_stop_sequence is not None`, call `predict_future_delay_range(...)` and print:

```python
for stop_sequence, delay in predictions.items():
    print(f"Predicted delay at stop_sequence {stop_sequence}: {delay:.2f} seconds")
```

- [ ] **Step 4: Update `info_gather.py` prompts**

Prompt for:

```python
current_stop_sequence = _prompt_int("Current stop sequence")
end_stop_sequence = _prompt_int("End stop sequence")
```

Call `training.predict_future_delay_range(...)` and print each returned stop prediction.

- [ ] **Step 5: Run parser test and compile checks**

Run: `python -m pytest tests/test_training_multi_stop.py::test_predict_parser_accepts_end_stop_sequence -q`

Expected: PASS.

Run: `python -m py_compile info_gather.py process.py training.py`

Expected: no output and exit code 0.

---

### Task 3: Full Verification and Commit

**Files:**
- Modify: `training.py`
- Modify: `info_gather.py`
- Create: `tests/test_training_multi_stop.py`
- Create: `docs/superpowers/specs/2026-07-28-multi-stop-delay-prediction-design.md`
- Create: `docs/superpowers/plans/2026-07-28-multi-stop-delay-prediction.md`

**Interfaces:**
- Consumes: completed Tasks 1 and 2.
- Produces: verified multi-stop prediction feature.

- [ ] **Step 1: Run all focused tests**

Run: `python -m pytest tests/test_training_multi_stop.py -q`

Expected: all tests pass.

- [ ] **Step 2: Run compile verification**

Run: `python -m py_compile info_gather.py process.py training.py tests/test_training_multi_stop.py`

Expected: no output and exit code 0.

- [ ] **Step 3: Run single-stop smoke command**

Run: `python training.py predict --route-id 100001 --stop-sequence 8 --current-delay 120 --model-path models/bus_model_checkpoint.pt`

Expected: prints one predicted delay or a clear data validation error for unavailable route/stop data.

- [ ] **Step 4: Run multi-stop smoke command**

Run: `python training.py predict --route-id 100001 --stop-sequence 3 --end-stop-sequence 9 --current-delay 120 --model-path models/bus_model_checkpoint.pt`

Expected: prints one predicted delay per stop from 4 through 9, or a clear data validation error for unavailable route/stop data.

- [ ] **Step 5: Commit**

```bash
git add training.py info_gather.py tests/test_training_multi_stop.py docs/superpowers/specs/2026-07-28-multi-stop-delay-prediction-design.md docs/superpowers/plans/2026-07-28-multi-stop-delay-prediction.md
git commit -m "feat: add multi-stop delay prediction"
```
