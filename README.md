# Seattle Bus Delay Predictor

This project collects King County Metro vehicle snapshots from the OneBusAway API
and experiments with neural networks that predict a vehicle's next reported delay.
It supports an embedding-based LSTM and a feed-forward MLP candidate. Route,
direction, and next-stop IDs are categorical embeddings; numeric features include
current delay, distance, stop sequence, speed, delay change, update age, and cyclic
time of day.

# Performance
Due to the lack of data, the current matrics are:
MAE: 13.18 seconds; RMSE: 95.12 seconds; Smooth L1: 13.02.
More data is needed for improving the performance

The current snapshot collection is suitable for exercising the pipeline, but it
is too short to choose a reliable production model. The 146 available snapshots
cover about 87 minutes across four collection sessions. Continue collecting across
full days, weekdays, weekends, peak hours, and unusual traffic conditions before
comparing model quality.

## What the model predicts

Models trained from `data/oba_snapshots` predict the **next OneBusAway snapshot's
`scheduleDeviation`**, usually about 30 seconds ahead. This is not the same target
as measured arrival delay at a future stop. Snapshot checkpoints reject the legacy
future-stop rollout commands so the two targets cannot be mixed silently.

The old stop-based prediction functions and checkpoints remain loadable for
compatibility. A trustworthy future-stop model will need labels aligned to actual
stop arrival or departure events.

## Project layout

```text
bus_data/
├── bus_delay/
│   ├── models.py       # Legacy LSTM and embedded LSTM/MLP architectures
│   ├── trainer.py      # Training loop, evaluation, early stopping, checkpoints
│   ├── prediction.py   # Snapshot and legacy future-stop inference
│   └── cli.py          # Command-line parsing and dispatch
├── process.py          # GTFS/OBA loading, feature engineering, data splits
├── training.py         # Compatible CLI and public API re-exports
├── collect_oba.py      # Continuous OneBusAway snapshot collector
├── info_gather.py      # Interactive legacy future-stop prompt
├── tests/              # Functional and compatibility tests
├── docs/               # Design notes and experiment details
├── data/               # GTFS files, realtime feeds, and collected snapshots
└── models/             # Saved PyTorch checkpoints
```

`training.py` and `process.py` remain stable entry points for existing code. New
Python code can import focused modules such as `bus_delay.models` or
`bus_delay.prediction`.

## Setup

Use Python 3.11 or 3.12. Create a virtual environment and install the dependencies:

```powershell
py -3.12 -m venv .venv
./.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

The data directory must include matching GTFS `trips.txt` and `stop_times.txt`
files. Collected snapshots use this layout:

```text
data/oba_snapshots/YYYY-MM-DD/vehicles-YYYYMMDDTHHMMSSZ.json
```

The loader accepts the collector's wrapper format and raw OneBusAway payloads.
It skips incomplete JSON files with a warning, which is useful if processing runs
while the collector is writing its latest snapshot.

## Collect data

Set the API key and start the collector:

```powershell
$env:ONEBUSAWAY_API_KEY = "your-key"
python collect_oba.py
```

The collector polls every 30 seconds. Stop it with `Ctrl+C`. Avoid committing API
keys or treating a few dense minutes of snapshots as independent training days.

## Train candidates

Train candidates against identical chronological splits and use distinct output
paths:

```powershell
python training.py train --architecture lstm --model-path models/embedded_lstm.pt
python training.py train --architecture mlp --model-path models/embedded_mlp.pt
```

Defaults reserve 15% of observation timestamps for validation and 15% for test.
Features and windows are constructed independently in each partition. Sequences
never cross vehicle, trip, service-date, or gaps longer than 120 seconds. Early
stopping restores the epoch with the lowest validation MAE.

Each checkpoint stores:

- category vocabularies, with index zero reserved for unseen values;
- numeric feature means and standard deviations fitted only on training data;
- split boundaries, model configuration, and target definition;
- validation history, test metrics, and the persistence baseline.

Do not select a model from the current short collection. Once enough data exists,
use whole later days as validation and test periods and compare every candidate
against persistence (`next delay = latest delay`).

## Evaluate and predict

Re-evaluate a saved checkpoint on its frozen test interval:

```powershell
python training.py test --model-path models/embedded_lstm.pt
```

Predict the next saved-observation delay for a vehicle and static GTFS trip ID:

```powershell
python training.py predict-snapshot `
  --vehicle-id 1_1018 `
  --trip-id 635447420 `
  --model-path models/embedded_lstm.pt
```

This uses the latest matching history already on disk; it does not fetch live API
data. Use IDs that occur in the processed snapshots.

Legacy stop-target checkpoints can still use:

```powershell
python training.py predict `
  --route-id 10 `
  --stop-sequence 3 `
  --end-stop-sequence 8 `
  --current-delay 45 `
  --model-path models/bus_model_checkpoint.pt
```

## Tests

Run the functional suite with:

```powershell
python -m pytest -q
```

The suite checks snapshot parsing, leakage-safe splits, unseen categories,
embedding gradients, both model architectures, early stopping, inference,
recursive legacy predictions, and legacy checkpoint loading. Its tiny synthetic
training checks code behavior; they are not model-performance experiments.

More details about the embedding experiments and current data limitations are in
[docs/embedding-models.md](docs/embedding-models.md).
