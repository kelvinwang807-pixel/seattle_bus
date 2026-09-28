# Embedded bus delay models

Train separate candidates with the same data, seed and split fractions:

```powershell
python training.py train --architecture lstm --model-path models/embedded_lstm.pt
python training.py train --architecture mlp --model-path models/embedded_mlp.pt
python training.py test --model-path models/embedded_lstm.pt
```

Install dependencies with `python -m pip install -r requirements.txt` in a working Python environment.
The original checkpoint is not overwritten by the commands above. Old checkpoints
remain loadable, but cannot gain embeddings without retraining.

Both models embed route ID, direction ID and next-stop ID. Numeric inputs include
delay, distance, stop sequence, elapsed time, speed, delay change, update age and
Seattle-local cyclic time of day. The LSTM processes the sequence recurrently;
the MLP uses the flattened history window and two hidden layers. Both predict a
correction to the latest delay, starting at the persistence baseline. Neither
architecture is assumed to be more accurate until evaluated.

## Snapshot compatibility and target

The loader reads `data/oba_snapshots/**/vehicles-*.json`, including the collector's
`payload` wrapper and raw OneBusAway responses. It requires static `trips.txt` and
`stop_times.txt` from the matching GTFS feed. Invalid/incomplete JSON files are
skipped with a warning, allowing collection to continue. Unknown categories at
evaluation/inference map to index zero; vocabulary and numeric scaling are fitted
only on training data and saved in each checkpoint.

Snapshot labels are the next observed OneBusAway `scheduleDeviation`, not measured
arrival delays at a future stop. The horizon follows the collection cadence (the
collector defaults to 30 seconds); gaps over 120 seconds are excluded. Repeated
upstream values can make persistence particularly strong. A future-stop predictor
requires a separate stop-arrival labeling scheme and horizon evaluation.

Predict from saved history for a vehicle/trip:

```powershell
python training.py predict-snapshot --vehicle-id 1_1018 --trip-id 635447420 --model-path models/embedded_lstm.pt
```

Use actual IDs present in the saved data. This forecasts after that vehicle's last
saved observation, which may be historical. It does not fetch live data. In Python,
`build_prediction_window_from_history(history, checkpoint)` followed by
`predict(model, window, checkpoint)` accepts a processed vehicle history frame.
Snapshot checkpoints reject the old stop-rollout API instead of silently treating
one observation step as one stop. `info_gather.py` remains a stop-rollout interface
for legacy/GTFS-stop models; use `predict-snapshot` for these new snapshot models.

## Splits and selection

Defaults are 70% train, 15% validation, 15% test, divided by unique observation
timestamps. Windows and causal motion features are built separately in each
partition. Sequences never cross vehicle/trip/service-date boundaries or large
gaps. The same ongoing trip may appear in different chronological partitions,
but their observations and windows do not overlap. This measures future-time
forecasting on the network, not generalization to entirely unseen routes.

For static GTFS fallback data, whole trips are split reproducibly. With only a
single realtime feed, those labels may themselves be upstream predictions.

`process.to_tensor(df, validation_fraction=0.15, test_fraction=0.15)` returns six
tensors in train/validation/test order. Omitting validation_fraction preserves
the original four-tensor API. New training uses `prepare_embedding_data`, which
also returns the saved vocabularies and split boundaries.

Training stops after `--patience 8` epochs without validation MAE improvement and
restores the best validation epoch. Checkpoints include validation history, test
MAE/RMSE/Smooth L1, and test persistence MAE/RMSE. Select the candidate using
validation MAE, then inspect its test performance. Saved test boundaries remain
fixed when additional, later snapshots arrive; retraining establishes a new split.
Reproducible comparisons require the same snapshot collection (pause collection
or train against a fixed copy). Short partitions without a complete history plus
target fail explicitly; collect more data or reduce `--window`.

## Data sufficiency and smoke runs

This collection is sufficient for functional checks, but does not establish a
reliable comparison between model architectures. It contains only 146 distinct
observation times across four short collection sessions, totaling approximately
86.7 minutes of covered intervals (counting consecutive intervals up to 120
seconds). The dates span August 19–23, but these are not full days of data.
There are 2,454 vehicle/trip/service-date groups; only 1,248 have at least seven
rows. The 46,021 rows include 43,092 distinct vehicle/trip/service-date/update-time
observations. Median route coverage is 339 rows, and 35 of 119 routes have fewer
than 100 rows. Overlapping windows and buses sharing the same short time periods
do not constitute independent evidence across traffic conditions.

The validation interval is August 23, approximately 07:55–08:09 Seattle time;
the test interval is approximately 08:09–08:23. These short adjacent intervals
cannot support a dependable model choice. Treat the following numbers and saved
candidate checkpoints as smoke-run artifacts only. Further real-data training
and model comparisons were stopped; functional tests and checkpoint/inference
checks already passed. Collect broader coverage across days, times and routes
before resuming performance experiments, and reserve whole later days for that
evaluation.

On the 146 available snapshot files (46,021 usable rows, 119 routes), the default
window and fractions produced 22,255 training, 4,002 validation and 4,105 test
windows. A run capped at 10 epochs with seed 42 produced these results after
restoring the best validation epoch:

| Candidate | Validation MAE (s) | Test MAE (s) | Test RMSE (s) |
| --- | ---: | ---: | ---: |
| Embedded LSTM | 13.936 | 19.487 | 106.069 |
| Embedded MLP | 13.900 | 19.470 | 106.341 |
| Persistence | — | 19.443 | 106.075 |

Neither learned candidate improved test MAE over persistence in this short run.
Training MAE decreased while validation MAE increased, so the best checkpoints
were early epochs. The new split differs from the previous random-trip evaluation;
these test scores cannot be compared directly to the earlier 13.18-second MAE.
Saved candidates are `models/embedded_lstm.pt` and `models/embedded_mlp.pt`.
