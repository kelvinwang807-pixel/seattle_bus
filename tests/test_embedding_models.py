"""Functional tests for embedded models and snapshot preprocessing."""

import json

import pandas as pd
import pytest
import torch

import process
import training


def observations():
    return pd.DataFrame([
        dict(observation_time=1_780_000_000 + i * 30, vehicle_id="v1", trip_id="t1",
             service_date="day1", route_id="r1" if i < 70 else "unseen",
             direction_id=0, next_stop_id="s1" if i < 70 else "new-stop",
             arrival_min=100+i/2, shape_dist_traveled=i*100, delay=10+i%5,
             stop_sequence=i//3, route_idx=0)
        for i in range(100)
    ])


def test_chronological_split_unknown_categories_and_frozen_test():
    df = observations()
    partitions, metadata = process.prepare_embedding_data(df, window=3)
    assert [len(x) for x, _ in partitions] == [67, 12, 12]
    assert "unseen" not in metadata["vocabularies"]["route_id"]
    assert torch.all(partitions[1][0][..., -3] == 0)
    assert partitions[0][0][..., 0].max() < partitions[1][0][..., 0].min()
    later = df.copy()
    later.observation_time += 100000
    new_parts, _ = process.prepare_embedding_data(pd.concat([df, later]), window=3, metadata=metadata)
    assert torch.equal(new_parts[2][0], partitions[2][0])
    six = process.to_tensor(df, window=3, validation_fraction=.15, test_fraction=.15)
    assert len(six) == 6


def test_windows_never_cross_gaps_or_service_dates():
    df = observations().iloc[:8].copy()
    df.loc[df.index[4:], "service_date"] = "day2"
    x, _ = process.sequence_tensors(df, process.FEATURE_COLUMNS, window=3)
    assert len(x) == 2
    df.service_date = "day1"
    df.loc[df.index[4:], "observation_time"] += 1000
    x, _ = process.sequence_tensors(df, process.FEATURE_COLUMNS, window=3)
    assert len(x) == 2


@pytest.mark.parametrize("architecture", ["lstm", "mlp"])
def test_train_reload_and_predict_with_embeddings(tmp_path, monkeypatch, architecture):
    torch.set_num_threads(1)
    df = observations()
    monkeypatch.setattr(process, "process_features", lambda _: df)
    path = tmp_path / "candidate.pt"
    cp = training.train_and_save(model_path=path, window=3, epochs=2, architecture=architecture)
    model, loaded = training.load_checkpoint(path)
    assert loaded["model_config"]["architecture"] == architecture
    assert len(cp["validation_losses"]) == 2
    assert cp["validation_metrics"]["mae_seconds"] == pytest.approx(min(cp["validation_losses"]), abs=1e-5)
    assert cp["feature_mean"][..., -3:].eq(0).all()
    assert cp["feature_std"][..., -3:].eq(1).all()
    history = df.iloc[-6:]
    x = training.build_prediction_window_from_history(history, loaded)
    output = training.predict(model, x, loaded)
    assert output.shape == (1, 1) and torch.isfinite(output).all()
    metrics = training.test_accuracy(path)
    assert metrics["mae_seconds"] == pytest.approx(cp["test_metrics"]["mae_seconds"])
    # Unknown IDs have a safe zero embedding; known IDs receive gradients.
    model.train()
    parts, _ = process.prepare_embedding_data(df, window=3)
    x = training._standardize(parts[0][0], cp["feature_mean"], cp["feature_std"])
    model(x).sum().backward()
    assert model.embeddings[0].weight.grad[1].abs().sum() > 0


def test_snapshot_wrapper_matches_collector(tmp_path, monkeypatch):
    schedule = pd.DataFrame([dict(trip_id="t1", stop_id="s1", stop_sequence=1,
                                  route_id="r1", direction_id=0)])
    monkeypatch.setattr(process, "load_schedule", lambda _: schedule)
    directory = tmp_path / "oba_snapshots" / "2026-08-19"
    directory.mkdir(parents=True)
    status = dict(predicted=True, lastUpdateTime=1_780_000_000_000,
                  nextStop="1_s1", scheduleDeviation=25, distanceAlongTrip=100,
                  totalDistanceAlongTrip=1000, serviceDate=1_779_984_000_000)
    payload = dict(code=200, currentTime=1_780_000_030_000,
                   data={"list": [{"tripId": "1_t1", "vehicleId": "1_v1", "tripStatus": status},
                                  {"tripId": "1_t1", "vehicleId": "1_bad", "tripStatus": None}]})
    (directory / "vehicles-test.json").write_text(json.dumps({"payload": payload}), encoding="utf-8")
    df = process.process_features(tmp_path)
    assert len(df) == 1
    assert df.iloc[0].route_id == "r1"
    assert df.iloc[0].next_stop_id == "s1"
    assert df.iloc[0].delay == 25
    assert df.attrs["source"] == "onebusaway_snapshots"


def test_snapshot_checkpoint_rejects_stop_rollout():
    with pytest.raises(ValueError, match="next snapshot"):
        training.build_future_stop_window("r", 3, 10, {"target": "next_observation_delay"})


@pytest.mark.parametrize("architecture", ["lstm", "mlp"])
def test_initial_model_is_persistence(architecture):
    model = training.EmbeddedDelayModel([3, 3, 3], window=3, architecture=architecture,
                                        delay_mean=25, delay_std=10)
    x = torch.zeros(2, 3, len(process.EMBEDDING_COLUMNS))
    x[:, -1, process.NUMERIC_COLUMNS.index("delay")] = torch.tensor([2., -1.])
    assert torch.equal(model(x), torch.tensor([[45.], [15.]]))


def test_validation_restores_best_epoch():
    # Training pushes toward +10 while validation prefers zero. Epoch 1 is
    # therefore best, and patience=1 must stop after epoch 2 then restore it.
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    x = torch.ones(2, 1)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, x*10), batch_size=2)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, x*0), batch_size=2)
    losses = training.train(model, torch.optim.SGD(model.parameters(), lr=.1), train_loader,
                            epochs=10, validation_loader=val_loader, patience=1)
    assert len(losses) == 2
    assert model.weight.item() == pytest.approx(.1, abs=1e-6)


def test_legacy_checkpoint_still_loads(tmp_path):
    original = training.LSTM()
    original.eval()
    path = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": original.state_dict(), "feature_columns": process.FEATURE_COLUMNS}, path)
    loaded, _ = training.load_checkpoint(path)
    x = torch.zeros(1, 6, 6)
    assert torch.equal(original(x), loaded(x))
