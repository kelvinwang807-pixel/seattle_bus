import sys

import pandas as pd
import torch
from unittest.mock import patch

import process
import training


class LastDelayPlusOneModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delay_index = process.FEATURE_COLUMNS.index("delay")
        return x[:, -1, delay_index].unsqueeze(1) + 1.0


class NoNanLastDelayPlusOneModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert not torch.isnan(x).any()
        delay_index = process.FEATURE_COLUMNS.index("delay")
        return x[:, -1, delay_index].unsqueeze(1) + 1.0


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
            {
                "trip_id": "t1",
                "route_id": "10",
                "arrival_min": 1,
                "shape_dist_traveled": 0.0,
                "delay": 0.0,
                "stop_sequence": 1,
                "route_idx": 0,
                "direction_id": 0,
            },
            {
                "trip_id": "t1",
                "route_id": "10",
                "arrival_min": 2,
                "shape_dist_traveled": 1.0,
                "delay": 0.0,
                "stop_sequence": 2,
                "route_idx": 0,
                "direction_id": 0,
            },
            {
                "trip_id": "t1",
                "route_id": "10",
                "arrival_min": 3,
                "shape_dist_traveled": 2.0,
                "delay": 0.0,
                "stop_sequence": 3,
                "route_idx": 0,
                "direction_id": 0,
            },
            {
                "trip_id": "t1",
                "route_id": "10",
                "arrival_min": 4,
                "shape_dist_traveled": 3.0,
                "delay": 0.0,
                "stop_sequence": 4,
                "route_idx": 0,
                "direction_id": 0,
            },
            {
                "trip_id": "t1",
                "route_id": "10",
                "arrival_min": 5,
                "shape_dist_traveled": 4.0,
                "delay": 0.0,
                "stop_sequence": 5,
                "route_idx": 0,
                "direction_id": 0,
            },
        ]
    )


def test_predict_future_delay_range_feeds_each_prediction_into_the_next_window():
    with (
        patch.object(
            training,
            "load_or_train",
            lambda model_path, data_dir: (LastDelayPlusOneModel(), checkpoint()),
        ),
        patch.object(
            training,
            "_representative_route_rows",
            lambda route_id, stop_sequence, route_to_idx, data_dir, direction_id: route_rows(),
        ),
    ):
        result = training.predict_future_delay_range(
            route_id="10",
            current_stop_sequence=3,
            end_stop_sequence=5,
            current_delay=10.0,
        )

    assert result == {4: 11.0, 5: 12.0}

def test_predict_future_delay_range_rejects_end_stop_before_next_stop():
    try:
        training.predict_future_delay_range(
            route_id="10",
            current_stop_sequence=3,
            end_stop_sequence=3,
            current_delay=10.0,
        )
    except ValueError as exc:
        assert "end_stop_sequence must be greater" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid prediction range")


def test_predict_parser_accepts_end_stop_sequence():
    with patch.object(
        sys,
        "argv",
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
    ):
        args = training._parse_args()

    assert args.end_stop_sequence == 5

def test_predict_future_delay_range_fills_missing_history_delays_before_predicting():
    rows_without_delay = route_rows().drop(columns=["delay"])

    with (
        patch.object(
            training,
            "load_or_train",
            lambda model_path, data_dir: (NoNanLastDelayPlusOneModel(), checkpoint()),
        ),
        patch.object(
            training,
            "_representative_route_rows",
            lambda route_id, stop_sequence, route_to_idx, data_dir, direction_id: rows_without_delay,
        ),
    ):
        result = training.predict_future_delay_range(
            route_id="10",
            current_stop_sequence=3,
            end_stop_sequence=5,
            current_delay=10.0,
        )

    assert result == {4: 11.0, 5: 12.0}

def test_to_tensor_keeps_existing_window_target_behavior():
    df = route_rows()
    df["delay"] = [0.0, 1.0, 2.0, 3.0, 4.0]

    x, y = process.to_tensor(df, window=2)

    delay_index = process.FEATURE_COLUMNS.index("delay")
    assert x.shape == (3, 2, len(process.FEATURE_COLUMNS))
    assert y.squeeze(1).tolist() == [2.0, 3.0, 4.0]
    assert x[0, :, delay_index].tolist() == [0.0, 1.0]
