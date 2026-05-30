import datetime

import pandas as pd

from kego.cli.commands.ls import format_table


def test_format_table_basic():
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    runs = pd.DataFrame(
        [
            {
                "tags.kego_id": "abc123",
                "tags.mlflow.runName": "soundscape-v8",
                "tags.kego_target": "cluster",
                "metrics.cmAP": 0.8821,
                "status": "FINISHED",
                "start_time": now - datetime.timedelta(hours=2),
            },
            {
                "tags.kego_id": "def456",
                "tags.mlflow.runName": "soundscape-v7",
                "tags.kego_target": "cluster",
                "metrics.cmAP": 0.8794,
                "status": "FINISHED",
                "start_time": now - datetime.timedelta(days=1),
            },
        ]
    )
    lines = format_table(runs, primary_metric="cmAP")
    assert len(lines) >= 3  # header + separator + rows
    assert "abc123" in lines[2]
    assert "soundscape-v8" in lines[2]
    assert "0.8821" in lines[2]
    assert "def456" in lines[3]


def test_format_table_empty():
    lines = format_table(pd.DataFrame(), primary_metric="cmAP")
    assert lines == ["No experiments found."]


def test_format_table_multi_competition():
    """Runs from two competitions each show their own metric value."""
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    runs = pd.DataFrame(
        [
            {
                "experiment_id": "1",
                "tags.kego_id": "aaa111",
                "tags.mlflow.runName": "soundscape-v9",
                "tags.kego_target": "cluster",
                "tags.kego_primary_metric": "cmAP",
                "metrics.cmAP": 0.9120,
                "status": "FINISHED",
                "start_time": now - datetime.timedelta(hours=1),
            },
            {
                "experiment_id": "2",
                "tags.kego_id": "bbb222",
                "tags.mlflow.runName": "retrain-full-v2",
                "tags.kego_target": "cluster",
                "tags.kego_primary_metric": "auc",
                "metrics.auc": 0.9538,
                "status": "FINISHED",
                "start_time": now - datetime.timedelta(hours=3),
            },
        ]
    )
    exp_names = {"1": "birdclef-2026", "2": "playground-s6e2"}
    lines = format_table(runs, primary_metric="cmAP", exp_names=exp_names)

    birdclef_row = lines[2]
    playground_row = lines[3]

    assert "0.9120" in birdclef_row
    assert "birdclef-2026" in birdclef_row
    # playground run shows its own auc, not cmAP
    assert "0.9538" in playground_row
    assert "playground-s6e2" in playground_row
    # neither value appears in the wrong row
    assert "0.9538" not in birdclef_row
    assert "0.9120" not in playground_row
