import datetime

import pandas as pd

from kego.cli.commands.ls import _ago, format_table


def test_ago_hours():
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    start = now - datetime.timedelta(hours=3)
    assert _ago(start) == "3h"


def test_ago_minutes():
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    start = now - datetime.timedelta(minutes=45)
    assert _ago(start) == "45m"


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
