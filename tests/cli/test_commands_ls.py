import argparse
import datetime
import secrets

import mlflow
import pandas as pd
import pytest

from kego.cli.commands.ls import _ls, format_table

# ---------------------------------------------------------------------------
# Helpers shared by _ls filter tests
# ---------------------------------------------------------------------------


def _make_ls_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        name=None,
        status=None,
        target=None,
        competition=None,
        limit=50,
        show_all=False,
        show_metric_name=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture
def mlflow_db(tmp_path, monkeypatch):
    """Isolated SQLite MLflow backend; resets global tracking URI after test."""
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri("")


def _create_run(
    name: str,
    *,
    target: str = "cluster",
    status: str = "FINISHED",
    experiment: str = "test-exp",
) -> None:
    mlflow.set_experiment(experiment)
    mlflow.start_run(
        run_name=name,
        tags={"kego_id": secrets.token_hex(3), "kego_target": target, "kego_debug": "false"},
    )
    mlflow.end_run(status=status)


# ---------------------------------------------------------------------------
# _ls filter behaviour tests (real SQLite MLflow backend)
# ---------------------------------------------------------------------------


def test_ls_status_filter_shows_only_matching_status(mlflow_db, capsys):
    _create_run("finished-run", status="FINISHED")
    _create_run("failed-run", status="FAILED")

    rc = _ls(_make_ls_args(status="finished"), [])
    out = capsys.readouterr().out
    assert rc == 0
    assert "finished-run" in out
    assert "failed-run" not in out


def test_ls_target_filter_shows_only_matching_target(mlflow_db, capsys):
    _create_run("cluster-run", target="cluster")
    _create_run("local-run", target="local")

    _ls(_make_ls_args(target="cluster"), [])
    out = capsys.readouterr().out
    assert "cluster-run" in out
    assert "local-run" not in out


def test_ls_name_filter_shows_only_matching_prefix(mlflow_db, capsys):
    _create_run("soundscape-v7")
    _create_run("retrain-full-v2")

    _ls(_make_ls_args(name="soundscape"), [])
    out = capsys.readouterr().out
    assert "soundscape-v7" in out
    assert "retrain-full-v2" not in out


def test_ls_competition_filter_scopes_to_experiment(mlflow_db, capsys):
    _create_run("bird-run", experiment="birdclef-2026")
    _create_run("play-run", experiment="playground-s6e2")

    rc = _ls(_make_ls_args(competition="birdclef-2026"), [])
    out = capsys.readouterr().out
    assert rc == 0
    assert "bird-run" in out
    assert "play-run" not in out


def test_ls_competition_not_found_returns_zero_with_message(mlflow_db, capsys):
    rc = _ls(_make_ls_args(competition="nonexistent-comp"), [])
    out = capsys.readouterr().out
    assert rc == 0
    assert "nonexistent-comp" in out


def test_ls_limit_caps_number_of_runs_shown(mlflow_db, capsys):
    names = [f"limit-run-{i}" for i in range(5)]
    for name in names:
        _create_run(name)

    rc = _ls(_make_ls_args(limit=2), [])
    out = capsys.readouterr().out
    assert rc == 0
    assert sum(name in out for name in names) <= 2


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
