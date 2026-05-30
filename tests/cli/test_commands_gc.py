import argparse
import datetime

import mlflow
import pytest

from kego.cli.commands.gc import _gc


def _make_gc_args(**overrides) -> argparse.Namespace:
    defaults = dict(older_than="1h", dry_run=False)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture
def mlflow_db(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri("")


def _running_run(name: str, *, hours_ago: float = 0) -> str:
    """Create a RUNNING run (not terminated). Returns run_id."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    exp = mlflow.set_experiment("test-exp")
    start_ms = int(
        (datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(hours=hours_ago)).timestamp() * 1000
    )
    run = client.create_run(exp.experiment_id, run_name=name, start_time=start_ms)
    return run.info.run_id


def test_gc_kills_old_running_run(mlflow_db):
    from mlflow.tracking import MlflowClient

    run_id = _running_run("zombie-run", hours_ago=3)

    rc = _gc(_make_gc_args(older_than="1h"), [])
    assert rc == 0
    assert MlflowClient().get_run(run_id).info.status == "KILLED"


def test_gc_spares_recent_running_run(mlflow_db):
    from mlflow.tracking import MlflowClient

    run_id = _running_run("active-run", hours_ago=0)

    rc = _gc(_make_gc_args(older_than="1h"), [])
    assert rc == 0
    assert MlflowClient().get_run(run_id).info.status == "RUNNING"


def test_gc_dry_run_makes_no_changes(mlflow_db):
    from mlflow.tracking import MlflowClient

    run_id = _running_run("zombie-run", hours_ago=3)

    rc = _gc(_make_gc_args(older_than="1h", dry_run=True), [])
    assert rc == 0
    assert MlflowClient().get_run(run_id).info.status == "RUNNING"


def test_gc_no_zombies_returns_zero(mlflow_db, capsys):
    rc = _gc(_make_gc_args(older_than="1h"), [])
    out = capsys.readouterr().out
    assert rc == 0
    assert "No zombie" in out


def test_gc_invalid_duration_returns_error(mlflow_db, capsys):
    rc = _gc(_make_gc_args(older_than="2weeks"), [])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Invalid" in out
