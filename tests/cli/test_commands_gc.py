import argparse
import datetime
from unittest.mock import patch

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


@pytest.fixture(autouse=True)
def no_ray():
    """Default: Ray returns no jobs, so reconcile is a no-op unless a test overrides it."""
    with patch("kego.cli.commands.gc._ray_job_statuses", return_value={}):
        yield


def _running_run(name: str, *, hours_ago: float = 0, submission_id: str | None = None) -> str:
    """Create a RUNNING run (not terminated). Returns run_id."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    exp = mlflow.set_experiment("test-exp")
    start_ms = int(
        (datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(hours=hours_ago)).timestamp() * 1000
    )
    tags = {"ray_submission_id": submission_id} if submission_id else None
    run = client.create_run(exp.experiment_id, run_name=name, start_time=start_ms, tags=tags)
    return run.info.run_id


def _status(run_id: str) -> str:
    from mlflow.tracking import MlflowClient

    return MlflowClient().get_run(run_id).info.status


# --- time-based fallback (no live Ray job) ---


def test_gc_kills_old_running_run_with_no_ray_job(mlflow_db):
    run_id = _running_run("zombie-run", hours_ago=3)
    rc = _gc(_make_gc_args(older_than="1h"), [])
    assert rc == 0
    assert _status(run_id) == "KILLED"


def test_gc_spares_recent_running_run_with_no_ray_job(mlflow_db):
    run_id = _running_run("active-run", hours_ago=0)
    _gc(_make_gc_args(older_than="1h"), [])
    assert _status(run_id) == "RUNNING"


def test_gc_dry_run_makes_no_changes(mlflow_db):
    run_id = _running_run("zombie-run", hours_ago=3)
    _gc(_make_gc_args(older_than="1h", dry_run=True), [])
    assert _status(run_id) == "RUNNING"


# --- Ray reconciliation (the zombie that's recent but Ray says done) ---


def test_gc_reconciles_stopped_ray_job_to_killed(mlflow_db):
    # Recent run (time-kill would NOT catch it) but its Ray job stopped → must be killed.
    run_id = _running_run("v7-fold0", hours_ago=0, submission_id="raysubmit_abc")
    with patch("kego.cli.commands.gc._ray_job_statuses", return_value={"raysubmit_abc": "STOPPED"}):
        _gc(_make_gc_args(), [])
    assert _status(run_id) == "KILLED"


def test_gc_reconciles_succeeded_ray_job_to_finished(mlflow_db):
    run_id = _running_run("v7-fold3", hours_ago=0, submission_id="raysubmit_ok")
    with patch("kego.cli.commands.gc._ray_job_statuses", return_value={"raysubmit_ok": "SUCCEEDED"}):
        _gc(_make_gc_args(), [])
    assert _status(run_id) == "FINISHED"


def test_gc_leaves_run_with_live_ray_job(mlflow_db):
    # Old enough to time-kill, but Ray says it's still RUNNING → must be spared.
    run_id = _running_run("still-going", hours_ago=3, submission_id="raysubmit_live")
    with patch("kego.cli.commands.gc._ray_job_statuses", return_value={"raysubmit_live": "RUNNING"}):
        _gc(_make_gc_args(older_than="1h"), [])
    assert _status(run_id) == "RUNNING"


# --- edge cases ---


def test_gc_no_running_runs_returns_zero(mlflow_db, capsys):
    rc = _gc(_make_gc_args(), [])
    out = capsys.readouterr().out
    assert rc == 0
    assert "No RUNNING runs" in out


def test_gc_invalid_duration_returns_error(mlflow_db, capsys):
    rc = _gc(_make_gc_args(older_than="2weeks"), [])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Invalid" in out
