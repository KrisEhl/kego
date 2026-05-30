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
        since=None,
        limit=50,
        show_all=False,
        show_metric_name=False,
        show_children=False,
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
    hours_ago: float = 0,
) -> None:
    from mlflow.tracking import MlflowClient

    exp = mlflow.set_experiment(experiment)
    if hours_ago:
        # Back-date the run using the low-level client so --since can be tested.
        import datetime as dt

        start_ms = int((dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(hours=hours_ago)).timestamp() * 1000)
        client = MlflowClient()
        run = client.create_run(
            experiment_id=exp.experiment_id,
            run_name=name,
            start_time=start_ms,
            tags={
                "kego_id": secrets.token_hex(3),
                "kego_target": target,
                "kego_debug": "false",
            },
        )
        client.set_terminated(run.info.run_id, status=status)
    else:
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


def test_ls_since_filter_excludes_older_runs(mlflow_db, capsys):
    _create_run("recent-run", hours_ago=0)
    _create_run("old-run", hours_ago=3)

    rc = _ls(_make_ls_args(since="1h"), [])
    out = capsys.readouterr().out
    assert rc == 0
    assert "recent-run" in out
    assert "old-run" not in out


def test_ls_since_invalid_format_returns_error(mlflow_db, capsys):
    rc = _ls(_make_ls_args(since="2weeks"), [])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Invalid" in out


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


def test_format_table_old_run_shows_date_not_hours():
    """Runs older than 24h display a date string, not '48h'."""
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    runs = pd.DataFrame(
        [
            {
                "tags.kego_id": "abc123",
                "tags.mlflow.runName": "old-run",
                "tags.kego_target": "cluster",
                "metrics.auc": 0.9,
                "status": "FINISHED",
                "start_time": now - datetime.timedelta(days=2),
            }
        ]
    )
    lines = format_table(runs, primary_metric="auc")
    assert "48h" not in lines[2]
    assert "202" in lines[2]  # year prefix e.g. 2026-…


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


def test_ls_hides_child_runs_by_default(mlflow_db, capsys):
    """Child runs (mlflow.parentRunId set) are hidden unless --children is passed."""
    import secrets

    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    exp = mlflow.set_experiment("test-exp")

    parent = client.create_run(
        exp.experiment_id,
        run_name="parent-run",
        tags={
            "kego_id": secrets.token_hex(3),
            "kego_target": "cluster",
            "kego_debug": "false",
            "kego_is_parent": "true",
            "kego_fold_count": "2",
        },
    )
    client.set_terminated(parent.info.run_id, status="FINISHED")

    for fold in range(2):
        child = client.create_run(
            exp.experiment_id,
            run_name=f"parent-run fold={fold}",
            tags={
                "kego_id": secrets.token_hex(3),
                "kego_target": "cluster",
                "kego_debug": "false",
                "mlflow.parentRunId": parent.info.run_id,
            },
        )
        client.log_param(child.info.run_id, "fold", str(fold))
        client.set_terminated(child.info.run_id, status="FINISHED")

    _ls(_make_ls_args(), [])
    out = capsys.readouterr().out
    assert "parent-run" in out
    # Child rows must not appear
    assert "fold=0" not in out
    assert "fold=1" not in out


def test_ls_children_flag_shows_fold_rows(mlflow_db, capsys):
    """--children shows child fold runs with a FOLD column."""
    import secrets

    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    exp = mlflow.set_experiment("test-exp")

    parent = client.create_run(
        exp.experiment_id,
        run_name="parent-run",
        tags={
            "kego_id": secrets.token_hex(3),
            "kego_target": "cluster",
            "kego_debug": "false",
            "kego_is_parent": "true",
            "kego_fold_count": "2",
        },
    )
    client.set_terminated(parent.info.run_id, status="FINISHED")

    for fold in range(2):
        child = client.create_run(
            exp.experiment_id,
            run_name=f"parent-run fold={fold}",
            tags={
                "kego_id": secrets.token_hex(3),
                "kego_target": "cluster",
                "kego_debug": "false",
                "mlflow.parentRunId": parent.info.run_id,
            },
        )
        client.log_param(child.info.run_id, "fold", str(fold))
        client.set_terminated(child.info.run_id, status="FINISHED")

    _ls(_make_ls_args(show_children=True), [])
    out = capsys.readouterr().out
    # FOLD column header present
    assert "FOLD" in out
    # Parent shows fold count
    assert "2×" in out  # noqa: RUF001
    # Children show fold numbers
    assert "f0" in out
    assert "f1" in out
