import argparse
import datetime
import secrets
from unittest.mock import patch

import mlflow
import pandas as pd

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
        show_ray=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


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


def test_ls_competition_filter_uses_competition_primary_metric(mlflow_db, tmp_path, monkeypatch, capsys):
    root = tmp_path / "repo"
    comp_dir = root / "competitions" / "metric-comp"
    comp_dir.mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "kego.toml").write_text(
        """
[cluster]
ray_address = "http://192.168.1.1:8265"
mlflow_uri = "http://192.168.1.1:5000"
"""
    )
    (comp_dir / "kego.toml").write_text(
        """
[competition]
slug = "metric-comp"
kaggle_user = "aldisued"
enable_gpu = false
submit_file = "submission.csv"
pattern = "script"
inference_notebook = "train.py"
checkpoint_dir = "outputs"
primary_metric = "post_ps_rmse"
"""
    )
    monkeypatch.chdir(root)

    exp = mlflow.set_experiment("metric-comp")
    with mlflow.start_run(experiment_id=exp.experiment_id, run_name="metric-run"):
        mlflow.set_tag("kego_id", secrets.token_hex(3))
        mlflow.set_tag("kego_target", "cluster")
        mlflow.set_tag("kego_debug", "false")
        mlflow.set_tag("kego_primary_metric", "oof_rmse")
        mlflow.log_metric("oof_rmse", 99.0)
        mlflow.log_metric("post_ps_rmse", 1.25)

    rc = _ls(_make_ls_args(competition="metric-comp", show_metric_name=True), [])
    out = capsys.readouterr().out
    assert rc == 0
    assert "metric-run" in out
    assert "1.2500" in out
    assert "post_ps_rmse" in out
    assert "99.0000" not in out


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
                "end_time": now - datetime.timedelta(days=2) + datetime.timedelta(minutes=5),  # ran 5m
            }
        ]
    )
    lines = format_table(runs, primary_metric="auc")
    assert "202" in lines[2]  # AGO shows a date for >24h-old runs, e.g. 2026-…
    assert "5m" in lines[2]  # DURATION column shows the real 5m runtime


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


def _setup_parent_child_runs(mlflow_db: str, fold_count: int = 2) -> None:
    """Create one parent run and `fold_count` child runs in the test MLflow DB."""
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
            "kego_fold_count": str(fold_count),
        },
    )
    client.set_terminated(parent.info.run_id, status="FINISHED")

    for fold in range(fold_count):
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


def test_ls_shows_nested_runs_with_fold_column(mlflow_db, capsys):
    """Parent and child fold runs both appear; FOLD column is auto-added; children indented."""
    _setup_parent_child_runs(mlflow_db)
    _ls(_make_ls_args(), [])
    out = capsys.readouterr().out
    assert "FOLD" in out
    assert "parent-run" in out
    assert "2×" in out  # noqa: RUF001
    assert "f0" in out
    assert "f1" in out
    assert "└─" in out  # child rows are visually indented


def test_ls_filter_fetches_parent_when_only_children_match(mlflow_db, capsys):
    """When a filter returns children but not their parent, the parent is fetched and shown."""
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
    client.set_terminated(parent.info.run_id, status="FINISHED")  # parent is FINISHED

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
        # Leave children RUNNING so --status running returns them but not the parent

    # Filter to running only — parent is FINISHED so it won't be in the MLflow result
    _ls(_make_ls_args(status="running"), [])
    out = capsys.readouterr().out
    # Parent must be injected so children render with context
    assert "parent-run" in out
    assert "└─" in out


def test_ls_parent_status_is_derived_from_children(mlflow_db, capsys):
    """A pre-created FINISHED parent displays RUNNING while any child is still running."""
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

    running_child = client.create_run(
        exp.experiment_id,
        run_name="parent-run fold=0",
        tags={
            "kego_id": secrets.token_hex(3),
            "kego_target": "cluster",
            "kego_debug": "false",
            "mlflow.parentRunId": parent.info.run_id,
        },
    )
    client.log_param(running_child.info.run_id, "fold", "0")

    finished_child = client.create_run(
        exp.experiment_id,
        run_name="parent-run fold=1",
        tags={
            "kego_id": secrets.token_hex(3),
            "kego_target": "cluster",
            "kego_debug": "false",
            "mlflow.parentRunId": parent.info.run_id,
        },
    )
    client.log_param(finished_child.info.run_id, "fold", "1")
    client.set_terminated(finished_child.info.run_id, status="FINISHED")

    _ls(_make_ls_args(), [])
    out = capsys.readouterr().out
    parent_rows = [
        line for line in out.splitlines() if "parent-run" in line and "RUNNING" in line and "fold=0" not in line
    ]
    assert parent_rows, "parent run row should show RUNNING status"


def test_ls_ray_column_surfaces_mlflow_vs_ray_mismatch(mlflow_db, capsys):
    """--ray adds a RAY column so a zombie (MLflow RUNNING, Ray STOPPED) is visible inline."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    exp = mlflow.set_experiment("test-exp")
    client.create_run(
        exp.experiment_id,
        run_name="zombie-run",
        tags={
            "kego_id": secrets.token_hex(3),
            "kego_target": "cluster",
            "kego_debug": "false",
            "ray_submission_id": "raysubmit_x",
        },
    )  # left RUNNING in MLflow

    with patch("kego.cli.ray.job_statuses", return_value={"raysubmit_x": "STOPPED"}):
        _ls(_make_ls_args(show_ray=True), [])
    out = capsys.readouterr().out

    assert "RAY" in out  # column header present
    row = next(line for line in out.splitlines() if "zombie-run" in line)
    assert "RUNNING" in row  # MLflow status
    assert "STOPPED" in row  # Ray state alongside → mismatch obvious


def test_ls_no_ray_column_by_default(mlflow_db, capsys):
    from mlflow.tracking import MlflowClient

    exp = mlflow.set_experiment("test-exp")
    MlflowClient().create_run(
        exp.experiment_id,
        run_name="plain-run",
        tags={"kego_id": secrets.token_hex(3), "kego_target": "cluster", "kego_debug": "false"},
    )
    _ls(_make_ls_args(), [])
    out = capsys.readouterr().out
    assert "RAY" not in out


def test_format_table_duration_shows_run_wall_clock():
    """DURATION = how long the run took (end-start), distinct from AGO (age since start)."""
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    runs = pd.DataFrame(
        [
            {
                "tags.kego_id": "abc123",
                "tags.mlflow.runName": "slow-run",
                "tags.kego_target": "cluster",
                "metrics.auc": 0.9,
                "status": "FINISHED",
                "start_time": now - datetime.timedelta(hours=2),  # started 2h ago
                "end_time": now - datetime.timedelta(minutes=15),  # ran 1h45m
            }
        ]
    )
    lines = format_table(runs, primary_metric="auc")
    assert "DURATION" in lines[0]
    assert "1h45m" in lines[2]  # ran 1h45m — slowness is obvious
    assert lines[2].rstrip().endswith("2h")  # AGO = age since start (2h), separate column
