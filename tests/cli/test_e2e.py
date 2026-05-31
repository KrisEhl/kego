"""End-to-end tests for the kego CLI.

These tests run the real CLI via subprocess and verify that:
  - kego run executes a script and logs to MLflow
  - kego ls shows the logged experiment

No external services needed — uses a local SQLite MLflow backend.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _run_kego(args: list[str], env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    kego_bin = Path(sys.executable).parent / "kego"
    return subprocess.run(  # noqa: S603
        [str(kego_bin), *args],
        env=env,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )


def test_run_and_ls(tmp_path: Path, repo_root: Path) -> None:
    """kego run logs a metric; kego ls shows the experiment."""
    mlflow_uri = f"sqlite:///{tmp_path}/mlflow.db"
    env = _base_env(repo_root, mlflow_uri)

    script = tmp_path / "train.py"
    script.write_text("print('KEGO_METRIC fold_auc 0.9123')\nprint('KEGO_PARAM backbone test_net')\nprint('Done!')\n")

    result = _run_kego(
        ["run", str(script), "--name", "e2e-test"],
        env=env,
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr
    assert "KEGO_METRIC fold_auc 0.9123" in result.stdout
    assert "Done!" in result.stdout

    ls_result = _run_kego(["ls"], env=env, cwd=repo_root)
    assert ls_result.returncode == 0, ls_result.stderr
    assert "FINISHED" in ls_result.stdout
    assert "local" in ls_result.stdout
    assert "0.9123" in ls_result.stdout


def test_run_tags_run_with_git_sha(tmp_path: Path, repo_root: Path) -> None:
    """Each run records the commit it ran from, so a stale-code node is detectable."""
    import subprocess

    import mlflow

    mlflow_uri = f"sqlite:///{tmp_path}/mlflow.db"
    env = _base_env(repo_root, mlflow_uri)

    script = tmp_path / "train.py"
    script.write_text("print('KEGO_METRIC x 1.0')\n")

    result = _run_kego(["run", str(script), "--name", "sha-test"], env=env, cwd=repo_root)
    assert result.returncode == 0, result.stderr

    expected_sha = subprocess.run(  # noqa: S603
        ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
    ).stdout.strip()

    mlflow.set_tracking_uri(mlflow_uri)
    runs = mlflow.search_runs(experiment_names=["sha-test"])
    assert runs["tags.git_sha"].iloc[0] == expected_sha


def test_run_debug_excluded_from_ls(tmp_path: Path, repo_root: Path) -> None:
    """Debug runs are excluded from kego ls by default."""
    mlflow_uri = f"sqlite:///{tmp_path}/mlflow.db"
    env = _base_env(repo_root, mlflow_uri)

    script = tmp_path / "train.py"
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--debug', action='store_true')\n"
        "args = parser.parse_args()\n"
        "print('KEGO_METRIC fold_auc 0.5000')\n"
    )

    result = _run_kego(
        ["run", str(script), "--debug"],
        env=env,
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr

    # Default ls hides debug runs
    ls_result = _run_kego(["ls"], env=env, cwd=repo_root)
    assert ls_result.returncode == 0, ls_result.stderr
    assert "No experiments found." in ls_result.stdout

    # --all shows them
    ls_all = _run_kego(["ls", "--all"], env=env, cwd=repo_root)
    assert ls_all.returncode == 0, ls_all.stderr
    assert "FINISHED" in ls_all.stdout


def test_run_nonzero_exit_still_logs(tmp_path: Path, repo_root: Path) -> None:
    """A script that exits non-zero still logs whatever metrics it printed."""
    mlflow_uri = f"sqlite:///{tmp_path}/mlflow.db"
    env = _base_env(repo_root, mlflow_uri)

    script = tmp_path / "train.py"
    script.write_text("print('KEGO_METRIC fold_auc 0.7500')\nraise RuntimeError('intentional failure')\n")

    result = _run_kego(["run", str(script), "--name", "failing-run"], env=env, cwd=repo_root)
    assert result.returncode != 0

    ls_result = _run_kego(["ls", "--all"], env=env, cwd=repo_root)
    assert "FAILED" in ls_result.stdout  # non-zero exit → FAILED status in MLflow


def test_ls_offline_mlflow_fails_fast(tmp_path: Path, repo_root: Path) -> None:
    """kego ls prints a useful error (not a hang) when MLflow is unreachable."""
    env = _base_env(repo_root, "http://127.0.0.1:19999")  # nothing listening here

    ls_result = _run_kego(["ls"], env=env, cwd=repo_root)
    assert ls_result.returncode == 1
    assert "Cannot reach MLflow at http://127.0.0.1:19999" in ls_result.stdout


def test_ls_rejects_typoed_flag(tmp_path: Path, repo_root: Path) -> None:
    """A mistyped flag (----since) must error, not silently run unfiltered."""
    mlflow_uri = f"sqlite:///{tmp_path}/mlflow.db"
    env = _base_env(repo_root, mlflow_uri)

    result = _run_kego(["ls", "----since", "30m"], env=env, cwd=repo_root)
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr


# ---------------------------------------------------------------------------
# kego logs
# ---------------------------------------------------------------------------


def test_logs_unknown_id(tmp_path: Path, repo_root: Path) -> None:
    """kego logs with an unknown ID prints a clear error."""
    mlflow_uri = f"sqlite:///{tmp_path}/mlflow.db"
    env = _base_env(repo_root, mlflow_uri)

    result = _run_kego(["logs", "xxxxxx"], env=env, cwd=repo_root)
    assert result.returncode == 1
    assert "No runs found" in result.stdout


def test_logs_local_run_with_no_capture_dir_falls_back_to_message(tmp_path: Path, repo_root: Path) -> None:
    """A local run whose captured-stdout file is missing prints a clear message, not a crash."""
    mlflow_uri = f"sqlite:///{tmp_path}/mlflow.db"
    env = _base_env(repo_root, mlflow_uri)
    env["KEGO_LOG_DIR"] = str(tmp_path / "logs")

    script = tmp_path / "train.py"
    script.write_text("print('KEGO_METRIC fold_auc 0.9')\n")
    run_result = _run_kego(["run", str(script), "--name", "logs-test"], env=env, cwd=repo_root)
    assert run_result.returncode == 0, run_result.stderr
    kego_id = run_result.stdout.split("[")[1].split("]")[0]

    # Wipe the capture dir to simulate a run that predates local-log capture
    import shutil

    shutil.rmtree(tmp_path / "logs", ignore_errors=True)

    result = _run_kego(["logs", kego_id], env=env, cwd=repo_root)
    assert result.returncode == 0
    assert "No captured stdout" in result.stdout


def test_logs_unknown_submission_id(tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """kego logs with a fake ray_submission_id gets a 404 and exits non-zero."""
    import mlflow

    mlflow_uri = f"sqlite:///{tmp_path}/mlflow.db"
    env = _base_env(repo_root, mlflow_uri)

    # Set via monkeypatch so this env var is restored after the test
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow_uri)

    # Manually create a run with a bogus ray_submission_id tag
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("test-exp")
    with mlflow.start_run(
        run_name="ray-run",
        tags={
            "kego_id": "aabbcc",
            "kego_target": "cluster",
            "kego_debug": "false",
            "ray_submission_id": "raysubmit_fake123",
        },
    ):
        pass

    result = _run_kego(["logs", "aabbcc"], env=env, cwd=repo_root)
    assert result.returncode == 1
    # Either Ray returned an HTTP error (cluster up, ID not found) or the cluster
    # is unreachable — both are valid "can't fetch logs" outcomes.
    assert "Ray API error" in result.stdout or "Cannot reach Ray cluster" in result.stdout


def test_logs_replays_local_run_stdout(tmp_path: Path, repo_root: Path) -> None:
    """A local run's stdout is captured so `kego logs <id>` can replay it (no Ray job)."""
    mlflow_uri = f"sqlite:///{tmp_path}/mlflow.db"
    env = _base_env(repo_root, mlflow_uri)
    env["KEGO_LOG_DIR"] = str(tmp_path / "logs")

    script = tmp_path / "train.py"
    script.write_text("print('UNIQUE_MARKER_LINE_42')\nprint('KEGO_METRIC x 1.0')\n")

    run_result = _run_kego(["run", str(script), "--name", "local-log-test"], env=env, cwd=repo_root)
    assert run_result.returncode == 0, run_result.stderr
    kego_id = run_result.stdout.split("[")[1].split("]")[0]

    logs_result = _run_kego(["logs", kego_id], env=env, cwd=repo_root)
    assert logs_result.returncode == 0
    assert "UNIQUE_MARKER_LINE_42" in logs_result.stdout


def test_run_multifold_creates_parent_and_children(tmp_path: Path) -> None:
    """Multi-fold _pre_create_runs creates one parent + N child MLflow runs."""
    import os

    import mlflow
    from mlflow.tracking import MlflowClient

    from kego.cli.commands.run import _pre_create_runs
    from kego.cli.config import ClusterConfig, CompetitionConfig, KegoConfig

    mlflow_uri = f"sqlite:///{tmp_path}/mlflow.db"
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri

    try:
        config = KegoConfig(
            cluster=ClusterConfig(ray_address="http://x:8265", mlflow_uri=mlflow_uri),
            competition=CompetitionConfig(
                slug="test-comp",
                kaggle_user="u",
                enable_gpu=False,
                submit_file="s.csv",
                pattern="script",
                inference_notebook="t.py",
                checkpoint_dir="out",
                primary_metric="rmse",
            ),
            repo_root=tmp_path,
            competition_dir=None,
        )

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("test-comp")

        fold_run_ids = _pre_create_runs(config, "test-comp", "my-run", "abc123", {}, [0, 1, 2])

        client = MlflowClient()
        exp_id = mlflow.get_experiment_by_name("test-comp").experiment_id
        all_runs = client.search_runs(experiment_ids=[exp_id])

        parent_runs = [r for r in all_runs if r.data.tags.get("kego_is_parent") == "true"]
        child_runs = [r for r in all_runs if r.data.tags.get("mlflow.parentRunId")]

        assert len(parent_runs) == 1
        assert len(child_runs) == 3
        assert parent_runs[0].data.tags["kego_fold_count"] == "3"
        parent_id = parent_runs[0].info.run_id
        for child in child_runs:
            assert child.data.tags["mlflow.parentRunId"] == parent_id
        assert set(fold_run_ids.keys()) == {0, 1, 2}
        assert set(fold_run_ids.values()) == {r.info.run_id for r in child_runs}
    finally:
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        mlflow.set_tracking_uri("")


def test_run_singlefold_creates_no_parent(tmp_path: Path) -> None:
    """Single-fold _pre_create_runs creates one run with no parent wrapper."""
    import os

    import mlflow
    from mlflow.tracking import MlflowClient

    from kego.cli.commands.run import _pre_create_runs
    from kego.cli.config import ClusterConfig, CompetitionConfig, KegoConfig

    mlflow_uri = f"sqlite:///{tmp_path}/mlflow.db"
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri

    try:
        config = KegoConfig(
            cluster=ClusterConfig(ray_address="http://x:8265", mlflow_uri=mlflow_uri),
            competition=CompetitionConfig(
                slug="test-comp",
                kaggle_user="u",
                enable_gpu=False,
                submit_file="s.csv",
                pattern="script",
                inference_notebook="t.py",
                checkpoint_dir="out",
                primary_metric="rmse",
            ),
            repo_root=tmp_path,
            competition_dir=None,
        )

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("test-comp")

        fold_run_ids = _pre_create_runs(config, "test-comp", "my-run", "abc123", {}, [0])

        client = MlflowClient()
        exp_id = mlflow.get_experiment_by_name("test-comp").experiment_id
        all_runs = client.search_runs(experiment_ids=[exp_id])

        assert len(all_runs) == 1
        run = all_runs[0]
        assert run.data.tags.get("kego_is_parent") != "true"
        assert "mlflow.parentRunId" not in run.data.tags
        assert fold_run_ids == {0: run.info.run_id}
    finally:
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        mlflow.set_tracking_uri("")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_env(repo_root: Path, mlflow_uri: str) -> dict[str, str]:
    """Minimal environment for CLI subprocesses."""
    import os

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = mlflow_uri
    env["PYTHONPATH"] = str(repo_root)
    return env
