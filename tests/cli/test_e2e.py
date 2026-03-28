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


def _run_kego(
    args: list[str], env: dict[str, str], cwd: Path
) -> subprocess.CompletedProcess[str]:
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
    script.write_text(
        "print('KEGO_METRIC fold_auc 0.9123')\n"
        "print('KEGO_PARAM backbone test_net')\n"
        "print('Done!')\n"
    )

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
    script.write_text(
        "print('KEGO_METRIC fold_auc 0.7500')\n"
        "raise RuntimeError('intentional failure')\n"
    )

    result = _run_kego(
        ["run", str(script), "--name", "failing-run"], env=env, cwd=repo_root
    )
    assert result.returncode != 0

    ls_result = _run_kego(["ls", "--all"], env=env, cwd=repo_root)
    assert "FAILED" in ls_result.stdout  # non-zero exit → FAILED status in MLflow


def test_ls_offline_mlflow_fails_fast(tmp_path: Path, repo_root: Path) -> None:
    """kego ls prints a useful error (not a hang) when MLflow is unreachable."""
    env = _base_env(repo_root, "http://127.0.0.1:19999")  # nothing listening here

    ls_result = _run_kego(["ls"], env=env, cwd=repo_root)
    assert ls_result.returncode == 1
    assert "Cannot reach MLflow at http://127.0.0.1:19999" in ls_result.stdout


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


def test_logs_no_ray_submission_id(tmp_path: Path, repo_root: Path) -> None:
    """kego logs shows a helpful message when run has no ray_submission_id tag."""
    mlflow_uri = f"sqlite:///{tmp_path}/mlflow.db"
    env = _base_env(repo_root, mlflow_uri)

    # Create a run via kego run (local — no ray_submission_id tag)
    script = tmp_path / "train.py"
    script.write_text("print('KEGO_METRIC fold_auc 0.9')\n")
    run_result = _run_kego(
        ["run", str(script), "--name", "logs-test"], env=env, cwd=repo_root
    )
    assert run_result.returncode == 0, run_result.stderr

    # Extract the kego ID from the run output
    kego_id = run_result.stdout.split("[")[1].split("]")[0]

    result = _run_kego(["logs", kego_id], env=env, cwd=repo_root)
    assert result.returncode == 0
    assert "ray_submission_id" in result.stdout


def test_logs_unknown_submission_id(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    assert "Ray API error" in result.stdout


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
