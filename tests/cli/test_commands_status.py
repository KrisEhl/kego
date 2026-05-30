import argparse
import json
from io import BytesIO
from unittest.mock import patch

from kego.cli.commands.status import _status


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(status=None, show_all=False)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _fake_job(
    run_name: str = "soundscape-v7",
    fold: int = 0,
    status: str = "RUNNING",
    kego_id: str = "abc123",
    start_time: int = 1_700_000_000_000,
    message: str = "",
) -> dict:
    return {
        "submission_id": f"raysubmit_{run_name}_{fold}",
        "status": status,
        "start_time": start_time,
        "end_time": None,
        "message": message,
        "runtime_env": {
            "env_vars": {
                "KEGO_RUN_NAME": run_name,
                "KEGO_EXPERIMENT_ID": kego_id,
                "KEGO_CLI_PARAMS": json.dumps({"fold": str(fold)}),
            }
        },
    }


def _mock_ray(jobs: list[dict]):
    """Context manager that patches urllib to return `jobs` from the Ray API."""
    body = BytesIO(json.dumps(jobs).encode())
    body.read = body.read  # already has read()
    return patch(
        "urllib.request.urlopen",
        return_value=body,
    )


# ---------------------------------------------------------------------------
# Behaviour tests
# ---------------------------------------------------------------------------


def test_status_shows_run_name_and_fold(capsys):
    with _mock_ray([_fake_job("soundscape-v7", fold=2)]):
        rc = _status(_make_args(), [])
    out = capsys.readouterr().out
    assert rc == 0
    assert "soundscape-v7" in out
    assert "2" in out


def test_status_shows_job_status(capsys):
    with _mock_ray([_fake_job(status="RUNNING"), _fake_job(status="FAILED", fold=1)]):
        _status(_make_args(), [])
    out = capsys.readouterr().out
    assert "RUNNING" in out
    assert "FAILED" in out


def test_status_filter_excludes_non_matching(capsys):
    jobs = [
        _fake_job("running-job", status="RUNNING"),
        _fake_job("failed-job", status="FAILED", fold=1),
    ]
    with _mock_ray(jobs):
        _status(_make_args(status="running"), [])
    out = capsys.readouterr().out
    assert "running-job" in out
    assert "failed-job" not in out


def test_status_failed_job_shows_error_hint(capsys):
    job = _fake_job(
        status="FAILED",
        message="Job failed.\nSome logs here.\nuv: command not found\n",
    )
    with _mock_ray([job]):
        _status(_make_args(), [])
    out = capsys.readouterr().out
    assert "uv: command not found" in out


def test_status_ray_offline_returns_error(capsys):
    with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        rc = _status(_make_args(), [])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Cannot reach Ray" in out


def test_status_no_jobs_prints_message(capsys):
    with _mock_ray([]):
        rc = _status(_make_args(), [])
    out = capsys.readouterr().out
    assert rc == 0
    assert "No jobs found" in out


def test_status_default_caps_at_20_jobs(capsys):
    names = [f"run-{i:03d}" for i in range(25)]
    jobs = [_fake_job(names[i], fold=i, start_time=1_700_000_000_000 + i) for i in range(25)]
    with _mock_ray(jobs):
        _status(_make_args(show_all=False), [])
    out = capsys.readouterr().out
    assert sum(n in out for n in names) <= 20


def test_status_all_flag_shows_beyond_20(capsys):
    names = [f"run-{i:03d}" for i in range(25)]
    jobs = [_fake_job(names[i], fold=i, start_time=1_700_000_000_000 + i) for i in range(25)]
    with _mock_ray(jobs):
        _status(_make_args(show_all=True), [])
    out = capsys.readouterr().out
    assert sum(n in out for n in names) == 25
