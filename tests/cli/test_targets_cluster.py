import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kego.cli.config import ClusterConfig, KegoConfig
from kego.cli.targets.cluster import build_ray_command, submit, submit_fold


@pytest.fixture
def config():
    return KegoConfig(
        cluster=ClusterConfig(
            ray_address="http://192.168.1.1:8265",
            mlflow_uri="http://192.168.1.1:5000",
        ),
        competition=None,
        repo_root=Path("/repo"),
        competition_dir=None,
    )


def test_build_ray_command_structure(config):
    cmd = build_ray_command(
        script="train_cnn.py",
        script_args=["--fold", "0"],
        config=config,
        experiment_name="soundscape-v8",
        experiment_id="abc123",
        cli_params={"fold": "0"},
    )
    cmd_str = " ".join(cmd)
    assert "ray" in cmd_str
    assert "job" in cmd_str
    assert "submit" in cmd_str
    assert "192.168.1.1:8265" in cmd_str
    assert "kego.cli.runner" in cmd_str
    assert "train_cnn.py" in cmd_str


def test_build_ray_command_includes_mlflow_env(config):
    cmd = build_ray_command(
        script="train_cnn.py",
        script_args=["--fold", "0"],
        config=config,
        experiment_name="test",
        experiment_id="abc123",
        cli_params={},
    )
    # runtime-env-json should contain mlflow URI
    json_idx = cmd.index("--runtime-env-json") + 1
    runtime_env = json.loads(cmd[json_idx])
    assert runtime_env["env_vars"]["MLFLOW_TRACKING_URI"] == "http://192.168.1.1:5000"
    assert runtime_env["env_vars"]["KEGO_EXPERIMENT_ID"] == "abc123"


def test_submit_fans_out_folds(config):
    submitted = []

    def fake_submit_fold(
        script, script_args, config, experiment_name, experiment_id, cli_params
    ):
        submitted.append({"script_args": script_args, "cli_params": cli_params})
        return f"raysubmit_fold{len(submitted)}"

    with patch("kego.cli.targets.cluster.submit_fold", side_effect=fake_submit_fold):
        job_ids = submit(
            script="train_cnn.py",
            folds=[0, 1, 2, 3],
            base_args=["--epochs", "30"],
            config=config,
            experiment_name="soundscape-v8",
            experiment_id="abc123",
            cli_params={"epochs": "30"},
        )

    assert len(job_ids) == 4
    assert len(submitted) == 4
    # Each fold submission should include --fold N
    for i, call in enumerate(submitted):
        assert "--fold" in call["script_args"]
        assert str(i) in call["script_args"]


def test_submit_fold_parses_job_id(config):
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Job submission: raysubmit_ABCD1234\nDone."

    with patch("subprocess.run", return_value=mock_result):
        job_id = submit_fold(
            script="train_cnn.py",
            script_args=["--fold", "0"],
            config=config,
            experiment_name="test",
            experiment_id="abc123",
            cli_params={},
        )
    assert job_id == "raysubmit_ABCD1234"


def test_submit_fold_raises_on_failure(config):
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "Connection refused"

    with (
        patch("subprocess.run", return_value=mock_result),
        pytest.raises(RuntimeError, match="ray job submit failed"),
    ):
        submit_fold(
            script="train_cnn.py",
            script_args=["--fold", "0"],
            config=config,
            experiment_name="test",
            experiment_id="abc123",
            cli_params={},
        )
