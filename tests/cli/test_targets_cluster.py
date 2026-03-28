from pathlib import Path
from unittest.mock import patch

import pytest

from kego.cli.config import ClusterConfig, KegoConfig
from kego.cli.targets.cluster import (
    _build_runtime_env,
    _cluster_script_path,
    submit,
    submit_fold,
)


@pytest.fixture
def config():
    return KegoConfig(
        cluster=ClusterConfig(
            ray_address="http://192.168.1.1:8265",
            mlflow_uri="http://192.168.1.1:5000",
            repo_path="~/projects/kego",
        ),
        competition=None,
        repo_root=Path("/home/user/projects/kego"),
        competition_dir=None,
    )


def test_cluster_script_path_inside_repo(config):
    local = "/home/user/projects/kego/competitions/birdclef-2026/training/train_cnn.py"
    result = _cluster_script_path(local, config)
    assert result.endswith("competitions/birdclef-2026/training/train_cnn.py")
    assert result.startswith("~/projects/kego")


def test_cluster_script_path_outside_repo(config):
    result = _cluster_script_path("/var/tmp/random_script.py", config)  # noqa: S108
    assert result.endswith("random_script.py")


def test_build_runtime_env_contains_required_keys(config):
    env = _build_runtime_env(config, "birdclef-2026", "soundscape-v8", "abc123", {})
    vars_ = env["env_vars"]
    assert vars_["MLFLOW_TRACKING_URI"] == "http://192.168.1.1:5000"
    assert vars_["KEGO_EXPERIMENT_NAME"] == "birdclef-2026"
    assert vars_["KEGO_RUN_NAME"] == "soundscape-v8"
    assert vars_["KEGO_EXPERIMENT_ID"] == "abc123"
    assert vars_["KEGO_TARGET"] == "cluster"


def test_submit_fold_calls_http_api(config):
    with patch(
        "kego.cli.targets.cluster._submit_http", return_value="raysubmit_ABC"
    ) as mock_http:
        job_id = submit_fold(
            script="/home/user/projects/kego/train.py",
            script_args=["--fold", "0"],
            config=config,
            experiment_name="birdclef-2026",
            run_name="soundscape-v8",
            experiment_id="abc123",
            cli_params={},
        )
    assert job_id == "raysubmit_ABC"
    mock_http.assert_called_once()
    _, entrypoint, _ = mock_http.call_args[0]
    assert "kego.cli.runner" in entrypoint
    assert "--fold" in entrypoint


def test_submit_fans_out_folds(config):
    submitted = []

    def fake_submit_fold(
        script,
        script_args,
        config,
        experiment_name,
        run_name,
        experiment_id,
        cli_params,
        mlflow_run_id=None,
    ):
        submitted.append({"script_args": script_args, "cli_params": cli_params})
        return f"raysubmit_fold{len(submitted)}"

    with patch("kego.cli.targets.cluster.submit_fold", side_effect=fake_submit_fold):
        job_ids = submit(
            script="train_cnn.py",
            folds=[0, 1, 2, 3],
            base_args=["--epochs", "30"],
            config=config,
            experiment_name="birdclef-2026",
            run_name="soundscape-v8",
            experiment_id="abc123",
            cli_params={"epochs": "30"},
        )

    assert len(job_ids) == 4
    assert len(submitted) == 4
    for i, call in enumerate(submitted):
        assert "--fold" in call["script_args"]
        assert str(i) in call["script_args"]
