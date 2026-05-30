from pathlib import Path
from unittest.mock import patch

import pytest

from kego.cli.config import ClusterConfig, KegoConfig
from kego.cli.targets.cluster import submit, submit_fold


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


def test_submit_fold_returns_job_id(config):
    with patch("kego.cli.targets.cluster._submit_http", return_value="raysubmit_ABC"):
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


def test_submit_fans_out_one_job_per_fold(config):
    with patch("kego.cli.targets.cluster.submit_fold", return_value="raysubmit_x"):
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
