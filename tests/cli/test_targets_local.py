import os
from pathlib import Path

import pytest

from kego.cli.config import ClusterConfig, KegoConfig
from kego.cli.targets.local import run


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


def test_run_sets_env_vars(config, monkeypatch):
    captured_env: dict = {}

    def fake_runner_run(argv):
        captured_env.update(os.environ.copy())
        return 0

    monkeypatch.setattr("kego.cli.targets.local.runner.run", fake_runner_run)
    run(
        script="dummy.py",
        script_args=["--fold", "0"],
        config=config,
        experiment_name="test-exp",
        run_name="soundscape-v8",
        experiment_id="abc123",
        cli_params={"fold": "0"},
    )

    assert captured_env["MLFLOW_TRACKING_URI"] == "http://192.168.1.1:5000"
    assert captured_env["KEGO_EXPERIMENT_NAME"] == "test-exp"
    assert captured_env["KEGO_RUN_NAME"] == "soundscape-v8"
    assert captured_env["KEGO_EXPERIMENT_ID"] == "abc123"
