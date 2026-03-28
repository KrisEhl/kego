"""Local execution target — runs script in current process via runner."""

from __future__ import annotations

import json
import os
import sys

from kego.cli import runner
from kego.cli.config import KegoConfig


def _build_command(script: str, script_args: list[str]) -> list[str]:
    """Build command to invoke training script via runner.

    Args:
        script: Path to training script
        script_args: Script arguments

    Returns:
        Command line as list of strings
    """
    return [sys.executable, "-m", "kego.cli.runner", script, *script_args]


def run(
    script: str,
    script_args: list[str],
    config: KegoConfig,
    experiment_name: str,
    experiment_id: str,
    cli_params: dict[str, str],
) -> int:
    """Execute script locally with MLflow tracking via runner.

    Sets environment variables, calls runner.run(), then restores original env.

    Args:
        script: Path to training script
        script_args: Script arguments
        config: KegoConfig with cluster settings
        experiment_name: MLflow experiment name
        experiment_id: 6-char experiment ID for tagging
        cli_params: CLI parameters to pre-log as MLflow params

    Returns:
        Exit code from script
    """
    env_patch = {
        "MLFLOW_TRACKING_URI": config.cluster.mlflow_uri,
        "KEGO_EXPERIMENT_NAME": experiment_name,
        "KEGO_EXPERIMENT_ID": experiment_id,
        "KEGO_CLI_PARAMS": json.dumps(cli_params),
        "KEGO_TARGET": "local",
        "KEGO_DEBUG": "true" if "--debug" in script_args else "false",
    }
    old = {k: os.environ.get(k) for k in env_patch}
    os.environ.update(env_patch)
    try:
        return runner.run([script, *script_args])
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
