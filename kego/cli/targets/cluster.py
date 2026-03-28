"""Ray cluster execution target — submits jobs to Ray cluster."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from kego.cli.config import KegoConfig

_JOB_ID_RE = re.compile(r"(raysubmit_\w+)")


def build_ray_command(
    script: str,
    script_args: list[str],
    config: KegoConfig,
    experiment_name: str,
    experiment_id: str,
    cli_params: dict[str, str],
) -> list[str]:
    """Build the ray job submit command list.

    Args:
        script: Path to training script
        script_args: Script arguments
        config: KegoConfig with cluster settings
        experiment_name: MLflow experiment name
        experiment_id: 6-char experiment ID for tagging
        cli_params: CLI parameters to pre-log as MLflow params

    Returns:
        Ray job submit command as list of strings
    """
    runtime_env = {
        "env_vars": {
            "MLFLOW_TRACKING_URI": config.cluster.mlflow_uri,
            "KEGO_EXPERIMENT_NAME": experiment_name,
            "KEGO_EXPERIMENT_ID": experiment_id,
            "KEGO_CLI_PARAMS": json.dumps(cli_params),
            "KEGO_PATH_DATA": os.environ.get(
                "KEGO_PATH_DATA",
                str(Path.home() / "projects/kego/data"),
            ),
            "KEGO_TARGET": "cluster",
            "KEGO_DEBUG": "false",
        },
    }

    entrypoint = [
        "uv",
        "run",
        "python",
        "-m",
        "kego.cli.runner",
        script,
        *script_args,
    ]

    return [
        "uv",
        "run",
        "ray",
        "job",
        "submit",
        "--address",
        config.cluster.ray_address,
        "--runtime-env-json",
        json.dumps(runtime_env),
        "--",
        *entrypoint,
    ]


def submit_fold(
    script: str,
    script_args: list[str],
    config: KegoConfig,
    experiment_name: str,
    experiment_id: str,
    cli_params: dict[str, str],
) -> str:
    """Submit one fold as a Ray job. Returns the Ray submission ID.

    Args:
        script: Path to training script
        script_args: Script arguments
        config: KegoConfig with cluster settings
        experiment_name: MLflow experiment name
        experiment_id: 6-char experiment ID for tagging
        cli_params: CLI parameters to pre-log as MLflow params

    Returns:
        Ray submission ID (e.g. 'raysubmit_ABCD1234')

    Raises:
        RuntimeError: If ray job submit fails or job ID cannot be parsed
    """
    cmd = build_ray_command(
        script, script_args, config, experiment_name, experiment_id, cli_params
    )
    result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
    if result.returncode != 0:
        raise RuntimeError(f"ray job submit failed:\n{result.stderr}")

    for line in result.stdout.splitlines():
        if m := _JOB_ID_RE.search(line):
            return m.group(1)

    raise RuntimeError(f"Could not parse job ID from output:\n{result.stdout}")


def submit(
    script: str,
    folds: list[int],
    base_args: list[str],
    config: KegoConfig,
    experiment_name: str,
    experiment_id: str,
    cli_params: dict[str, str],
) -> list[str]:
    """Submit one Ray job per fold. Returns list of Ray submission IDs.

    Args:
        script: Path to training script
        folds: List of fold indices to submit
        base_args: Base script arguments (without fold-specific args)
        config: KegoConfig with cluster settings
        experiment_name: MLflow experiment name
        experiment_id: 6-char experiment ID for tagging
        cli_params: CLI parameters to pre-log as MLflow params

    Returns:
        List of Ray submission IDs (e.g. ['raysubmit_A', 'raysubmit_B', ...])
    """
    job_ids: list[str] = []
    for fold in folds:
        fold_args = [*base_args, "--fold", str(fold)]
        fold_params = {**cli_params, "fold": str(fold)}
        job_id = submit_fold(
            script, fold_args, config, experiment_name, experiment_id, fold_params
        )
        print(f"  fold {fold}: {job_id}", flush=True)
        job_ids.append(job_id)
    return job_ids
