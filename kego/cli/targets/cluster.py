"""Ray cluster execution target — submits jobs via Ray Jobs HTTP API.

No `ray` binary required on the local machine. Uses stdlib urllib to POST
directly to the Ray dashboard at port 8265.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from kego.cli.config import KegoConfig


def _cluster_script_path(local_script: str, config: KegoConfig) -> str:
    """Convert a local absolute script path to its equivalent on the cluster.

    Computes the path relative to the local repo root, then prepends the
    cluster repo path from config (e.g. ~/projects/kego).
    """
    local_path = Path(local_script)
    try:
        rel = local_path.relative_to(config.repo_root)
    except ValueError:
        rel = Path(local_path.name)
    cluster_root = Path(config.cluster.repo_path).expanduser()
    return str(cluster_root / rel)


def _build_runtime_env(
    config: KegoConfig,
    experiment_name: str,
    run_name: str,
    experiment_id: str,
    cli_params: dict[str, str],
) -> dict:
    return {
        "env_vars": {
            "MLFLOW_TRACKING_URI": config.cluster.mlflow_uri,
            "KEGO_EXPERIMENT_NAME": experiment_name,
            "KEGO_RUN_NAME": run_name,
            "KEGO_EXPERIMENT_ID": experiment_id,
            "KEGO_CLI_PARAMS": json.dumps(cli_params),
            "KEGO_PATH_DATA": os.environ.get(
                "KEGO_PATH_DATA",
                str(Path(config.cluster.repo_path).expanduser() / "data"),
            ),
            "KEGO_TARGET": "cluster",
            "KEGO_DEBUG": "false",
        },
    }


def _submit_http(config: KegoConfig, entrypoint: str, runtime_env: dict) -> str:
    """Submit a Ray job via the HTTP API. Returns the submission ID."""
    # Ray address is http://host:8265 — jobs API lives at /api/jobs/
    base = config.cluster.ray_address.rstrip("/")
    url = f"{base}/api/jobs/"

    resources = config.cluster.default_resources
    body = {
        "entrypoint": entrypoint,
        "runtime_env": runtime_env,
        "entrypoint_num_gpus": resources.get("num_gpus", 0),
        "entrypoint_resources": {k: v for k, v in resources.items() if k != "num_gpus"},
    }

    data = json.dumps(body).encode()
    req = urllib.request.Request(  # noqa: S310
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Ray job submission failed (HTTP {e.code}): {e.read().decode()}"
        ) from e
    except OSError as e:
        raise RuntimeError(
            f"Cannot reach Ray cluster at {config.cluster.ray_address} — "
            "is the cluster online?\n"
            "  Start cluster : make cluster-start"
        ) from e

    submission_id = result.get("submission_id")
    if not submission_id:
        raise RuntimeError(f"No submission_id in response: {result}")
    return submission_id


def submit_fold(
    script: str,
    script_args: list[str],
    config: KegoConfig,
    experiment_name: str,
    run_name: str,
    experiment_id: str,
    cli_params: dict[str, str],
) -> str:
    """Submit one fold as a Ray job. Returns the Ray submission ID."""
    cluster_script = _cluster_script_path(script, config)
    runtime_env = _build_runtime_env(
        config, experiment_name, run_name, experiment_id, cli_params
    )
    args_str = " ".join(script_args)
    entrypoint = (
        f"cd {Path(config.cluster.repo_path).expanduser()} && "
        f"uv run python -m kego.cli.runner {cluster_script} {args_str}"
    )
    return _submit_http(config, entrypoint, runtime_env)


def submit(
    script: str,
    folds: list[int],
    base_args: list[str],
    config: KegoConfig,
    experiment_name: str,
    run_name: str,
    experiment_id: str,
    cli_params: dict[str, str],
) -> list[str]:
    """Submit one Ray job per fold. Returns list of Ray submission IDs."""
    job_ids: list[str] = []
    for fold in folds:
        fold_args = [*base_args, "--fold", str(fold)]
        fold_params = {**cli_params, "fold": str(fold)}
        job_id = submit_fold(
            script,
            fold_args,
            config,
            experiment_name,
            run_name,
            experiment_id,
            fold_params,
        )
        print(f"  fold {fold}: {job_id}", flush=True)
        job_ids.append(job_id)
    return job_ids
