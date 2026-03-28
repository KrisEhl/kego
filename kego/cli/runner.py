"""
Subprocess wrapper that parses KEGO_METRIC / KEGO_PARAM stdout lines
and logs them to MLflow. Runs on any compute target (local or cluster).

Invoked as: python -m kego.cli.runner <script> [script_args...]

Environment variables (injected by kego run):
    MLFLOW_TRACKING_URI    — MLflow server URI (empty string = no logging)
    KEGO_EXPERIMENT_NAME   — MLflow experiment name (competition slug if available)
    KEGO_RUN_NAME          — MLflow run name (--name or auto-generated)
    KEGO_EXPERIMENT_ID     — 6-char experiment ID stored as MLflow tag
    KEGO_CLI_PARAMS        — JSON dict of CLI args to pre-log as params
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys

_METRIC_RE = re.compile(r"^KEGO_METRIC\s+(\S+)\s+(\S+)\s*$")
_PARAM_RE = re.compile(r"^KEGO_PARAM\s+(\S+)\s+(.+?)\s*$")


def parse_kego_lines(
    lines: list[str],
) -> tuple[dict[str, float], dict[str, str]]:
    """Parse KEGO_METRIC and KEGO_PARAM lines. Returns (metrics, params)."""
    metrics: dict[str, float] = {}
    params: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if m := _METRIC_RE.match(stripped):
            with contextlib.suppress(ValueError):
                metrics[m.group(1)] = float(m.group(2))
        elif m := _PARAM_RE.match(stripped):
            params[m.group(1)] = m.group(2)
    return metrics, params


def _log_to_mlflow(
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    experiment_id: str,
    cli_params: dict[str, str],
    metrics: dict[str, float],
    params: dict[str, str],
    extra_tags: dict[str, str] | None = None,
) -> None:
    """Log everything to MLflow. No-op if tracking_uri is empty."""
    if not tracking_uri:
        return
    import logging

    import mlflow

    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.WARNING)

    tags = {"kego_id": experiment_id, **(extra_tags or {})}
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name, tags=tags):
        if cli_params:
            mlflow.log_params(cli_params)
        if params:
            mlflow.log_params(params)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)


def run(argv: list[str]) -> int:
    """Run script at argv[0] with args argv[1:], tracking KEGO_ lines."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    experiment_name = os.environ.get("KEGO_EXPERIMENT_NAME", "kego-default")
    run_name = os.environ.get("KEGO_RUN_NAME", experiment_name)
    experiment_id = os.environ.get("KEGO_EXPERIMENT_ID", "unknown")
    cli_params = json.loads(os.environ.get("KEGO_CLI_PARAMS", "{}"))

    cmd = [sys.executable, *list(argv)]
    process = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    collected_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        collected_lines.append(line)

    process.wait()

    metrics, params = parse_kego_lines(collected_lines)

    target = os.environ.get("KEGO_TARGET", "local")
    debug = os.environ.get("KEGO_DEBUG", "false")

    _log_to_mlflow(
        tracking_uri,
        experiment_name,
        run_name,
        experiment_id,
        cli_params,
        metrics,
        params,
        extra_tags={
            "kego_exit_code": str(process.returncode),
            "kego_target": target,
            "kego_debug": debug,
        },
    )

    return process.returncode


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
