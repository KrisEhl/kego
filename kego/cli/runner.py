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
import logging
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


def run(argv: list[str]) -> int:
    """Run script at argv[0] with args argv[1:], tracking KEGO_ lines."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    experiment_name = os.environ.get("KEGO_EXPERIMENT_NAME", "kego-default")
    run_name = os.environ.get("KEGO_RUN_NAME", experiment_name)
    experiment_id = os.environ.get("KEGO_EXPERIMENT_ID", "unknown")
    cli_params = json.loads(os.environ.get("KEGO_CLI_PARAMS", "{}"))
    target = os.environ.get("KEGO_TARGET", "local")
    debug = os.environ.get("KEGO_DEBUG", "false")

    # Open the MLflow run before launching the script so it appears as RUNNING
    active_run = None
    if tracking_uri:
        import mlflow

        logging.getLogger("mlflow").setLevel(logging.WARNING)
        logging.getLogger("alembic").setLevel(logging.WARNING)

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        tags = {
            "kego_id": experiment_id,
            "kego_target": target,
            "kego_debug": debug,
        }
        active_run = mlflow.start_run(run_name=run_name, tags=tags)
        if cli_params:
            mlflow.log_params(cli_params)

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

    if active_run is not None:
        import mlflow

        if params:
            mlflow.log_params(params)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        mlflow.set_tag("kego_exit_code", str(process.returncode))
        mlflow.end_run(status="FINISHED" if process.returncode == 0 else "FAILED")

    return process.returncode


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
