from __future__ import annotations

import argparse
import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def _ago(start: datetime.datetime) -> str:
    delta = datetime.datetime.now(tz=datetime.timezone.utc) - start
    hours = int(delta.total_seconds() // 3600)
    if hours > 0:
        return f"{hours}h"
    return f"{int(delta.total_seconds() // 60)}m"


def _resolve_metric(runs: pd.DataFrame, primary_metric: str) -> str:
    """Return primary_metric if it has data, otherwise the first metric col that does."""

    preferred = f"metrics.{primary_metric}"
    if preferred in runs.columns and runs[preferred].notna().any():
        return primary_metric
    for col in runs.columns:
        if col.startswith("metrics.") and runs[col].notna().any():
            return col[len("metrics.") :]
    return primary_metric


def format_table(runs: pd.DataFrame, primary_metric: str) -> list[str]:
    """Format experiment runs into a table. Returns list of lines."""
    import pandas as pd

    if runs.empty:
        return ["No experiments found."]

    primary_metric = _resolve_metric(runs, primary_metric)
    metric_col = f"metrics.{primary_metric}"

    header = f"{'ID':<8} {'NAME':<26} {'TARGET':<8} {primary_metric.upper():>8} {'STATUS':<10} {'AGO'}"
    sep = "-" * len(header)
    lines = [header, sep]

    for _, row in runs.iterrows():
        exp_id = str(row.get("tags.kego_id", "?"))[:6]
        name = str(row.get("tags.mlflow.runName", "?"))[:26]
        target = str(row.get("tags.kego_target", "local"))[:8]
        raw_metric = row.get(metric_col)
        metric_str = f"{raw_metric:.4f}" if pd.notna(raw_metric) else "—"
        status = str(row.get("status", "?"))[:10]
        start = row.get("start_time")
        ago = _ago(start) if start is not None and pd.notna(start) else "?"
        lines.append(
            f"{exp_id:<8} {name:<26} {target:<8} {metric_str:>8} {status:<10} {ago}"
        )

    return lines


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("ls", help="List and compare experiments")
    p.add_argument("--name", help="Filter by experiment name")
    p.add_argument(
        "--all",
        action="store_true",
        dest="show_all",
        help="Include debug runs",
    )
    p.set_defaults(func=_ls)


def _ls(args: argparse.Namespace, extra_args: list[str]) -> int:
    import logging
    import os

    import mlflow

    from kego.cli import config as cfg_module

    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.WARNING)

    config = cfg_module.load_config()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or config.cluster.mlflow_uri
    mlflow.set_tracking_uri(tracking_uri)

    filter_parts: list[str] = []
    if args.name:
        filter_parts.append(f"tags.`mlflow.runName` LIKE '{args.name}%'")

    filter_string = " AND ".join(filter_parts) if filter_parts else ""

    try:
        runs = mlflow.search_runs(
            search_all_experiments=True,
            filter_string=filter_string,
            order_by=["start_time DESC"],
            max_results=50,
        )
    except Exception:
        print(
            f"Cannot reach MLflow at {tracking_uri} — is the cluster online?\n"
            "  Start cluster : uv run kego cluster start\n"
            "  Test locally  : MLFLOW_TRACKING_URI=sqlite:///local.db uv run kego ls"
        )
        return 1

    # Filter debug runs in Python — MLflow filter doesn't handle missing tags correctly
    if not args.show_all and "tags.kego_debug" in runs.columns:
        runs = runs[runs["tags.kego_debug"] != "true"]

    primary_metric = "metric"
    if config.competition:
        primary_metric = config.competition.primary_metric

    for line in format_table(runs, primary_metric):
        print(line)

    return 0
