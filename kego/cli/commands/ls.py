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


_SKIP_METRICS = {"epoch", "loss", "train_loss", "val_loss", "lr", "learning_rate"}
_STATUSES = ["running", "finished", "failed", "killed"]


def _resolve_metric(runs: pd.DataFrame, primary_metric: str) -> str:
    """Return primary_metric if it has data, otherwise the first non-bookkeeping metric."""
    preferred = f"metrics.{primary_metric}"
    if preferred in runs.columns and runs[preferred].notna().any():
        return primary_metric
    for col in runs.columns:
        name = col[len("metrics.") :]
        if col.startswith("metrics.") and name not in _SKIP_METRICS and runs[col].notna().any():
            return name
    return primary_metric


def _metric_str(row: pd.Series, fallback_metric: str) -> str:
    """Return formatted metric value for a single run."""
    import pandas as pd

    # Prefer per-run kego_primary_metric tag, fall back to table-level resolved metric
    metric_name = str(row.get("tags.kego_primary_metric") or fallback_metric)
    if not metric_name:
        metric_name = fallback_metric
    val = row.get(f"metrics.{metric_name}")
    return f"{val:.4f}" if val is not None and pd.notna(val) else "—"


def format_table(
    runs: pd.DataFrame,
    primary_metric: str,
    exp_names: dict[str, str] | None = None,
    show_metric_name: bool = False,
) -> list[str]:
    """Format experiment runs into a table. Returns list of lines."""
    import pandas as pd

    if runs.empty:
        return ["No experiments found."]

    fallback_metric = _resolve_metric(runs, primary_metric)

    metric_name_col = f" {'METRIC_NAME':<10}" if show_metric_name else ""
    header = (
        f"{'ID':<8} {'NAME':<26} {'COMPETITION':<20} {'TARGET':<8}"
        f" {'METRIC':>8}{metric_name_col} {'STATUS':<10} {'AGO'}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for _, row in runs.iterrows():
        exp_id = str(row.get("tags.kego_id", "?"))[:6]
        name = str(row.get("tags.mlflow.runName", "?"))[:26]
        mlflow_exp_id = str(row.get("experiment_id", ""))
        competition = (exp_names or {}).get(mlflow_exp_id, "?")[:20]
        target = str(row.get("tags.kego_target", "local"))[:8]
        metric_name = str(row.get("tags.kego_primary_metric") or fallback_metric)
        metric = _metric_str(row, fallback_metric)
        status = str(row.get("status", "?"))[:10]
        start = row.get("start_time")
        ago = _ago(start) if start is not None and pd.notna(start) else "?"
        metric_name_cell = f" {metric_name:<10}" if show_metric_name else ""
        lines.append(
            f"{exp_id:<8} {name:<26} {competition:<20} {target:<8} {metric:>8}{metric_name_cell} {status:<10} {ago}"
        )

    return lines


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("ls", help="List and compare experiments")
    p.add_argument("--name", metavar="SUBSTR", help="Filter by run name prefix")
    p.add_argument(
        "--status",
        choices=_STATUSES,
        metavar="STATUS",
        help=f"Filter by run status: {', '.join(_STATUSES)}",
    )
    p.add_argument(
        "--target",
        choices=["local", "cluster"],
        metavar="TARGET",
        help="Filter by compute target: local, cluster",
    )
    p.add_argument(
        "--competition",
        metavar="SLUG",
        help="Filter by competition slug (e.g. birdclef-2026)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=50,
        metavar="N",
        help="Max runs to show (default: 50)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        dest="show_all",
        help="Include debug runs",
    )
    p.add_argument(
        "--metric-name",
        action="store_true",
        dest="show_metric_name",
        help="Show metric name column",
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
    if args.status:
        filter_parts.append(f"attributes.status = '{args.status.upper()}'")
    if args.target:
        filter_parts.append(f"tags.kego_target = '{args.target}'")

    filter_string = " AND ".join(filter_parts) if filter_parts else ""

    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()

        # Scope experiment lookup when --competition is given to avoid fetching all
        if args.competition:
            scoped = client.search_experiments(filter_string=f"name = '{args.competition}'")
            if not scoped:
                print(f"No experiments found for competition: {args.competition}")
                return 0
            exp_names = {e.experiment_id: e.name for e in scoped}
            experiment_ids: list[str] | None = list(exp_names)
        else:
            all_exps = client.search_experiments()
            exp_names = {e.experiment_id: e.name for e in all_exps}
            experiment_ids = None

        search_kwargs: dict = {
            "filter_string": filter_string,
            "order_by": ["start_time DESC"],
            "max_results": args.limit,
        }
        if experiment_ids is not None:
            search_kwargs["experiment_ids"] = experiment_ids
        else:
            search_kwargs["search_all_experiments"] = True

        runs = mlflow.search_runs(**search_kwargs)
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

    for line in format_table(runs, primary_metric, exp_names, args.show_metric_name):
        print(line)

    return 0
