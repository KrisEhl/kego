"""kego gc — kill zombie RUNNING runs older than a threshold."""

from __future__ import annotations

import argparse
import datetime


def _ms_cutoff(value: str) -> int:
    """Parse a duration string (30m / 2h / 7d) to a UTC millisecond cutoff."""
    units = {"m": "minutes", "h": "hours", "d": "days"}
    unit = value[-1]
    if unit not in units or not value[:-1].isdigit():
        raise argparse.ArgumentTypeError(f"Invalid duration '{value}'. Use e.g. 30m, 2h, 7d.")
    delta = datetime.timedelta(**{units[unit]: int(value[:-1])})
    cutoff = datetime.datetime.now(tz=datetime.timezone.utc) - delta
    return int(cutoff.timestamp() * 1000)


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "gc",
        help="Kill zombie RUNNING runs older than a threshold",
        description=(
            "Find all RUNNING MLflow runs older than --older-than and mark them KILLED. "
            "Useful after a cluster restart leaves runs stuck in RUNNING state."
        ),
        epilog=(
            "Examples:\n"
            "  # Kill RUNNING runs older than 1 hour (default)\n"
            "  uv run kego gc\n"
            "\n"
            "  # Preview without making changes\n"
            "  uv run kego gc --dry-run\n"
            "\n"
            "  # Custom threshold\n"
            "  uv run kego gc --older-than 30m\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--older-than",
        default="1h",
        metavar="DURATION",
        dest="older_than",
        help="Kill RUNNING runs older than this (default: 1h). Format: 30m, 2h, 7d",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Show what would be killed without making any changes",
    )
    p.set_defaults(func=_gc)


def _gc(args: argparse.Namespace, extra_args: list[str]) -> int:
    import logging
    import os

    import mlflow
    from mlflow.tracking import MlflowClient

    from kego.cli import config as cfg_module

    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.WARNING)

    try:
        cutoff_ms = _ms_cutoff(args.older_than)
    except argparse.ArgumentTypeError as e:
        print(str(e))
        return 1

    config = cfg_module.load_config()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or config.cluster.mlflow_uri
    mlflow.set_tracking_uri(tracking_uri)

    try:
        client = MlflowClient()
        runs = mlflow.search_runs(
            search_all_experiments=True,
            filter_string=(f"attributes.status = 'RUNNING' AND attributes.start_time < {cutoff_ms}"),
            order_by=["start_time ASC"],
        )
    except Exception:
        print(
            f"Cannot reach MLflow at {tracking_uri} — is the cluster online?\n"
            "  Start cluster : uv run kego cluster start"
        )
        return 1

    if runs.empty:
        print(f"No zombie RUNNING runs older than {args.older_than}.")
        return 0

    verb = "Would kill" if args.dry_run else "Killing"
    print(f"{verb} {len(runs)} zombie run(s) older than {args.older_than}:")
    for _, r in runs.iterrows():
        name = str(r.get("tags.mlflow.runName", "?"))
        start = r.get("start_time")
        ts = start.strftime("%Y-%m-%d %H:%M") if start is not None else "?"
        print(f"  {r['run_id'][:8]}  {name:<26}  {ts}")
        if not args.dry_run:
            client.set_terminated(r["run_id"], status="KILLED")

    if args.dry_run:
        print("\nDry run — no changes made. Remove --dry-run to kill.")

    return 0
