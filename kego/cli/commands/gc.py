"""kego gc — reconcile zombie RUNNING MLflow runs against Ray job state."""

from __future__ import annotations

import argparse
import datetime

# Ray terminal job status → MLflow run status to set.
_RAY_TERMINAL = {"SUCCEEDED": "FINISHED", "STOPPED": "KILLED", "FAILED": "FAILED"}


def _ms_cutoff(value: str) -> int:
    """Parse a duration string (30m / 2h / 7d) to a UTC millisecond cutoff."""
    units = {"m": "minutes", "h": "hours", "d": "days"}
    unit = value[-1]
    if unit not in units or not value[:-1].isdigit():
        raise argparse.ArgumentTypeError(f"Invalid duration '{value}'. Use e.g. 30m, 2h, 7d.")
    delta = datetime.timedelta(**{units[unit]: int(value[:-1])})
    cutoff = datetime.datetime.now(tz=datetime.timezone.utc) - delta
    return int(cutoff.timestamp() * 1000)


def _ray_job_statuses(ray_address: str) -> dict[str, str]:
    """Map Ray submission_id → status. Empty dict if Ray is unreachable."""
    from kego.cli import ray

    return ray.job_statuses(ray_address)


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "gc",
        help="Reconcile zombie RUNNING runs against Ray job state",
        description=(
            "Fix RUNNING MLflow runs that are actually finished. First reconciles each "
            "RUNNING run against its Ray job (SUCCEEDED→FINISHED, STOPPED/FAILED→KILLED). "
            "Then kills any remaining RUNNING run older than --older-than (catches local "
            "zombies and cluster-restart orphans with no live Ray job)."
        ),
        epilog=(
            "Examples:\n"
            "  # Reconcile against Ray + kill RUNNING runs older than 1h\n"
            "  uv run kego gc\n"
            "\n"
            "  # Preview without making changes\n"
            "  uv run kego gc --dry-run\n"
            "\n"
            "  # Custom time threshold for the fallback kill\n"
            "  uv run kego gc --older-than 30m\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--older-than",
        default="1h",
        metavar="DURATION",
        dest="older_than",
        help="Time-kill RUNNING runs with no terminal Ray job older than this (default: 1h)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Show what would change without making any changes",
    )
    p.set_defaults(func=_gc)


def _gc(args: argparse.Namespace, extra_args: list[str]) -> int:
    # Lazy imports (function top, once): defer heavy deps so light CLI commands stay fast.
    import logging
    import os

    import mlflow
    import pandas as pd
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
            filter_string="attributes.status = 'RUNNING'",
            order_by=["start_time ASC"],
        )
    except Exception:
        print(
            f"Cannot reach MLflow at {tracking_uri} — is the cluster online?\n"
            "  Start cluster : uv run kego cluster start"
        )
        return 1

    if runs.empty:
        print("No RUNNING runs to reconcile.")
        return 0

    ray_status = _ray_job_statuses(config.cluster.ray_address)

    # New status for one RUNNING run: Ray terminal state wins; else time-kill if old + no live job.
    def _decide(r) -> tuple[str, str] | None:  # (new_status, reason) or None to leave alone
        if (sub := r.get("tags.ray_submission_id")) is not None and pd.notna(sub) and sub in ray_status:
            return (m, f"ray {ray_status[sub]}") if (m := _RAY_TERMINAL.get(ray_status[sub])) else None
        start = r.get("start_time")
        if start is not None and pd.notna(start) and int(start.timestamp() * 1000) < cutoff_ms:
            return "KILLED", f"older than {args.older_than}, no live Ray job"
        return None

    actions = [
        (r["run_id"], str(r.get("tags.mlflow.runName", "?"))[:26], *d) for _, r in runs.iterrows() if (d := _decide(r))
    ]

    if not actions:
        print("No zombie runs — all RUNNING runs have live Ray jobs and are recent.")
        return 0

    verb = "Would update" if args.dry_run else "Updating"
    print(f"{verb} {len(actions)} zombie run(s):")
    for run_id, name, new_status, reason in actions:
        print(f"  {run_id[:8]}  {name:<26}  → {new_status:<9} ({reason})")
        if not args.dry_run:
            client.set_terminated(run_id, status=new_status)

    if args.dry_run:
        print("\nDry run — no changes made. Remove --dry-run to apply.")

    return 0
