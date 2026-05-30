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
    import json
    import urllib.request

    url = ray_address.rstrip("/") + "/api/jobs/"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            jobs = json.loads(resp.read())
    except Exception:
        return {}
    return {j["submission_id"]: j.get("status", "") for j in jobs if j.get("submission_id")}


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

    # Decide a new status for each RUNNING run: Ray terminal state wins, else time-kill.
    actions: list[tuple[str, str, str, str]] = []  # (run_id, name, new_status, reason)
    for _, r in runs.iterrows():
        sub_id = r.get("tags.ray_submission_id")
        run_id = r["run_id"]
        name = str(r.get("tags.mlflow.runName", "?"))[:26]
        start = r.get("start_time")

        import pandas as pd

        if sub_id is not None and pd.notna(sub_id) and sub_id in ray_status:
            mapped = _RAY_TERMINAL.get(ray_status[sub_id])
            if mapped:
                actions.append((run_id, name, mapped, f"ray {ray_status[sub_id]}"))
                continue
            # Ray says still RUNNING/PENDING — leave it.
            continue
        if start is not None and pd.notna(start) and int(start.timestamp() * 1000) < cutoff_ms:
            actions.append((run_id, name, "KILLED", f"older than {args.older_than}, no live Ray job"))

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
