"""kego logs — stream Ray job logs for a given kego experiment ID."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

from kego.cli import config as cfg_module

_TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "STOPPED"}
_POLL_INTERVAL = 2  # seconds between log polls


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("logs", help="Show logs for a cluster job")
    p.add_argument("id", help="Kego experiment ID (or prefix)")
    p.add_argument(
        "--no-follow",
        action="store_true",
        help="Print logs once and exit (default: follow until job finishes)",
    )
    p.set_defaults(func=_logs)


def _ray_get(base: str, path: str) -> dict:
    """GET a Ray API endpoint. Raises OSError or HTTPError on failure."""
    req = urllib.request.Request(f"{base}{path}")  # noqa: S310
    with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
        return json.loads(resp.read())


def _stream_job_logs(base: str, submission_id: str, follow: bool) -> int:
    """Print logs for one Ray job, optionally following until it finishes."""
    offset = 0
    while True:
        try:
            result = _ray_get(base, f"/api/jobs/{submission_id}/logs")
        except urllib.error.HTTPError as e:
            print(f"  Ray API error (HTTP {e.code}): {e.read().decode()}")
            return 1 if e.code == 404 else 0
        except OSError:
            print(
                f"  Cannot reach Ray cluster at {base} — "
                "is the cluster online?\n"
                "  Start cluster: make cluster-start"
            )
            return 1

        logs = result.get("logs", "")
        if len(logs) > offset:
            print(logs[offset:], end="", flush=True)
            offset = len(logs)
        elif offset == 0:
            print("(no logs yet)", flush=True)

        if not follow:
            break

        # Check whether the job has reached a terminal state
        try:
            job = _ray_get(base, f"/api/jobs/{submission_id}")
        except (urllib.error.HTTPError, OSError):
            break

        if job.get("status") in _TERMINAL_STATUSES:
            # One final log flush
            try:
                result = _ray_get(base, f"/api/jobs/{submission_id}/logs")
                logs = result.get("logs", "")
                if len(logs) > offset:
                    print(logs[offset:], end="", flush=True)
            except (urllib.error.HTTPError, OSError):
                pass
            print(f"\n[job {job.get('status', 'DONE')}]", flush=True)
            break

        time.sleep(_POLL_INTERVAL)

    return 0


def _logs(args: argparse.Namespace, extra_args: list[str]) -> int:
    config = cfg_module.load_config()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or config.cluster.mlflow_uri

    try:
        import logging

        import mlflow

        logging.getLogger("mlflow").setLevel(logging.WARNING)
        logging.getLogger("alembic").setLevel(logging.WARNING)

        mlflow.set_tracking_uri(tracking_uri)
        from mlflow.tracking import MlflowClient

        client = MlflowClient()

        runs = client.search_runs(
            experiment_ids=[e.experiment_id for e in client.search_experiments()],
            filter_string=f"tags.kego_id LIKE '{args.id}%'",
            max_results=10,
        )
    except Exception as e:
        print(f"Error reaching MLflow at {tracking_uri}: {e}")
        return 1

    if not runs:
        print(f"No runs found matching ID: {args.id}")
        return 1

    follow = not args.no_follow
    base = config.cluster.ray_address.rstrip("/")

    for run in runs:
        submission_id = run.data.tags.get("ray_submission_id")
        fold = run.data.params.get("fold", "?")
        print(f"=== {run.info.run_name} fold={fold} ===")

        if not submission_id:
            print(
                "  No ray_submission_id tag — job may have been submitted before log tracking was added."
            )
            continue

        rc = _stream_job_logs(base, submission_id, follow=follow)
        if rc != 0:
            return rc

    return 0
