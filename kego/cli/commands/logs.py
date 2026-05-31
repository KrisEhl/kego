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
    p.add_argument("id", nargs="?", help="Kego experiment ID/prefix or run name (default: most recent run)")
    p.add_argument("--fold", type=int, help="Show logs for a specific fold only")
    p.add_argument(
        "--no-follow",
        action="store_true",
        help="Print logs once and exit (default: follow until job finishes)",
    )
    p.set_defaults(func=_logs)


def _latest_job_run(client, experiment_ids: list[str]) -> list:
    """Most recent run with a Ray job, preferring RUNNING. Returns [run] or []."""

    def _pick(filter_string: str):
        runs = client.search_runs(
            experiment_ids, filter_string=filter_string, order_by=["start_time DESC"], max_results=20
        )
        return [r for r in runs if r.data.tags.get("kego_is_parent") != "true" and r.data.tags.get("ray_submission_id")]

    candidates = _pick("attributes.status = 'RUNNING'") or _pick("")
    return candidates[:1]


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
            print(f"  Cannot reach Ray cluster at {base} — is the cluster online?\n  Start cluster: make cluster-start")
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

        from kego.cli.experiment import resolve_runs

        experiment_ids = [e.experiment_id for e in client.search_experiments()]
        if args.id:
            runs = resolve_runs(args.id, client, experiment_ids)
        else:
            runs = _latest_job_run(client, experiment_ids)
    except Exception as e:
        print(f"Error reaching MLflow at {tracking_uri}: {e}")
        return 1

    if not runs:
        msg = f"No runs found matching ID: {args.id}" if args.id else "No cluster jobs found to show logs for."
        print(msg)
        return 1

    if not args.id:
        r0 = runs[0]
        print(
            f"(latest run: {r0.data.tags.get('mlflow.runName', '?')} [{r0.data.tags.get('kego_id', '?')}])", flush=True
        )

    runs = [r for r in runs if r.data.tags.get("kego_is_parent") != "true"]

    if args.fold is not None:
        runs = [r for r in runs if r.data.params.get("fold") == str(args.fold)]
        if not runs:
            print(f"No runs found for fold {args.fold}")
            return 1

    follow = not args.no_follow
    base = config.cluster.ray_address.rstrip("/")

    for run in runs:
        submission_id = run.data.tags.get("ray_submission_id")
        fold = run.data.params.get("fold", "?")
        print(f"=== {run.info.run_name} fold={fold} ===")

        if not submission_id:
            # Local run (no Ray job) — replay the captured stdout file.
            _print_local_log(run.info.run_id, run.data.tags.get("kego_target"))
            continue

        rc = _stream_job_logs(base, submission_id, follow=follow)
        if rc != 0:
            return rc

    return 0


def _print_local_log(run_id: str, target: str | None) -> None:
    """Print a local run's captured stdout (tee'd by the runner)."""
    from kego.cli.runner import local_log_path

    path = local_log_path(run_id)
    if path.exists():
        print(path.read_text(), end="")
    elif target == "cluster":
        print("  No ray_submission_id tag — submitted before log tracking was added.")
    else:
        print("  No captured stdout for this local run (it may predate local-log capture).")
