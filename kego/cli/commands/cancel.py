"""kego cancel — cancel Ray jobs for a given kego experiment ID or run name."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

from kego.cli import config as cfg_module

_TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "STOPPED"}


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("cancel", help="Cancel running cluster jobs")
    p.add_argument("id", help="Kego experiment ID (or prefix) or run name")
    p.set_defaults(func=_cancel)


def _ray_stop(base: str, submission_id: str) -> None:
    """Send POST .../stop to the Ray Jobs API to stop a running job."""
    url = f"{base}/api/jobs/{submission_id}/stop"
    req = urllib.request.Request(url, data=b"", method="POST")  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=15):  # noqa: S310
            pass
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ray API error (HTTP {e.code}): {e.read().decode()}") from e
    except OSError as e:
        raise RuntimeError(
            f"Cannot reach Ray cluster at {base} — is the cluster online?"
        ) from e


def _cancel(args: argparse.Namespace, extra_args: list[str]) -> int:
    import logging

    import mlflow

    config = cfg_module.load_config()
    base = config.cluster.ray_address.rstrip("/")

    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.WARNING)

    try:
        mlflow.set_tracking_uri(config.cluster.mlflow_uri)
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        experiment_ids = [e.experiment_id for e in client.search_experiments()]
        # MLflow doesn't support OR — run two queries and merge by run_id
        by_id = client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=f"tags.kego_id LIKE '{args.id}%'",
            max_results=10,
        )
        by_name = client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=f"tags.`mlflow.runName` LIKE '%{args.id}%'",
            max_results=10,
        )
        seen: set[str] = set()
        runs = []
        for r in [*by_id, *by_name]:
            if r.info.run_id not in seen:
                seen.add(r.info.run_id)
                runs.append(r)
    except Exception as e:
        print(f"Error reaching MLflow: {e}")
        return 1

    if not runs:
        print(f"No runs found matching: {args.id}")
        return 1

    cancelled = 0
    for run in runs:
        submission_id = run.data.tags.get("ray_submission_id")
        fold = run.data.params.get("fold", "?")

        if not submission_id:
            print(f"  fold {fold}: no ray_submission_id tag, skipping")
            continue

        # Check current Ray status
        try:
            url = f"{base}/api/jobs/{submission_id}"
            req = urllib.request.Request(url)  # noqa: S310
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
                job = json.loads(resp.read())
            status = job.get("status", "UNKNOWN")
        except (urllib.error.HTTPError, OSError) as e:
            print(f"  fold {fold}: could not fetch status ({e}), skipping")
            continue

        if status in _TERMINAL_STATUSES:
            print(f"  fold {fold}: already {status}, skipping")
            continue

        try:
            _ray_stop(base, submission_id)
            print(f"  fold {fold}: cancelled ({submission_id})")
            cancelled += 1
        except RuntimeError as e:
            print(f"  fold {fold}: {e}")

    print(f"\nCancelled {cancelled} job(s).")
    return 0
