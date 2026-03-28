"""kego logs — stream Ray job logs for a given kego experiment ID."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

from kego.cli import config as cfg_module


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("logs", help="Show logs for a cluster job")
    p.add_argument("id", help="Kego experiment ID (or prefix)")
    p.set_defaults(func=_logs)


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

        # Search for runs matching the kego_id tag prefix
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

    for run in runs:
        submission_id = run.data.tags.get("ray_submission_id")
        fold = run.data.params.get("fold", "?")
        run_name = run.info.run_name
        print(f"=== {run_name} fold={fold} ===")

        if not submission_id:
            print(
                "  No ray_submission_id tag — job may have been submitted before log tracking was added."
            )
            continue

        base = config.cluster.ray_address.rstrip("/")
        url = f"{base}/api/jobs/{submission_id}/logs"
        req = urllib.request.Request(url)  # noqa: S310
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
                result = json.loads(resp.read())
                logs = result.get("logs", "")
                print(logs or "(no logs yet)")
        except urllib.error.HTTPError as e:
            print(f"  Ray API error (HTTP {e.code}): {e.read().decode()}")
        except OSError:
            print(
                f"  Cannot reach Ray cluster at {config.cluster.ray_address} — "
                "is the cluster online?\n"
                "  Start cluster: make cluster-start"
            )
            return 1

    return 0
