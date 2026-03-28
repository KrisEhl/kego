"""kego kernel-status — show Kaggle kernel status for a submitted experiment."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

from kego.cli import config as cfg_module


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "kernel-status", help="Show Kaggle kernel status for a submitted experiment"
    )
    p.add_argument(
        "experiment",
        metavar="ID_OR_NAME",
        help="Kego experiment ID (prefix) or run name",
    )
    p.add_argument(
        "--competition",
        metavar="SLUG",
        help="Competition slug (auto-detected from cwd if omitted)",
    )
    p.add_argument(
        "--output",
        action="store_true",
        help="Also fetch kernel output log",
    )
    p.set_defaults(func=_kernel_status)


def _kernel_status(args: argparse.Namespace, extra_args: list[str]) -> int:
    import mlflow
    from mlflow.tracking import MlflowClient

    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.WARNING)

    competition_dir = None
    if args.competition:
        competition_dir = cfg_module.find_competition_dir_by_slug(args.competition)
        if competition_dir is None:
            print(f"Error: competition '{args.competition}' not found")
            return 1
    config = cfg_module.load_config(competition_dir=competition_dir)

    if config.competition is None or config.competition_dir is None:
        print(
            "Error: no competition config found. "
            "Run from a competition directory or pass --competition."
        )
        return 1

    mlflow.set_tracking_uri(config.cluster.mlflow_uri)
    client = MlflowClient()

    try:
        from kego.cli.experiment import resolve_runs

        experiment_ids = [e.experiment_id for e in client.search_experiments()]
        runs = resolve_runs(args.experiment, client, experiment_ids)
    except Exception as e:
        print(f"Error reaching MLflow: {e}")
        return 1

    if not runs:
        print(f"No runs found matching: {args.experiment}")
        return 1

    run = runs[0]
    run_name = run.data.tags.get("mlflow.runName", args.experiment)
    version = run.data.tags.get("kaggle_kernel_version")

    if not version:
        print(f"Run '{run_name}' has not been submitted as a Kaggle kernel.")
        return 1

    # Resolve kernel ref from kernel-metadata.json
    comp = config.competition
    inference_dir = config.competition_dir / Path(comp.inference_notebook).parent
    metadata_path = inference_dir / "kernel-metadata.json"
    if not metadata_path.exists():
        print(f"Error: kernel-metadata.json not found at {metadata_path}")
        return 1

    with open(metadata_path) as f:
        kernel_meta = json.load(f)
    kernel_ref = kernel_meta["id"]

    print(f"Run     : {run_name}")
    print(f"Kernel  : {kernel_ref}  (v{version})")
    print()

    result = subprocess.run(  # noqa: S603
        ["kaggle", "kernels", "status", kernel_ref],  # noqa: S607
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip() or result.stderr.strip())

    if args.output:
        print()
        out = subprocess.run(  # noqa: S603
            ["kaggle", "kernels", "output", kernel_ref],  # noqa: S607
            capture_output=True,
            text=True,
        )
        print(out.stdout.strip() or out.stderr.strip())

    return 0 if result.returncode == 0 else 1
