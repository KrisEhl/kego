"""kego kernel-list — show MLflow runs that have been submitted as Kaggle kernels."""

from __future__ import annotations

import argparse
import logging

from kego.cli import config as cfg_module


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "kernel-list", help="List runs submitted as Kaggle kernels"
    )
    p.add_argument(
        "--competition",
        metavar="SLUG",
        help="Competition slug (auto-detected from cwd if omitted)",
    )
    p.set_defaults(func=_kernel_list)


def _kernel_list(args: argparse.Namespace, extra_args: list[str]) -> int:
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

    mlflow.set_tracking_uri(config.cluster.mlflow_uri)
    client = MlflowClient()

    try:
        experiment_ids = [e.experiment_id for e in client.search_experiments()]
        runs = client.search_runs(
            experiment_ids=experiment_ids,
            filter_string="tags.kaggle_kernel_version != ''",
            order_by=["start_time DESC"],
            max_results=50,
        )
    except Exception as e:
        print(f"Error reaching MLflow: {e}")
        return 1

    if not runs:
        print("No submitted kernels found.")
        return 0

    print(f"{'RUN NAME':<35} {'KEGO ID':<12} {'KERNEL VERSION'}")
    print("-" * 65)
    for r in runs:
        name = r.data.tags.get("mlflow.runName", r.info.run_id)
        kego_id = r.data.tags.get("kego_id", "?")
        version = r.data.tags.get("kaggle_kernel_version", "?")
        print(f"{name:<35} {kego_id:<12} v{version}")

    return 0
