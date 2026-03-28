"""kego push — package checkpoints and upload as a Kaggle dataset."""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from kego.cli import config as cfg_module


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "push", help="Upload model checkpoints to Kaggle Datasets"
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
    p.set_defaults(func=_push)


def dataset_slug(kaggle_user: str, competition_slug: str, run_name: str) -> str:
    """Build Kaggle dataset ID: {user}/{slugclean}-{run_name}."""
    slug_clean = re.sub(r"[^a-z0-9]", "", competition_slug.lower())
    return f"{kaggle_user}/{slug_clean}-{run_name}"


def _push(args: argparse.Namespace, extra_args: list[str]) -> int:
    import mlflow
    from mlflow.tracking import MlflowClient

    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.WARNING)

    # Resolve competition
    competition_dir: Path | None = None
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

    comp = config.competition

    # Resolve experiment
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
    kego_id = run.data.tags.get("kego_id", "unknown")

    # Find checkpoints
    ckpt_dir = config.competition_dir / comp.checkpoint_dir
    checkpoints = sorted(ckpt_dir.glob(f"{run_name}_fold*.pt"))
    if not checkpoints:
        print(f"No checkpoints found: {ckpt_dir}/{run_name}_fold*.pt")
        return 1

    print(f"Packaging {len(checkpoints)} checkpoint(s) for '{run_name}' [{kego_id}]:")
    for f in checkpoints:
        print(f"  {f.name}")

    ds = dataset_slug(comp.kaggle_user, comp.slug, run_name)
    print(f"\nTarget dataset: {ds}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        for f in checkpoints:
            shutil.copy2(f, tmppath / f.name)

        metadata = {
            "title": f"{comp.slug} {run_name} checkpoints",
            "id": ds,
            "licenses": [{"name": "CC0-1.0"}],
        }
        (tmppath / "dataset-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

        # Try version (dataset exists) → fall back to create
        result = subprocess.run(  # noqa: S603
            ["kaggle", "datasets", "version", "-p", tmpdir, "-m", f"kego_id={kego_id}"],  # noqa: S607
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 and (
            "404" in result.stderr or "not found" in result.stderr.lower()
        ):
            print("Dataset not found — creating new dataset...")
            result = subprocess.run(  # noqa: S603
                ["kaggle", "datasets", "create", "-p", tmpdir],  # noqa: S607
                capture_output=True,
                text=True,
            )

        if result.returncode != 0:
            print(f"Error: {result.stderr or result.stdout}")
            return 1

        print(result.stdout.strip())

    print(f"\nDone. To submit:\n  uv run kego submit --experiment {kego_id}")
    return 0
