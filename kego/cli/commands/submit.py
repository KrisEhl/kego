"""kego submit — push inference kernel to Kaggle and poll for completion."""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from kego.cli import config as cfg_module
from kego.cli.commands.push import dataset_slug

_POLL_INTERVAL = 30  # seconds


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("submit", help="Push inference kernel to Kaggle")
    p.add_argument(
        "--experiment",
        required=True,
        metavar="ID_OR_NAME",
        help="Kego experiment ID (prefix) or run name",
    )
    p.add_argument(
        "--competition",
        metavar="SLUG",
        help="Competition slug (auto-detected from cwd if omitted)",
    )
    p.add_argument("-m", "--message", default="", help="Submission message")
    p.add_argument(
        "--no-wait",
        action="store_true",
        help="Push and return without polling for completion",
    )
    p.set_defaults(func=_submit)


def _poll_kernel(kernel_ref: str) -> str:
    """Poll kaggle kernels status until terminal state. Returns 'complete' or 'error'."""
    time.sleep(_POLL_INTERVAL)  # initial delay — kernel won't start instantly
    while True:
        result = subprocess.run(  # noqa: S603
            ["kaggle", "kernels", "status", kernel_ref],  # noqa: S607
            capture_output=True,
            text=True,
        )
        output = (result.stdout + result.stderr).lower()
        if "complete" in output:
            return "complete"
        if "error" in output or "cancel" in output or "failed" in output:
            return "error"
        print(
            f"  Kernel running... (next check in {_POLL_INTERVAL}s)",
            flush=True,
        )
        time.sleep(_POLL_INTERVAL)


def _submit(args: argparse.Namespace, extra_args: list[str]) -> int:
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

    # Find kernel-metadata.json in the inference directory
    inference_dir = config.competition_dir / Path(comp.inference_notebook).parent
    metadata_path = inference_dir / "kernel-metadata.json"
    if not metadata_path.exists():
        print(f"Error: kernel-metadata.json not found at {metadata_path}")
        return 1

    with open(metadata_path) as f:
        kernel_meta = json.load(f)

    # Find the notebook file
    notebook_file = inference_dir / kernel_meta["code_file"]
    if not notebook_file.exists():
        print(f"Error: notebook not found at {notebook_file}")
        return 1

    # Update dataset_sources: replace all {user}/{slugclean}-* entries with the new one
    checkpoint_ds = dataset_slug(comp.kaggle_user, comp.slug, run_name)
    slug_clean = re.sub(r"[^a-z0-9]", "", comp.slug.lower())
    checkpoint_prefix = f"{comp.kaggle_user}/{slug_clean}-"
    existing = kernel_meta.get("dataset_sources", [])
    other = [s for s in existing if not s.startswith(checkpoint_prefix)]
    kernel_meta["dataset_sources"] = [*other, checkpoint_ds]
    kernel_meta["enable_gpu"] = comp.enable_gpu

    kernel_ref = kernel_meta["id"]
    message = args.message or f"{run_name} [{kego_id}]"

    print(f"Submitting: {kernel_ref}")
    print(f"  Notebook : {notebook_file.name}")
    print(f"  Datasets : {kernel_meta['dataset_sources']}")
    print(f"  Message  : {message}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        shutil.copy2(notebook_file, tmppath / notebook_file.name)
        (tmppath / "kernel-metadata.json").write_text(
            json.dumps(kernel_meta, indent=2) + "\n"
        )

        result = subprocess.run(  # noqa: S603
            ["kaggle", "kernels", "push", "-p", tmpdir],  # noqa: S607
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Error pushing kernel:\n{result.stderr or result.stdout}")
            return 1

        print(result.stdout.strip())

    if args.no_wait:
        print(f"\nKernel submitted. Check: kaggle kernels status {kernel_ref}")
        return 0

    print("\nPolling for completion...", flush=True)
    status = _poll_kernel(kernel_ref)

    if status == "complete":
        print(f"\nComplete. Output: kaggle kernels output {kernel_ref}")
        return 0

    print(f"\nKernel finished with status: {status}")
    return 1
