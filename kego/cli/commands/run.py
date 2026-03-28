from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from kego.cli import config as cfg_module
from kego.cli import experiment as exp_module
from kego.cli.targets import cluster as cluster_target
from kego.cli.targets import local as local_target


def _pre_create_runs(
    config: cfg_module.KegoConfig,
    experiment_name: str,
    run_name: str,
    experiment_id: str,
    cli_params: dict[str, str],
    folds: list[int],
) -> dict[int, str]:
    """Create one MLflow run per fold so cluster jobs appear immediately in kego ls.

    Returns a dict mapping fold index → mlflow run_id.
    Falls back gracefully if MLflow is unreachable.
    """
    import mlflow

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or config.cluster.mlflow_uri
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.WARNING)

    run_ids: dict[int, str] = {}
    try:
        from mlflow.entities import Param
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        exp = mlflow.set_experiment(experiment_name)
        for fold in folds:
            fold_params = {**cli_params, "fold": str(fold)}
            tags = {
                "kego_id": experiment_id,
                "kego_target": "cluster",
                "kego_debug": "false",
                "mlflow.runName": run_name,
            }
            # Create run via low-level client so it stays RUNNING (no context manager).
            # The cluster runner will resume it by run_id and call set_terminated().
            run = client.create_run(
                experiment_id=exp.experiment_id,
                run_name=run_name,
                tags=tags,
            )
            client.log_batch(
                run.info.run_id,
                params=[Param(k, str(v)) for k, v in fold_params.items()],
            )
            run_ids[fold] = run.info.run_id
    except Exception as e:
        print(
            f"  Warning: could not pre-create MLflow runs ({e})"
            " — jobs won't appear in kego ls until they start",
            flush=True,
        )

    return run_ids


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("run", help="Dispatch a training script")
    p.add_argument("script", help="Path to training script")
    p.add_argument(
        "--target",
        choices=["local", "cluster"],
        default="local",
        help="Compute target (default: local)",
    )
    p.add_argument("--name", help="Experiment name (auto-generated if omitted)")
    p.add_argument("--fold", type=int, help="Single fold index")
    p.add_argument("--folds", help="Comma-separated fold indices, e.g. 0,1,2,3")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Smoke-test mode (forwards --debug to script)",
    )
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace, extra_args: list[str]) -> int:
    script_path = Path(args.script).resolve()
    if not script_path.exists():
        print(f"Error: script not found: {args.script}")
        return 1

    config = cfg_module.load_config()

    # Resolve folds
    if args.folds:
        folds: list[int] | None = [int(f) for f in args.folds.split(",")]
    elif args.fold is not None:
        folds = [args.fold]
    else:
        folds = None

    # Build script args: pass through extra_args + debug flag
    script_args = list(extra_args)
    if args.debug:
        script_args.append("--debug")

    # Parse extra_args into a params dict for MLflow logging
    cli_params: dict[str, str] = {}
    i = 0
    while i < len(extra_args):
        a = extra_args[i]
        if a.startswith("--"):
            key = a[2:]
            if "=" in key:
                k, v = key.split("=", 1)
                cli_params[k] = v
            elif i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
                cli_params[key] = extra_args[i + 1]
                i += 1
            else:
                cli_params[key] = "true"
        i += 1

    script = str(script_path)
    run_name = exp_module.build_experiment_name(script, args.name, cli_params)
    experiment_name = config.competition.slug if config.competition else run_name
    experiment_id = exp_module.generate_id()

    print(f"kego run: {run_name} [{experiment_id}] → {args.target}", flush=True)

    if args.target == "local":
        if folds and len(folds) > 1:
            print(
                f"Note: local target runs single fold only (fold {folds[0]}). Use --target cluster for fan-out.",
                flush=True,
            )
        fold_args = script_args[:]
        if folds:
            fold_args += ["--fold", str(folds[0])]
        fold_cli_params = {**cli_params, **({"fold": str(folds[0])} if folds else {})}
        return local_target.run(
            script=script,
            script_args=fold_args,
            config=config,
            experiment_name=experiment_name,
            run_name=run_name,
            experiment_id=experiment_id,
            cli_params=fold_cli_params,
        )

    elif args.target == "cluster":
        resolved_folds = folds if folds is not None else [0]

        # Pre-create one MLflow run per fold so jobs appear immediately in kego ls.
        mlflow_run_ids = _pre_create_runs(
            config=config,
            experiment_name=experiment_name,
            run_name=run_name,
            experiment_id=experiment_id,
            cli_params=cli_params,
            folds=resolved_folds,
        )

        job_ids = cluster_target.submit(
            script=script,
            folds=resolved_folds,
            base_args=script_args,
            config=config,
            experiment_name=experiment_name,
            run_name=run_name,
            experiment_id=experiment_id,
            cli_params=cli_params,
            mlflow_run_ids=mlflow_run_ids,
        )

        # Tag each pre-created MLflow run with its Ray submission ID for kego logs.
        if mlflow_run_ids and job_ids:
            try:
                import mlflow

                tracking_uri = (
                    os.environ.get("MLFLOW_TRACKING_URI") or config.cluster.mlflow_uri
                )
                mlflow.set_tracking_uri(tracking_uri)
                from mlflow.tracking import MlflowClient

                client = MlflowClient()
                for fold, job_id in zip(resolved_folds, job_ids):
                    run_id = mlflow_run_ids.get(fold)
                    if run_id:
                        client.set_tag(run_id, "ray_submission_id", job_id)
            except Exception as e:
                print(
                    f"  Warning: could not tag runs with ray_submission_id ({e})",
                    flush=True,
                )

        print(
            f"\nSubmitted {len(job_ids)} job(s). Track with: uv run kego ls", flush=True
        )
        return 0

    return 1
