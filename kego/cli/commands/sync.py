"""kego sync — copy offline (local sqlite) MLflow runs to the cluster server.

When the cluster is down, run locally against a local sqlite tracking store
(MLFLOW_TRACKING_URI=sqlite:///~/kego-offline.db). Once the cluster is back,
`kego sync` copies those runs — params, tags, full metric history, status,
timestamps — into the cluster MLflow server. Idempotent: each copied run is
tagged `kego_synced_from=<source_run_id>`, so re-running skips what's already there.
"""

from __future__ import annotations

import argparse
import os


def _expand(uri: str) -> str:
    """Expand ~ in a sqlite:/// path so sqlite:///~/x.db works."""
    prefix = "sqlite:///"
    if uri.startswith(prefix):
        return prefix + os.path.expanduser(uri[len(prefix) :])
    return uri


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "sync",
        help="Copy offline (local sqlite) runs to the cluster MLflow server",
        description=(
            "Copy runs logged offline (while the cluster was down) into the cluster MLflow "
            "server once it's back. Preserves params, tags, full metric history, status and "
            "timestamps. Idempotent — already-synced runs are skipped."
        ),
        epilog=(
            "Examples:\n"
            "  # Preview what would sync from the offline DB to the cluster\n"
            "  uv run kego sync --from sqlite:///~/kego-offline.db --dry-run\n"
            "\n"
            "  # Sync for real (dest defaults to cluster mlflow_uri in kego.toml)\n"
            "  uv run kego sync --from sqlite:///~/kego-offline.db\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--from", dest="source", required=True, metavar="URI", help="Source tracking URI (sqlite:///…)")
    p.add_argument("--to", dest="dest", metavar="URI", help="Dest tracking URI (default: cluster mlflow_uri)")
    p.add_argument("--dry-run", action="store_true", dest="dry_run", help="Show what would sync without writing")
    p.set_defaults(func=_sync)


def _sync(args: argparse.Namespace, extra_args: list[str]) -> int:
    import logging

    from mlflow.tracking import MlflowClient

    from kego.cli import config as cfg_module

    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.WARNING)

    source = _expand(args.source)
    dest = _expand(args.dest) if args.dest else cfg_module.load_config().cluster.mlflow_uri

    src = MlflowClient(tracking_uri=source)
    dst = MlflowClient(tracking_uri=dest)

    try:
        src_exps = [e for e in src.search_experiments() if e.name != "Default"]
    except Exception as e:
        print(f"Cannot read source {source}: {e}")
        return 1

    # Collect source-run-ids already present on dest, so the sync is idempotent.
    try:
        already: set[str] = set()
        for e in dst.search_experiments():
            for r in dst.search_runs([e.experiment_id], max_results=50000):
                if sid := r.data.tags.get("kego_synced_from"):
                    already.add(sid)
    except Exception as e:
        print(f"Cannot reach dest {dest} — is the cluster MLflow up? ({e})")
        return 1

    copied = skipped = 0
    for exp in src_exps:
        for run in src.search_runs([exp.experiment_id], max_results=50000):
            if run.info.run_id in already:
                skipped += 1
                continue
            label = run.data.tags.get("mlflow.runName") or run.info.run_id[:8]
            print(f"  {exp.name} / {label}")
            if args.dry_run:
                copied += 1
                continue

            dexp = dst.get_experiment_by_name(exp.name)
            dexp_id = dexp.experiment_id if dexp else dst.create_experiment(exp.name)
            new = dst.create_run(
                dexp_id,
                start_time=run.info.start_time,
                run_name=run.data.tags.get("mlflow.runName"),
                tags={**run.data.tags, "kego_synced_from": run.info.run_id},
            )
            new_id = new.info.run_id
            for k, v in run.data.params.items():
                dst.log_param(new_id, k, v)
            # Replay full metric history so learning curves survive the copy.
            for metric_key in run.data.metrics:
                for m in src.get_metric_history(run.info.run_id, metric_key):
                    dst.log_metric(new_id, metric_key, m.value, timestamp=m.timestamp, step=m.step)
            dst.set_terminated(new_id, status=run.info.status, end_time=run.info.end_time)
            copied += 1

    verb = "Would copy" if args.dry_run else "Copied"
    print(f"\n{verb} {copied} run(s); skipped {skipped} already-synced.")
    return 0
