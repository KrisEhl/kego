#!/usr/bin/env python3
"""Promote MLflow runs into named ensembles via tags.

Usage:
    # Search: browse runs matching constraints
    python promote.py search --all
    python promote.py search --experiment full-v1 --folds 10 --model xgboost catboost

    # Auto-promote: best holdout_auc per model from experiment(s)
    python promote.py auto submit-v1 --experiment full-v1 gbdt-v1 --folds 10
    python promote.py auto submit-v1 --all --folds 10

    # Manual: tag specific run IDs
    python promote.py add submit-v1 --run-id abc123 def456

    # List runs in an ensemble
    python promote.py list submit-v1

    # Remove runs from an ensemble
    python promote.py remove submit-v1 --run-id abc123

Runs are tagged with `ensemble:<name>` in MLflow.
"""

import argparse
import os
import sys

import mlflow

TAG_PREFIX = "ensemble"


def _learner_id(row):
    """Build learner ID from a run row. Falls back to bare model name."""
    model = row.get("params.model", "")
    feat = row.get("params.feature_set", "")
    folds = row.get("params.folds_n", "")
    if feat and folds:
        return f"{model}/{feat}/{folds}f"
    return model


def _get_client():
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri:
        print("Error: MLFLOW_TRACKING_URI not set", file=sys.stderr)
        sys.exit(1)
    mlflow.set_tracking_uri(uri)
    return mlflow.tracking.MlflowClient()


def _tag_key(ensemble_name):
    return f"{TAG_PREFIX}:{ensemble_name}"


def _collect_and_filter(args):
    """Collect runs from experiments and apply constraint filters.

    Returns filtered DataFrame sorted by model then holdout_auc descending.
    """
    if not args.all and not args.experiment:
        print("Error: specify --experiment or --all", file=sys.stderr)
        sys.exit(1)

    if args.all:
        runs_df = mlflow.search_runs(search_all_experiments=True)
    else:
        import pandas as pd

        all_runs = []
        for exp_name in args.experiment:
            exp = mlflow.get_experiment_by_name(exp_name)
            if exp is None:
                exp = mlflow.get_experiment_by_name(f"playground-s6e2-{exp_name}")
            if exp is None:
                print(f"Warning: experiment '{exp_name}' not found, skipping")
                continue
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            all_runs.append(runs)

        if not all_runs:
            print("No experiments found")
            sys.exit(1)

        runs_df = pd.concat(all_runs, ignore_index=True)

    # Filter out ensemble summary runs
    runs_df = runs_df[
        ~runs_df["tags.mlflow.runName"].str.startswith("ensemble_", na=True)
    ]

    # Check which runs have prediction artifacts
    client = mlflow.tracking.MlflowClient()
    has_preds = []
    for _, row in runs_df.iterrows():
        arts = client.list_artifacts(row.run_id, "predictions")
        has_preds.append(len(arts) > 0)
    runs_df = runs_df.copy()
    runs_df["_has_predictions"] = has_preds

    # Apply constraints
    if args.folds:
        runs_df = runs_df[runs_df["params.folds_n"].astype(float) == args.folds]
        if runs_df.empty:
            print(f"No runs with folds_n={args.folds}")
            sys.exit(1)

    if args.seeds:
        runs_df = runs_df[runs_df["params.seed"].astype(float).isin(args.seeds)]
        if runs_df.empty:
            print(f"No runs with seeds in {args.seeds}")
            sys.exit(1)

    if args.model:
        runs_df = runs_df[runs_df["params.model"].isin(args.model)]
        if runs_df.empty:
            print(f"No runs for models: {args.model}")
            sys.exit(1)

    if args.features:
        col = "params.feature_set"
        if col in runs_df.columns:
            runs_df = runs_df[runs_df[col] == args.features]
        else:
            # Column missing means no runs have feature_set logged
            runs_df = runs_df.iloc[0:0]
        if runs_df.empty:
            print(f"No runs with feature_set={args.features}")
            sys.exit(1)

    # Build learner ID for grouping
    runs_df = runs_df.copy()
    runs_df["_learner_id"] = runs_df.apply(_learner_id, axis=1)

    return runs_df.sort_values(
        ["_learner_id", "metrics.holdout_auc"], ascending=[True, False]
    )


def _print_runs_table(runs_df):
    """Print runs grouped by (learner_id, experiment), showing seeds per experiment."""
    exp_cache = {}

    def _exp_name(eid):
        if eid not in exp_cache:
            exp = mlflow.get_experiment(eid)
            exp_cache[eid] = exp.name if exp else "?"
        return exp_cache[eid]

    runs_df = runs_df.copy()
    runs_df["_exp_name"] = runs_df["experiment_id"].map(_exp_name)
    if "_learner_id" not in runs_df.columns:
        runs_df["_learner_id"] = runs_df.apply(_learner_id, axis=1)

    # Check prediction artifacts if not already done
    if "_has_predictions" not in runs_df.columns:
        client = mlflow.tracking.MlflowClient()
        runs_df["_has_predictions"] = [
            len(client.list_artifacts(rid, "predictions")) > 0
            for rid in runs_df["run_id"]
        ]

    # Find max experiment name length for alignment
    max_exp_len = max(len(n) for n in runs_df["_exp_name"].unique())

    for lid, learner_group in sorted(
        runs_df.groupby("_learner_id"), key=lambda x: x[0]
    ):
        avg_all = learner_group["metrics.holdout_auc"].mean()
        avg_all_str = f"{avg_all:.4f}" if avg_all == avg_all else "?"
        print(
            f"\033[1m{lid}\033[0m  (avg holdout={avg_all_str} across all experiments)"
        )

        exp_groups = sorted(
            learner_group.groupby("_exp_name"),
            key=lambda x: -x[1]["metrics.holdout_auc"].mean(),
        )
        # Best avg only among experiments with predictions
        valid_groups = [g for _, g in exp_groups if g["_has_predictions"].all()]
        best_avg = (
            valid_groups[0]["metrics.holdout_auc"].mean() if valid_groups else None
        )

        for exp_name, exp_group in exp_groups:
            exp_group = exp_group.sort_values("params.seed")
            n_seeds = len(exp_group)
            avg_auc = exp_group["metrics.holdout_auc"].mean()
            avg_str = f"{avg_auc:.4f}" if avg_auc == avg_auc else "?"
            has_preds = exp_group["_has_predictions"].all()

            if not has_preds:
                # Dim runs without prediction artifacts
                print(
                    f"  \033[2m{exp_name:<{max_exp_len}s}  "
                    f"seeds: {n_seeds:<4d} avg holdout: {avg_str}  "
                    f"(no artifacts)\033[0m"
                )
            else:
                is_best = best_avg is not None and avg_auc == best_avg
                bold = "\033[1;32m" if is_best else ""
                reset = "\033[0m" if is_best else ""
                print(
                    f"  {bold}{exp_name:<{max_exp_len}s}  "
                    f"seeds: {n_seeds:<4d} avg holdout: {avg_str}{reset}"
                )

            for _, row in exp_group.iterrows():
                seed = row.get("params.seed") or "?"
                holdout_auc = row.get("metrics.holdout_auc")
                auc_str = f"{holdout_auc:.4f}" if holdout_auc is not None else "?"
                rid = row["run_id"][:8]
                dim = "\033[2m" if not has_preds else ""
                reset = "\033[0m" if not has_preds else ""
                print(
                    f"  {dim}{'':<{max_exp_len}s}  "
                    f"  seed: {seed:<5s} "
                    f"holdout: {auc_str}  {rid}{reset}"
                )
        print()


def cmd_search(args):
    """Search and display runs matching constraints."""
    _get_client()
    runs_df = _collect_and_filter(args)

    print(f"{len(runs_df)} runs found:\n")
    _print_runs_table(runs_df)


def cmd_auto(args):
    """Auto-promote best experiment per model (all seeds within it).

    For each model type, finds the experiment with the highest average
    holdout AUC across seeds, then promotes all seed runs from that
    experiment. Use --seeds or --folds to constrain which runs qualify.
    """
    client = _get_client()
    tag_key = _tag_key(args.ensemble)
    runs_df = _collect_and_filter(args)

    # Resolve experiment names
    exp_cache = {}

    def _exp_name(eid):
        if eid not in exp_cache:
            exp = mlflow.get_experiment(eid)
            exp_cache[eid] = exp.name if exp else "?"
        return exp_cache[eid]

    # Only consider runs with prediction artifacts
    valid_df = runs_df[runs_df["_has_predictions"]]
    if valid_df.empty:
        print("No runs with prediction artifacts found")
        sys.exit(1)

    skipped = len(runs_df) - len(valid_df)
    if skipped:
        print(f"  Skipping {skipped} runs without prediction artifacts\n")

    if "_learner_id" not in valid_df.columns:
        valid_df = valid_df.copy()
        valid_df["_learner_id"] = valid_df.apply(_learner_id, axis=1)

    promoted = 0
    for lid, learner_group in sorted(
        valid_df.groupby("_learner_id"), key=lambda x: x[0]
    ):
        # Find experiment with best avg holdout AUC for this learner
        best_exp_id = (
            learner_group.groupby("experiment_id")["metrics.holdout_auc"]
            .mean()
            .idxmax()
        )
        best_runs = learner_group[learner_group["experiment_id"] == best_exp_id]

        for _, row in best_runs.iterrows():
            client.set_tag(row["run_id"], tag_key, "true")
            promoted += 1

        avg_auc = best_runs["metrics.holdout_auc"].mean()
        avg_str = f"{avg_auc:.4f}" if avg_auc == avg_auc else "?"
        exp = _exp_name(best_exp_id)
        print(f"  {lid}: {len(best_runs)} seeds from {exp} " f"(avg holdout={avg_str})")

    print(f"\nPromoted {promoted} runs to ensemble '{args.ensemble}'")


def cmd_add(args):
    """Manually tag specific run IDs."""
    client = _get_client()
    tag_key = _tag_key(args.ensemble)

    for run_id in args.run_id:
        try:
            run = client.get_run(run_id)
            client.set_tag(run_id, tag_key, "true")
            name = run.data.tags.get("mlflow.runName", "?")
            print(f"  Tagged {name} (run_id={run_id[:8]}...)")
        except Exception as e:
            print(f"  Error tagging {run_id}: {e}", file=sys.stderr)

    print(f"\nTagged {len(args.run_id)} run(s) for ensemble '{args.ensemble}'")


def cmd_list(args):
    """List all runs in a named ensemble."""
    _get_client()
    tag_key = _tag_key(args.ensemble)

    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=f"tags.`{tag_key}` = 'true'",
    )

    if runs.empty:
        print(f"No runs in ensemble '{args.ensemble}'")
        return

    # Filter out ensemble summary runs
    runs = runs[~runs["tags.mlflow.runName"].str.startswith("ensemble_", na=True)]

    print(f"Ensemble '{args.ensemble}' ({len(runs)} runs):\n")
    _print_runs_table(runs)


def cmd_remove(args):
    """Remove runs from a named ensemble."""
    client = _get_client()
    tag_key = _tag_key(args.ensemble)

    for run_id in args.run_id:
        try:
            client.delete_tag(run_id, tag_key)
            print(f"  Removed {run_id[:8]}... from ensemble '{args.ensemble}'")
        except Exception as e:
            print(f"  Error removing {run_id}: {e}", file=sys.stderr)


def cmd_clear(args):
    """Remove all runs from a named ensemble."""
    client = _get_client()
    tag_key = _tag_key(args.ensemble)

    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=f"tags.`{tag_key}` = 'true'",
    )

    if runs.empty:
        print(f"No runs in ensemble '{args.ensemble}'")
        return

    removed = 0
    for _, row in runs.iterrows():
        try:
            client.delete_tag(row.run_id, tag_key)
            removed += 1
        except Exception as e:
            print(f"  Error removing {row.run_id[:8]}...: {e}", file=sys.stderr)

    print(f"Cleared ensemble '{args.ensemble}' ({removed} runs removed)")


def _add_filter_args(parser):
    """Add shared filter arguments to a subparser."""
    parser.add_argument(
        "--experiment",
        "-e",
        nargs="+",
        default=None,
        help="Experiment name(s) to select from",
    )
    parser.add_argument(
        "--all", "-a", action="store_true", help="Search across all experiments"
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=None,
        help="Only consider runs with this many folds",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Only consider runs with these seeds",
    )
    parser.add_argument(
        "--model", nargs="+", default=None, help="Only consider these model types"
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Only consider runs with this feature set (e.g. ablation-pruned)",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Promote MLflow runs into named ensembles"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # search
    p_search = sub.add_parser("search", help="Browse runs matching constraints")
    _add_filter_args(p_search)

    # auto
    p_auto = sub.add_parser(
        "auto", help="Auto-promote best run per model from experiment(s)"
    )
    p_auto.add_argument("ensemble", help="Ensemble name (e.g. submit-v1)")
    _add_filter_args(p_auto)

    # add
    p_add = sub.add_parser("add", help="Manually tag specific run IDs")
    p_add.add_argument("ensemble", help="Ensemble name")
    p_add.add_argument("--run-id", "-r", nargs="+", required=True)

    # list
    p_list = sub.add_parser("list", help="List runs in an ensemble")
    p_list.add_argument("ensemble", help="Ensemble name")

    # remove
    p_remove = sub.add_parser("remove", help="Remove runs from an ensemble")
    p_remove.add_argument("ensemble", help="Ensemble name")
    p_remove.add_argument("--run-id", "-r", nargs="+", required=True)

    # clear
    p_clear = sub.add_parser("clear", help="Remove all runs from an ensemble")
    p_clear.add_argument("ensemble", help="Ensemble name")

    args = parser.parse_args()

    commands = {
        "search": cmd_search,
        "auto": cmd_auto,
        "add": cmd_add,
        "list": cmd_list,
        "remove": cmd_remove,
        "clear": cmd_clear,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
