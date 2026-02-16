"""Analyze model disagreement and best-performing runs across experiments.

Uses OOF predictions from MLflow, averaged across seeds per model.
Prints disagreement matrix, unique correct counts, and best runs per model.

Usage:
    python analyze_disagreement.py
    python analyze_disagreement.py --folds 5
    python analyze_disagreement.py --experiment full-v1 gbdt-v2
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

project_root = Path(__file__).resolve().parents[2]

DATA_DIR = (
    Path(os.environ.get("KEGO_PATH_DATA", project_root / "data"))
    / "playground"
    / "playground-series-s6e2"
)
TARGET = "Heart Disease"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://192.168.178.32:5000")
EXPERIMENT = "playground-s6e2-full"


def load_oof_predictions(features=None):
    """Load OOF predictions from MLflow, averaged across seeds per model."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    runs = runs[~runs["tags.mlflow.runName"].str.startswith("ensemble_", na=True)]

    if features and "params.feature_set" in runs.columns:
        runs = runs[runs["params.feature_set"] == features]

    oof_by_model = defaultdict(list)
    for _, run in runs.iterrows():
        model_name = run.get("params.model")
        if model_name is None:
            continue
        artifact_dir = client.download_artifacts(run.run_id, "predictions")
        oof = np.load(os.path.join(artifact_dir, "oof.npy"))
        oof_by_model[model_name].append(oof)

    # Average across seeds
    averaged = {}
    for name, arrays in oof_by_model.items():
        averaged[name] = np.mean(arrays, axis=0)
        print(f"  {name}: {len(arrays)} seeds")

    return averaged


def load_labels():
    """Load train labels using the same split as training.

    Reproduces the split from train_s6e2_baseline.py:
    split_dataset(train_full, train_size=0.8, validate_size=0.2, stratify_column=TARGET)
    which uses train_test_split with random_state=42.
    """
    train_full = pd.read_csv(DATA_DIR / "train.csv")
    original = pd.read_csv(DATA_DIR / "Heart_Disease_Prediction.csv")

    train_full[TARGET] = train_full[TARGET].map({"Presence": 1, "Absence": 0})
    original[TARGET] = original[TARGET].map({"Presence": 1, "Absence": 0})
    original["id"] = -1
    train_full = pd.concat([train_full, original], ignore_index=True)

    # Replicate split_dataset(train_size=0.8, validate_size=0.2)
    train, _ = train_test_split(
        train_full, train_size=0.8, random_state=42, stratify=train_full[TARGET]
    )
    train = train.reset_index(drop=True)
    return train[TARGET].values


def load_best_runs(folds, experiments=None, features=None):
    """Find best experiment per model by avg holdout_auc.

    Returns dict: model_name -> {experiment, avg_holdout_auc, avg_oof_auc, n_seeds, runs}
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    if experiments:
        all_runs = []
        for exp_name in experiments:
            exp = mlflow.get_experiment_by_name(exp_name)
            if exp is None:
                exp = mlflow.get_experiment_by_name(f"playground-s6e2-{exp_name}")
            if exp is None:
                print(f"  Warning: experiment '{exp_name}' not found, skipping")
                continue
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            all_runs.append(runs)
        if not all_runs:
            return {}
        runs_df = pd.concat(all_runs, ignore_index=True)
    else:
        runs_df = mlflow.search_runs(search_all_experiments=True)

    # Filter out ensemble summary runs
    runs_df = runs_df[
        ~runs_df["tags.mlflow.runName"].str.startswith("ensemble_", na=True)
    ]

    # Filter by folds
    if folds is not None:
        runs_df = runs_df[runs_df["params.folds_n"].astype(float) == folds]

    # Filter by feature set
    if features and "params.feature_set" in runs_df.columns:
        runs_df = runs_df[runs_df["params.feature_set"] == features]

    # Only runs with prediction artifacts
    has_preds = []
    for _, row in runs_df.iterrows():
        arts = client.list_artifacts(row.run_id, "predictions")
        has_preds.append(len(arts) > 0)
    runs_df = runs_df[has_preds].copy()

    if runs_df.empty:
        return {}

    # Resolve experiment names
    exp_cache = {}

    def _exp_name(eid):
        if eid not in exp_cache:
            exp = mlflow.get_experiment(eid)
            exp_cache[eid] = exp.name if exp else "?"
        return exp_cache[eid]

    runs_df["_exp_name"] = runs_df["experiment_id"].map(_exp_name)

    # For each model, find best experiment by avg holdout_auc
    best = {}
    for model_name, model_group in runs_df.groupby("params.model"):
        exp_avgs = model_group.groupby("_exp_name").agg(
            avg_holdout=("metrics.holdout_auc", "mean"),
            avg_oof=("metrics.oof_auc", "mean"),
            n_seeds=("run_id", "count"),
        )
        best_exp = exp_avgs["avg_holdout"].idxmax()
        best_row = exp_avgs.loc[best_exp]

        # Collect individual run details for the best experiment
        best_runs = model_group[model_group["_exp_name"] == best_exp]

        best[model_name] = {
            "experiment": best_exp,
            "avg_holdout_auc": best_row["avg_holdout"],
            "avg_oof_auc": best_row["avg_oof"],
            "n_seeds": int(best_row["n_seeds"]),
            "runs": best_runs,
        }

    return best


def build_disagreement_matrix(oof_preds, labels, threshold=0.5):
    """Build matrix where (i,j) = count of samples where model i is correct and model j is wrong."""
    model_names = sorted(oof_preds.keys())
    n = len(model_names)

    # Binary correctness per model
    correct = {}
    for name in model_names:
        preds = (oof_preds[name] >= threshold).astype(int)
        correct[name] = preds == labels

    # Disagreement matrix: (i,j) = i correct AND j wrong
    matrix = np.zeros((n, n), dtype=int)
    for i, name_i in enumerate(model_names):
        for j, name_j in enumerate(model_names):
            if i == j:
                continue
            matrix[i, j] = np.sum(correct[name_i] & ~correct[name_j])

    return model_names, matrix, correct


def main():
    parser = argparse.ArgumentParser(
        description="Analyze model disagreement and best runs"
    )
    parser.add_argument(
        "--folds", type=int, default=10, help="Filter runs by folds_n (default: 10)"
    )
    parser.add_argument(
        "--experiment",
        nargs="+",
        default=None,
        help="Experiment name(s) to search (default: all)",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Filter by feature set (e.g. ablation-pruned)",
    )
    args = parser.parse_args()

    feat_str = f", features={args.features}" if args.features else ""
    print(f"Loading OOF predictions from MLflow{feat_str}...")
    oof_preds = load_oof_predictions(features=args.features)

    print("\nLoading labels...")
    labels = load_labels()

    print(
        f"\nLoading best runs per model "
        f"(folds={args.folds}, experiments={'all' if not args.experiment else args.experiment}"
        f"{feat_str})..."
    )
    best_runs = load_best_runs(args.folds, args.experiment, features=args.features)

    print(f"\n{len(oof_preds)} models with OOF predictions, {len(labels)} samples")

    model_names, matrix, correct = build_disagreement_matrix(oof_preds, labels)

    # Compute unique correct per model
    all_correct = np.column_stack([correct[n] for n in model_names])
    unique_correct = {}
    for i, name in enumerate(model_names):
        others_correct = np.delete(all_correct, i, axis=1).any(axis=1)
        unique_correct[name] = int((correct[name] & ~others_correct).sum())

    # === Combined summary table ===
    print("\n=== Model Summary ===")
    print(
        f"  {'model':25s}  {'OOF acc':>8s}  {'holdout':>8s}  "
        f"{'oof_auc':>8s}  {'seeds':>5s}  {'unique':>6s}  {'experiment'}"
    )
    print("-" * 105)

    # Sort by holdout AUC descending (fall back to OOF accuracy if no runs)
    def sort_key(name):
        if name in best_runs:
            return -best_runs[name]["avg_holdout_auc"]
        return -correct[name].mean()

    for name in sorted(model_names, key=sort_key):
        acc = correct[name].mean()
        uniq = unique_correct[name]
        if name in best_runs:
            info = best_runs[name]
            print(
                f"  {name:25s}  {acc:8.4f}  {info['avg_holdout_auc']:8.4f}  "
                f"{info['avg_oof_auc']:8.4f}  {info['n_seeds']:5d}  "
                f"{uniq:6d}  {info['experiment']}"
            )
        else:
            print(
                f"  {name:25s}  {acc:8.4f}  {'—':>8s}  "
                f"{'—':>8s}  {'—':>5s}  {uniq:6d}  (no matching runs)"
            )

    # Models in best_runs but not in OOF predictions (e.g. different experiment)
    extra = set(best_runs.keys()) - set(model_names)
    if extra:
        print()
        for name in sorted(extra):
            info = best_runs[name]
            print(
                f"  {name:25s}  {'—':>8s}  {info['avg_holdout_auc']:8.4f}  "
                f"{info['avg_oof_auc']:8.4f}  {info['n_seeds']:5d}  "
                f"{'—':>6s}  {info['experiment']}"
            )

    # Print disagreement matrix
    print("\n=== Disagreement Matrix ===")
    print(
        "Cell (row, col) = # samples where ROW model is correct but COL model is wrong\n"
    )

    short_names = [n[:12] for n in model_names]
    header = f"{'':>14s}" + "".join(f"{s:>13s}" for s in short_names)
    print(header)
    print("-" * len(header))

    for i, name in enumerate(model_names):
        row = f"{short_names[i]:>14s}"
        for j in range(len(model_names)):
            if i == j:
                row += f"{'—':>13s}"
            else:
                row += f"{matrix[i, j]:>13d}"
        print(row)

    # Symmetric disagreement
    print("\n=== Symmetric Disagreement (total unique info per pair) ===")
    print("Higher = more diverse pair (better for ensembling)\n")

    sym_matrix = matrix + matrix.T
    header = f"{'':>14s}" + "".join(f"{s:>13s}" for s in short_names)
    print(header)
    print("-" * len(header))

    for i, name in enumerate(model_names):
        row = f"{short_names[i]:>14s}"
        for j in range(len(model_names)):
            if i == j:
                row += f"{'—':>13s}"
            else:
                row += f"{sym_matrix[i, j]:>13d}"
        print(row)

    # Most/least redundant pairs
    print("\n=== Top 10 Most Diverse Pairs ===")
    pairs = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            pairs.append((model_names[i], model_names[j], sym_matrix[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    for name_i, name_j, count in pairs[:10]:
        print(f"  {name_i:25s} vs {name_j:25s}  {count:5d} disagreements")

    print("\n=== Top 10 Most Redundant Pairs ===")
    for name_i, name_j, count in pairs[-10:]:
        print(f"  {name_i:25s} vs {name_j:25s}  {count:5d} disagreements")

    # Unique correct (already computed)
    print("\n=== Unique Correct Predictions (right when ALL others are wrong) ===")
    for name in model_names:
        print(f"  {name:25s}  {unique_correct[name]:4d} unique correct")


if __name__ == "__main__":
    main()
