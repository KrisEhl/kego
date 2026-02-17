"""Greedy ensemble member selection and leave-one-out analysis.

Tests whether each model actually improves the ensemble or just adds noise.

Modes:
  1. Greedy forward selection (default): Starting from empty, adds models one
     at a time picking the one with highest AUC gain.
  2. Leave-one-out (--check): Removes each model one at a time from an existing
     ensemble to measure each member's contribution.

Usage:
    python analyze_ensemble.py --from-experiment playground-s6e2-full
    python analyze_ensemble.py --from-ensemble submit-v1
    python analyze_ensemble.py --from-ensemble submit-v1 --check
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from kego.datasets.split import split_dataset  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = (
    Path(os.environ.get("KEGO_PATH_DATA", project_root / "data"))
    / "playground"
    / "playground-series-s6e2"
)
TARGET = "Heart Disease"


# ---------------------------------------------------------------------------
# MLflow prediction loading (copied from train_s6e2_baseline.py — can't
# import directly because that module imports torch/ray/catboost at module level)
# ---------------------------------------------------------------------------


def _load_predictions_from_runs(runs_df, tracking_uri):
    """Load and average predictions from a DataFrame of MLflow runs."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    all_oof = {}
    all_holdout = {}
    all_test = {}
    seed_counts = {}

    for _, run in runs_df.iterrows():
        model_name = run.get("params.model")
        if model_name is None:
            continue

        artifact_dir = client.download_artifacts(run.run_id, "predictions")
        oof = np.load(os.path.join(artifact_dir, "oof.npy"))
        holdout = np.load(os.path.join(artifact_dir, "holdout.npy"))
        test = np.load(os.path.join(artifact_dir, "test.npy"))

        if model_name not in all_oof:
            all_oof[model_name] = np.zeros_like(oof)
            all_holdout[model_name] = np.zeros_like(holdout)
            all_test[model_name] = np.zeros_like(test)
            seed_counts[model_name] = 0

        all_oof[model_name] += oof
        all_holdout[model_name] += holdout
        all_test[model_name] += test
        seed_counts[model_name] += 1

        seed = run.get("params.seed", "?")
        logger.info(f"  Loaded {model_name} seed={seed}")

    # Average across seeds
    for name in all_oof:
        n = seed_counts[name]
        all_oof[name] /= n
        all_holdout[name] /= n
        all_test[name] /= n
        logger.info(f"{name}: averaged over {n} seed(s)")

    model_names = list(all_oof.keys())
    logger.info(f"Total models loaded: {len(model_names)}")
    return model_names, all_oof, all_holdout, all_test


def _load_predictions_from_mlflow(experiment_names, tracking_uri):
    """Load per-model averaged predictions from MLflow experiments."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)

    all_runs = []
    for exp_name in experiment_names:
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            logger.warning(f"Experiment '{exp_name}' not found, skipping")
            continue

        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        runs = runs[~runs["tags.mlflow.runName"].str.startswith("ensemble_", na=True)]
        logger.info(f"Experiment '{exp_name}': {len(runs)} model runs")
        all_runs.append(runs)

    if not all_runs:
        return [], {}, {}, {}

    runs_df = pd.concat(all_runs, ignore_index=True)
    return _load_predictions_from_runs(runs_df, tracking_uri)


def _load_predictions_from_ensemble(ensemble_name, tracking_uri):
    """Load predictions from runs tagged with a named ensemble."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)

    tag_key = f"ensemble:{ensemble_name}"
    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=f"tags.`{tag_key}` = 'true'",
    )

    if runs.empty:
        logger.error(f"No runs found in ensemble '{ensemble_name}'")
        return [], {}, {}, {}

    runs = runs[~runs["tags.mlflow.runName"].str.startswith("ensemble_", na=True)]
    logger.info(f"Ensemble '{ensemble_name}': {len(runs)} model runs")

    return _load_predictions_from_runs(runs, tracking_uri)


# ---------------------------------------------------------------------------
# Analysis algorithms
# ---------------------------------------------------------------------------


def greedy_forward_selection(model_names, all_oof, all_holdout, holdout_labels):
    """Greedy forward selection: add models one at a time by highest AUC gain."""
    available = set(model_names)
    ensemble = []
    ensemble_oof = None
    ensemble_holdout = None
    current_auc = 0.0

    selected_rows = []
    rejected_rows = []

    step = 0
    while available:
        best_candidate = None
        best_auc = current_auc
        best_corr = None
        candidate_results = []

        for candidate in sorted(available):
            n = len(ensemble)
            if n == 0:
                blended_oof = all_oof[candidate]
                blended_holdout = all_holdout[candidate]
            else:
                blended_oof = (ensemble_oof * n + all_oof[candidate]) / (n + 1)
                blended_holdout = (ensemble_holdout * n + all_holdout[candidate]) / (
                    n + 1
                )

            auc = roc_auc_score(holdout_labels, blended_holdout)
            delta = auc - current_auc

            if n == 0:
                corr = None
            else:
                r, _ = spearmanr(rankdata(all_oof[candidate]), rankdata(ensemble_oof))
                corr = r

            candidate_results.append((candidate, auc, delta, corr))

            if auc > best_auc:
                best_candidate = candidate
                best_auc = auc
                best_corr = corr

        if best_candidate is not None:
            step += 1
            n = len(ensemble)
            if n == 0:
                ensemble_oof = all_oof[best_candidate].copy()
                ensemble_holdout = all_holdout[best_candidate].copy()
            else:
                ensemble_oof = (ensemble_oof * n + all_oof[best_candidate]) / (n + 1)
                ensemble_holdout = (
                    ensemble_holdout * n + all_holdout[best_candidate]
                ) / (n + 1)

            ensemble.append(best_candidate)
            available.remove(best_candidate)
            current_auc = best_auc

            selected_rows.append(
                (
                    step,
                    best_candidate,
                    best_auc,
                    best_auc
                    - (
                        current_auc
                        if step == 1
                        else selected_rows[-2][2] if len(selected_rows) >= 2 else 0
                    ),
                    best_corr,
                    len(ensemble),
                )
            )
        else:
            # No model improves the ensemble — record all remaining as rejected
            for candidate, auc, delta, corr in candidate_results:
                rejected_rows.append((candidate, delta, corr))
            break

    # Recalculate deltas properly for display
    display_rows = []
    for i, (step_n, name, auc, _, corr, n_models) in enumerate(selected_rows):
        if i == 0:
            delta = auc  # first model: delta is the AUC itself
        else:
            delta = auc - selected_rows[i - 1][2]
        display_rows.append((step_n, name, auc, delta, corr, n_models))

    return display_rows, rejected_rows


def leave_one_out_analysis(model_names, all_oof, all_holdout, holdout_labels):
    """Leave-one-out: remove each model and measure AUC change."""
    n = len(model_names)

    # Full ensemble
    full_holdout = np.mean(
        np.column_stack([all_holdout[name] for name in model_names]), axis=1
    )
    full_auc = roc_auc_score(holdout_labels, full_holdout)

    rows = []
    for name in model_names:
        others = [m for m in model_names if m != name]
        reduced_holdout = np.mean(
            np.column_stack([all_holdout[m] for m in others]), axis=1
        )
        reduced_auc = roc_auc_score(holdout_labels, reduced_holdout)
        delta = full_auc - reduced_auc  # positive = model helps

        # Spearman between this model's OOF ranks and ensemble-without-it OOF ranks
        reduced_oof = np.mean(np.column_stack([all_oof[m] for m in others]), axis=1)
        r, _ = spearmanr(rankdata(all_oof[name]), rankdata(reduced_oof))

        if delta > 0.00005:
            verdict = "helpful"
        elif delta < -0.00005:
            verdict = "HARMFUL"
        else:
            verdict = "neutral"

        rows.append((name, reduced_auc, delta, r, verdict))

    # Sort by delta ascending (most harmful first, most helpful last)
    rows.sort(key=lambda x: x[2])

    return full_auc, n, rows


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_forward_selection(display_rows, rejected_rows):
    """Print greedy forward selection results."""
    print(f"\n{'='*80}")
    print("GREEDY FORWARD SELECTION")
    print(f"{'='*80}")
    print(
        f"{'Step':<6}{'Model Added':<25}{'Ensemble AUC':>14}"
        f"{'Delta':>11}{'Spearman r':>12}{'Models':>8}"
    )
    print("-" * 80)

    for step_n, name, auc, delta, corr, n_models in display_rows:
        corr_str = f"{corr:.3f}" if corr is not None else "\u2014"
        print(
            f"{step_n:<6}{name:<25}{auc:>14.5f}"
            f"{delta:>+11.5f}{corr_str:>12}{n_models:>8}"
        )

    if rejected_rows:
        print("-" * 80)
        for name, delta, corr in rejected_rows:
            corr_str = f"{corr:.3f}" if corr is not None else "\u2014"
            print(
                f"{'x':<6}{name:<25}{'\u2014':>14}"
                f"{delta:>+11.5f}{corr_str:>12}{'(rejected)':>12}"
            )

    print(f"{'='*80}")

    if display_rows:
        final = display_rows[-1]
        print(f"\nFinal ensemble: {final[5]} models, AUC: {final[2]:.5f}")
        if rejected_rows:
            print(f"Rejected: {len(rejected_rows)} models (would decrease AUC)")


def print_leave_one_out(full_auc, n_models, rows):
    """Print leave-one-out analysis results."""
    print(f"\n{'='*80}")
    print("LEAVE-ONE-OUT ANALYSIS")
    print(f"{'='*80}")
    print(f"Full ensemble AUC: {full_auc:.5f} ({n_models} models)\n")
    print(
        f"{'Model':<25}{'AUC without':>13}{'Delta':>11}"
        f"{'Spearman r':>12}{'Verdict':>10}"
    )
    print("-" * 80)

    for name, reduced_auc, delta, corr, verdict in rows:
        print(
            f"{name:<25}{reduced_auc:>13.5f}{delta:>+11.5f}"
            f"{corr:>12.3f}{verdict:>10}"
        )

    print(f"{'='*80}")

    harmful = [r for r in rows if r[4] == "HARMFUL"]
    helpful = [r for r in rows if r[4] == "helpful"]
    print(
        f"\nSummary: {len(helpful)} helpful, "
        f"{n_models - len(helpful) - len(harmful)} neutral, "
        f"{len(harmful)} harmful"
    )
    if harmful:
        print(f"Consider removing: {', '.join(r[0] for r in harmful)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Greedy ensemble member selection and leave-one-out analysis"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--from-experiment",
        nargs="+",
        metavar="EXPERIMENT",
        help="Load predictions from MLflow experiment(s)",
    )
    source.add_argument(
        "--from-ensemble",
        type=str,
        metavar="ENSEMBLE",
        help="Load predictions from a curated ensemble (tagged runs)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Leave-one-out analysis instead of greedy forward selection",
    )
    args = parser.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        logger.error("MLFLOW_TRACKING_URI must be set")
        sys.exit(1)

    # --- Load predictions ---
    if args.from_ensemble:
        logger.info(f"Loading predictions from ensemble: {args.from_ensemble}")
        model_names, all_oof, all_holdout, _ = _load_predictions_from_ensemble(
            args.from_ensemble, tracking_uri
        )
    else:
        logger.info(f"Loading predictions from experiments: {args.from_experiment}")
        model_names, all_oof, all_holdout, _ = _load_predictions_from_mlflow(
            args.from_experiment, tracking_uri
        )

    if not model_names:
        logger.error("No predictions loaded, exiting")
        sys.exit(1)

    # --- Load holdout labels (same split as training) ---
    train_full = pd.read_csv(DATA_DIR / "train.csv")
    original = pd.read_csv(DATA_DIR / "Heart_Disease_Prediction.csv")

    train_full[TARGET] = train_full[TARGET].map({"Presence": 1, "Absence": 0})
    original[TARGET] = original[TARGET].map({"Presence": 1, "Absence": 0})

    original["id"] = -1
    train_full = pd.concat([train_full, original], ignore_index=True)

    _, holdout, _ = split_dataset(
        train_full,
        train_size=0.8,
        validate_size=0.2,
        stratify_column=TARGET,
    )
    holdout = holdout.reset_index(drop=True)
    holdout_labels = holdout[TARGET].values

    # --- Run analysis ---
    if args.check:
        full_auc, n_models, rows = leave_one_out_analysis(
            model_names, all_oof, all_holdout, holdout_labels
        )
        print_leave_one_out(full_auc, n_models, rows)
    else:
        display_rows, rejected_rows = greedy_forward_selection(
            model_names, all_oof, all_holdout, holdout_labels
        )
        print_forward_selection(display_rows, rejected_rows)


if __name__ == "__main__":
    main()
