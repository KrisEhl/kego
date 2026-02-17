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
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import RidgeCV
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
    """Load and average predictions from a DataFrame of MLflow runs.

    Groups by learner ID (model/feature_set/folds_nf) when params are available,
    falls back to bare model_name for backward compatibility.
    """
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

        # Build learner ID from params (backward compat: fall back to model_name)
        feature_set = run.get("params.feature_set", "")
        folds_n = run.get("params.folds_n", "")
        if feature_set and folds_n:
            learner_id = f"{model_name}/{feature_set}/{folds_n}f"
        else:
            learner_id = model_name

        artifact_dir = client.download_artifacts(run.run_id, "predictions")
        oof = np.load(os.path.join(artifact_dir, "oof.npy"))
        holdout = np.load(os.path.join(artifact_dir, "holdout.npy"))
        test = np.load(os.path.join(artifact_dir, "test.npy"))

        if learner_id not in all_oof:
            all_oof[learner_id] = np.zeros_like(oof)
            all_holdout[learner_id] = np.zeros_like(holdout)
            all_test[learner_id] = np.zeros_like(test)
            seed_counts[learner_id] = 0

        all_oof[learner_id] += oof
        all_holdout[learner_id] += holdout
        all_test[learner_id] += test
        seed_counts[learner_id] += 1

        seed = run.get("params.seed", "?")
        logger.info(f"  Loaded {learner_id} seed={seed}")

    # Average across seeds
    for name in all_oof:
        n = seed_counts[name]
        all_oof[name] /= n
        all_holdout[name] /= n
        all_test[name] /= n
        logger.info(f"{name}: averaged over {n} seed(s)")

    learner_names = list(all_oof.keys())
    logger.info(f"Total learners loaded: {len(learner_names)}")
    return learner_names, all_oof, all_holdout, all_test


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


def _hill_climbing(oof_matrix, labels, n_iterations=50):
    """Find ensemble weights by greedy hill climbing on AUC.

    Uses a larger step size (0.05) for faster convergence with many models.
    """
    n_models = oof_matrix.shape[1]
    best_weights = np.ones(n_models) / n_models
    best_preds = oof_matrix @ best_weights
    best_auc = roc_auc_score(labels, best_preds)
    step = 0.05

    for _ in range(n_iterations):
        improved = False
        for i in range(n_models):
            for j in range(i + 1, n_models):
                # Try shifting weight from j to i
                for di, dj in [(step, -step), (-step, step)]:
                    weights = best_weights.copy()
                    weights[i] += di
                    weights[j] += dj
                    if weights[i] < 0 or weights[j] < 0:
                        continue
                    weights /= weights.sum()
                    preds = oof_matrix @ weights
                    if not np.all(np.isfinite(preds)):
                        continue
                    auc = roc_auc_score(labels, preds)
                    if auc > best_auc:
                        best_auc = auc
                        best_weights = weights
                        improved = True
        if not improved:
            break

    return best_weights


def _rank_blend(matrix):
    """Convert predictions to percentile ranks per model, then average."""
    n = matrix.shape[0]
    ranked = np.column_stack(
        [rankdata(matrix[:, i]) / n for i in range(matrix.shape[1])]
    )
    return np.mean(ranked, axis=1)


def _evaluate_strategies(oof_matrix, holdout_matrix, holdout_labels, oof_labels):
    """Evaluate all blending strategies and return {strategy: auc} dict."""
    with (
        warnings.catch_warnings(),
        np.errstate(divide="ignore", over="ignore", invalid="ignore"),
    ):
        warnings.simplefilter("ignore", RuntimeWarning)

        strategies = {}

        # 1. Simple mean (always)
        strategies["mean"] = roc_auc_score(holdout_labels, holdout_matrix.mean(axis=1))

        # 2. Rank mean (always)
        strategies["rank"] = roc_auc_score(holdout_labels, _rank_blend(holdout_matrix))

        # 3 & 4: only with oof_labels and >=2 models
        n_models = holdout_matrix.shape[1]
        if oof_labels is not None and n_models >= 2:
            # Ridge stacking (fit on OOF, evaluate on holdout)
            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            ridge.fit(oof_matrix, oof_labels)
            ridge_preds = ridge.predict(holdout_matrix)
            if np.all(np.isfinite(ridge_preds)):
                strategies["ridge"] = roc_auc_score(holdout_labels, ridge_preds)

            # Hill climbing (optimize weights on OOF, apply to holdout)
            # Skip for large ensembles — O(n^2) per iteration is too expensive
            if n_models <= 15:
                weights = _hill_climbing(oof_matrix, oof_labels)
                holdout_preds = holdout_matrix @ weights
                if np.all(np.isfinite(holdout_preds)):
                    strategies["hill"] = roc_auc_score(holdout_labels, holdout_preds)

        return strategies


def greedy_forward_selection(
    model_names, all_oof, all_holdout, holdout_labels, oof_labels=None
):
    """Greedy forward selection: add models one at a time by highest AUC gain.

    Evaluates multiple blending strategies (mean, rank, ridge, hill climbing)
    at each step and picks the (candidate, strategy) pair with highest AUC.
    """
    available = set(model_names)
    ensemble = []
    current_auc = 0.0

    selected_rows = []
    rejected_rows = []

    step = 0
    while available:
        best_candidate = None
        best_auc = current_auc
        best_corr = None
        best_strategy = None
        candidate_results = []

        for candidate in sorted(available):
            members = ensemble + [candidate]
            oof_matrix = np.column_stack([all_oof[m] for m in members])
            holdout_matrix = np.column_stack([all_holdout[m] for m in members])

            strategies = _evaluate_strategies(
                oof_matrix, holdout_matrix, holdout_labels, oof_labels
            )

            # Pick best strategy for this candidate
            cand_strategy = max(strategies, key=strategies.get)
            auc = strategies[cand_strategy]
            delta = auc - current_auc

            if len(ensemble) == 0:
                corr = None
            else:
                ens_oof = np.column_stack([all_oof[m] for m in ensemble])
                r, _ = spearmanr(
                    rankdata(all_oof[candidate]), rankdata(ens_oof.mean(axis=1))
                )
                corr = r

            candidate_results.append((candidate, auc, delta, corr, cand_strategy))

            if auc > best_auc:
                best_candidate = candidate
                best_auc = auc
                best_corr = corr
                best_strategy = cand_strategy

        if best_candidate is not None:
            step += 1
            ensemble.append(best_candidate)
            available.remove(best_candidate)
            prev_auc = current_auc
            current_auc = best_auc

            selected_rows.append(
                (
                    step,
                    best_candidate,
                    best_strategy,
                    best_auc,
                    best_auc - prev_auc if step > 1 else best_auc,
                    best_corr,
                    len(ensemble),
                )
            )
        else:
            # No model improves the ensemble — record all remaining as rejected
            for candidate, auc, delta, corr, strat in candidate_results:
                rejected_rows.append((candidate, delta, corr, strat))
            break

    return selected_rows, rejected_rows


def leave_one_out_analysis(
    model_names, all_oof, all_holdout, holdout_labels, oof_labels=None
):
    """Leave-one-out: remove each model and measure AUC change.

    Evaluates all blending strategies on the full ensemble to find the best one,
    then measures each model's contribution under that strategy.
    """
    n = len(model_names)

    # Full ensemble — find best strategy
    full_oof_matrix = np.column_stack([all_oof[name] for name in model_names])
    full_holdout_matrix = np.column_stack([all_holdout[name] for name in model_names])
    full_strategies = _evaluate_strategies(
        full_oof_matrix, full_holdout_matrix, holdout_labels, oof_labels
    )
    best_full_strategy = max(full_strategies, key=full_strategies.get)
    full_auc = full_strategies[best_full_strategy]

    rows = []
    for name in model_names:
        others = [m for m in model_names if m != name]
        reduced_oof_matrix = np.column_stack([all_oof[m] for m in others])
        reduced_holdout_matrix = np.column_stack([all_holdout[m] for m in others])

        reduced_strategies = _evaluate_strategies(
            reduced_oof_matrix, reduced_holdout_matrix, holdout_labels, oof_labels
        )
        reduced_auc = reduced_strategies[best_full_strategy]
        delta = full_auc - reduced_auc  # positive = model helps

        # Spearman between this model's OOF ranks and ensemble-without-it OOF ranks
        reduced_oof = reduced_oof_matrix.mean(axis=1)
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

    return full_auc, best_full_strategy, n, rows


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_forward_selection(display_rows, rejected_rows):
    """Print greedy forward selection results."""
    w = 100
    print(f"\n{'=' * w}")
    print("GREEDY FORWARD SELECTION (multi-strategy)")
    print(f"{'=' * w}")
    print(
        f"{'Step':<6}{'Learner Added':<30}{'Strategy':>10}{'Ensemble AUC':>14}"
        f"{'Delta':>11}{'Spearman r':>12}{'Models':>8}"
    )
    print("-" * w)

    for step_n, name, strategy, auc, delta, corr, n_models in display_rows:
        corr_str = f"{corr:.3f}" if corr is not None else "\u2014"
        print(
            f"{step_n:<6}{name:<30}{strategy:>10}{auc:>14.5f}"
            f"{delta:>+11.5f}{corr_str:>12}{n_models:>8}"
        )

    if rejected_rows:
        print("-" * w)
        for name, delta, corr, strat in rejected_rows:
            corr_str = f"{corr:.3f}" if corr is not None else "\u2014"
            print(
                f"{'x':<6}{name:<30}{strat:>10}{'\u2014':>14}"
                f"{delta:>+11.5f}{corr_str:>12}{'(rejected)':>12}"
            )

    print(f"{'=' * w}")

    if display_rows:
        final = display_rows[-1]
        print(
            f"\nFinal ensemble: {final[6]} models, AUC: {final[3]:.5f}"
            f" (strategy: {final[2]})"
        )
        if rejected_rows:
            print(f"Rejected: {len(rejected_rows)} models (would decrease AUC)")


def print_leave_one_out(full_auc, strategy, n_models, rows):
    """Print leave-one-out analysis results."""
    print(f"\n{'='*90}")
    print("LEAVE-ONE-OUT ANALYSIS")
    print(f"{'='*90}")
    print(
        f"Full ensemble AUC: {full_auc:.5f} ({n_models} models,"
        f" strategy: {strategy})\n"
    )
    print(
        f"{'Learner':<35}{'AUC without':>13}{'Delta':>11}"
        f"{'Spearman r':>12}{'Verdict':>10}"
    )
    print("-" * 90)

    for name, reduced_auc, delta, corr, verdict in rows:
        print(
            f"{name:<35}{reduced_auc:>13.5f}{delta:>+11.5f}"
            f"{corr:>12.3f}{verdict:>10}"
        )

    print(f"{'='*90}")

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

    train_split, holdout, _ = split_dataset(
        train_full,
        train_size=0.8,
        validate_size=0.2,
        stratify_column=TARGET,
    )
    holdout = holdout.reset_index(drop=True)
    holdout_labels = holdout[TARGET].values
    oof_labels = train_split[TARGET].values

    # --- Run analysis ---
    if args.check:
        full_auc, strategy, n_models, rows = leave_one_out_analysis(
            model_names, all_oof, all_holdout, holdout_labels, oof_labels=oof_labels
        )
        print_leave_one_out(full_auc, strategy, n_models, rows)
    else:
        display_rows, rejected_rows = greedy_forward_selection(
            model_names, all_oof, all_holdout, holdout_labels, oof_labels=oof_labels
        )
        print_forward_selection(display_rows, rejected_rows)


if __name__ == "__main__":
    main()
