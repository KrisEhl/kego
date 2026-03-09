"""Ensemble member analysis: greedy forward selection and leave-one-out.

These functions operate on dicts of OOF/holdout predictions and require no
competition-specific code. They are the algorithmic core extracted from
analyze_ensemble.py.
"""

import warnings

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score

from .weights import hill_climbing


def rank_blend(matrix: np.ndarray) -> np.ndarray:
    """Convert predictions to percentile ranks per model, then average."""
    n = matrix.shape[0]
    ranked = np.column_stack(
        [rankdata(matrix[:, i]) / n for i in range(matrix.shape[1])]
    )
    return np.mean(ranked, axis=1)


def evaluate_blending_strategies(
    oof_matrix: np.ndarray,
    holdout_matrix: np.ndarray,
    holdout_labels: np.ndarray,
    oof_labels: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate all blending strategies and return {strategy: auc} dict.

    Strategies evaluated:
    - mean: simple average of predictions
    - rank: average of per-model percentile ranks
    - ridge: RidgeCV fit on OOF, evaluated on holdout (requires oof_labels, >=2 models)
    - hill: hill climbing weights on OOF, evaluated on holdout (<=15 models only)

    Args:
        oof_matrix: Shape (n_train, n_models).
        holdout_matrix: Shape (n_holdout, n_models).
        holdout_labels: Ground truth for holdout set.
        oof_labels: Ground truth for OOF set. Required for ridge and hill strategies.
    """
    with (
        warnings.catch_warnings(),
        np.errstate(divide="ignore", over="ignore", invalid="ignore"),
    ):
        warnings.simplefilter("ignore", RuntimeWarning)

        strategies: dict[str, float] = {}
        strategies["mean"] = roc_auc_score(holdout_labels, holdout_matrix.mean(axis=1))
        strategies["rank"] = roc_auc_score(holdout_labels, rank_blend(holdout_matrix))

        n_models = holdout_matrix.shape[1]
        if oof_labels is not None and n_models >= 2:
            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            ridge.fit(oof_matrix, oof_labels)
            ridge_preds = ridge.predict(holdout_matrix)
            if np.all(np.isfinite(ridge_preds)):
                strategies["ridge"] = roc_auc_score(holdout_labels, ridge_preds)

            # Hill climbing is O(n^2) per iteration — skip for large ensembles
            if n_models <= 15:
                weights = hill_climbing(
                    oof_matrix, oof_labels, [str(i) for i in range(n_models)], step=0.05
                )
                holdout_preds = holdout_matrix @ weights
                if np.all(np.isfinite(holdout_preds)):
                    strategies["hill"] = roc_auc_score(holdout_labels, holdout_preds)

        return strategies


def greedy_forward_selection(
    model_names: list[str],
    all_oof: dict[str, np.ndarray],
    all_holdout: dict[str, np.ndarray],
    holdout_labels: np.ndarray,
    oof_labels: np.ndarray | None = None,
) -> tuple[list[tuple], list[tuple]]:
    """Greedy forward selection: add models one at a time by highest AUC gain.

    Evaluates multiple blending strategies (mean, rank, ridge, hill climbing)
    at each step and picks the (candidate, strategy) pair with highest AUC.

    Args:
        model_names: Candidate model/learner names.
        all_oof: Dict mapping name -> OOF prediction array.
        all_holdout: Dict mapping name -> holdout prediction array.
        holdout_labels: Ground truth for holdout set.
        oof_labels: Ground truth for OOF set. Enables ridge/hill strategies.

    Returns:
        selected_rows: List of tuples (step, name, strategy, auc, delta, corr, n_models)
            for each model added to the ensemble.
        rejected_rows: List of tuples (name, delta, corr, strategy) for models that
            didn't improve AUC.
    """
    available = set(model_names)
    ensemble: list[str] = []
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

            strategies = evaluate_blending_strategies(
                oof_matrix, holdout_matrix, holdout_labels, oof_labels
            )
            cand_strategy = max(strategies, key=lambda k: strategies[k])
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
            for candidate, auc, delta, corr, strat in candidate_results:
                rejected_rows.append((candidate, delta, corr, strat))
            break

    return selected_rows, rejected_rows


def leave_one_out_analysis(
    model_names: list[str],
    all_oof: dict[str, np.ndarray],
    all_holdout: dict[str, np.ndarray],
    holdout_labels: np.ndarray,
    oof_labels: np.ndarray | None = None,
) -> tuple[float, str, int, list[tuple]]:
    """Leave-one-out: remove each model and measure AUC change.

    Evaluates all blending strategies on the full ensemble to find the best one,
    then measures each model's contribution under that strategy.

    Args:
        model_names: All model/learner names in the ensemble.
        all_oof: Dict mapping name -> OOF prediction array.
        all_holdout: Dict mapping name -> holdout prediction array.
        holdout_labels: Ground truth for holdout set.
        oof_labels: Ground truth for OOF set. Enables ridge/hill strategies.

    Returns:
        full_auc: AUC of the full ensemble under the best strategy.
        best_strategy: Name of the best blending strategy.
        n_models: Total number of models.
        rows: List of (name, auc_without, delta, spearman_r, verdict) tuples,
            sorted by delta ascending (most harmful first).
    """
    full_oof_matrix = np.column_stack([all_oof[n] for n in model_names])
    full_holdout_matrix = np.column_stack([all_holdout[n] for n in model_names])
    full_strategies = evaluate_blending_strategies(
        full_oof_matrix, full_holdout_matrix, holdout_labels, oof_labels
    )
    best_strategy = max(full_strategies, key=lambda k: full_strategies[k])
    full_auc = full_strategies[best_strategy]

    rows = []
    for name in model_names:
        others = [m for m in model_names if m != name]
        reduced_oof = np.column_stack([all_oof[m] for m in others])
        reduced_holdout = np.column_stack([all_holdout[m] for m in others])

        reduced_strategies = evaluate_blending_strategies(
            reduced_oof, reduced_holdout, holdout_labels, oof_labels
        )
        reduced_auc = reduced_strategies[best_strategy]
        delta = full_auc - reduced_auc  # positive = model helps

        reduced_oof_mean = reduced_oof.mean(axis=1)
        r, _ = spearmanr(rankdata(all_oof[name]), rankdata(reduced_oof_mean))

        if delta > 0.00005:
            verdict = "helpful"
        elif delta < -0.00005:
            verdict = "HARMFUL"
        else:
            verdict = "neutral"

        rows.append((name, reduced_auc, delta, r, verdict))

    rows.sort(key=lambda x: x[2])
    return full_auc, best_strategy, len(model_names), rows


def print_forward_selection(
    selected_rows: list[tuple], rejected_rows: list[tuple]
) -> None:
    """Print greedy forward selection results in a formatted table."""
    w = 100
    print(f"\n{'=' * w}")
    print("GREEDY FORWARD SELECTION (multi-strategy)")
    print(f"{'=' * w}")
    print(
        f"{'Step':<6}{'Learner Added':<30}{'Strategy':>10}{'Ensemble AUC':>14}"
        f"{'Delta':>11}{'Spearman r':>12}{'Models':>8}"
    )
    print("-" * w)

    for step_n, name, strategy, auc, delta, corr, n_models in selected_rows:
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

    if selected_rows:
        final = selected_rows[-1]
        print(
            f"\nFinal ensemble: {final[6]} models, AUC: {final[3]:.5f}"
            f" (strategy: {final[2]})"
        )
        if rejected_rows:
            print(f"Rejected: {len(rejected_rows)} models (would decrease AUC)")


def print_leave_one_out(
    full_auc: float, strategy: str, n_models: int, rows: list[tuple]
) -> None:
    """Print leave-one-out analysis results in a formatted table."""
    print(f"\n{'=' * 90}")
    print("LEAVE-ONE-OUT ANALYSIS")
    print(f"{'=' * 90}")
    print(
        f"Full ensemble AUC: {full_auc:.5f} ({n_models} models, strategy: {strategy})\n"
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

    print(f"{'=' * 90}")

    harmful = [r for r in rows if r[4] == "HARMFUL"]
    helpful = [r for r in rows if r[4] == "helpful"]
    print(
        f"\nSummary: {len(helpful)} helpful, "
        f"{n_models - len(helpful) - len(harmful)} neutral, "
        f"{len(harmful)} harmful"
    )
    if harmful:
        print(f"Consider removing: {', '.join(r[0] for r in harmful)}")
