"""Feature selection via permutation importance, drop-one ablation, and forward selection.

All functions are model-agnostic: pass a `model_factory` callable that returns
a fitted sklearn-compatible estimator. The only requirement is that the model
exposes `predict_proba(X)[:, 1]`.

Typical usage::

    import lightgbm as lgb
    from kego.features.selection import drop_one_ablation, forward_selection


    def lgbm_factory(seed):
        return lgb.LGBMClassifier(n_estimators=500, random_state=seed, verbosity=-1)


    results = drop_one_ablation(
        X_train,
        y_train,
        X_holdout,
        y_holdout,
        features=feature_list,
        seeds=[42, 123, 777],
        model_factory=lgbm_factory,
    )
    for feat, auc_without, delta in results:
        print(f"{feat}: {delta:+.5f}")
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def eval_multiseed(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_holdout: pd.DataFrame,
    y_holdout: np.ndarray,
    features: list[str],
    seeds: list[int],
    model_factory: Callable[[int], Any],
    fit_kwargs: dict | None = None,
) -> float:
    """Evaluate a feature subset with multi-seed averaging.

    Args:
        X_train: Training features (all columns available; subset selected by features).
        y_train: Training labels.
        X_holdout: Holdout features.
        y_holdout: Holdout labels.
        features: Feature subset to evaluate.
        seeds: List of random seeds. One model trained per seed.
        model_factory: Callable(seed) -> unfitted sklearn-compatible classifier.
        fit_kwargs: Optional extra keyword arguments passed to model.fit().

    Returns:
        Mean holdout AUC across seeds.
    """
    fit_kwargs = fit_kwargs or {}
    aucs = []
    for seed in seeds:
        model = model_factory(seed)
        model.fit(X_train[features], y_train, **fit_kwargs)  # type: ignore[union-attr]
        preds = model.predict_proba(X_holdout[features])[:, 1]  # type: ignore[union-attr]
        aucs.append(roc_auc_score(y_holdout, preds))
    return float(np.mean(aucs))


def drop_one_ablation(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_holdout: pd.DataFrame,
    y_holdout: np.ndarray,
    features: list[str],
    seeds: list[int],
    model_factory: Callable[[int], Any],
    fit_kwargs: dict | None = None,
    verbose: bool = True,
) -> list[tuple[str, float, float]]:
    """Drop-one-at-a-time ablation: train on all-but-one feature, measure AUC delta.

    A positive delta means removing that feature *improves* AUC — the feature is harmful.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_holdout: Holdout features.
        y_holdout: Holdout labels.
        features: Full feature list.
        seeds: Random seeds for multi-seed averaging.
        model_factory: Callable(seed) -> unfitted classifier.
        fit_kwargs: Optional extra kwargs for model.fit().
        verbose: Print progress.

    Returns:
        List of (feature, auc_without, delta) tuples, sorted by delta descending
        (most harmful features first).
    """
    baseline = eval_multiseed(
        X_train,
        y_train,
        X_holdout,
        y_holdout,
        features,
        seeds,
        model_factory,
        fit_kwargs,
    )
    if verbose:
        print(
            f"Baseline AUC ({len(features)} features, {len(seeds)} seeds): {baseline:.5f}"
        )

    results = []
    for i, feat in enumerate(features):
        reduced = [f for f in features if f != feat]
        auc_without = eval_multiseed(
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            reduced,
            seeds,
            model_factory,
            fit_kwargs,
        )
        delta = auc_without - baseline
        results.append((feat, auc_without, delta))
        if verbose:
            print(
                f"  [{i + 1}/{len(features)}] -{feat:<30} AUC={auc_without:.5f} delta={delta:+.5f}"
            )

    results.sort(key=lambda x: x[2], reverse=True)

    if verbose:
        print(f"\n{'Feature':<30} {'AUC without':>12} {'Delta':>10} {'Verdict':>10}")
        print("-" * 66)
        for feat, auc_without, delta in results:
            verdict = "HARMFUL" if delta > 0 else "helpful"
            print(f"{feat:<30} {auc_without:>12.5f} {delta:>+10.5f} {verdict:>10}")
        harmful = [f for f, _, d in results if d > 0]
        print(f"\nHarmful (removal improves AUC): {harmful}")

    return results


def forward_selection(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_holdout: pd.DataFrame,
    y_holdout: np.ndarray,
    features_ordered: list[str],
    seeds: list[int],
    model_factory: Callable[[int], Any],
    fit_kwargs: dict | None = None,
    verbose: bool = True,
) -> tuple[list[str], float]:
    """Forward selection: add features one at a time in the given order.

    Designed to be used after permutation importance ranking — pass features
    sorted by decreasing importance. The function evaluates cumulative AUC at
    each prefix and returns the optimal prefix.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_holdout: Holdout features.
        y_holdout: Holdout labels.
        features_ordered: Features in priority order (e.g. by permutation importance).
        seeds: Random seeds for multi-seed averaging.
        model_factory: Callable(seed) -> unfitted classifier.
        fit_kwargs: Optional extra kwargs for model.fit().
        verbose: Print progress.

    Returns:
        selected_features: Feature subset with highest holdout AUC.
        best_auc: AUC of the selected subset.
    """
    history = []
    if verbose:
        print(f"\n{'N':>3} {'Added feature':<30} {'AUC':>10} {'Delta':>10}")
        print("-" * 57)

    prev_auc = 0.0
    for i, _ in enumerate(features_ordered, start=1):
        subset = features_ordered[:i]
        auc = eval_multiseed(
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            subset,
            seeds,
            model_factory,
            fit_kwargs,
        )
        history.append((i, subset[-1], auc))
        if verbose:
            delta = auc - prev_auc if i > 1 else 0.0
            print(f"{i:>3} {subset[-1]:<30} {auc:>10.5f} {delta:>+10.5f}")
        prev_auc = auc

    best_n, _, best_auc = max(history, key=lambda x: x[2])
    selected = features_ordered[:best_n]
    if verbose:
        print(f"\nOptimal: {best_n} features, AUC={best_auc:.5f}")
        print(f"Features: {selected}")

    return selected, best_auc


def greedy_add_one_screening(
    baseline_features: list[str],
    candidate_features: list[str],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_holdout: pd.DataFrame,
    y_holdout: np.ndarray,
    seeds: list[int],
    model_factory: Callable[[int], Any],
    baseline_auc: float | None = None,
    fit_kwargs: dict | None = None,
    verbose: bool = True,
) -> list[tuple[str, float, float]]:
    """Test each candidate feature by adding it to the baseline feature set.

    Used for research feature screening: given a curated baseline set, test
    whether each candidate adds signal. Each candidate is tested independently
    (not greedily accumulated).

    Args:
        baseline_features: Current best feature set.
        candidate_features: Features to test one at a time.
        X_train: Training features.
        y_train: Training labels.
        X_holdout: Holdout features.
        y_holdout: Holdout labels.
        seeds: Random seeds for multi-seed averaging.
        model_factory: Callable(seed) -> unfitted classifier.
        baseline_auc: Pre-computed baseline AUC. If None, computed from baseline_features.
        fit_kwargs: Optional extra kwargs for model.fit().
        verbose: Print progress.

    Returns:
        List of (feature, auc_with, delta) tuples, sorted by delta descending.
    """
    if baseline_auc is None:
        baseline_auc = eval_multiseed(
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            baseline_features,
            seeds,
            model_factory,
            fit_kwargs,
        )
    if verbose:
        print(f"Baseline AUC: {baseline_auc:.5f} ({len(baseline_features)} features)")
        print(f"Screening {len(candidate_features)} candidates...\n")

    results = []
    for i, feat in enumerate(candidate_features):
        augmented = [*baseline_features, feat]
        auc = eval_multiseed(
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            augmented,
            seeds,
            model_factory,
            fit_kwargs,
        )
        delta = auc - baseline_auc
        results.append((feat, auc, delta))
        if verbose:
            print(
                f"  [{i + 1}/{len(candidate_features)}] +{feat:<30} AUC={auc:.5f} delta={delta:+.5f}"
            )

    results.sort(key=lambda x: x[2], reverse=True)

    if verbose:
        print(f"\n{'Feature':<30} {'AUC with':>10} {'Delta':>10}")
        print("-" * 54)
        for feat, auc, delta in results:
            print(f"{feat:<30} {auc:>10.5f} {delta:>+10.5f}")
        helpful = [f for f, _, d in results if d > 0]
        print(f"\nHelpful candidates ({len(helpful)}): {helpful}")

    return results
