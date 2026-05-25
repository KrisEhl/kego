"""Feature selection via permutation importance, drop-one ablation, and forward selection.

All functions are model-agnostic: pass an unfitted sklearn-compatible estimator.
The model is cloned per seed with ``random_state`` set automatically.

Typical usage::

    import lightgbm as lgb
    from kego.features.selection import drop_one_ablation, forward_selection

    model = lgb.LGBMClassifier(n_estimators=500, verbosity=-1)

    result = drop_one_ablation(
        X_train,
        y_train,
        X_holdout,
        y_holdout,
        features=feature_list,
        seeds=[42, 123, 777],
        metric="roc_auc",
        model=model,
    )
    print(f"Baseline: {result.baseline_score:.5f}")
    print(f"Optimized: {result.selected_score:.5f} ({len(result.selected_features)} features)")
    for entry in result.feature_results:
        print(f"  {entry['feature']}: {entry['delta']:+.5f}")
"""

import contextlib
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import get_scorer

# Reserve 2 cores for the OS; use the rest for parallel seed training.
_TOTAL_CORES = os.cpu_count() or 4
_USABLE_CORES = max(1, _TOTAL_CORES - 2)


@dataclass
class SelectionResult:
    """Result from a feature selection method."""

    selected_features: list[str]
    selected_score: float
    baseline_score: float
    feature_results: list[dict] = field(default_factory=list)


def _train_and_score(
    model: BaseEstimator,
    seed: int,
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: np.ndarray | pd.Series,
    features: list[str],
    metric: str,
    fit_kwargs: dict,
    n_jobs_per_model: int,
) -> tuple[float, BaseEstimator]:
    """Train a single seeded model and return (score, fitted_model)."""
    seeded_model = clone(model).set_params(random_state=seed)
    with contextlib.suppress(ValueError, TypeError):
        seeded_model.set_params(n_jobs=n_jobs_per_model)
    fit_kwargs = dict(fit_kwargs)
    if "eval_set" in fit_kwargs:
        fit_kwargs["eval_set"] = [(X_holdout[features], y_holdout)]
    seeded_model.fit(X_train[features], y_train, **fit_kwargs)
    score = float(get_scorer(metric)(seeded_model, X_holdout[features], y_holdout))
    return score, seeded_model


def eval_multiseed(
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: np.ndarray | pd.Series,
    features: list[str],
    seeds: list[int],
    metric: str,
    model: BaseEstimator,
    fit_kwargs: dict | None = None,
) -> tuple[float, int]:
    """Evaluate a feature subset with multi-seed averaging.

    Seeds are trained in parallel. Each model gets an equal share of the
    available cores (total CPU minus 2 reserved).

    Args:
        X_train: Training features (all columns available; subset selected by features).
        y_train: Training labels.
        X_holdout: Holdout features.
        y_holdout: Holdout labels.
        features: Feature subset to evaluate.
        seeds: List of random seeds. One model trained per seed.
        model: Unfitted sklearn-compatible estimator. Cloned per seed.
        fit_kwargs: Optional extra keyword arguments passed to model.fit().

    Returns:
        Tuple of (mean_score, mean_iterations) across seeds.
    """
    fit_kwargs = dict(fit_kwargs) if fit_kwargs else {}
    if "categorical_feature" in fit_kwargs:
        fit_kwargs["categorical_feature"] = [
            column for column in fit_kwargs["categorical_feature"] if column in features
        ]

    n_parallel_seeds = min(len(seeds), _USABLE_CORES)
    n_jobs_per_model = max(1, _USABLE_CORES // n_parallel_seeds)

    results = Parallel(n_jobs=n_parallel_seeds, backend="threading")(
        delayed(_train_and_score)(
            model,
            seed,
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            features,
            metric,
            fit_kwargs,
            n_jobs_per_model,
        )
        for seed in seeds
    )
    scores = [score for score, _ in results]
    fitted_models = [fitted_model for _, fitted_model in results]
    iterations = [
        getattr(fitted_model, "best_iteration_", getattr(fitted_model, "n_estimators", 0))
        for fitted_model in fitted_models
    ]
    return float(np.mean(scores)), int(np.mean(iterations))


def drop_one_ablation(
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: np.ndarray | pd.Series,
    features: list[str],
    seeds: list[int],
    metric: str,
    model: BaseEstimator,
    fit_kwargs: dict | None = None,
    verbose: bool = True,
) -> SelectionResult:
    """Drop-one-at-a-time ablation: train on all-but-one feature, measure score delta.

    A positive delta means removing that feature *improves* the score — the feature
    is harmful. The optimized feature set excludes all harmful features.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_holdout: Holdout features.
        y_holdout: Holdout labels.
        features: Full feature list.
        seeds: Random seeds for multi-seed averaging.
        metric: Sklearn scorer name (e.g. "roc_auc").
        model: Unfitted sklearn-compatible estimator. Cloned per seed.
        fit_kwargs: Optional extra kwargs for model.fit().
        verbose: Print progress.

    Returns:
        SelectionResult with selected_features excluding harmful features.
        feature_results contain {"feature", "score", "delta"} per feature.
    """
    baseline, baseline_iters = eval_multiseed(
        X_train,
        y_train,
        X_holdout,
        y_holdout,
        features,
        seeds,
        metric,
        model,
        fit_kwargs,
    )
    if verbose:
        print(
            f"Baseline {metric} ({len(features)} features, {len(seeds)} seeds): {baseline:.5f} ({baseline_iters} iters)"
        )

    feature_results = []
    for i, feat in enumerate(features):
        reduced = [feature for feature in features if feature != feat]
        score_without, iters = eval_multiseed(
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            reduced,
            seeds,
            metric,
            model,
            fit_kwargs,
        )
        delta = score_without - baseline
        feature_results.append(
            {
                "feature": feat,
                "score": score_without,
                "delta": delta,
                "iterations": iters,
            }
        )
        if verbose:
            print(
                f"  [{i + 1}/{len(features)}] -{feat:<30} "
                f"{metric}={score_without:.5f} delta={delta:+.5f} ({iters} iters)"
            )

    feature_results.sort(key=lambda x: x["delta"], reverse=True)

    if verbose:
        score_header = f"{metric} without"
        score_width = max(len(score_header), 12)
        print(f"\n{'Feature':<30} {score_header:>{score_width}} {'Delta':>10} {'Iters':>7} {'Verdict':>10}")
        print("-" * (30 + score_width + 10 + 7 + 10 + 4))
        for entry in feature_results:
            verdict = "HARMFUL" if entry["delta"] > 0 else "helpful"
            print(
                f"{entry['feature']:<30} {entry['score']:>{score_width}.5f} "
                f"{entry['delta']:>+10.5f} {entry['iterations']:>7} {verdict:>10}"
            )
        harmful = [entry["feature"] for entry in feature_results if entry["delta"] > 0]
        print(f"\nHarmful (removal improves {metric}): {harmful}")

    selected = [entry["feature"] for entry in feature_results if entry["delta"] <= 0]
    if selected != features:
        selected_score, _ = eval_multiseed(
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            selected,
            seeds,
            metric,
            model,
            fit_kwargs,
        )
    else:
        selected_score = baseline

    return SelectionResult(
        selected_features=selected,
        selected_score=selected_score,
        baseline_score=baseline,
        feature_results=feature_results,
    )


def forward_selection(
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: np.ndarray | pd.Series,
    features_ordered: list[str],
    seeds: list[int],
    metric: str,
    model: BaseEstimator,
    fit_kwargs: dict | None = None,
    verbose: bool = True,
) -> SelectionResult:
    """Forward selection: add features one at a time in the given order.

    Designed to be used after permutation importance ranking — pass features
    sorted by decreasing importance. The function evaluates cumulative score at
    each prefix and returns the optimal prefix.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_holdout: Holdout features.
        y_holdout: Holdout labels.
        features_ordered: Features in priority order (e.g. by permutation importance).
        seeds: Random seeds for multi-seed averaging.
        metric: Sklearn scorer name (e.g. "roc_auc").
        model: Unfitted sklearn-compatible estimator. Cloned per seed.
        fit_kwargs: Optional extra kwargs for model.fit().
        verbose: Print progress.

    Returns:
        SelectionResult with the optimal feature prefix.
        feature_results contain {"feature", "score", "delta"} per step.
    """
    feature_results = []
    score_width = max(len(metric), 10)
    if verbose:
        print(f"\n{'N':>3} {'Added feature':<30} {metric:>{score_width}} {'Delta':>10} {'Iters':>7}")
        print("-" * (3 + 1 + 30 + 1 + score_width + 1 + 10 + 1 + 7))

    previous_score = 0.0
    for i, _ in enumerate(features_ordered, start=1):
        subset = features_ordered[:i]
        score, iters = eval_multiseed(
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            subset,
            seeds,
            metric,
            model,
            fit_kwargs,
        )
        delta = score - previous_score if i > 1 else 0.0
        feature_results.append(
            {
                "feature": subset[-1],
                "score": score,
                "delta": delta,
                "iterations": iters,
            }
        )
        if verbose:
            print(f"{i:>3} {subset[-1]:<30} {score:>{score_width}.5f} {delta:>+10.5f} {iters:>7}")
        previous_score = score

    best_index = max(range(len(feature_results)), key=lambda i: feature_results[i]["score"])
    selected = features_ordered[: best_index + 1]
    best_score = feature_results[best_index]["score"]

    if verbose:
        print(f"\nOptimal: {len(selected)} features, {metric}={best_score:.5f}")
        print(f"Features: {selected}")

    return SelectionResult(
        selected_features=selected,
        selected_score=best_score,
        baseline_score=feature_results[-1]["score"],
        feature_results=feature_results,
    )


def greedy_add_one_screening(
    baseline_features: list[str],
    candidate_features: list[str],
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: np.ndarray | pd.Series,
    seeds: list[int],
    metric: str,
    model: BaseEstimator,
    baseline_score: float | None = None,
    fit_kwargs: dict | None = None,
    verbose: bool = True,
) -> SelectionResult:
    """Test each candidate feature by adding it to the baseline feature set.

    Used for research feature screening: given a curated baseline set, test
    whether each candidate adds signal. Each candidate is tested independently
    (not greedily accumulated). The optimized feature set is the baseline plus
    all candidates with positive delta.

    Args:
        baseline_features: Current best feature set.
        candidate_features: Features to test one at a time.
        X_train: Training features.
        y_train: Training labels.
        X_holdout: Holdout features.
        y_holdout: Holdout labels.
        seeds: Random seeds for multi-seed averaging.
        metric: Sklearn scorer name (e.g. "roc_auc").
        model: Unfitted sklearn-compatible estimator. Cloned per seed.
        baseline_score: Pre-computed baseline score. If None, computed from baseline_features.
        fit_kwargs: Optional extra kwargs for model.fit().
        verbose: Print progress.

    Returns:
        SelectionResult with baseline + helpful candidates as selected_features.
        feature_results contain {"feature", "score", "delta"} per candidate.
    """
    if baseline_score is None:
        baseline_score, _ = eval_multiseed(
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            baseline_features,
            seeds,
            metric,
            model,
            fit_kwargs,
        )
    if verbose:
        print(f"Baseline {metric}: {baseline_score:.5f} ({len(baseline_features)} features)")
        print(f"Screening {len(candidate_features)} candidates...\n")

    feature_results = []
    for i, feat in enumerate(candidate_features):
        augmented = [*baseline_features, feat]
        score, iters = eval_multiseed(
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            augmented,
            seeds,
            metric,
            model,
            fit_kwargs,
        )
        delta = score - baseline_score
        feature_results.append(
            {
                "feature": feat,
                "score": score,
                "delta": delta,
                "iterations": iters,
            }
        )
        if verbose:
            print(
                f"  [{i + 1}/{len(candidate_features)}] +{feat:<30} "
                f"{metric}={score:.5f} delta={delta:+.5f} ({iters} iters)"
            )

    feature_results.sort(key=lambda x: x["delta"], reverse=True)

    if verbose:
        score_header = f"{metric} with"
        score_width = max(len(score_header), 10)
        print(f"\n{'Feature':<30} {score_header:>{score_width}} {'Delta':>10} {'Iters':>7}")
        print("-" * (30 + score_width + 10 + 7 + 3))
        for entry in feature_results:
            print(
                f"{entry['feature']:<30} {entry['score']:>{score_width}.5f} "
                f"{entry['delta']:>+10.5f} {entry['iterations']:>7}"
            )
        helpful = [entry["feature"] for entry in feature_results if entry["delta"] > 0]
        print(f"\nHelpful candidates ({len(helpful)}): {helpful}")

    helpful_candidates = [entry["feature"] for entry in feature_results if entry["delta"] > 0]
    selected = baseline_features + helpful_candidates
    if helpful_candidates:
        selected_score, _ = eval_multiseed(
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            selected,
            seeds,
            metric,
            model,
            fit_kwargs,
        )
    else:
        selected_score = baseline_score

    return SelectionResult(
        selected_features=selected,
        selected_score=selected_score,
        baseline_score=baseline_score,
        feature_results=feature_results,
    )
