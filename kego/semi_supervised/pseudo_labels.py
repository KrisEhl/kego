"""Soft pseudo-labeling for semi-supervised learning.

Self-training loop: train on labeled data, generate soft labels for unlabeled
data (e.g. test set), retrain combining both with configurable sample weights,
optionally iterate multiple rounds.

The training function is passed as a callable, making this model-agnostic.

Example::

    from kego.semi_supervised.pseudo_labels import soft_pseudo_label_experiment

    def my_train_fn(X_train, y_train, X_holdout, y_holdout, X_test, seed, sample_weight):
        # train your model, return (holdout_auc, test_predictions)
        ...

    results = soft_pseudo_label_experiment(
        X_train, y_train, X_holdout, y_holdout, X_test,
        seeds=[42, 123, 777],
        train_fn=my_train_fn,
        test_weights=[1.0, 0.3],
        n_rounds=2,
    )
    # results is a dict like {"baseline": 0.956, "round1_w1.0": 0.955, ...}
"""

from collections.abc import Callable

import numpy as np
import pandas as pd


def self_training_round(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    X_unlabeled: pd.DataFrame,
    soft_labels: np.ndarray | None,
    test_weight: float,
    seeds: list[int],
    train_fn: Callable,
    label: str = "",
) -> tuple[float, np.ndarray]:
    """One round of soft pseudo-labeling with multi-seed averaging.

    If soft_labels is None, trains on labeled data only (baseline). Otherwise,
    appends the unlabeled rows with soft labels and the given sample weight.

    Args:
        X_train: Labeled training features.
        y_train: Labeled training targets (float, for soft labels in later rounds).
        X_holdout: Holdout features for evaluation.
        y_holdout: Holdout labels.
        X_unlabeled: Unlabeled data (e.g. test set) to be pseudo-labeled.
        soft_labels: Soft probability labels for X_unlabeled. None for baseline.
        test_weight: Sample weight applied to pseudo-labeled rows (e.g. 0.3 to down-weight).
        seeds: Random seeds. One model trained per seed; results averaged.
        train_fn: Callable with signature
            ``(X_train, y_train, X_holdout, y_holdout, X_unlabeled, seed, sample_weight)
            -> (holdout_auc: float, unlabeled_preds: np.ndarray)``.
        label: Descriptive label for progress printing.

    Returns:
        mean_holdout_auc: Mean holdout AUC across seeds.
        avg_unlabeled_preds: Averaged predictions for X_unlabeled across seeds.
    """
    aucs = []
    all_unlabeled_preds = []

    for seed in seeds:
        if soft_labels is not None:
            X_combined = pd.concat(
                [X_train, X_unlabeled.reset_index(drop=True)], ignore_index=True
            )
            y_combined = pd.concat(
                [
                    y_train.reset_index(drop=True),
                    pd.Series(soft_labels, name=y_train.name),
                ],
                ignore_index=True,
            )
            sw = np.ones(len(X_combined))
            sw[len(X_train) :] = test_weight
        else:
            X_combined = X_train
            y_combined = y_train
            sw = None

        auc, unlabeled_preds = train_fn(
            X_combined, y_combined, X_holdout, y_holdout, X_unlabeled, seed, sw
        )
        aucs.append(auc)
        all_unlabeled_preds.append(unlabeled_preds)

        prefix = f"  {label} | " if label else "  "
        print(f"{prefix}seed={seed} | holdout AUC: {auc:.5f}", flush=True)

    mean_auc = float(np.mean(aucs))
    mean_preds = np.mean(all_unlabeled_preds, axis=0)
    prefix = f"  {label} | " if label else "  "
    print(f"{prefix}MEAN: {mean_auc:.5f}\n", flush=True)

    return mean_auc, mean_preds


def soft_pseudo_label_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    X_unlabeled: pd.DataFrame,
    seeds: list[int],
    train_fn: Callable,
    test_weights: list[float] | None = None,
    n_rounds: int = 2,
) -> dict[str, float]:
    """Full multi-round soft pseudo-labeling experiment.

    Runs a baseline (no pseudo-labels), then for each test_weight runs n_rounds
    of self-training with soft labels from the previous round.

    Args:
        X_train: Labeled training features.
        y_train: Labeled training targets.
        X_holdout: Holdout features.
        y_holdout: Holdout labels.
        X_unlabeled: Unlabeled data (e.g. test set).
        seeds: Random seeds for multi-seed averaging.
        train_fn: Training callable — see self_training_round for signature.
        test_weights: Sample weights to try for pseudo-labeled rows. Default: [1.0, 0.3].
        n_rounds: Number of self-training rounds per weight.

    Returns:
        Dict mapping label -> mean_holdout_auc. Keys:
        "baseline", "round1_w{weight}", "round2_w{weight}", ...
    """
    if test_weights is None:
        test_weights = [1.0, 0.3]

    results: dict[str, float] = {}

    print("=== Baseline (no pseudo-labels) ===", flush=True)
    baseline_auc, base_preds = self_training_round(
        X_train,
        y_train,
        X_holdout,
        y_holdout,
        X_unlabeled,
        soft_labels=None,
        test_weight=1.0,
        seeds=seeds,
        train_fn=train_fn,
        label="baseline",
    )
    results["baseline"] = baseline_auc

    for tw in test_weights:
        prev_preds = base_preds
        for round_n in range(1, n_rounds + 1):
            key = f"round{round_n}_w{tw}"
            print(
                f"=== Round {round_n}: soft pseudo-labels, test_weight={tw} ===",
                flush=True,
            )
            round_auc, prev_preds = self_training_round(
                X_train,
                y_train,
                X_holdout,
                y_holdout,
                X_unlabeled,
                soft_labels=prev_preds,
                test_weight=tw,
                seeds=seeds,
                train_fn=train_fn,
                label=key,
            )
            results[key] = round_auc
            print(
                f"  weight={tw} round={round_n}: "
                f"baseline={baseline_auc:.5f} | {key}={round_auc:.5f} "
                f"({round_auc - baseline_auc:+.5f})",
                flush=True,
            )

    print("=" * 60, flush=True)
    print("FINAL SUMMARY", flush=True)
    for key, auc in results.items():
        delta = f" ({auc - baseline_auc:+.5f})" if key != "baseline" else ""
        print(f"  {key:<25} {auc:.5f}{delta}", flush=True)

    return results
