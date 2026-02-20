from dataclasses import dataclass, field

import numpy as np
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score

from .stacking import l2_stacking
from .weights import hill_climbing


@dataclass
class EnsembleMethodResult:
    name: str
    oof_preds: np.ndarray
    holdout_preds: np.ndarray
    test_preds: np.ndarray
    auc: float
    metadata: dict = field(default_factory=dict)


@dataclass
class EnsembleResult:
    methods: list[EnsembleMethodResult]
    best_method: str
    best_test_preds: np.ndarray
    best_auc: float
    calibrated: bool
    all_aucs: dict[str, float]


def compute_ensemble(
    model_names,
    oof_preds,
    holdout_preds,
    test_preds,
    train_labels,
    holdout_labels=None,
    l2_feature_configs=None,
) -> EnsembleResult:
    """Run all ensemble methods. No logging, no side effects.

    Args:
        model_names: List of model/learner names.
        oof_preds: Dict mapping name -> OOF prediction array.
        holdout_preds: Dict mapping name -> holdout prediction array.
        test_preds: Dict mapping name -> test prediction array.
        train_labels: Ground truth labels for training set.
        holdout_labels: Ground truth labels for holdout set (optional).
        l2_feature_configs: List of (name, train_feat, holdout_feat, test_feat) tuples
            for L2 stacking with extra features. Pass None to skip feature-augmented
            stacking. Each tuple's feat arrays can be None for preds-only stacking.
    """
    oof_matrix = np.column_stack([oof_preds[n] for n in model_names])
    holdout_matrix = np.column_stack([holdout_preds[n] for n in model_names])
    test_matrix = np.column_stack([test_preds[n] for n in model_names])

    methods = []

    # --- Simple average ---
    avg_oof = np.mean(oof_matrix, axis=1)
    avg_holdout = np.mean(holdout_matrix, axis=1)
    avg_test = np.mean(test_matrix, axis=1)
    methods.append(
        EnsembleMethodResult(
            name="average",
            oof_preds=avg_oof,
            holdout_preds=avg_holdout,
            test_preds=avg_test,
            auc=_eval_auc(avg_oof, avg_holdout, train_labels, holdout_labels),
        )
    )

    # --- Ridge stacking ---
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge.fit(oof_matrix, train_labels)
    ridge_oof = ridge.predict(oof_matrix)
    ridge_holdout = ridge.predict(holdout_matrix)
    ridge_test = ridge.predict(test_matrix)
    methods.append(
        EnsembleMethodResult(
            name="ridge",
            oof_preds=ridge_oof,
            holdout_preds=ridge_holdout,
            test_preds=ridge_test,
            auc=_eval_auc(ridge_oof, ridge_holdout, train_labels, holdout_labels),
            metadata={
                "alpha": ridge.alpha_,
                "weights": dict(zip(model_names, ridge.coef_)),
            },
        )
    )

    # --- Hill Climbing ---
    best_weights = hill_climbing(oof_matrix, train_labels, model_names)
    hc_oof = oof_matrix @ best_weights
    hc_holdout = holdout_matrix @ best_weights
    hc_test = test_matrix @ best_weights
    methods.append(
        EnsembleMethodResult(
            name="hill_climbing",
            oof_preds=hc_oof,
            holdout_preds=hc_holdout,
            test_preds=hc_test,
            auc=_eval_auc(hc_oof, hc_holdout, train_labels, holdout_labels),
            metadata={"weights": dict(zip(model_names, best_weights))},
        )
    )

    # --- Rank Blending ---
    def _rank_blend(matrix):
        n = matrix.shape[0]
        ranked = np.column_stack(
            [rankdata(matrix[:, i]) / n for i in range(matrix.shape[1])]
        )
        return np.mean(ranked, axis=1)

    rb_oof = _rank_blend(oof_matrix)
    rb_holdout = _rank_blend(holdout_matrix)
    rb_test = _rank_blend(test_matrix)
    methods.append(
        EnsembleMethodResult(
            name="rank_blending",
            oof_preds=rb_oof,
            holdout_preds=rb_holdout,
            test_preds=rb_test,
            auc=_eval_auc(rb_oof, rb_holdout, train_labels, holdout_labels),
        )
    )

    # --- L2 Stacking ---
    l2_configs = [("preds_only", None, None, None)]
    if l2_feature_configs:
        l2_configs += l2_feature_configs

    for fs_name, train_feat, holdout_feat, test_feat in l2_configs:
        l2_oof, l2_holdout, l2_test = l2_stacking(
            oof_matrix,
            holdout_matrix,
            test_matrix,
            train_labels,
            train_feat,
            holdout_feat,
            test_feat,
        )
        methods.append(
            EnsembleMethodResult(
                name=f"l2_{fs_name}",
                oof_preds=l2_oof,
                holdout_preds=l2_holdout,
                test_preds=l2_test,
                auc=_eval_auc(l2_oof, l2_holdout, train_labels, holdout_labels),
            )
        )

    # Pick best ensemble method
    all_aucs = {m.name: m.auc for m in methods}
    best_name = max(all_aucs, key=lambda k: all_aucs[k])
    best_method = next(m for m in methods if m.name == best_name)

    best_oof = best_method.oof_preds
    best_holdout_out = best_method.holdout_preds
    best_test_out = best_method.test_preds
    best_auc = best_method.auc

    # --- Post-processing: Isotonic calibration ---
    calibrated = False
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(best_oof, train_labels)
    cal_holdout = iso.predict(best_holdout_out)
    cal_test = iso.predict(best_test_out)

    if holdout_labels is not None:
        cal_auc = roc_auc_score(holdout_labels, cal_holdout)
        if cal_auc > best_auc:
            best_test_out = cal_test
            best_auc = cal_auc
            calibrated = True

    return EnsembleResult(
        methods=methods,
        best_method=best_name,
        best_test_preds=best_test_out,
        best_auc=best_auc,
        calibrated=calibrated,
        all_aucs=all_aucs,
    )


def _eval_auc(oof, holdout, train_labels, holdout_labels):
    """Evaluate AUC using holdout labels if available, else OOF."""
    if holdout_labels is not None:
        return roc_auc_score(holdout_labels, holdout)
    return roc_auc_score(train_labels, oof)
