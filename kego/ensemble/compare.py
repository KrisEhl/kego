"""Stacking method comparison: simple average vs Ridge vs LightGBM meta-models.

Use this to quickly decide whether a non-linear meta-learner adds value over
simple averaging. If the gap is < 0.001 AUC, stick with averaging.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict


def compare_stacking_methods(
    model_names: list[str],
    all_oof: dict[str, np.ndarray],
    all_holdout: dict[str, np.ndarray],
    train_labels: np.ndarray,
    holdout_labels: np.ndarray,
    train_features: np.ndarray | pd.DataFrame | None = None,
    holdout_features: np.ndarray | pd.DataFrame | None = None,
) -> list[tuple[str, float, float]]:
    """Compare average / Ridge / LightGBM-preds / LightGBM-preds+features.

    All LightGBM meta-models are intentionally simple (shallow, regularized)
    to avoid overfitting on the small number of meta-features.

    Args:
        model_names: Ordered list of model/learner names.
        all_oof: Dict mapping name -> OOF prediction array (train rows).
        all_holdout: Dict mapping name -> holdout prediction array.
        train_labels: Binary labels for the training set.
        holdout_labels: Binary labels for the holdout set.
        train_features: Optional extra features for the LightGBM preds+features method.
        holdout_features: Matching holdout rows for train_features.

    Returns:
        List of (method_name, holdout_auc, oof_auc) tuples, sorted by holdout_auc desc.
        Also prints a formatted comparison table with verdict.
    """
    import lightgbm as lgb

    oof_matrix = np.column_stack([all_oof[n] for n in model_names])
    holdout_matrix = np.column_stack([all_holdout[n] for n in model_names])

    results: list[tuple[str, float, float]] = []

    # 1. Simple average
    avg_oof = np.mean(oof_matrix, axis=1)
    avg_holdout = np.mean(holdout_matrix, axis=1)
    oof_auc_avg = roc_auc_score(train_labels, avg_oof)
    holdout_auc_avg = roc_auc_score(holdout_labels, avg_holdout)
    results.append(("Simple Average", holdout_auc_avg, oof_auc_avg))

    # 2. Ridge
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge.fit(oof_matrix, train_labels)
    ridge_holdout = ridge.predict(holdout_matrix)
    ridge_oof = cross_val_predict(
        RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]),
        oof_matrix,
        train_labels,
        cv=5,
    )
    results.append(
        (
            "Ridge",
            roc_auc_score(holdout_labels, ridge_holdout),
            roc_auc_score(train_labels, ridge_oof),
        )
    )

    # 3. LightGBM (preds only)
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "max_depth": 3,
        "num_leaves": 7,
        "min_child_samples": 200,
        "n_estimators": 500,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbosity": -1,
    }
    lgb_preds = lgb.LGBMClassifier(**lgb_params)
    lgb_preds.fit(oof_matrix, train_labels)
    lgb_preds_holdout = lgb_preds.predict_proba(holdout_matrix)[:, 1]
    lgb_preds_oof = cross_val_predict(
        lgb.LGBMClassifier(**lgb_params),
        oof_matrix,
        train_labels,
        cv=5,
        method="predict_proba",
    )[:, 1]
    results.append(
        (
            "LightGBM (preds only)",
            roc_auc_score(holdout_labels, lgb_preds_holdout),
            roc_auc_score(train_labels, lgb_preds_oof),
        )
    )

    # 4. LightGBM (preds + features)
    has_features = train_features is not None and holdout_features is not None
    lgb_full = None
    if has_features:
        train_feat_np = (
            train_features.values
            if isinstance(train_features, pd.DataFrame)
            else np.asarray(train_features)
        )
        holdout_feat_np = (
            holdout_features.values
            if isinstance(holdout_features, pd.DataFrame)
            else np.asarray(holdout_features)
        )
        oof_plus = np.hstack([oof_matrix, train_feat_np])
        holdout_plus = np.hstack([holdout_matrix, holdout_feat_np])

        lgb_full_params = {
            **lgb_params,
            "max_depth": 4,
            "num_leaves": 15,
            "colsample_bytree": 0.5,
            "reg_lambda": 2.0,
        }
        lgb_full = lgb.LGBMClassifier(**lgb_full_params)
        lgb_full.fit(oof_plus, train_labels)
        lgb_full_holdout = lgb_full.predict_proba(holdout_plus)[:, 1]
        lgb_full_oof = cross_val_predict(
            lgb.LGBMClassifier(**lgb_full_params),
            oof_plus,
            train_labels,
            cv=5,
            method="predict_proba",
        )[:, 1]
        results.append(
            (
                "LightGBM (preds+features)",
                roc_auc_score(holdout_labels, lgb_full_holdout),
                roc_auc_score(train_labels, lgb_full_oof),
            )
        )

    # --- Print comparison table ---
    print("\n" + "=" * 70)
    print("STACKING COMPARISON")
    print("=" * 70)
    print(f"{'Method':<30} {'Holdout AUC':>12} {'OOF AUC':>12} {'Delta':>10}")
    print("-" * 70)
    for method, holdout_auc, oof_auc in results:
        delta = holdout_auc - holdout_auc_avg
        delta_str = f"{delta:+.5f}" if method != "Simple Average" else "baseline"
        print(f"{method:<30} {holdout_auc:>12.5f} {oof_auc:>12.5f} {delta_str:>10}")
    print("-" * 70)

    best_method, best_auc, _ = max(results, key=lambda x: x[1])
    gap = best_auc - holdout_auc_avg
    print(f"\nBest method: {best_method} (holdout AUC: {best_auc:.5f})")
    print(f"Gap vs simple average: {gap:+.5f}")
    if gap < 0.001:
        print("VERDICT: Gap < 0.001 — stacking NOT worth the added complexity.")
    else:
        print(f"VERDICT: Gap >= 0.001 — {best_method} is worthwhile.")

    # Ridge weights
    print(f"\n{'=' * 70}")
    print(f"RIDGE WEIGHTS  (alpha={ridge.alpha_:.2f})")
    print(f"{'=' * 70}")
    for name, w in sorted(
        zip(model_names, ridge.coef_), key=lambda x: abs(x[1]), reverse=True
    ):
        print(f"  {name:<30} {w:+.4f}")

    # LightGBM preds-only importances
    print(f"\n{'=' * 70}")
    print("LIGHTGBM (PREDS ONLY) — FEATURE IMPORTANCES")
    print(f"{'=' * 70}")
    for name, imp in sorted(
        zip(model_names, lgb_preds.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"  {name:<30} {imp:>6}")

    # LightGBM preds+features importances
    if has_features and lgb_full is not None:
        feat_cols = (
            list(train_features.columns)  # type: ignore[union-attr]
            if isinstance(train_features, pd.DataFrame)
            else [f"feat_{i}" for i in range(train_feat_np.shape[1])]
        )
        all_feat = model_names + feat_cols
        print(f"\n{'=' * 70}")
        print("LIGHTGBM (PREDS+FEATURES) — TOP 20 FEATURE IMPORTANCES")
        print(f"{'=' * 70}")
        for name, imp in sorted(
            zip(all_feat, lgb_full.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )[:20]:
            marker = " (pred)" if name in model_names else ""
            print(f"  {name:<35}{marker:<8} {imp:>6}")

    print(f"\n{'=' * 70}")

    results.sort(key=lambda x: x[1], reverse=True)
    return results
