"""Feature selection via permutation importance and fine-grained ablation.

Runs entirely on CPU (no Ray/torch). Tests whether engineered features help
or hurt, with separate recommendations for tree models vs neural networks.

Steps:
  1. Permutation importance on holdout (N repeats)
  2. Drop-one-at-a-time ablation — train on all-but-one, measure AUC delta
  3. Forward selection — add features one at a time in importance order
  4. Compare feature sets (all / raw / forward-selected / ablation-pruned)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from kego.datasets.split import split_dataset  # noqa: E402

DATA_DIR = (
    Path(os.environ.get("KEGO_PATH_DATA", project_root / "data"))
    / "playground"
    / "playground-series-s6e2"
)
TARGET = "Heart Disease"

RAW_FEATURES = [
    "Age",
    "Sex",
    "Chest pain type",
    "BP",
    "Cholesterol",
    "FBS over 120",
    "EKG results",
    "Max HR",
    "Exercise angina",
    "ST depression",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
]

CAT_FEATURES = [
    "Sex",
    "Chest pain type",
    "FBS over 120",
    "EKG results",
    "Exercise angina",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
]


# ---------------------------------------------------------------------------
# Data helpers (copied from train_s6e2_baseline.py — can't import directly
# because that module imports torch/ray/catboost at module level)
# ---------------------------------------------------------------------------


def _impute_cholesterol(df: pd.DataFrame) -> pd.DataFrame:
    """Replace Cholesterol=0 (missing) with grouped median by Sex and Age bin."""
    df = df.copy()
    if (df["Cholesterol"] == 0).any():
        df["_age_bin"] = pd.cut(df["Age"], bins=[0, 40, 50, 60, 100])
        median_map = (
            df[df["Cholesterol"] > 0]
            .groupby(["Sex", "_age_bin"])["Cholesterol"]
            .median()
        )
        mask = df["Cholesterol"] == 0
        for idx in df[mask].index:
            key = (df.loc[idx, "Sex"], df.loc[idx, "_age_bin"])
            if key in median_map.index:
                df.loc[idx, "Cholesterol"] = median_map[key]
            else:
                df.loc[idx, "Cholesterol"] = df.loc[~mask, "Cholesterol"].median()
        df = df.drop(columns=["_age_bin"])
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven interaction and composite features."""
    df = df.copy()

    # --- Thallium interactions (Thallium is the #1 predictor) ---
    df["thallium_x_chestpain"] = df["Thallium"] * df["Chest pain type"]
    df["thallium_x_slope"] = df["Thallium"] * df["Slope of ST"]
    df["thallium_x_sex"] = df["Thallium"] * df["Sex"]
    df["thallium_x_stdep"] = df["Thallium"] * df["ST depression"]
    df["thallium_abnormal"] = (df["Thallium"] >= 6).astype(int)

    # --- Other strong interactions ---
    df["chestpain_x_slope"] = df["Chest pain type"] * df["Slope of ST"]
    df["chestpain_x_angina"] = df["Chest pain type"] * df["Exercise angina"]
    df["vessels_x_thallium"] = df["Number of vessels fluro"] * df["Thallium"]
    df["angina_x_stdep"] = df["Exercise angina"] * df["ST depression"]

    # --- Composite risk scores ---
    df["top4_sum"] = (
        df["Thallium"]
        + df["Chest pain type"]
        + df["Number of vessels fluro"]
        + df["Exercise angina"]
    )
    df["abnormal_count"] = (
        (df["Thallium"] >= 6).astype(int)
        + (df["Number of vessels fluro"] >= 1).astype(int)
        + (df["Chest pain type"] >= 3).astype(int)
        + (df["Exercise angina"] == 1).astype(int)
        + (df["Slope of ST"] >= 2).astype(int)
        + (df["ST depression"] > 1).astype(int)
        + (df["Sex"] == 1).astype(int)
    )
    df["risk_score"] = (
        3 * (df["Thallium"] >= 6).astype(int)
        + 2 * (df["Number of vessels fluro"] >= 1).astype(int)
        + 2 * (df["Chest pain type"] >= 3).astype(int)
        + 2 * (df["Exercise angina"] == 1).astype(int)
        + (df["Slope of ST"] >= 2).astype(int)
        + (df["ST depression"] > 1).astype(int)
    )

    # --- Ratio features ---
    df["maxhr_per_age"] = df["Max HR"] / df["Age"]
    df["hr_reserve_pct"] = df["Max HR"] / (220 - df["Age"])
    df["age_x_stdep"] = df["Age"] * df["ST depression"]
    df["age_x_maxhr"] = df["Age"] * df["Max HR"]
    df["heart_load"] = df["BP"] * df["Cholesterol"] / df["Max HR"].clip(lower=1)

    # --- Grouped deviation features (individual risk vs demographic peers) ---
    for col in ["Cholesterol", "BP", "Max HR", "ST depression"]:
        grp_mean = df.groupby("Sex")[col].transform("mean")
        df[f"{col}_dev_sex"] = df[col] - grp_mean

    # --- Signal conflict: top predictors disagree on risk direction ---
    df["signal_conflict"] = (
        (df["Thallium"] >= 6) & (df["Chest pain type"] <= 3)
    ).astype(int) + ((df["Thallium"] == 3) & (df["Chest pain type"] == 4)).astype(int)

    return df


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

LGBM_PARAMS = {
    "n_estimators": 1500,
    "max_depth": 4,
    "num_leaves": 15,
    "learning_rate": 0.08,
    "metric": "auc",
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 3,
    "reg_alpha": 0.01,
    "reg_lambda": 0.1,
    "random_state": 123,
    "verbosity": -1,
}


def _train_lgbm(X_train, y_train, X_holdout, y_holdout, features, cat_feats=None):
    """Train LightGBM and return holdout AUC."""
    import lightgbm as lgb

    model = LGBMClassifier(**LGBM_PARAMS)
    fit_cat = [c for c in (cat_feats or CAT_FEATURES) if c in features]
    model.fit(
        X_train[features],
        y_train,
        eval_set=[(X_holdout[features], y_holdout)],
        categorical_feature=fit_cat,
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)],
    )
    preds = model.predict_proba(X_holdout[features])[:, 1]
    auc = roc_auc_score(y_holdout, preds)
    return model, auc


def _eval_lgbm_multiseed(X_train, y_train, X_holdout, y_holdout, features, seeds):
    """Train LightGBM once per seed and return mean holdout AUC."""
    import lightgbm as lgb

    fit_cat = [c for c in CAT_FEATURES if c in features]
    aucs = []
    for seed in seeds:
        params = {**LGBM_PARAMS, "random_state": seed}
        model = LGBMClassifier(**params)
        model.fit(
            X_train[features],
            y_train,
            eval_set=[(X_holdout[features], y_holdout)],
            categorical_feature=fit_cat,
            callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)],
        )
        preds = model.predict_proba(X_holdout[features])[:, 1]
        aucs.append(roc_auc_score(y_holdout, preds))
    return float(np.mean(aucs))


def _train_logreg(X_train, y_train, X_holdout, y_holdout, features):
    """Train scaled LogisticRegression and return holdout AUC."""
    pipe = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    )
    pipe.fit(X_train[features].values, y_train)
    preds = pipe.predict_proba(X_holdout[features].values)[:, 1]
    auc = roc_auc_score(y_holdout, preds)
    return pipe, auc


def _eval_logreg_multiseed(X_train, y_train, X_holdout, y_holdout, features, seeds):
    """Train LogisticRegression once per seed and return mean holdout AUC."""
    aucs = []
    for seed in seeds:
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0, random_state=seed),
        )
        pipe.fit(X_train[features].values, y_train)
        preds = pipe.predict_proba(X_holdout[features].values)[:, 1]
        aucs.append(roc_auc_score(y_holdout, preds))
    return float(np.mean(aucs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Feature selection via permutation importance"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of permutation importance repeats (default: 10)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,777",
        help="Comma-separated seeds for multi-seed averaging (default: 42,123,777)",
    )
    parser.add_argument(
        "--train-sample",
        type=int,
        default=50000,
        help="Subsample training set to N rows (default: 50000, 0=no sampling)",
    )
    parser.add_argument(
        "--holdout-sample",
        type=int,
        default=20000,
        help="Subsample holdout set to N rows (default: 20000, 0=no sampling)",
    )
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    # --- Load & prepare data (same pipeline as training script) ---
    # Sample early to avoid OOM — full 630K dataset not needed for feature selection.
    total_sample = (args.train_sample or 0) + (args.holdout_sample or 0)

    train_full = pd.read_csv(DATA_DIR / "train.csv")
    original = pd.read_csv(DATA_DIR / "Heart_Disease_Prediction.csv")

    train_full[TARGET] = train_full[TARGET].map({"Presence": 1, "Absence": 0})
    original[TARGET] = original[TARGET].map({"Presence": 1, "Absence": 0})

    original["id"] = -1
    train_full = pd.concat([train_full, original], ignore_index=True)

    # Downsample before split to keep memory low
    if total_sample and len(train_full) > total_sample:
        train_full = train_full.sample(n=total_sample, random_state=42).reset_index(
            drop=True
        )

    train, holdout, _ = split_dataset(
        train_full,
        train_size=0.8,
        validate_size=0.2,
        stratify_column=TARGET,
    )
    del train_full  # free memory
    train = train.reset_index(drop=True)
    holdout = holdout.reset_index(drop=True)

    train = _impute_cholesterol(train)
    holdout = _impute_cholesterol(holdout)

    train = _engineer_features(train)
    holdout = _engineer_features(holdout)

    all_features = [c for c in train.columns if c not in ["id", TARGET]]
    engineered_features = [c for c in all_features if c not in RAW_FEATURES]
    y_train = train[TARGET].values
    y_holdout = holdout[TARGET].values

    print(f"Total features: {len(all_features)}")
    print(f"  Raw: {len(RAW_FEATURES)}")
    print(f"  Engineered: {len(engineered_features)}")
    print(f"Train: {len(train)}, Holdout: {len(holdout)}")
    print(f"Seeds: {seeds}")

    # ===================================================================
    # Step 1: Permutation importance baseline
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"STEP 1: PERMUTATION IMPORTANCE ({args.repeats} repeats)")
    print(f"{'='*70}")

    model_baseline, auc_baseline = _train_lgbm(
        train, y_train, holdout, y_holdout, all_features
    )
    print(f"\nBaseline LightGBM holdout AUC: {auc_baseline:.5f}")

    result = permutation_importance(
        model_baseline,
        holdout[all_features],
        y_holdout,
        n_repeats=args.repeats,
        scoring="roc_auc",
        random_state=42,
        n_jobs=1,
    )

    # Build ranked table
    imp_df = pd.DataFrame(
        {
            "feature": all_features,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "is_engineered": [f not in RAW_FEATURES for f in all_features],
        }
    ).sort_values("importance_mean", ascending=False)
    imp_df["significant"] = imp_df["importance_mean"] > 2 * imp_df["importance_std"]

    print(f"\n{'Feature':<30} {'Mean':>10} {'Std':>10} {'Sig':>5} {'Type':>6}")
    print("-" * 65)
    for _, row in imp_df.iterrows():
        sig = "***" if row["significant"] else ""
        ftype = "ENG" if row["is_engineered"] else "RAW"
        sign = "-" if row["importance_mean"] < 0 else ""
        print(
            f"{row['feature']:<30} {sign}{abs(row['importance_mean']):>9.5f} "
            f"{row['importance_std']:>10.5f} {sig:>5} {ftype:>6}"
        )

    negative_features = imp_df[imp_df["importance_mean"] < 0]["feature"].tolist()
    zero_features = imp_df[(imp_df["importance_mean"] >= 0) & (~imp_df["significant"])][
        "feature"
    ].tolist()
    print(f"\nNegative importance ({len(negative_features)}): {negative_features}")
    print(f"Non-significant ({len(zero_features)}): {zero_features}")

    # Features ranked by descending importance (used in step 3)
    features_by_importance = imp_df["feature"].tolist()

    # ===================================================================
    # Step 2: Drop-one-at-a-time ablation (multi-seed)
    # ===================================================================
    print(f"\n{'='*70}")
    print(
        f"STEP 2: DROP-ONE-AT-A-TIME ABLATION ({len(all_features)} features x {len(seeds)} seeds)"
    )
    print(f"{'='*70}")

    baseline_ms = _eval_lgbm_multiseed(
        train, y_train, holdout, y_holdout, all_features, seeds
    )
    print(f"\nMulti-seed baseline AUC: {baseline_ms:.5f}")

    ablation_results = []
    for i, feat in enumerate(all_features):
        reduced = [f for f in all_features if f != feat]
        auc_without = _eval_lgbm_multiseed(
            train, y_train, holdout, y_holdout, reduced, seeds
        )
        delta = auc_without - baseline_ms
        ablation_results.append((feat, auc_without, delta))
        print(
            f"  [{i+1}/{len(all_features)}] -{feat:<30} AUC={auc_without:.5f} (delta={delta:+.5f})"
        )

    # Sort by delta descending (features whose removal helps most at top)
    ablation_results.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'Feature':<30} {'AUC without':>12} {'Delta':>10} {'Verdict':>10}")
    print("-" * 66)
    for feat, auc_without, delta in ablation_results:
        verdict = "HARMFUL" if delta > 0 else "helpful"
        print(f"{feat:<30} {auc_without:>12.5f} {delta:>+10.5f} {verdict:>10}")

    harmful_features = [f for f, _, d in ablation_results if d > 0]
    print(f"\nHarmful features (removal improves AUC): {harmful_features}")

    ablation_pruned = [f for f in all_features if f not in harmful_features]

    # ===================================================================
    # Step 3: Forward selection by importance order (multi-seed)
    # ===================================================================
    print(f"\n{'='*70}")
    print(
        f"STEP 3: FORWARD SELECTION ({len(all_features)} features x {len(seeds)} seeds)"
    )
    print(f"{'='*70}")

    forward_history = []
    for i, _ in enumerate(features_by_importance, start=1):
        subset = features_by_importance[:i]
        auc_fwd = _eval_lgbm_multiseed(
            train, y_train, holdout, y_holdout, subset, seeds
        )
        forward_history.append((i, subset[-1], auc_fwd))
        print(
            f"  [{i}/{len(features_by_importance)}] +{subset[-1]:<30} AUC={auc_fwd:.5f}"
        )

    print(f"\n{'N':>3} {'Added feature':<30} {'AUC':>10} {'Delta':>10}")
    print("-" * 57)
    prev_auc = 0.0
    for n, feat, auc in forward_history:
        delta = auc - prev_auc if n > 1 else 0.0
        print(f"{n:>3} {feat:<30} {auc:>10.5f} {delta:>+10.5f}")
        prev_auc = auc

    # Find optimal feature count (highest AUC)
    best_n, best_feat, best_fwd_auc = max(forward_history, key=lambda x: x[2])
    forward_selected = features_by_importance[:best_n]
    print(f"\nOptimal: {best_n} features, AUC={best_fwd_auc:.5f}")
    print(f"Features: {forward_selected}")

    # ===================================================================
    # Step 4: Feature set comparison (4 sets x 2 model types, multi-seed)
    # ===================================================================
    print(f"\n{'='*70}")
    print("STEP 4: FEATURE SET COMPARISON (LightGBM + LogReg, multi-seed)")
    print(f"{'='*70}")

    feature_sets = {
        f"All features ({len(all_features)})": all_features,
        f"Raw only ({len(RAW_FEATURES)})": RAW_FEATURES,
        f"Forward-selected ({len(forward_selected)})": forward_selected,
        f"Ablation-pruned ({len(ablation_pruned)})": ablation_pruned,
    }

    print(f"\n{'Feature set':<35} {'LightGBM AUC':>14} {'LogReg AUC':>14}")
    print("-" * 67)

    for name, features in feature_sets.items():
        auc_lgbm = _eval_lgbm_multiseed(
            train, y_train, holdout, y_holdout, features, seeds
        )
        logreg_features = [f for f in features if f not in CAT_FEATURES]
        auc_logreg = _eval_logreg_multiseed(
            train, y_train, holdout, y_holdout, logreg_features, seeds
        )
        print(f"{name:<35} {auc_lgbm:>14.5f} {auc_logreg:>14.5f}")

    # ===================================================================
    # Summary & recommendations
    # ===================================================================
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")

    print(f"\nForward-selected feature set ({len(forward_selected)} features):")
    for f in forward_selected:
        marker = " (engineered)" if f not in RAW_FEATURES else ""
        print(f"  - {f}{marker}")

    print(f"\nAblation-pruned feature set ({len(ablation_pruned)} features):")
    for f in ablation_pruned:
        marker = " (engineered)" if f not in RAW_FEATURES else ""
        print(f"  - {f}{marker}")

    if harmful_features:
        print(f"\nFeatures to DROP (removal improved AUC):")
        for f in harmful_features:
            marker = " (engineered)" if f not in RAW_FEATURES else ""
            print(f"  - {f}{marker}")

    print(f"\nFor TREES (LightGBM/XGBoost/CatBoost):")
    print(
        f"  Use forward-selected ({len(forward_selected)}) or ablation-pruned ({len(ablation_pruned)}),"
    )
    print(f"  whichever scored higher in step 4.")

    print(f"\nFor NNs (ResNet/FTTransformer/RealMLP):")
    print(
        f"  Compare forward-selected vs raw-only — NNs are more sensitive to noisy features."
    )
    print(f"  LogReg AUC above serves as a proxy for NN feature sensitivity.")


if __name__ == "__main__":
    main()
