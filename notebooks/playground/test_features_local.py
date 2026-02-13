"""Local CPU test: compare old vs new feature engineering on LightGBM + LogReg.

Run: uv run python notebooks/playground/test_features_local.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

DATA_DIR = (
    Path(os.environ.get("KEGO_PATH_DATA", project_root / "data"))
    / "playground"
    / "playground-series-s6e2"
)
TARGET = "Heart Disease"

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

TE_FEATURES = ["Thallium", "Chest pain type", "Slope of ST", "EKG results"]

LGB_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 5,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "metric": "auc",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}


# ── Feature engineering variants ────────────────────────────────────────


def engineer_old(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_x_maxhr"] = df["Age"] * df["Max HR"]
    df["age_x_stdep"] = df["Age"] * df["ST depression"]
    df["maxhr_x_stdep"] = df["Max HR"] * df["ST depression"]
    df["hr_reserve"] = 220 - df["Age"] - df["Max HR"]
    df["bp_x_chol"] = df["BP"] * df["Cholesterol"]
    df["age_x_chol"] = df["Age"] * df["Cholesterol"]
    df["bp_x_age"] = df["BP"] * df["Age"]
    df["chol_per_age"] = df["Cholesterol"] / df["Age"]
    df["maxhr_per_age"] = df["Max HR"] / df["Age"]
    df["angina_x_stdep"] = df["Exercise angina"] * df["ST depression"]
    df["vessels_x_thallium"] = df["Number of vessels fluro"] * df["Thallium"]
    df["chestpain_x_angina"] = df["Chest pain type"] * df["Exercise angina"]
    return df


def engineer_new(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Thallium interactions
    df["thallium_x_chestpain"] = df["Thallium"] * df["Chest pain type"]
    df["thallium_x_slope"] = df["Thallium"] * df["Slope of ST"]
    df["thallium_x_sex"] = df["Thallium"] * df["Sex"]
    df["thallium_x_stdep"] = df["Thallium"] * df["ST depression"]
    df["thallium_abnormal"] = (df["Thallium"] >= 6).astype(int)

    # Other strong interactions
    df["chestpain_x_slope"] = df["Chest pain type"] * df["Slope of ST"]
    df["chestpain_x_angina"] = df["Chest pain type"] * df["Exercise angina"]
    df["vessels_x_thallium"] = df["Number of vessels fluro"] * df["Thallium"]
    df["angina_x_stdep"] = df["Exercise angina"] * df["ST depression"]

    # Composite risk scores
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

    # Ratio features
    df["maxhr_per_age"] = df["Max HR"] / df["Age"]
    df["hr_reserve_pct"] = df["Max HR"] / (220 - df["Age"])
    df["age_x_stdep"] = df["Age"] * df["ST depression"]
    df["age_x_maxhr"] = df["Age"] * df["Max HR"]
    df["heart_load"] = df["BP"] * df["Cholesterol"] / df["Max HR"].clip(lower=1)

    # Grouped deviation features
    for col in ["Cholesterol", "BP", "Max HR", "ST depression"]:
        grp_mean = df.groupby("Sex")[col].transform("mean")
        df[f"{col}_dev_sex"] = df[col] - grp_mean

    # Signal conflict: top predictors disagree on risk direction
    df["signal_conflict"] = (
        (df["Thallium"] >= 6) & (df["Chest pain type"] <= 3)
    ).astype(int) + ((df["Thallium"] == 3) & (df["Chest pain type"] == 4)).astype(int)

    return df


def engineer_new_plus_te(df: pd.DataFrame) -> pd.DataFrame:
    """New features + target encoding (applied within CV, see evaluate_with_te)."""
    return engineer_new(df)


def apply_target_encoding(X_tr, y_tr, X_val, te_features, drop_original=False):
    """Apply target encoding from train fold to val fold (no leakage)."""
    X_tr = X_tr.copy()
    X_val = X_val.copy()
    for col in te_features:
        if col not in X_tr.columns:
            continue
        means = y_tr.groupby(X_tr[col]).mean()
        global_mean = y_tr.mean()
        X_tr[f"{col}_te"] = X_tr[col].map(means).fillna(global_mean)
        X_val[f"{col}_te"] = X_val[col].map(means).fillna(global_mean)
    if drop_original:
        cols_to_drop = [c for c in te_features if c in X_tr.columns]
        X_tr = X_tr.drop(columns=cols_to_drop)
        X_val = X_val.drop(columns=cols_to_drop)
    return X_tr, X_val


# ── Evaluation functions ────────────────────────────────────────────────


def evaluate_lgb(train_df, name, te_features=None):
    """5-fold CV with LightGBM. Optionally applies target encoding per fold."""
    features = [c for c in train_df.columns if c not in ["id", TARGET]]
    X = train_df[features]
    y = train_df[TARGET]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    importances_dict = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if te_features:
            X_tr, X_val = apply_target_encoding(X_tr, y_tr, X_val, te_features)

        model = LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            categorical_feature=[c for c in CAT_FEATURES if c in X_tr.columns],
            callbacks=[
                __import__("lightgbm").early_stopping(50),
                __import__("lightgbm").log_evaluation(0),
            ],
        )
        preds = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, preds))

        for feat, imp in zip(X_tr.columns, model.feature_importances_):
            importances_dict[feat] = importances_dict.get(feat, 0) + imp

    for k in importances_dict:
        importances_dict[k] /= 5

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  5-Fold CV AUC: {mean_auc:.5f} (+/- {std_auc:.5f})")
    n_feats = len(importances_dict) if importances_dict else len(features)
    print(f"  Features: {n_feats}")
    print(f"\n  Top 15 features by importance:")
    ranked = sorted(importances_dict.items(), key=lambda x: -x[1])
    for i, (feat, imp) in enumerate(ranked[:15]):
        print(f"    {i+1:2d}. {feat:<30s} {imp:>8.0f}")

    return mean_auc


def evaluate_logreg(train_df, name, te_features=None, drop_original=False):
    """5-fold CV with LogisticRegression. Optionally applies TE per fold."""
    features = [c for c in train_df.columns if c not in ["id", TARGET]]
    X = train_df[features]
    y = train_df[TARGET]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if te_features:
            X_tr, X_val = apply_target_encoding(
                X_tr, y_tr, X_val, te_features, drop_original=drop_original
            )

        model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, C=1.0)
        )
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, preds))

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"  {name:<48s} AUC={mean_auc:.5f} (+/- {std_auc:.5f})")
    return mean_auc


def main():
    train_full = pd.read_csv(DATA_DIR / "train.csv")
    original = pd.read_csv(DATA_DIR / "Heart_Disease_Prediction.csv")

    train_full[TARGET] = train_full[TARGET].map({"Presence": 1, "Absence": 0})
    original[TARGET] = original[TARGET].map({"Presence": 1, "Absence": 0})

    original["id"] = -1
    combined = pd.concat([train_full, original], ignore_index=True)

    print(
        f"Data: {len(combined)} rows ({len(train_full)} synthetic + {len(original)} original)"
    )

    # ── LightGBM ──
    print(f"\n{'='*60}")
    print(f"  LightGBM COMPARISON")
    print(f"{'='*60}")

    lgb_raw = evaluate_lgb(combined, "LGB: Raw features")
    lgb_old = evaluate_lgb(engineer_old(combined), "LGB: Old FE")
    lgb_new = evaluate_lgb(
        engineer_new(combined), "LGB: New FE (thallium + risk + deviation)"
    )
    lgb_new_te = evaluate_lgb(
        engineer_new(combined), "LGB: New FE + Target Encoding", te_features=TE_FEATURES
    )

    print(f"\n  LightGBM Summary:")
    for label, auc in sorted(
        [
            ("Raw", lgb_raw),
            ("Old FE", lgb_old),
            ("New FE", lgb_new),
            ("New FE + TE", lgb_new_te),
        ],
        key=lambda x: -x[1],
    ):
        print(f"    {label:<20s} AUC={auc:.5f}  ({auc - lgb_raw:+.5f})")

    # ── LogReg ──
    print(f"\n{'='*60}")
    print(f"  LogisticRegression COMPARISON")
    print(f"{'='*60}")

    lr_raw = evaluate_logreg(combined, "LR: Raw features")
    lr_old = evaluate_logreg(engineer_old(combined), "LR: Old FE")
    lr_new = evaluate_logreg(engineer_new(combined), "LR: New FE")
    lr_new_te = evaluate_logreg(
        engineer_new(combined), "LR: New FE + TE (keep cat)", te_features=TE_FEATURES
    )
    lr_new_te_drop = evaluate_logreg(
        engineer_new(combined),
        "LR: New FE + TE (drop cat)",
        te_features=TE_FEATURES,
        drop_original=True,
    )

    print(f"\n  LogReg Summary:")
    for label, auc in sorted(
        [
            ("Raw", lr_raw),
            ("Old FE", lr_old),
            ("New FE", lr_new),
            ("TE keep cat", lr_new_te),
            ("TE drop cat", lr_new_te_drop),
        ],
        key=lambda x: -x[1],
    ):
        print(f"    {label:<20s} AUC={auc:.5f}  ({auc - lr_raw:+.5f})")

    # ── Overall ──
    print(f"\n{'='*60}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*60}")
    print(
        f"  {'Model':<12s} {'Raw':>10s} {'Old FE':>10s} {'New FE':>10s} {'TE+keep':>10s} {'TE+drop':>10s}"
    )
    print(
        f"  {'LightGBM':<12s} {lgb_raw:>10.5f} {lgb_old:>10.5f} {lgb_new:>10.5f} {lgb_new_te:>10.5f} {'N/A':>10s}"
    )
    print(
        f"  {'LogReg':<12s} {lr_raw:>10.5f} {lr_old:>10.5f} {lr_new:>10.5f} {lr_new_te:>10.5f} {lr_new_te_drop:>10.5f}"
    )
    print()
    print(f"  Best LGB improvement:   {max(lgb_new, lgb_new_te) - lgb_raw:+.5f}")
    print(
        f"  Best LogReg improvement: {max(lr_new, lr_new_te, lr_new_te_drop) - lr_raw:+.5f}"
    )


if __name__ == "__main__":
    main()
