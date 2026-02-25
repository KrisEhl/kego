"""
Quantify the impact of adding the original 270 UCI rows to training data.
Runs LightGBM (tuned params) with and without the original rows and compares
holdout AUC, using 5 seeds for stability.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from kego.constants import PATH_DATA

DATA_DIR = PATH_DATA / "playground" / "playground-series-s6e2"
TARGET = "Heart Disease"

ABLATION_PRUNED_FEATURES = [
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
    # Engineered (ablation-pruned set)
    "abnormal_count",
    "top4_sum",
    "risk_score",
    "angina_x_stdep",
    "chestpain_x_slope",
    "ST depression_dev_sex",
    "signal_conflict",
    "chestpain_x_angina",
]

LGBM_PARAMS = {
    "n_estimators": 2000,
    "num_leaves": 16,
    "max_depth": 12,
    "learning_rate": 0.0206,
    "min_child_samples": 25,
    "subsample": 0.563,
    "subsample_freq": 1,
    "colsample_bytree": 0.466,
    "reg_alpha": 0.328,
    "reg_lambda": 1e-7,
    "path_smooth": 73.7,
    "min_split_gain": 0.611,
    "metric": "auc",
    "verbosity": -1,
}

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

SEEDS = [42, 123, 777]
N_FOLDS = 5


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["abnormal_count"] = (
        (df["Thallium"] >= 6).astype(int)
        + (df["Number of vessels fluro"] > 0).astype(int)
        + (df["Exercise angina"] == 1).astype(int)
        + (df["Chest pain type"] == 4).astype(int)
    )
    df["top4_sum"] = (
        df["Thallium"]
        + df["Chest pain type"]
        + df["Number of vessels fluro"]
        + df["Exercise angina"]
    )
    df["risk_score"] = (
        df["Age"] / 10
        + df["Chest pain type"] * 2
        + df["Number of vessels fluro"] * 1.5
        + df["Thallium"] / 2
    )
    df["angina_x_stdep"] = df["Exercise angina"] * df["ST depression"]
    df["chestpain_x_slope"] = df["Chest pain type"] * df["Slope of ST"]
    sex_mean_stdep = df.groupby("Sex")["ST depression"].transform("mean")
    df["ST depression_dev_sex"] = df["ST depression"] - sex_mean_stdep
    df["signal_conflict"] = (
        (df["Exercise angina"] == 1) & (df["ST depression"] == 0)
    ).astype(int)
    df["chestpain_x_angina"] = df["Chest pain type"] * df["Exercise angina"]
    return df


def impute_cholesterol(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask = df["Cholesterol"] == 0
    if mask.any():
        median = df.loc[~mask, "Cholesterol"].median()
        df.loc[mask, "Cholesterol"] = median
    return df


def cross_val_auc(X, y, seed: int) -> tuple[float, float]:
    """Returns (oof_auc, mean_val_auc)."""
    oof_preds = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(**LGBM_PARAMS, random_state=seed)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            categorical_feature=CAT_FEATURES,
            callbacks=[early_stopping(100), log_evaluation(-1)],
        )
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    return roc_auc_score(y, oof_preds)


def run_experiment(df: pd.DataFrame, label: str) -> list[tuple[float, float]]:
    df = impute_cholesterol(df)
    df = engineer_features(df)

    # Use only ablation-pruned features that exist in the dataframe
    features = [f for f in ABLATION_PRUNED_FEATURES if f in df.columns]
    X = df[features]
    y = df[TARGET]

    aucs = []
    for seed in SEEDS:
        auc = cross_val_auc(X, y, seed)
        aucs.append(auc)
        print(f"  {label} | seed={seed} | OOF AUC: {auc:.5f}", flush=True)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"  {label} | MEAN: {mean_auc:.5f} ± {std_auc:.5f}\n", flush=True)
    return aucs


def main():
    print("Loading data...", flush=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    original = pd.read_csv(DATA_DIR / "Heart_Disease_Prediction.csv")

    train[TARGET] = train[TARGET].map({"Presence": 1, "Absence": 0})
    original[TARGET] = original[TARGET].map({"Presence": 1, "Absence": 0})

    print(f"Synthetic train: {len(train):,} rows", flush=True)
    print(f"Original UCI:    {len(original):,} rows\n", flush=True)

    # Experiment A: without original data
    print("=== Experiment A: Synthetic data only ===", flush=True)
    aucs_without = run_experiment(train.copy(), "without_original")

    # Experiment B: with original data appended
    original_aligned = original.copy()
    original_aligned["id"] = -1
    combined = pd.concat([train, original_aligned], ignore_index=True)

    print("=== Experiment B: Synthetic + original (270 rows) ===", flush=True)
    aucs_with = run_experiment(combined, "with_original")

    # Summary
    delta = np.mean(aucs_with) - np.mean(aucs_without)
    print("=" * 55, flush=True)
    print("SUMMARY", flush=True)
    print(
        f"  Without original: {np.mean(aucs_without):.5f} ± {np.std(aucs_without):.5f}",
        flush=True,
    )
    print(
        f"  With original:    {np.mean(aucs_with):.5f} ± {np.std(aucs_with):.5f}",
        flush=True,
    )
    print(f"  Delta:            {delta:+.5f}", flush=True)
    print(f"  (+{abs(delta)*100:.4f}% AUC)", flush=True)


if __name__ == "__main__":
    main()
