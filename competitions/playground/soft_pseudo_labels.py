"""
Soft pseudo-labeling experiment.

Previous attempt (experiment #4): hard labels (0/1), 136K confident samples,
early weak model → no improvement.

This experiment:
  - Uses continuous probability soft labels (not hard 0/1)
  - Uses ALL 270K test rows (not just high-confidence ones)
  - Tests two sample weight variants: full weight and reduced weight (0.3)
  - Two rounds of self-training to see if iteration helps

Workflow:
  1. Baseline: 5-fold CV on train split → holdout AUC
  2. Round 1: train baseline → get soft labels for test
             retrain on train + soft-labeled test → holdout AUC
  3. Round 2: use round 1 model → new soft labels → retrain again
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

DATA_DIR = Path.home() / "projects/kego/data/playground/playground-series-s6e2"
TARGET = "Heart Disease"

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

ABLATION_PRUNED = [
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
    "abnormal_count",
    "top4_sum",
    "risk_score",
    "angina_x_stdep",
    "chestpain_x_slope",
    "ST depression_dev_sex",
    "signal_conflict",
    "chestpain_x_angina",
]

SEEDS = [42, 123, 777]
N_FOLDS = 5
TEST_WEIGHTS = [1.0, 0.3]  # full weight vs down-weighted test rows


def engineer(df: pd.DataFrame) -> pd.DataFrame:
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
    sex_mean = df.groupby("Sex")["ST depression"].transform("mean")
    df["ST depression_dev_sex"] = df["ST depression"] - sex_mean
    df["signal_conflict"] = (
        (df["Exercise angina"] == 1) & (df["ST depression"] == 0)
    ).astype(int)
    df["chestpain_x_angina"] = df["Chest pain type"] * df["Exercise angina"]
    return df


def impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask = df["Cholesterol"] == 0
    if mask.any():
        df.loc[mask, "Cholesterol"] = df.loc[~mask, "Cholesterol"].median()
    return df


def train_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    X_test: pd.DataFrame,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """
    5-fold CV on X_train/y_train. Returns (holdout_auc, test_predictions).
    sample_weight applies to X_train rows only.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    holdout_preds_list = []
    test_preds = np.zeros(len(X_test))

    # Use hard y for CV splits (soft labels still stratify by integer rounding)
    y_hard = y_train.round().astype(int)

    lgb_params = {
        **{k: v for k, v in LGBM_PARAMS.items() if k not in ("metric", "verbosity")},
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": seed,
    }

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_hard)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr = y_train.iloc[tr_idx]
        y_val_hard = y_hard.iloc[val_idx]
        sw_tr = sample_weight[tr_idx] if sample_weight is not None else None

        dtrain = lgb.Dataset(
            X_tr,
            label=y_tr,
            weight=sw_tr,
            categorical_feature=CAT_FEATURES,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_val,
            label=y_val_hard,
            reference=dtrain,
            free_raw_data=False,
        )
        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=LGBM_PARAMS["n_estimators"],
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
        )
        test_preds += model.predict(X_test) / N_FOLDS
        holdout_preds_list.append(model.predict(X_holdout))

    holdout_pred = np.mean(holdout_preds_list, axis=0)
    holdout_auc = roc_auc_score(y_holdout, holdout_pred)
    return holdout_auc, test_preds


def run_round(
    label: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    X_test: pd.DataFrame,
    soft_labels: np.ndarray | None,
    test_weight: float,
    seeds: list[int],
) -> tuple[float, np.ndarray]:
    """
    Run multi-seed CV. If soft_labels provided, appends test rows with those
    labels and given weight. Returns (mean_holdout_auc, avg_test_preds).
    """
    aucs = []
    all_test_preds = []

    for seed in seeds:
        if soft_labels is not None:
            # Combine train + soft-labeled test
            X_combined = pd.concat(
                [X_train, X_test.reset_index(drop=True)], ignore_index=True
            )
            y_combined = pd.concat(
                [y_train.reset_index(drop=True), pd.Series(soft_labels, name=TARGET)],
                ignore_index=True,
            )
            n_train = len(X_train)
            sw = np.ones(len(X_combined))
            sw[n_train:] = test_weight
        else:
            X_combined = X_train
            y_combined = y_train
            sw = None

        auc, test_preds = train_cv(
            X_combined, y_combined, X_holdout, y_holdout, X_test, seed, sw
        )
        aucs.append(auc)
        all_test_preds.append(test_preds)
        print(f"  {label} | seed={seed} | Holdout AUC: {auc:.5f}", flush=True)

    mean_auc = float(np.mean(aucs))
    mean_test_preds = np.mean(all_test_preds, axis=0)
    print(f"  {label} | MEAN: {mean_auc:.5f}\n", flush=True)
    return mean_auc, mean_test_preds


def main() -> None:
    print("Loading data...", flush=True)
    train_full = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    original = pd.read_csv(DATA_DIR / "Heart_Disease_Prediction.csv")

    train_full[TARGET] = train_full[TARGET].map({"Presence": 1, "Absence": 0})
    original[TARGET] = original[TARGET].map({"Presence": 1, "Absence": 0})
    original["id"] = -1
    train_full = pd.concat([train_full, original], ignore_index=True)

    # Fixed 80/20 split
    from sklearn.model_selection import train_test_split

    train_df, holdout_df = train_test_split(
        train_full, test_size=0.2, stratify=train_full[TARGET], random_state=42
    )
    train_df = train_df.reset_index(drop=True)
    holdout_df = holdout_df.reset_index(drop=True)

    for df in [train_df, holdout_df, test]:
        df = impute(df)  # modifies copy — reapply below
    train_df = engineer(impute(train_df))
    holdout_df = engineer(impute(holdout_df))
    test = engineer(impute(test))

    features = [f for f in ABLATION_PRUNED if f in train_df.columns]
    X_train = train_df[features]
    y_train = train_df[TARGET].astype(float)
    X_holdout = holdout_df[features]
    y_holdout = holdout_df[TARGET]
    X_test = test[features]

    print(
        f"Train: {len(X_train):,} | Holdout: {len(X_holdout):,} | Test: {len(X_test):,}\n",
        flush=True,
    )

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("=== Baseline (no pseudo-labels) ===", flush=True)
    baseline_auc, base_test_preds = run_round(
        "baseline",
        X_train,
        y_train,
        X_holdout,
        y_holdout,
        X_test,
        soft_labels=None,
        test_weight=1.0,
        seeds=SEEDS,
    )

    # ── Round 1 ───────────────────────────────────────────────────────────────
    for tw in TEST_WEIGHTS:
        label = f"round1_w{tw}"
        print(f"=== Round 1: soft pseudo-labels, test_weight={tw} ===", flush=True)
        r1_auc, r1_test_preds = run_round(
            label,
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            X_test,
            soft_labels=base_test_preds,
            test_weight=tw,
            seeds=SEEDS,
        )

        # ── Round 2 ───────────────────────────────────────────────────────────
        label2 = f"round2_w{tw}"
        print(
            f"=== Round 2: iterate with round-1 labels, test_weight={tw} ===",
            flush=True,
        )
        r2_auc, _ = run_round(
            label2,
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            X_test,
            soft_labels=r1_test_preds,
            test_weight=tw,
            seeds=SEEDS,
        )

        print(
            f"  weight={tw} summary: baseline={baseline_auc:.5f} | "
            f"round1={r1_auc:.5f} ({r1_auc-baseline_auc:+.5f}) | "
            f"round2={r2_auc:.5f} ({r2_auc-baseline_auc:+.5f})\n",
            flush=True,
        )

    print("=" * 60, flush=True)
    print("FINAL SUMMARY", flush=True)
    print(f"  Baseline:  {baseline_auc:.5f}", flush=True)


if __name__ == "__main__":
    main()
