"""Domain-driven feature research for S6E2 Heart Disease.

Generates ~80 static feature candidates (13 raw + 22 existing + ~45 research) plus
~45 per-fold features from clinical literature, advanced encodings,
and competition techniques. Evaluates via clean-slate forward selection with
LightGBM (CPU only, no Ray).

Usage:
    uv run python notebooks/playground/research_features.py
    uv run python notebooks/playground/research_features.py --train-sample 5000 --holdout-sample 2000
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier  # noqa: F401 — used in Task 3
from sklearn.metrics import roc_auc_score  # noqa: F401 — used in Task 3

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

CONTINUOUS_FEATURES = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]

FEATURES_ABLATION_PRUNED = [
    "Age",
    "Sex",
    "Chest pain type",
    "Cholesterol",
    "FBS over 120",
    "EKG results",
    "Max HR",
    "ST depression",
    "Slope of ST",
    "Number of vessels fluro",
    "thallium_x_slope",
    "chestpain_x_slope",
    "angina_x_stdep",
    "top4_sum",
    "abnormal_count",
    "risk_score",
    "age_x_stdep",
    "Cholesterol_dev_sex",
    "BP_dev_sex",
    "ST depression_dev_sex",
    "signal_conflict",
]

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


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _impute_cholesterol(df: pd.DataFrame) -> pd.DataFrame:
    """Replace Cholesterol=0 (missing) with grouped median by Sex and Age bin."""
    df = df.copy()
    if (df["Cholesterol"] == 0).any():
        df["_age_bin"] = pd.cut(df["Age"], bins=[0, 40, 50, 60, 200])
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


# ---------------------------------------------------------------------------
# Feature engineering: existing features (from select_features.py)
# ---------------------------------------------------------------------------


def _engineer_existing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven interaction and composite features (22 features)."""
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
# Feature engineering: research features (~44 new features)
# ---------------------------------------------------------------------------


def _engineer_research_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate ~44 research feature candidates from clinical literature."""
    df = df.copy()

    # === Clinical Scores (5) ===
    # framingham_partial
    log_age = np.log(df["Age"].clip(lower=20))
    log_chol = np.log(df["Cholesterol"].clip(lower=100))
    log_bp = np.log(df["BP"].clip(lower=80))
    df["framingham_partial"] = np.where(
        df["Sex"] == 1,
        3.06 * log_age + 1.12 * log_chol + 1.93 * log_bp + 0.57 * df["FBS over 120"],
        2.33 * log_age + 1.21 * log_chol + 2.76 * log_bp + 0.69 * df["FBS over 120"],
    )

    # heart_score_partial
    age_pts = np.where(df["Age"] < 45, 0, np.where(df["Age"] < 65, 1, 2))
    ekg_pts = np.where(
        df["EKG results"] == 0, 0, np.where(df["EKG results"] == 1, 1, 2)
    )
    risk_pts = np.minimum(df["FBS over 120"] + (df["BP"] > 140).astype(int), 2)
    df["heart_score_partial"] = age_pts + ekg_pts + risk_pts

    # duke_treadmill_approx
    est_exercise_min = ((df["Max HR"] - 80) / 8).clip(0, 21)
    df["duke_treadmill_approx"] = (
        est_exercise_min - 5 * df["ST depression"] - 4 * df["Exercise angina"] * 2
    )

    # modified_duke
    angina_index = np.where(
        df["Exercise angina"] == 0,
        0,
        np.where(df["Chest pain type"] == 4, 2, 1),
    )
    est_time = ((df["Max HR"] - 60) / 20).clip(0, 21)
    df["modified_duke"] = est_time - 5 * df["ST depression"] - 4 * angina_index

    # timi_partial
    df["timi_partial"] = (
        (df["Age"] >= 65).astype(int)
        + df["FBS over 120"]
        + (df["BP"] > 140).astype(int)
        + (df["ST depression"] > 0).astype(int)
    )

    # === Exercise Physiology (11) ===
    resting_hr = 60 + 0.2 * df["BP"]
    predicted_max = 220 - df["Age"]

    df["chronotropic_incompetence"] = (df["Max HR"] < 0.80 * predicted_max).astype(int)
    denom = (predicted_max - resting_hr).clip(lower=1)
    df["chronotropic_response_index"] = (df["Max HR"] - resting_hr) / denom
    df["hr_reserve_pct_tanaka"] = df["Max HR"] / (208 - 0.7 * df["Age"])
    df["hr_reserve_absolute"] = predicted_max - df["Max HR"]
    df["st_hr_index"] = (df["ST depression"] * 1000) / df["Max HR"].clip(lower=60)
    hr_delta = (df["Max HR"] - resting_hr).clip(lower=1)
    df["st_hr_hysteresis"] = df["ST depression"] / hr_delta
    df["rate_pressure_product"] = df["Max HR"] * df["BP"]
    df["rpp_normalized"] = (df["rate_pressure_product"] - 10000) / 30000
    df["supply_demand_mismatch"] = (
        df["Max HR"]
        * df["BP"]
        / 10000
        * df["ST depression"]
        * (1 + df["Exercise angina"])
    )
    df["estimated_mets"] = 0.05 * df["Max HR"] - 1.0
    df["poor_exercise_capacity"] = (df["estimated_mets"] < 5).astype(int)

    # === Clinical Categories (6) ===
    df["age_risk_category"] = pd.cut(
        df["Age"], bins=[0, 44, 54, 64, 200], labels=[0, 1, 2, 3]
    ).astype(int)
    df["age_sex_risk"] = np.where(
        df["Sex"] == 1, (df["Age"] >= 45).astype(int), (df["Age"] >= 55).astype(int)
    )
    df["bp_category"] = pd.cut(
        df["BP"], bins=[0, 119, 129, 139, 500], labels=[0, 1, 2, 3]
    ).astype(int)
    df["cholesterol_category"] = pd.cut(
        df["Cholesterol"].clip(lower=1), bins=[0, 199, 239, 1000], labels=[0, 1, 2]
    ).astype(int)
    pct = df["Max HR"] / predicted_max
    df["hr_achievement_category"] = pd.cut(
        pct, bins=[0, 0.60, 0.80, 0.85, 5.0], labels=[0, 1, 2, 3]
    ).astype(int)
    df["st_depression_category"] = pd.cut(
        df["ST depression"], bins=[-0.1, 0, 1, 2, 100], labels=[0, 1, 2, 3]
    ).astype(int)

    # === Domain Interactions (10) ===
    df["diabetes_hypertension"] = df["FBS over 120"] * (df["BP"] > 140).astype(int)
    df["multivessel_ischemia"] = (df["Number of vessels fluro"] >= 2).astype(int) * (
        df["ST depression"] + df["Exercise angina"]
    )
    df["anatomic_severity"] = df["Number of vessels fluro"] * (
        df["Thallium"] >= 6
    ).astype(int)
    df["exercise_test_positive"] = (
        (df["ST depression"] >= 1).astype(int)
        + (df["Slope of ST"] >= 2).astype(int)
        + df["Exercise angina"]
    )
    df["age_sex_interaction"] = df["Age"] * df["Sex"]
    df["triple_threat"] = (
        (df["Chest pain type"] == 4).astype(int)
        * (df["Thallium"] >= 6).astype(int)
        * (df["Number of vessels fluro"] >= 1).astype(int)
    )
    df["cholesterol_age_risk"] = df["Cholesterol"] * (df["Age"] > 50).astype(int)
    df["cardiac_efficiency"] = df["Max HR"] / df["BP"].clip(lower=80)
    df["rest_exercise_concordance"] = (df["EKG results"] >= 1).astype(int) * (
        (df["ST depression"] > 0).astype(int) + df["Exercise angina"]
    )
    df["ekg_with_hypertension"] = (df["EKG results"] >= 1).astype(int) * (
        df["BP"] > 140
    ).astype(int)

    # === Composites (4) ===
    slope_weight = np.where(
        df["Slope of ST"] == 2, 2, np.where(df["Slope of ST"] == 1, 1, 0)
    )
    df["ischemic_burden"] = (
        df["ST depression"] * slope_weight
        + 2 * df["Exercise angina"]
        + 3 * (df["Thallium"] >= 6).astype(int)
    )
    df["risk_factor_count"] = (
        df["FBS over 120"]
        + (df["BP"] > 140).astype(int)
        + (df["Cholesterol"] > 240).astype(int)
        + (df["Age"] > 55).astype(int)
        + df["Sex"]
    )
    df["thallium_severity"] = (
        np.where(
            df["Thallium"] == 3,
            0,
            np.where(df["Thallium"] == 6, 2, np.where(df["Thallium"] == 7, 3, 1)),
        )
        + df["Exercise angina"]
        + (df["ST depression"] > 2).astype(int)
    )
    supply_proxy = (
        (df["Thallium"] == 3).astype(int) * (4 - df["Number of vessels fluro"]) / 4
    )
    df["o2_supply_demand"] = (df["Max HR"] * df["BP"] / 10000) * (1 - supply_proxy)

    # === Competition Tricks (~8) ===
    for col in ["Chest pain type", "EKG results", "Slope of ST", "Thallium"]:
        freq = df[col].value_counts(normalize=True)
        df[f"{col}_freq"] = df[col].map(freq)
    df["age_squared"] = df["Age"] ** 2
    df["cholesterol_squared"] = df["Cholesterol"] ** 2
    df["st_depression_squared"] = df["ST depression"] ** 2

    # risk_logodds — only compute if risk_score exists (from _engineer_existing_features)
    if "risk_score" in df.columns:
        risk_prob = (df["risk_score"] + 0.5) / (df["risk_score"].max() + 1)
        df["risk_logodds"] = np.log(risk_prob / (1 - risk_prob))

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Domain-driven feature research for S6E2 Heart Disease"
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

    # --- Load & prepare data (same pipeline as select_features.py) ---
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

    train = _engineer_existing_features(train)
    holdout = _engineer_existing_features(holdout)

    train = _engineer_research_features(train)
    holdout = _engineer_research_features(holdout)

    all_features = [c for c in train.columns if c not in ["id", TARGET]]
    existing_features = [c for c in all_features if c not in RAW_FEATURES]
    y_train = train[TARGET].values
    y_holdout = holdout[TARGET].values

    print(f"Total features: {len(all_features)}")
    print(f"  Raw: {len(RAW_FEATURES)}")
    print(f"  Engineered (existing + research): {len(existing_features)}")
    print(f"  Ablation-pruned baseline: {len(FEATURES_ABLATION_PRUNED)}")
    print(f"Train: {len(train)}, Holdout: {len(holdout)}")
    print(f"Seeds: {seeds}")

    # TODO: Evaluation pipeline (forward selection) added in Task 3.


if __name__ == "__main__":
    main()
