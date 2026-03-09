"""Compare stacking approaches: simple average vs learned meta-models.

Loads OOF + holdout predictions from MLflow, loads original features,
and compares 4 stacking methods on holdout AUC:
  1. Simple Average
  2. Ridge Regression (learned linear weights)
  3. LightGBM (predictions only)
  4. LightGBM (predictions + original features)

If the gap between the best meta-model and simple average is < 0.001 AUC,
the added complexity isn't worth it.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from kego.datasets.split import split_dataset  # noqa: E402
from kego.ensemble.compare import compare_stacking_methods  # noqa: E402
from kego.tracking import load_predictions_from_mlflow  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = (
    Path(os.environ.get("KEGO_PATH_DATA", project_root / "data"))
    / "playground"
    / "playground-series-s6e2"
)
TARGET = "Heart Disease"


# ---------------------------------------------------------------------------
# S6E2-specific data helpers
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

    df["thallium_x_chestpain"] = df["Thallium"] * df["Chest pain type"]
    df["thallium_x_slope"] = df["Thallium"] * df["Slope of ST"]
    df["thallium_x_sex"] = df["Thallium"] * df["Sex"]
    df["thallium_x_stdep"] = df["Thallium"] * df["ST depression"]
    df["thallium_abnormal"] = (df["Thallium"] >= 6).astype(int)

    df["chestpain_x_slope"] = df["Chest pain type"] * df["Slope of ST"]
    df["chestpain_x_angina"] = df["Chest pain type"] * df["Exercise angina"]
    df["vessels_x_thallium"] = df["Number of vessels fluro"] * df["Thallium"]
    df["angina_x_stdep"] = df["Exercise angina"] * df["ST depression"]

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

    df["maxhr_per_age"] = df["Max HR"] / df["Age"]
    df["hr_reserve_pct"] = df["Max HR"] / (220 - df["Age"])
    df["age_x_stdep"] = df["Age"] * df["ST depression"]
    df["age_x_maxhr"] = df["Age"] * df["Max HR"]
    df["heart_load"] = df["BP"] * df["Cholesterol"] / df["Max HR"].clip(lower=1)

    for col in ["Cholesterol", "BP", "Max HR", "ST depression"]:
        grp_mean = df.groupby("Sex")[col].transform("mean")
        df[f"{col}_dev_sex"] = df[col] - grp_mean

    df["signal_conflict"] = (
        (df["Thallium"] >= 6) & (df["Chest pain type"] <= 3)
    ).astype(int) + ((df["Thallium"] == 3) & (df["Chest pain type"] == 4)).astype(int)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare stacking approaches on holdout AUC"
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=10,
        help="Filter MLflow runs by folds_n (default: 10)",
    )
    parser.add_argument(
        "--experiment",
        nargs="+",
        default=["playground-s6e2-full"],
        help="MLflow experiment name(s) to load from (default: playground-s6e2-full)",
    )
    args = parser.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        logger.error("MLFLOW_TRACKING_URI must be set")
        sys.exit(1)

    logger.info(
        f"Loading predictions from experiments: {args.experiment} "
        f"(folds_n={args.folds})"
    )
    model_names, all_oof, all_holdout, _ = load_predictions_from_mlflow(
        args.experiment, tracking_uri, folds=args.folds
    )
    if not model_names:
        logger.error("No predictions loaded, exiting")
        sys.exit(1)

    # Load and prepare original features
    train_full = pd.read_csv(DATA_DIR / "train.csv")
    original = pd.read_csv(DATA_DIR / "Heart_Disease_Prediction.csv")

    train_full[TARGET] = train_full[TARGET].map({"Presence": 1, "Absence": 0})
    original[TARGET] = original[TARGET].map({"Presence": 1, "Absence": 0})
    original["id"] = -1
    train_full = pd.concat([train_full, original], ignore_index=True)

    train, holdout, _ = split_dataset(
        train_full, train_size=0.8, validate_size=0.2, stratify_column=TARGET
    )
    train = train.reset_index(drop=True)
    holdout = holdout.reset_index(drop=True)

    train = _impute_cholesterol(train)
    holdout = _impute_cholesterol(holdout)
    train = _engineer_features(train)
    holdout = _engineer_features(holdout)

    features = [c for c in train.columns if c not in ["id", TARGET]]
    train_labels = train[TARGET].values
    holdout_labels = holdout[TARGET].values

    logger.info(
        f"Train: {len(train)}, Holdout: {len(holdout)}, Features: {len(features)}"
    )
    logger.info(f"Models: {model_names}")

    compare_stacking_methods(
        model_names,
        all_oof,
        all_holdout,
        train_labels,
        holdout_labels,
        train_features=train[features],
        holdout_features=holdout[features],
    )


if __name__ == "__main__":
    main()
