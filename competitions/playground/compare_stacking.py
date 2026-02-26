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

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from kego.datasets.split import split_dataset  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = (
    Path(os.environ.get("KEGO_PATH_DATA", project_root / "data"))
    / "playground"
    / "playground-series-s6e2"
)
TARGET = "Heart Disease"


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
# MLflow prediction loading
# ---------------------------------------------------------------------------


def _load_predictions_from_runs(runs_df, tracking_uri):
    """Load and average predictions from a DataFrame of MLflow runs.

    Groups by learner ID (model/feature_set/folds_nf) when params are available,
    falls back to bare model_name for backward compatibility.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    all_oof = {}
    all_holdout = {}
    all_test = {}
    seed_counts = {}

    for _, run in runs_df.iterrows():
        model_name = run.get("params.model")
        if model_name is None:
            continue

        # Build learner ID from params (backward compat: fall back to model_name)
        feature_set = run.get("params.feature_set", "")
        folds_n = run.get("params.folds_n", "")
        if feature_set and folds_n:
            learner_id = f"{model_name}/{feature_set}/{folds_n}f"
        else:
            learner_id = model_name

        artifact_dir = client.download_artifacts(run.run_id, "predictions")
        oof = np.load(os.path.join(artifact_dir, "oof.npy"))
        holdout = np.load(os.path.join(artifact_dir, "holdout.npy"))
        test = np.load(os.path.join(artifact_dir, "test.npy"))

        if learner_id not in all_oof:
            all_oof[learner_id] = np.zeros_like(oof)
            all_holdout[learner_id] = np.zeros_like(holdout)
            all_test[learner_id] = np.zeros_like(test)
            seed_counts[learner_id] = 0

        all_oof[learner_id] += oof
        all_holdout[learner_id] += holdout
        all_test[learner_id] += test
        seed_counts[learner_id] += 1

        seed = run.get("params.seed", "?")
        logger.info(f"  Loaded {learner_id} seed={seed}")

    # Average across seeds
    for name in all_oof:
        n = seed_counts[name]
        all_oof[name] /= n
        all_holdout[name] /= n
        all_test[name] /= n
        logger.info(f"{name}: averaged over {n} seed(s)")

    learner_names = list(all_oof.keys())
    logger.info(f"Total learners loaded: {len(learner_names)}")
    return learner_names, all_oof, all_holdout, all_test


def _load_predictions_from_mlflow(experiment_names, tracking_uri, folds=None):
    """Load per-model averaged predictions from MLflow experiments.

    Args:
        experiment_names: List of experiment names to load from.
        tracking_uri: MLflow tracking URI.
        folds: If set, only load runs where params.folds_n matches this value.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)

    all_runs = []
    for exp_name in experiment_names:
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            logger.warning(f"Experiment '{exp_name}' not found, skipping")
            continue

        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        # Filter out ensemble runs
        runs = runs[~runs["tags.mlflow.runName"].str.startswith("ensemble_", na=True)]

        if folds is not None:
            folds_col = "params.folds_n"
            if folds_col in runs.columns:
                runs = runs[runs[folds_col] == str(folds)]
            else:
                logger.warning(
                    f"Column '{folds_col}' not found in runs, skipping folds filter"
                )

        logger.info(f"Experiment '{exp_name}': {len(runs)} model runs")
        all_runs.append(runs)

    if not all_runs:
        return [], {}, {}, {}

    runs_df = pd.concat(all_runs, ignore_index=True)
    return _load_predictions_from_runs(runs_df, tracking_uri)


# ---------------------------------------------------------------------------
# Stacking methods
# ---------------------------------------------------------------------------


def _compare_stacking(
    model_names,
    all_oof,
    all_holdout,
    train_labels,
    holdout_labels,
    train_features=None,
    holdout_features=None,
):
    """Compare 4 stacking approaches and print results."""
    oof_matrix = np.column_stack([all_oof[n] for n in model_names])
    holdout_matrix = np.column_stack([all_holdout[n] for n in model_names])

    results = []

    # --- 1. Simple Average ---
    avg_oof = np.mean(oof_matrix, axis=1)
    avg_holdout = np.mean(holdout_matrix, axis=1)
    oof_auc_avg = roc_auc_score(train_labels, avg_oof)
    holdout_auc_avg = roc_auc_score(holdout_labels, avg_holdout)
    results.append(("Simple Average", holdout_auc_avg, oof_auc_avg))

    # --- 2. Ridge Regression ---
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge.fit(oof_matrix, train_labels)
    ridge_holdout = ridge.predict(holdout_matrix)
    ridge_oof = cross_val_predict(
        RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]),
        oof_matrix,
        train_labels,
        cv=5,
    )
    oof_auc_ridge = roc_auc_score(train_labels, ridge_oof)
    holdout_auc_ridge = roc_auc_score(holdout_labels, ridge_holdout)
    results.append(("Ridge Regression", holdout_auc_ridge, oof_auc_ridge))

    # --- 3. LightGBM (preds only) ---
    lgb_preds_params = {
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
    lgb_preds = lgb.LGBMClassifier(**lgb_preds_params)
    lgb_preds.fit(oof_matrix, train_labels)
    lgb_preds_holdout = lgb_preds.predict_proba(holdout_matrix)[:, 1]
    lgb_preds_oof = cross_val_predict(
        lgb.LGBMClassifier(**lgb_preds_params),
        oof_matrix,
        train_labels,
        cv=5,
        method="predict_proba",
    )[:, 1]
    oof_auc_lgb_preds = roc_auc_score(train_labels, lgb_preds_oof)
    holdout_auc_lgb_preds = roc_auc_score(holdout_labels, lgb_preds_holdout)
    results.append(("LightGBM (preds only)", holdout_auc_lgb_preds, oof_auc_lgb_preds))

    # --- 4. LightGBM (preds + features) ---
    has_features = train_features is not None and holdout_features is not None
    if has_features:
        train_feat_np = (
            train_features.values
            if isinstance(train_features, pd.DataFrame)
            else train_features
        )
        holdout_feat_np = (
            holdout_features.values
            if isinstance(holdout_features, pd.DataFrame)
            else holdout_features
        )
        oof_plus_feat = np.hstack([oof_matrix, train_feat_np])
        holdout_plus_feat = np.hstack([holdout_matrix, holdout_feat_np])

        lgb_full_params = {
            "objective": "binary",
            "metric": "auc",
            "max_depth": 4,
            "num_leaves": 15,
            "min_child_samples": 200,
            "n_estimators": 500,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.5,
            "reg_lambda": 2.0,
            "random_state": 42,
            "verbosity": -1,
        }
        lgb_full = lgb.LGBMClassifier(**lgb_full_params)
        lgb_full.fit(oof_plus_feat, train_labels)
        lgb_full_holdout = lgb_full.predict_proba(holdout_plus_feat)[:, 1]
        lgb_full_oof = cross_val_predict(
            lgb.LGBMClassifier(**lgb_full_params),
            oof_plus_feat,
            train_labels,
            cv=5,
            method="predict_proba",
        )[:, 1]
        oof_auc_lgb_full = roc_auc_score(train_labels, lgb_full_oof)
        holdout_auc_lgb_full = roc_auc_score(holdout_labels, lgb_full_holdout)
        results.append(
            ("LightGBM (preds+features)", holdout_auc_lgb_full, oof_auc_lgb_full)
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

    # --- Verdict ---
    best_method, best_holdout_auc, _ = max(results, key=lambda x: x[1])
    gap = best_holdout_auc - holdout_auc_avg
    print(f"\nBest method: {best_method} (holdout AUC: {best_holdout_auc:.5f})")
    print(f"Gap vs simple average: {gap:+.5f}")
    if gap < 0.001:
        print("VERDICT: Gap < 0.001 — stacking is NOT worth the added complexity.")
        print("Stick with simple averaging.")
    else:
        print(f"VERDICT: Gap >= 0.001 — stacking with {best_method} is worthwhile.")

    # --- Ridge weights ---
    print(f"\n{'='*70}")
    print("RIDGE REGRESSION WEIGHTS")
    print(f"  alpha = {ridge.alpha_:.2f}")
    print(f"{'='*70}")
    weights = sorted(
        zip(model_names, ridge.coef_), key=lambda x: abs(x[1]), reverse=True
    )
    for name, w in weights:
        print(f"  {name:<30} {w:+.4f}")

    # --- LightGBM feature importances (preds only) ---
    print(f"\n{'='*70}")
    print("LIGHTGBM (PREDS ONLY) — FEATURE IMPORTANCES")
    print(f"{'='*70}")
    importances = lgb_preds.feature_importances_
    imp_sorted = sorted(zip(model_names, importances), key=lambda x: x[1], reverse=True)
    for name, imp in imp_sorted:
        print(f"  {name:<30} {imp:>6}")

    # --- LightGBM feature importances (preds + features) ---
    if has_features:
        print(f"\n{'='*70}")
        print("LIGHTGBM (PREDS+FEATURES) — TOP 20 FEATURE IMPORTANCES")
        print(f"{'='*70}")
        feature_cols = (
            list(train_features.columns)
            if isinstance(train_features, pd.DataFrame)
            else [f"feat_{i}" for i in range(train_feat_np.shape[1])]
        )
        all_feat_names = model_names + feature_cols
        importances_full = lgb_full.feature_importances_
        imp_full_sorted = sorted(
            zip(all_feat_names, importances_full), key=lambda x: x[1], reverse=True
        )
        for name, imp in imp_full_sorted[:20]:
            marker = " (pred)" if name in model_names else ""
            print(f"  {name:<35}{marker:<8} {imp:>6}")

    print(f"\n{'='*70}")


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

    # --- Load predictions from MLflow ---
    logger.info(
        f"Loading predictions from experiments: {args.experiment} "
        f"(folds_n={args.folds})"
    )
    model_names, all_oof, all_holdout, _ = _load_predictions_from_mlflow(
        args.experiment, tracking_uri, folds=args.folds
    )
    if not model_names:
        logger.error("No predictions loaded, exiting")
        sys.exit(1)

    # --- Load and prepare original features ---
    train_full = pd.read_csv(DATA_DIR / "train.csv")
    original = pd.read_csv(DATA_DIR / "Heart_Disease_Prediction.csv")

    train_full[TARGET] = train_full[TARGET].map({"Presence": 1, "Absence": 0})
    original[TARGET] = original[TARGET].map({"Presence": 1, "Absence": 0})

    original["id"] = -1
    train_full = pd.concat([train_full, original], ignore_index=True)

    # Split must match training script exactly (same default seed)
    train, holdout, _ = split_dataset(
        train_full,
        train_size=0.8,
        validate_size=0.2,
        stratify_column=TARGET,
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

    # --- Compare stacking approaches ---
    _compare_stacking(
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
