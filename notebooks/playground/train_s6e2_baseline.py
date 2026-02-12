import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ray
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from kego.datasets.split import build_xy, split_dataset  # noqa: E402
from kego.train import train_model_split  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = project_root / "data" / "playground" / "playground-series-s6e2"
TARGET = "Heart Disease"

SEEDS = [42, 123, 777]
GPU_MODEL_PREFIXES = {"xgboost", "catboost"}


class ScaledLogisticRegression:
    """LogisticRegression with StandardScaler preprocessing."""

    def __init__(self, **kwargs):
        self.pipe = make_pipeline(StandardScaler(), LogisticRegression(**kwargs))

    def fit(self, X, y, **kwargs):
        self.pipe.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)

    def predict(self, X):
        return self.pipe.predict(X)


def _hill_climbing(
    oof_matrix: np.ndarray,
    labels: np.ndarray,
    model_names: list[str],
    n_iterations: int = 100,
) -> np.ndarray:
    """Find ensemble weights by greedy hill climbing on AUC."""
    n_models = oof_matrix.shape[1]
    best_weights = np.ones(n_models) / n_models
    best_auc = roc_auc_score(labels, oof_matrix @ best_weights)
    step = 0.01

    for _ in range(n_iterations):
        improved = False
        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    continue
                weights = best_weights.copy()
                weights[i] += step
                weights[j] -= step
                if weights[j] < 0:
                    continue
                weights /= weights.sum()
                auc = roc_auc_score(labels, oof_matrix @ weights)
                if auc > best_auc:
                    best_auc = auc
                    best_weights = weights
                    improved = True
        if not improved:
            break

    return best_weights


def get_models(n_features: int) -> dict:
    """Build model configs with GPU acceleration for GBDT models."""
    return {
        # === CPU models ===
        "logistic_regression": {
            "model": ScaledLogisticRegression,
            "kwargs": {"max_iter": 1000, "C": 1.0, "random_state": 42},
            "seed_key": "random_state",
            "use_eval_set": False,
        },
        "random_forest": {
            "model": RandomForestClassifier,
            "kwargs": {
                "n_estimators": 500,
                "max_depth": None,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1,
            },
            "seed_key": "random_state",
            "use_eval_set": False,
        },
        "extra_trees": {
            "model": ExtraTreesClassifier,
            "kwargs": {
                "n_estimators": 500,
                "max_depth": None,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1,
            },
            "seed_key": "random_state",
            "use_eval_set": False,
        },
        # === XGBoost variants (GPU) ===
        "xgboost": {
            "model": XGBClassifier,
            "kwargs": {
                "n_estimators": 2000,
                "max_depth": 7,
                "learning_rate": 0.03,
                "eval_metric": "auc",
                "early_stopping_rounds": 100,
                "tree_method": "hist",
                "device": "cuda",
                "subsample": 0.7,
                "colsample_bytree": 0.6,
                "min_child_weight": 10,
                "reg_alpha": 0.5,
                "reg_lambda": 5.0,
                "random_state": 42,
            },
            "seed_key": "random_state",
            "kwargs_fit": {"verbose": 500},
        },
        "xgboost_reg": {
            "model": XGBClassifier,
            "kwargs": {
                "n_estimators": 1500,
                "max_depth": 4,
                "learning_rate": 0.05,
                "eval_metric": "auc",
                "early_stopping_rounds": 100,
                "tree_method": "hist",
                "device": "cuda",
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "min_child_weight": 20,
                "reg_alpha": 2.0,
                "reg_lambda": 10.0,
                "random_state": 99,
            },
            "seed_key": "random_state",
            "kwargs_fit": {"verbose": 500},
        },
        "xgboost_deep": {
            "model": XGBClassifier,
            "kwargs": {
                "n_estimators": 2000,
                "max_depth": 10,
                "learning_rate": 0.01,
                "eval_metric": "auc",
                "early_stopping_rounds": 100,
                "tree_method": "hist",
                "device": "cuda",
                "subsample": 0.6,
                "colsample_bytree": 0.6,
                "min_child_weight": 5,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42,
            },
            "seed_key": "random_state",
            "kwargs_fit": {"verbose": 500},
        },
        "xgboost_shallow": {
            "model": XGBClassifier,
            "kwargs": {
                "n_estimators": 500,
                "max_depth": 3,
                "learning_rate": 0.1,
                "eval_metric": "auc",
                "early_stopping_rounds": 50,
                "tree_method": "hist",
                "device": "cuda",
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 10,
                "reg_alpha": 0.5,
                "reg_lambda": 5.0,
                "random_state": 42,
            },
            "seed_key": "random_state",
            "kwargs_fit": {"verbose": 500},
        },
        "xgboost_dart": {
            "model": XGBClassifier,
            "kwargs": {
                "booster": "dart",
                "n_estimators": 1000,
                "max_depth": 6,
                "learning_rate": 0.05,
                "eval_metric": "auc",
                "tree_method": "hist",
                "device": "cuda",
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "min_child_weight": 5,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42,
            },
            "seed_key": "random_state",
            "kwargs_fit": {"verbose": 500},
            "use_eval_set": False,
        },
        # === LightGBM variants (GPU) ===
        "lightgbm": {
            "model": LGBMClassifier,
            "kwargs": {
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
            },
            "seed_key": "random_state",
            "kwargs_fit": {
                "callbacks": [
                    __import__("lightgbm").early_stopping(80),
                    __import__("lightgbm").log_evaluation(500),
                ],
            },
        },
        "lightgbm_dart": {
            "model": LGBMClassifier,
            "kwargs": {
                "boosting_type": "dart",
                "n_estimators": 500,
                "max_depth": 5,
                "num_leaves": 31,
                "learning_rate": 0.05,
                "metric": "auc",
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 55,
                "verbosity": -1,
            },
            "seed_key": "random_state",
            "kwargs_fit": {
                "callbacks": [
                    __import__("lightgbm").log_evaluation(500),
                ],
            },
        },
        "lightgbm_large": {
            "model": LGBMClassifier,
            "kwargs": {
                "n_estimators": 2000,
                "max_depth": 8,
                "num_leaves": 63,
                "learning_rate": 0.02,
                "metric": "auc",
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "min_child_weight": 5,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42,
                "verbosity": -1,
            },
            "seed_key": "random_state",
            "kwargs_fit": {
                "callbacks": [
                    __import__("lightgbm").early_stopping(100),
                    __import__("lightgbm").log_evaluation(500),
                ],
            },
        },
        "lightgbm_small": {
            "model": LGBMClassifier,
            "kwargs": {
                "n_estimators": 500,
                "max_depth": 3,
                "num_leaves": 7,
                "learning_rate": 0.1,
                "metric": "auc",
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "min_child_weight": 10,
                "reg_alpha": 0.5,
                "reg_lambda": 5.0,
                "random_state": 42,
                "verbosity": -1,
            },
            "seed_key": "random_state",
            "kwargs_fit": {
                "callbacks": [
                    __import__("lightgbm").early_stopping(50),
                    __import__("lightgbm").log_evaluation(500),
                ],
            },
        },
        # === CatBoost variants (GPU) ===
        "catboost": {
            "model": CatBoostClassifier,
            "kwargs": {
                "iterations": 2000,
                "depth": 6,
                "learning_rate": 0.05,
                "eval_metric": "AUC",
                "early_stopping_rounds": 100,
                "task_type": "GPU",
                "gpu_ram_part": 0.5,
                "subsample": 0.8,
                "bootstrap_type": "Bernoulli",
                "l2_leaf_reg": 3.0,
                "random_strength": 1.5,
                "random_seed": 77,
                "verbose": 500,
            },
            "seed_key": "random_seed",
        },
        "catboost_deep": {
            "model": CatBoostClassifier,
            "kwargs": {
                "iterations": 2000,
                "depth": 8,
                "learning_rate": 0.03,
                "eval_metric": "AUC",
                "early_stopping_rounds": 100,
                "task_type": "GPU",
                "gpu_ram_part": 0.5,
                "subsample": 0.7,
                "bootstrap_type": "Bernoulli",
                "l2_leaf_reg": 5.0,
                "random_strength": 2.0,
                "random_seed": 42,
                "verbose": 500,
            },
            "seed_key": "random_seed",
        },
        "catboost_shallow": {
            "model": CatBoostClassifier,
            "kwargs": {
                "iterations": 1000,
                "depth": 4,
                "learning_rate": 0.1,
                "eval_metric": "AUC",
                "early_stopping_rounds": 50,
                "task_type": "GPU",
                "gpu_ram_part": 0.5,
                "subsample": 0.9,
                "bootstrap_type": "Bernoulli",
                "l2_leaf_reg": 1.0,
                "random_strength": 0.5,
                "random_seed": 42,
                "verbose": 500,
            },
            "seed_key": "random_seed",
        },
    }


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction and ratio features."""
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


@ray.remote
def _train_single_model(
    train, test, holdout, features, target, model_name, model_config, seed, folds_n=10
):
    """Train one model with one seed on a Ray worker."""
    print(f"[{model_name}] Starting seed={seed}", flush=True)

    kwargs = model_config["kwargs"].copy()
    seed_key = model_config.get("seed_key", "random_state")
    kwargs[seed_key] = seed

    _, oof, holdout_pred, test_pred = train_model_split(
        model=model_config["model"],
        train=train,
        test=test,
        holdout=holdout,
        features=features,
        target=target,
        kwargs_model=kwargs,
        kwargs_fit=model_config.get("kwargs_fit", {}),
        folds_n=folds_n,
        use_probability=True,
        use_eval_set=model_config.get("use_eval_set", True),
        kfold_seed=seed,
    )
    print(f"[{model_name}] Finished seed={seed}", flush=True)
    return model_name, seed, oof, holdout_pred, test_pred


def _train_ensemble(train, holdout, test, features, models, tag="", folds_n=10):
    """Train all models with multiple seeds via Ray and return ensemble predictions."""
    # Share data via Ray object store (stored once, shared across all tasks)
    train_ref = ray.put(train)
    test_ref = ray.put(test)
    holdout_ref = ray.put(holdout)

    train_labels = train[TARGET].values
    holdout_labels = holdout[TARGET].values

    # Launch all (model, seed) tasks in parallel
    futures = []
    task_info = []
    for model_name, config in models.items():
        is_gpu = any(model_name.startswith(p) for p in GPU_MODEL_PREFIXES)
        for seed in SEEDS:
            if model_name.startswith("catboost"):
                opts = {"num_gpus": 1, "num_cpus": 1}
            elif is_gpu:
                opts = {"num_gpus": 0.25, "num_cpus": 1}
            else:
                opts = {"num_cpus": 2}
            future = _train_single_model.options(**opts).remote(
                train_ref,
                test_ref,
                holdout_ref,
                features,
                TARGET,
                model_name,
                config,
                seed,
                folds_n,
            )
            futures.append(future)
            task_info.append(f"{model_name} seed={seed} ({'GPU' if is_gpu else 'CPU'})")

    logger.info(f"Launched {len(futures)} Ray tasks:")
    for info in task_info:
        logger.info(f"  - {info}")

    # Collect results incrementally as they complete
    all_oof_preds = {}
    all_holdout_preds = {}
    all_test_preds = {}
    remaining = list(futures)
    completed = 0

    while remaining:
        done, remaining = ray.wait(remaining, num_returns=1, timeout=600)
        if not done:
            logger.warning(
                f"ray.wait timed out after 600s. "
                f"{completed}/{len(futures)} completed, "
                f"{len(remaining)} remaining."
            )
            continue
        try:
            model_name, seed, oof, holdout_pred, test_pred = ray.get(done[0])
        except Exception as e:
            completed += 1
            logger.error(f"[{completed}/{len(futures)}] Task failed: {e}")
            continue
        completed += 1

        if model_name not in all_oof_preds:
            all_oof_preds[model_name] = np.zeros(len(train))
            all_holdout_preds[model_name] = np.zeros(len(holdout))
            all_test_preds[model_name] = np.zeros(len(test))

        all_oof_preds[model_name] += oof
        all_holdout_preds[model_name] += holdout_pred
        all_test_preds[model_name] += test_pred

        auc = roc_auc_score(holdout_labels, holdout_pred)
        logger.info(
            f"[{completed}/{len(futures)}] {model_name} seed={seed} "
            f"— Holdout AUC: {auc:.4f}"
        )

    # Average across seeds and log per-model results
    for name in all_oof_preds:
        all_oof_preds[name] /= len(SEEDS)
        all_holdout_preds[name] /= len(SEEDS)
        all_test_preds[name] /= len(SEEDS)

        auc = roc_auc_score(holdout_labels, all_holdout_preds[name])
        acc = accuracy_score(
            holdout_labels, (all_holdout_preds[name] >= 0.5).astype(int)
        )
        logger.info(
            f"{name} (avg {len(SEEDS)} seeds) — "
            f"Holdout AUC: {auc:.4f}, Accuracy: {acc:.4f}"
        )

    # Build stacking matrices
    model_names = list(models.keys())
    oof_matrix = np.column_stack([all_oof_preds[n] for n in model_names])
    holdout_matrix = np.column_stack([all_holdout_preds[n] for n in model_names])
    test_matrix = np.column_stack([all_test_preds[n] for n in model_names])

    # --- Simple average ---
    avg_holdout = np.mean(holdout_matrix, axis=1)
    avg_test = np.mean(test_matrix, axis=1)
    auc = roc_auc_score(holdout_labels, avg_holdout)
    logger.info(f"\n{'='*50}")
    logger.info(f"{tag}Simple Average — Holdout AUC: {auc:.4f}")

    # --- Ridge stacking ---
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge.fit(oof_matrix, train_labels)
    ridge_holdout = ridge.predict(holdout_matrix)
    ridge_test = ridge.predict(test_matrix)
    auc = roc_auc_score(holdout_labels, ridge_holdout)
    logger.info(f"{tag}Ridge (alpha={ridge.alpha_:.2f}) — Holdout AUC: {auc:.4f}")
    logger.info(f"  Weights: {dict(zip(model_names, ridge.coef_))}")

    # --- Hill Climbing ---
    best_weights = _hill_climbing(oof_matrix, train_labels, model_names)
    hc_holdout = holdout_matrix @ best_weights
    hc_test = test_matrix @ best_weights
    auc = roc_auc_score(holdout_labels, hc_holdout)
    logger.info(f"{tag}Hill Climbing — Holdout AUC: {auc:.4f}")
    logger.info(f"  Weights: {dict(zip(model_names, best_weights))}")
    logger.info(f"{'='*50}")

    # Pick best ensemble method
    results = {
        "average": (avg_holdout, avg_test),
        "ridge": (ridge_holdout, ridge_test),
        "hill_climbing": (hc_holdout, hc_test),
    }
    best_name = max(results, key=lambda k: roc_auc_score(holdout_labels, results[k][0]))
    best_holdout, best_test = results[best_name]
    auc = roc_auc_score(holdout_labels, best_holdout)
    logger.info(f"{tag}Best method: {best_name} (AUC: {auc:.4f})")

    return best_test, best_name, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true", help="Quick run with small sample"
    )
    args = parser.parse_args()

    ray.init()

    # Load data
    train_full = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
    original = pd.read_csv(DATA_DIR / "Heart_Disease_Prediction.csv")

    # Map target to 0/1
    train_full[TARGET] = train_full[TARGET].map({"Presence": 1, "Absence": 0})
    original[TARGET] = original[TARGET].map({"Presence": 1, "Absence": 0})

    # Combine synthetic + original data
    original["id"] = -1  # placeholder, excluded from features
    train_full = pd.concat([train_full, original], ignore_index=True)

    if args.debug:
        train_full = train_full.sample(n=2000, random_state=42).reset_index(drop=True)
        test = test.head(500).reset_index(drop=True)
        sample_submission = sample_submission.head(500).reset_index(drop=True)
        logger.info(f"DEBUG MODE: train={len(train_full)}, test={len(test)}")

    logger.info(
        f"Combined train: {len(train_full)} rows " f"(+{len(original)} original)"
    )

    # Split into train (80%) and holdout (20%)
    train, holdout, _ = split_dataset(
        train_full,
        train_size=0.8,
        validate_size=0.2,
        stratify_column=TARGET,
    )
    train = train.reset_index(drop=True)
    holdout = holdout.reset_index(drop=True)

    # Engineer features
    train = _engineer_features(train)
    holdout = _engineer_features(holdout)
    test = _engineer_features(test)

    # Features = all columns except id and target
    features = [c for c in train.columns if c not in ["id", TARGET]]
    n_features = len(features)
    models = get_models(n_features)

    # Train ensemble with multi-seed averaging via Ray
    folds_n = 2 if args.debug else 10
    best_test, best_method, best_auc = _train_ensemble(
        train, holdout, test, features, models, folds_n=folds_n
    )

    # Generate submission
    submission = sample_submission.copy()
    submission[TARGET] = best_test
    output_path = Path(__file__).parent / "submission.csv"
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")
    logger.info(f"Mean prediction: {np.mean(best_test):.3f}")

    ray.shutdown()


if __name__ == "__main__":
    main()
