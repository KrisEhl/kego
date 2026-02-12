import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from pytabkit import RealMLP_TD_Classifier
from rtdl_revisiting_models import FTTransformer, ResNet
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetBinaryClassifier
from xgboost import XGBClassifier

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from kego.datasets.split import build_xy, split_dataset
from kego.train import train_model_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = project_root / "data" / "playground" / "playground-series-s6e2"
TARGET = "Heart Disease"


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


class FTTransformerWrapper(nn.Module):
    """Wraps FTTransformer to accept a single tensor (all continuous features)."""

    def __init__(
        self,
        n_cont_features,
        d_out=1,
        n_blocks=3,
        d_block=192,
        attention_n_heads=8,
        attention_dropout=0.2,
        ffn_d_hidden_multiplier=4 / 3,
        ffn_dropout=0.1,
        residual_dropout=0.0,
    ):
        super().__init__()
        self.ft = FTTransformer(
            n_cont_features=n_cont_features,
            cat_cardinalities=[],
            d_out=d_out,
            n_blocks=n_blocks,
            d_block=d_block,
            attention_n_heads=attention_n_heads,
            attention_dropout=attention_dropout,
            ffn_d_hidden_multiplier=ffn_d_hidden_multiplier,
            ffn_dropout=ffn_dropout,
            residual_dropout=residual_dropout,
        )

    def forward(self, X):
        return self.ft(X, None)


class SkorchBinaryClassifier:
    """Sklearn-compatible wrapper around skorch with float32 conversion."""

    def __init__(self, **kwargs):
        self.net = NeuralNetBinaryClassifier(**kwargs)

    def fit(self, X, y, **kwargs):
        self.net.fit(
            np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
            **kwargs,
        )
        return self

    def predict_proba(self, X):
        return self.net.predict_proba(np.asarray(X, dtype=np.float32))

    def predict(self, X):
        return self.net.predict(np.asarray(X, dtype=np.float32))


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
    """Build model configs. n_features needed for neural model dimensions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return {
        # === Tier 1: sklearn built-ins ===
        "logistic_regression": {
            "model": ScaledLogisticRegression,
            "kwargs": {"max_iter": 1000, "C": 1.0, "random_state": 42},
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
            "use_eval_set": False,
        },
        # === Tier 2: existing deps, new configs ===
        "xgboost": {
            "model": XGBClassifier,
            "kwargs": {
                "n_estimators": 2000,
                "max_depth": 7,
                "learning_rate": 0.03,
                "eval_metric": "auc",
                "early_stopping_rounds": 100,
                "tree_method": "hist",
                "subsample": 0.7,
                "colsample_bytree": 0.6,
                "min_child_weight": 10,
                "reg_alpha": 0.5,
                "reg_lambda": 5.0,
                "random_state": 42,
            },
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
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "min_child_weight": 20,
                "reg_alpha": 2.0,
                "reg_lambda": 10.0,
                "random_state": 99,
            },
            "kwargs_fit": {"verbose": 500},
        },
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
            "kwargs_fit": {
                "callbacks": [
                    __import__("lightgbm").log_evaluation(500),
                ],
            },
        },
        "catboost": {
            "model": CatBoostClassifier,
            "kwargs": {
                "iterations": 2000,
                "depth": 6,
                "learning_rate": 0.05,
                "eval_metric": "AUC",
                "early_stopping_rounds": 100,
                "subsample": 0.8,
                "bootstrap_type": "Bernoulli",
                "l2_leaf_reg": 3.0,
                "random_strength": 1.5,
                "random_seed": 77,
                "verbose": 500,
            },
        },
        # === Tier 3+4: Neural models (slow on CPU, uncomment with GPU) ===
        # "realmlp": {
        #     "model": RealMLP_TD_Classifier,
        #     "kwargs": {"device": device, "random_state": 42, "verbosity": 2},
        #     "use_eval_set": False,
        # },
        # "resnet": {
        #     "model": SkorchBinaryClassifier,
        #     "kwargs": {
        #         "module": ResNet, "module__d_in": n_features, "module__d_out": 1,
        #         "module__n_blocks": 2, "module__d_block": 192,
        #         "module__d_hidden_multiplier": 2.0, "module__dropout1": 0.3,
        #         "module__dropout2": 0.0, "max_epochs": 100, "lr": 0.001,
        #         "batch_size": 128, "device": device, "verbose": 0,
        #     },
        #     "use_eval_set": False,
        # },
        # "ft_transformer": {
        #     "model": SkorchBinaryClassifier,
        #     "kwargs": {
        #         "module": FTTransformerWrapper, "module__n_cont_features": n_features,
        #         "module__d_out": 1, "module__n_blocks": 3, "module__d_block": 192,
        #         "module__attention_n_heads": 8, "module__attention_dropout": 0.2,
        #         "module__ffn_dropout": 0.1, "max_epochs": 100, "lr": 1e-4,
        #         "batch_size": 128, "device": device, "verbose": 0,
        #     },
        #     "use_eval_set": False,
        # },
    }


def main():
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

    # Features = all columns except id and target
    features = [c for c in train.columns if c not in ["id", TARGET]]
    n_features = len(features)

    models = get_models(n_features)

    train_labels = train[TARGET].values
    holdout_labels = holdout[TARGET].values
    all_oof_preds = {}
    all_holdout_preds = {}
    all_test_preds = {}

    for name, config in models.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {name}")
        logger.info(f"{'='*50}")

        _, oof_preds, holdout_preds, test_preds = train_model_split(
            model=config["model"],
            train=train,
            test=test,
            holdout=holdout,
            features=features,
            target=TARGET,
            kwargs_model=config.get("kwargs", {}),
            kwargs_fit=config.get("kwargs_fit", {}),
            folds_n=10,
            use_probability=True,
            use_eval_set=config.get("use_eval_set", True),
        )

        auc = roc_auc_score(holdout_labels, holdout_preds)
        acc = accuracy_score(holdout_labels, (holdout_preds >= 0.5).astype(int))
        logger.info(f"{name} — Holdout AUC: {auc:.4f}, Accuracy: {acc:.4f}")

        all_oof_preds[name] = oof_preds
        all_holdout_preds[name] = holdout_preds
        all_test_preds[name] = test_preds

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
    logger.info(f"Simple Average — Holdout AUC: {auc:.4f}")

    # --- Ridge stacking ---
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge.fit(oof_matrix, train_labels)
    ridge_holdout = ridge.predict(holdout_matrix)
    ridge_test = ridge.predict(test_matrix)
    auc = roc_auc_score(holdout_labels, ridge_holdout)
    logger.info(f"Ridge (alpha={ridge.alpha_:.2f}) — Holdout AUC: {auc:.4f}")
    logger.info(f"  Weights: {dict(zip(model_names, ridge.coef_))}")

    # --- Hill Climbing ---
    best_weights = _hill_climbing(oof_matrix, train_labels, model_names)
    hc_holdout = holdout_matrix @ best_weights
    hc_test = test_matrix @ best_weights
    auc = roc_auc_score(holdout_labels, hc_holdout)
    logger.info(f"Hill Climbing — Holdout AUC: {auc:.4f}")
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
    logger.info(f"Best method: {best_name} (AUC: {auc:.4f})")

    # Generate submission
    submission = sample_submission.copy()
    submission[TARGET] = best_test
    output_path = Path(__file__).parent / "submission.csv"
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")
    logger.info(f"Mean prediction: {np.mean(best_test):.3f}")


if __name__ == "__main__":
    main()
