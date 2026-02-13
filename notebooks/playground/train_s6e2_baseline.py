import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import torch
import torch.nn as nn
import xgboost as xgb
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
from skorch.callbacks import EarlyStopping
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
GPU_MODEL_PREFIXES = {"xgboost", "catboost", "realmlp", "resnet", "ft_transformer"}
NEURAL_MODEL_PREFIXES = {"realmlp", "resnet", "ft_transformer"}


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


class GPUXGBClassifier(XGBClassifier):
    """XGBClassifier that uses DMatrix for GPU-native prediction."""

    def predict_proba(self, X, **kwargs):
        dmat = xgb.DMatrix(X)
        preds = self.get_booster().predict(dmat)
        return np.column_stack([1 - preds, preds])

    def predict(self, X, **kwargs):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class ScaledRealMLP:
    """RealMLP_TD_Classifier with StandardScaler preprocessing."""

    def __init__(self, random_state=42, **kwargs):
        self.random_state = random_state
        self.kwargs = kwargs

    def fit(self, X, y, **kwargs):
        self.scaler = StandardScaler()
        X_np = self.scaler.fit_transform(X.values if isinstance(X, pd.DataFrame) else X)
        y_np = y.values if hasattr(y, "values") else y
        self.model = RealMLP_TD_Classifier(
            random_state=self.random_state, **self.kwargs
        )
        self.model.fit(X_np, y_np)
        return self

    def predict_proba(self, X):
        X_np = self.scaler.transform(X.values if isinstance(X, pd.DataFrame) else X)
        return self.model.predict_proba(X_np)

    def predict(self, X):
        X_np = self.scaler.transform(X.values if isinstance(X, pd.DataFrame) else X)
        return self.model.predict(X_np)


class ResNetModule(nn.Module):
    """Wraps rtdl ResNet for skorch compatibility."""

    def __init__(
        self,
        d_in=1,
        d_out=1,
        n_blocks=3,
        d_block=192,
        d_hidden_multiplier=2.0,
        dropout1=0.15,
        dropout2=0.0,
    ):
        super().__init__()
        self.net = ResNet(
            d_in=d_in,
            d_out=d_out,
            n_blocks=n_blocks,
            d_block=d_block,
            d_hidden_multiplier=d_hidden_multiplier,
            dropout1=dropout1,
            dropout2=dropout2,
        )

    def forward(self, X):
        return self.net(X).squeeze(-1)


class SkorchResNet:
    """ResNet with StandardScaler, wrapped via skorch for sklearn API."""

    def __init__(
        self,
        d_block=192,
        n_blocks=3,
        d_hidden_multiplier=2.0,
        dropout1=0.15,
        dropout2=0.0,
        lr=1e-3,
        max_epochs=200,
        patience=20,
        batch_size=256,
        random_state=42,
    ):
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.d_hidden_multiplier = d_hidden_multiplier
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y, **kwargs):
        torch.manual_seed(self.random_state)
        self.scaler = StandardScaler()
        X_np = self.scaler.fit_transform(
            X.values if isinstance(X, pd.DataFrame) else X
        ).astype(np.float32)
        y_np = (y.values if hasattr(y, "values") else y).astype(np.float32)
        d_in = X_np.shape[1]

        self.net = NeuralNetBinaryClassifier(
            ResNetModule,
            module__d_in=d_in,
            module__d_out=1,
            module__n_blocks=self.n_blocks,
            module__d_block=self.d_block,
            module__d_hidden_multiplier=self.d_hidden_multiplier,
            module__dropout1=self.dropout1,
            module__dropout2=self.dropout2,
            criterion=nn.BCEWithLogitsLoss,
            optimizer=torch.optim.AdamW,
            lr=self.lr,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            device="cuda",
            callbacks=[
                EarlyStopping(patience=self.patience, monitor="valid_loss"),
            ],
            verbose=0,
        )
        self.net.fit(X_np, y_np)
        return self

    def predict_proba(self, X):
        X_np = self.scaler.transform(
            X.values if isinstance(X, pd.DataFrame) else X
        ).astype(np.float32)
        return self.net.predict_proba(X_np)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class FTTransformerModule(nn.Module):
    """Wraps rtdl FTTransformer for skorch (continuous features only)."""

    def __init__(
        self,
        n_cont_features=1,
        d_out=1,
        n_blocks=3,
        d_block=96,
        attention_n_heads=8,
        attention_dropout=0.2,
        ffn_d_hidden_multiplier=4 / 3,
        ffn_dropout=0.1,
        residual_dropout=0.0,
    ):
        super().__init__()
        self.net = FTTransformer(
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
        return self.net(X, x_cat=None).squeeze(-1)


class SkorchFTTransformer:
    """FTTransformer with StandardScaler, wrapped via skorch for sklearn API."""

    def __init__(
        self,
        n_blocks=3,
        d_block=96,
        attention_n_heads=8,
        attention_dropout=0.2,
        ffn_d_hidden_multiplier=4 / 3,
        ffn_dropout=0.1,
        residual_dropout=0.0,
        lr=1e-4,
        max_epochs=200,
        patience=20,
        batch_size=256,
        random_state=42,
    ):
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.ffn_d_hidden_multiplier = ffn_d_hidden_multiplier
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y, **kwargs):
        torch.manual_seed(self.random_state)
        self.scaler = StandardScaler()
        X_np = self.scaler.fit_transform(
            X.values if isinstance(X, pd.DataFrame) else X
        ).astype(np.float32)
        y_np = (y.values if hasattr(y, "values") else y).astype(np.float32)
        n_features = X_np.shape[1]

        self.net = NeuralNetBinaryClassifier(
            FTTransformerModule,
            module__n_cont_features=n_features,
            module__d_out=1,
            module__n_blocks=self.n_blocks,
            module__d_block=self.d_block,
            module__attention_n_heads=self.attention_n_heads,
            module__attention_dropout=self.attention_dropout,
            module__ffn_d_hidden_multiplier=self.ffn_d_hidden_multiplier,
            module__ffn_dropout=self.ffn_dropout,
            module__residual_dropout=self.residual_dropout,
            criterion=nn.BCEWithLogitsLoss,
            optimizer=torch.optim.AdamW,
            lr=self.lr,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            device="cuda",
            callbacks=[
                EarlyStopping(patience=self.patience, monitor="valid_loss"),
            ],
            verbose=0,
        )
        self.net.fit(X_np, y_np)
        return self

    def predict_proba(self, X):
        X_np = self.scaler.transform(
            X.values if isinstance(X, pd.DataFrame) else X
        ).astype(np.float32)
        return self.net.predict_proba(X_np)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


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
            "model": GPUXGBClassifier,
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
            "model": GPUXGBClassifier,
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
            "model": GPUXGBClassifier,
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
            "model": GPUXGBClassifier,
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
            "model": GPUXGBClassifier,
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
        # === Neural models (GPU) ===
        "realmlp": {
            "model": ScaledRealMLP,
            "kwargs": {
                "n_epochs": 256,
                "device": "cuda",
                "hidden_sizes": [256, 256, 256],
            },
            "seed_key": "random_state",
            "use_eval_set": False,
        },
        "realmlp_large": {
            "model": ScaledRealMLP,
            "kwargs": {
                "n_epochs": 256,
                "device": "cuda",
                "hidden_sizes": [512, 256, 128],
            },
            "seed_key": "random_state",
            "use_eval_set": False,
        },
        "resnet": {
            "model": SkorchResNet,
            "kwargs": {
                "d_block": 192,
                "n_blocks": 3,
                "d_hidden_multiplier": 2.0,
                "dropout1": 0.15,
                "dropout2": 0.0,
                "lr": 1e-3,
                "max_epochs": 200,
                "patience": 20,
            },
            "seed_key": "random_state",
            "use_eval_set": False,
        },
        "ft_transformer": {
            "model": SkorchFTTransformer,
            "kwargs": {
                "n_blocks": 3,
                "d_block": 96,
                "attention_n_heads": 8,
                "attention_dropout": 0.2,
                "ffn_dropout": 0.1,
                "lr": 1e-4,
                "max_epochs": 200,
                "patience": 20,
            },
            "seed_key": "random_state",
            "use_eval_set": False,
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
        is_neural = any(model_name.startswith(p) for p in NEURAL_MODEL_PREFIXES)
        for seed in SEEDS:
            if model_name.startswith("catboost"):
                opts = {"num_gpus": 1, "num_cpus": 1}
            elif is_neural:
                opts = {"num_gpus": 0.5, "num_cpus": 1}
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

    # Build stacking matrices (only models that succeeded)
    model_names = [n for n in models.keys() if n in all_oof_preds]
    logger.info(f"Models with predictions: {len(model_names)}/{len(models)}")
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
