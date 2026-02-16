import argparse
import logging
import os
import subprocess
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
from rtdl_num_embeddings import PeriodicEmbeddings
from rtdl_revisiting_models import FTTransformer, ResNet
from scipy.stats import rankdata
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import EarlyStopping
from xgboost import XGBClassifier

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from kego.datasets.split import build_xy, split_dataset  # noqa: E402
from kego.train import train_model_split  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_git_commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True,
    text=True,
    cwd=project_root,
).stdout.strip()
logger.info(f"Git commit: {_git_commit}")

DATA_DIR = (
    Path(os.environ.get("KEGO_PATH_DATA", project_root / "data"))
    / "playground"
    / "playground-series-s6e2"
)
TARGET = "Heart Disease"

SEEDS_FULL = [42, 123, 777]
SEEDS_FAST = [42]
FAST_MODELS = {
    "logistic_regression",
    "xgboost",
    "lightgbm",
    "catboost",
}
NEURAL_ONLY_MODELS = {
    "resnet",
    "ft_transformer",
    "realmlp",
}
GPU_MODEL_PREFIXES = {"xgboost", "catboost", "realmlp", "resnet", "ft_transformer"}
NEURAL_MODEL_PREFIXES = {"realmlp", "resnet", "ft_transformer"}
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

# Feature sets identified by select_features.py (multi-seed ablation + forward selection).
# See notebooks/playground/README.md "Feature Selection (Fine-Grained)" for details.
FEATURES_ABLATION_PRUNED = [
    # Raw features that help (10)
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
    # Engineered features that help (11)
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

FEATURES_FORWARD_SELECTED = [
    "abnormal_count",
    "top4_sum",
    "Max HR",
    "Chest pain type",
    "maxhr_per_age",
    "EKG results",
    "thallium_x_chestpain",
    "Sex",
    "risk_score",
    "ST depression",
    "Number of vessels fluro",
    "chestpain_x_slope",
    "Age",
    "BP",
    "Cholesterol",
    "chestpain_x_angina",
]

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

FEATURE_SETS = {
    "all": None,  # resolved at runtime from DataFrame columns
    "raw": RAW_FEATURES,
    "ablation-pruned": FEATURES_ABLATION_PRUNED,
    "forward-selected": FEATURES_FORWARD_SELECTED,
}


def make_te_preprocess(te_features, drop_original=False):
    """Create a fold_preprocess callback that applies target encoding per CV fold.

    Args:
        te_features: Columns to target-encode.
        drop_original: If True, drop the raw categorical columns after adding
            TE versions. Use for models that can't handle categoricals natively
            (LogReg, ResNet) to fix the ordinal fallacy.
    """

    def preprocess(x_train, y_train, x_valid, x_test, x_holdout):
        for col in te_features:
            if col not in x_train.columns:
                continue
            means = y_train.groupby(x_train[col]).mean()
            global_mean = y_train.mean()
            for df in [x_train, x_valid, x_test, x_holdout]:
                df[f"{col}_te"] = df[col].map(means).fillna(global_mean)
        if drop_original:
            cols_to_drop = [c for c in te_features if c in x_train.columns]
            for df in [x_train, x_valid, x_test, x_holdout]:
                df.drop(columns=cols_to_drop, inplace=True)
        return x_train, x_valid, x_test, x_holdout

    return preprocess


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
        dmat = xgb.DMatrix(
            X, enable_categorical=getattr(self, "enable_categorical", False)
        )
        preds = self.get_booster().predict(dmat)
        return np.column_stack([1 - preds, preds])

    def predict(self, X, **kwargs):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class ScaledRealMLP:
    """RealMLP_TD_Classifier with native categorical feature support."""

    def __init__(self, cat_features=None, random_state=42, **kwargs):
        self.cat_features = cat_features or []
        self.random_state = random_state
        self.kwargs = kwargs

    def _prepare(self, X):
        if isinstance(X, pd.DataFrame) and self.cat_features:
            X = X.copy()
            for c in self.cat_features:
                if c in X.columns:
                    X[c] = X[c].astype("category")
            return X
        return X.values if isinstance(X, pd.DataFrame) else X

    def fit(self, X, y, **kwargs):
        X_prep = self._prepare(X)
        y_np = y.values if hasattr(y, "values") else y
        self.model = RealMLP_TD_Classifier(
            random_state=self.random_state, **self.kwargs
        )
        self.model.fit(X_prep, y_np)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._prepare(X))

    def predict(self, X):
        return self.model.predict(self._prepare(X))


class GaussianNoise(nn.Module):
    """Adds Gaussian noise during training only. Regularization for synthetic data."""

    def __init__(self, std=0.01):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training and self.std > 0:
            return x + torch.randn_like(x) * self.std
        return x


class ResNetModule(nn.Module):
    """Wraps rtdl ResNet for skorch compatibility, with periodic numerical embeddings."""

    def __init__(
        self,
        d_in=1,
        d_out=1,
        n_blocks=3,
        d_block=192,
        d_hidden_multiplier=2.0,
        dropout1=0.15,
        dropout2=0.0,
        n_frequencies=48,
        frequency_init_scale=0.01,
        d_embedding=24,
        noise_std=0.01,
    ):
        super().__init__()
        self.noise = GaussianNoise(std=noise_std)
        self.num_embeddings = PeriodicEmbeddings(
            n_features=d_in,
            d_embedding=d_embedding,
            n_frequencies=n_frequencies,
            frequency_init_scale=frequency_init_scale,
            activation=True,
            lite=False,
        )
        self.net = ResNet(
            d_in=d_in * d_embedding,
            d_out=d_out,
            n_blocks=n_blocks,
            d_block=d_block,
            d_hidden_multiplier=d_hidden_multiplier,
            dropout1=dropout1,
            dropout2=dropout2,
        )

    def forward(self, X):
        X = self.noise(X)  # Add noise during training only
        X = self.num_embeddings(X)  # (B, n_feat) -> (B, n_feat, d_emb)
        X = X.flatten(1)  # (B, n_feat * d_emb)
        return self.net(X).squeeze(-1)


class AMPNeuralNetBinaryClassifier(NeuralNetBinaryClassifier):
    """NeuralNetBinaryClassifier with automatic mixed precision (fp16)."""

    def initialize(self):
        super().initialize()
        self.amp_scaler_ = torch.amp.GradScaler("cuda")
        return self

    def infer(self, x, **fit_params):
        with torch.amp.autocast("cuda"):
            return super().infer(x, **fit_params)

    def train_step_single(self, batch, **fit_params):
        try:
            from skorch.dataset import unpack_data
        except ImportError:
            from skorch.utils import unpack_data

        self._set_training(True)
        Xi, yi = unpack_data(batch)
        y_pred = self.infer(Xi, **fit_params)
        loss = self.get_loss(y_pred, yi, X=Xi, training=True)
        self.amp_scaler_.scale(loss).backward()
        return {"loss": loss, "y_pred": y_pred}

    def train_step(self, batch, **fit_params):
        self.optimizer_.zero_grad()
        step = self.train_step_single(batch, **fit_params)
        self.amp_scaler_.step(self.optimizer_)
        self.amp_scaler_.update()
        return step


class SkorchResNet:
    """ResNet with QuantileTransformer + periodic embeddings + Gaussian noise."""

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
        num_workers=0,
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
        self.num_workers = num_workers
        self.random_state = random_state

    def fit(self, X, y, **kwargs):
        torch.manual_seed(self.random_state)
        self.scaler = QuantileTransformer(
            output_distribution="normal", random_state=self.random_state
        )
        X_np = self.scaler.fit_transform(
            X.values if isinstance(X, pd.DataFrame) else X
        ).astype(np.float32)
        y_np = (y.values if hasattr(y, "values") else y).astype(np.float32)
        d_in = X_np.shape[1]

        self.net = AMPNeuralNetBinaryClassifier(
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
            iterator_train__pin_memory=True,
            iterator_valid__pin_memory=True,
            iterator_train__num_workers=self.num_workers,
            iterator_valid__num_workers=self.num_workers,
            callbacks=[
                EarlyStopping(patience=self.patience, monitor="valid_loss"),
            ],
            verbose=1,
        )
        self.net.initialize()
        self.net.module_ = torch.compile(self.net.module_)
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
    """Wraps rtdl FTTransformer for skorch (continuous + categorical features).

    Replaces the default LinearEmbeddings for continuous features with
    PeriodicEmbeddings for richer numerical representations.
    """

    def __init__(
        self,
        n_cont_features=1,
        cat_cardinalities=None,
        d_out=1,
        n_blocks=3,
        d_block=96,
        attention_n_heads=8,
        attention_dropout=0.2,
        ffn_d_hidden_multiplier=4 / 3,
        ffn_dropout=0.1,
        residual_dropout=0.0,
        n_frequencies=48,
        frequency_init_scale=0.01,
        noise_std=0.01,
    ):
        super().__init__()
        self.n_cont = n_cont_features
        self.noise = GaussianNoise(std=noise_std)
        self.net = FTTransformer(
            n_cont_features=n_cont_features,
            cat_cardinalities=cat_cardinalities or [],
            d_out=d_out,
            n_blocks=n_blocks,
            d_block=d_block,
            attention_n_heads=attention_n_heads,
            attention_dropout=attention_dropout,
            ffn_d_hidden_multiplier=ffn_d_hidden_multiplier,
            ffn_dropout=ffn_dropout,
            residual_dropout=residual_dropout,
        )
        # Replace the default LinearEmbeddings with PeriodicEmbeddings
        if n_cont_features > 0:
            self.net.cont_embeddings = PeriodicEmbeddings(
                n_features=n_cont_features,
                d_embedding=d_block,
                n_frequencies=n_frequencies,
                frequency_init_scale=frequency_init_scale,
                activation=True,
                lite=False,
            )

    def forward(self, X):
        x_cont = X[:, : self.n_cont]
        x_cont = self.noise(x_cont)  # Add noise to continuous features during training
        x_cat = X[:, self.n_cont :].long() if X.shape[1] > self.n_cont else None
        return self.net(x_cont, x_cat=x_cat).squeeze(-1)


class SkorchFTTransformer:
    """FTTransformer with categorical embeddings, wrapped via skorch."""

    def __init__(
        self,
        cat_features=None,
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
        num_workers=0,
        random_state=42,
    ):
        self.cat_features = cat_features or []
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
        self.num_workers = num_workers
        self.random_state = random_state

    def _prepare(self, X, fit=False):
        if isinstance(X, pd.DataFrame):
            cont_cols = [c for c in X.columns if c not in self.cat_features]
            cat_cols = [c for c in X.columns if c in self.cat_features]
        else:
            if fit:
                self.cont_cols = []
                self.cat_cols = []
                self.cat_cardinalities = []
                self.scaler = QuantileTransformer(
                    output_distribution="normal", random_state=self.random_state
                )
                return self.scaler.fit_transform(X).astype(np.float32)
            return self.scaler.transform(X).astype(np.float32)

        if fit:
            self.cont_cols = cont_cols
            self.cat_cols = cat_cols
            self.scaler = QuantileTransformer(
                output_distribution="normal", random_state=self.random_state
            )
            self.cat_encoders = {}
            for c in cat_cols:
                vals = sorted(X[c].unique())
                self.cat_encoders[c] = {v: i for i, v in enumerate(vals)}
            self.cat_cardinalities = [len(self.cat_encoders[c]) for c in cat_cols]
            X_cont = self.scaler.fit_transform(X[cont_cols].values).astype(np.float32)
        else:
            X_cont = self.scaler.transform(X[self.cont_cols].values).astype(np.float32)

        if self.cat_cols:
            X_cat = np.column_stack(
                [X[c].map(self.cat_encoders[c]).values for c in self.cat_cols]
            ).astype(np.float32)
            return np.hstack([X_cont, X_cat])
        return X_cont

    def fit(self, X, y, **kwargs):
        torch.manual_seed(self.random_state)
        X_prep = self._prepare(X, fit=True)
        y_np = (y.values if hasattr(y, "values") else y).astype(np.float32)
        n_cont = len(self.cont_cols) if self.cont_cols else X_prep.shape[1]

        self.net = AMPNeuralNetBinaryClassifier(
            FTTransformerModule,
            module__n_cont_features=n_cont,
            module__cat_cardinalities=self.cat_cardinalities,
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
            iterator_train__pin_memory=True,
            iterator_valid__pin_memory=True,
            iterator_train__num_workers=self.num_workers,
            iterator_valid__num_workers=self.num_workers,
            callbacks=[
                EarlyStopping(patience=self.patience, monitor="valid_loss"),
            ],
            verbose=1,
        )
        self.net.initialize()
        self.net.module_ = torch.compile(self.net.module_)
        self.net.fit(X_prep, y_np)
        return self

    def predict_proba(self, X):
        X_prep = self._prepare(X)
        return self.net.predict_proba(X_prep)

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


def get_models(n_features: int, fast: bool = False, neural: bool = False) -> dict:
    """Build model configs with GPU acceleration for GBDT models."""
    all_models = {
        # === CPU models ===
        "logistic_regression": {
            "model": ScaledLogisticRegression,
            "kwargs": {"max_iter": 1000, "C": 1.0, "random_state": 42},
            "seed_key": "random_state",
            "use_eval_set": False,
            "fold_preprocess": make_te_preprocess(TE_FEATURES, drop_original=True),
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
                "enable_categorical": True,
                "subsample": 0.7,
                "colsample_bytree": 0.6,
                "min_child_weight": 10,
                "reg_alpha": 0.5,
                "reg_lambda": 5.0,
                "random_state": 42,
            },
            "seed_key": "random_state",
            "kwargs_fit": {"verbose": 500},
            "convert_cat_dtype": True,
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
                "enable_categorical": True,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "min_child_weight": 20,
                "reg_alpha": 2.0,
                "reg_lambda": 10.0,
                "random_state": 99,
            },
            "seed_key": "random_state",
            "kwargs_fit": {"verbose": 500},
            "convert_cat_dtype": True,
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
                "enable_categorical": True,
                "subsample": 0.6,
                "colsample_bytree": 0.6,
                "min_child_weight": 5,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42,
            },
            "seed_key": "random_state",
            "kwargs_fit": {"verbose": 500},
            "convert_cat_dtype": True,
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
                "enable_categorical": True,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 10,
                "reg_alpha": 0.5,
                "reg_lambda": 5.0,
                "random_state": 42,
            },
            "seed_key": "random_state",
            "kwargs_fit": {"verbose": 500},
            "convert_cat_dtype": True,
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
                "enable_categorical": True,
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
            "convert_cat_dtype": True,
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
                "categorical_feature": CAT_FEATURES,
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
                "categorical_feature": CAT_FEATURES,
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
                "categorical_feature": CAT_FEATURES,
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
                "categorical_feature": CAT_FEATURES,
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
                "cat_features": CAT_FEATURES,
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
                "cat_features": CAT_FEATURES,
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
                "cat_features": CAT_FEATURES,
                "random_seed": 42,
                "verbose": 500,
            },
            "seed_key": "random_seed",
        },
        # === Neural models (GPU) ===
        "realmlp": {
            "model": ScaledRealMLP,
            "kwargs": {
                "cat_features": CAT_FEATURES,
                "n_epochs": 256,
                "batch_size": 1024,
                "device": "cuda",
                "hidden_sizes": [256, 256, 256],
            },
            "seed_key": "random_state",
            "use_eval_set": False,
            "fold_preprocess": make_te_preprocess(TE_FEATURES),
        },
        # "realmlp_large" removed — too slow for regular runs, marginal gain
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
                "batch_size": 4096,
            },
            "seed_key": "random_state",
            "use_eval_set": False,
            "fold_preprocess": make_te_preprocess(TE_FEATURES, drop_original=True),
        },
        "ft_transformer": {
            "model": SkorchFTTransformer,
            "kwargs": {
                "cat_features": CAT_FEATURES,
                "n_blocks": 3,
                "d_block": 96,
                "attention_n_heads": 8,
                "attention_dropout": 0.2,
                "ffn_dropout": 0.1,
                "lr": 1e-4,
                "max_epochs": 200,
                "patience": 20,
                "batch_size": 2048,
            },
            "seed_key": "random_state",
            "use_eval_set": False,
            "fold_preprocess": make_te_preprocess(TE_FEATURES),
        },
    }

    if fast:
        all_models = {k: v for k, v in all_models.items() if k in FAST_MODELS}
        # Reduce neural model epochs for faster iteration
        for name in all_models:
            if name.startswith("realmlp"):
                all_models[name]["kwargs"]["n_epochs"] = 64
            elif name in ("resnet", "ft_transformer"):
                all_models[name]["kwargs"]["max_epochs"] = 50
    elif neural:
        all_models = {k: v for k, v in all_models.items() if k in NEURAL_ONLY_MODELS}
        for name in all_models:
            if name.startswith("realmlp"):
                all_models[name]["kwargs"]["n_epochs"] = 64
            elif name in ("resnet", "ft_transformer"):
                all_models[name]["kwargs"]["max_epochs"] = 100

    return all_models


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


@ray.remote
def _train_single_model(
    train, test, holdout, features, target, model_name, model_config, seed, folds_n=10
):
    """Train one model with one seed on a Ray worker."""
    import os
    import sys
    import time

    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    t0 = time.time()
    print(f"[{model_name}] Starting seed={seed}", flush=True)

    # Convert cat features to pandas category dtype for models that need it (XGBoost)
    if model_config.get("convert_cat_dtype"):
        cat_convert = {c: "category" for c in CAT_FEATURES if c in train.columns}
        train = train.astype(cat_convert)
        test = test.astype(cat_convert)
        holdout = holdout.astype(cat_convert)

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
        fold_preprocess=model_config.get("fold_preprocess"),
    )

    # Compute metrics for MLflow logging
    from sklearn.metrics import roc_auc_score

    holdout_labels = holdout[target].values
    oof_auc = roc_auc_score(train[target].values, oof)
    holdout_auc = roc_auc_score(holdout_labels, holdout_pred)

    logging_data = {
        "params": {
            "model": model_name,
            "seed": seed,
            "folds_n": folds_n,
            **{
                k: v
                for k, v in model_config["kwargs"].items()
                if isinstance(v, (int, float, str, bool))
            },
        },
        "metrics": {
            "oof_auc": oof_auc,
            "holdout_auc": holdout_auc,
        },
        "oof": oof,
        "holdout": holdout_pred,
        "test": test_pred,
    }

    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    print(
        f"[{model_name}] Finished seed={seed} "
        f"— OOF AUC: {oof_auc:.4f}, Holdout AUC: {holdout_auc:.4f} "
        f"({mins}m{secs:02d}s)",
        flush=True,
    )
    logging_data["metrics"]["duration_seconds"] = elapsed
    return model_name, seed, oof, holdout_pred, test_pred, logging_data


def _ensemble_predictions(
    model_names,
    all_oof_preds,
    all_holdout_preds,
    all_test_preds,
    train_labels,
    holdout_labels,
    tag="",
):
    """Run ensemble methods on collected predictions and return best test preds."""
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

    # --- Rank Blending ---
    # Normalize predictions to ranks per model, then average.
    # Robust to different calibrations across models.
    def _rank_blend(matrix):
        n = matrix.shape[0]
        ranked = np.column_stack(
            [rankdata(matrix[:, i]) / n for i in range(matrix.shape[1])]
        )
        return np.mean(ranked, axis=1)

    rb_oof = _rank_blend(oof_matrix)
    rb_holdout = _rank_blend(holdout_matrix)
    rb_test = _rank_blend(test_matrix)
    auc = roc_auc_score(holdout_labels, rb_holdout)
    logger.info(f"{tag}Rank Blending — Holdout AUC: {auc:.4f}")
    logger.info(f"{'='*50}")

    # Pick best ensemble method
    results = {
        "average": (avg_holdout, avg_test, np.mean(oof_matrix, axis=1)),
        "ridge": (ridge_holdout, ridge_test, ridge.predict(oof_matrix)),
        "hill_climbing": (hc_holdout, hc_test, oof_matrix @ best_weights),
        "rank_blending": (rb_holdout, rb_test, rb_oof),
    }
    all_aucs = {k: roc_auc_score(holdout_labels, v[0]) for k, v in results.items()}
    best_name = max(all_aucs, key=all_aucs.get)
    best_holdout, best_test, best_oof = results[best_name]
    auc = all_aucs[best_name]
    logger.info(f"{tag}Best method: {best_name} (AUC: {auc:.4f})")

    # --- Post-processing: Isotonic calibration ---
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(best_oof, train_labels)
    cal_holdout = iso.predict(best_holdout)
    cal_test = iso.predict(best_test)
    cal_auc = roc_auc_score(holdout_labels, cal_holdout)
    logger.info(f"{tag}Calibrated ({best_name}) — Holdout AUC: {cal_auc:.4f}")

    if cal_auc > auc:
        logger.info(
            f"{tag}Calibration improved AUC by {cal_auc - auc:.5f}, using calibrated"
        )
        best_test = cal_test
        auc = cal_auc
    else:
        logger.info(
            f"{tag}Calibration did not improve AUC "
            f"({cal_auc:.4f} vs {auc:.4f}), using raw"
        )

    logger.info(f"{'='*50}")
    return best_test, best_name, auc, all_aucs


def _load_predictions_from_runs(runs_df, tracking_uri):
    """Load and average predictions from a DataFrame of MLflow runs.

    Shared logic for --from-experiment and --from-ensemble.
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

        artifact_dir = client.download_artifacts(run.run_id, "predictions")
        oof = np.load(os.path.join(artifact_dir, "oof.npy"))
        holdout = np.load(os.path.join(artifact_dir, "holdout.npy"))
        test = np.load(os.path.join(artifact_dir, "test.npy"))

        if model_name not in all_oof:
            all_oof[model_name] = np.zeros_like(oof)
            all_holdout[model_name] = np.zeros_like(holdout)
            all_test[model_name] = np.zeros_like(test)
            seed_counts[model_name] = 0

        all_oof[model_name] += oof
        all_holdout[model_name] += holdout
        all_test[model_name] += test
        seed_counts[model_name] += 1

        seed = run.get("params.seed", "?")
        logger.info(f"  Loaded {model_name} seed={seed}")

    # Average across seeds
    for name in all_oof:
        n = seed_counts[name]
        all_oof[name] /= n
        all_holdout[name] /= n
        all_test[name] /= n
        logger.info(f"{name}: averaged over {n} seed(s)")

    model_names = list(all_oof.keys())
    logger.info(f"Total models loaded: {len(model_names)}")
    return model_names, all_oof, all_holdout, all_test


def _load_predictions_from_mlflow(experiment_names, tracking_uri):
    """Load per-model averaged predictions from MLflow experiments."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)

    all_runs = []
    for exp_name in experiment_names:
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            logger.warning(f"Experiment '{exp_name}' not found, skipping")
            continue

        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        # Filter out ensemble runs (NOT LIKE not supported by MLflow API)
        runs = runs[~runs["tags.mlflow.runName"].str.startswith("ensemble_", na=True)]
        logger.info(f"Experiment '{exp_name}': {len(runs)} model runs")
        all_runs.append(runs)

    if not all_runs:
        return [], {}, {}, {}

    import pandas as pd

    runs_df = pd.concat(all_runs, ignore_index=True)
    return _load_predictions_from_runs(runs_df, tracking_uri)


def _load_predictions_from_ensemble(ensemble_name, tracking_uri):
    """Load predictions from runs tagged with a named ensemble."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)

    tag_key = f"ensemble:{ensemble_name}"
    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=f"tags.`{tag_key}` = 'true'",
    )

    if runs.empty:
        logger.error(f"No runs found in ensemble '{ensemble_name}'")
        return [], {}, {}, {}

    # Filter out ensemble summary runs
    runs = runs[~runs["tags.mlflow.runName"].str.startswith("ensemble_", na=True)]
    logger.info(f"Ensemble '{ensemble_name}': {len(runs)} model runs")

    return _load_predictions_from_runs(runs, tracking_uri)


def _train_ensemble(
    train, holdout, test, features, models, seeds, tag="full", folds_n=10
):
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
        for seed in seeds:
            if model_name.startswith("catboost"):
                opts = {"num_gpus": 1, "num_cpus": 1, "resources": {"heavy_gpu": 1}}
            elif model_name.startswith("realmlp"):
                opts = {"num_gpus": 1, "num_cpus": 2, "resources": {"heavy_gpu": 1}}
            elif model_name == "ft_transformer":
                opts = {"num_gpus": 1, "num_cpus": 2, "resources": {"heavy_gpu": 1}}
            elif is_neural:
                opts = {"num_gpus": 0.5, "num_cpus": 2, "resources": {"heavy_gpu": 0.5}}
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
            device = f"GPU {opts['num_gpus']}" if is_gpu else "CPU"
            task_info.append(f"{model_name} seed={seed} ({device})")

    logger.info(f"Launched {len(futures)} Ray tasks:")
    for info in task_info:
        logger.info(f"  - {info}")

    # Setup MLflow for incremental logging
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    mlflow_ready = False
    if tracking_uri:
        try:
            import tempfile

            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            experiment_name = f"playground-s6e2-{tag}"
            mlflow.set_experiment(experiment_name)
            mlflow_ready = True
            logger.info(f"MLflow: logging to experiment '{experiment_name}'")
        except Exception as e:
            logger.warning(f"MLflow setup failed (non-fatal): {e}")

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
            model_name, seed, oof, holdout_pred, test_pred, logging_data = ray.get(
                done[0]
            )
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

        # Log to MLflow immediately
        if mlflow_ready:
            try:
                run_name = f"{model_name}_seed{seed}"
                with mlflow.start_run(run_name=run_name):
                    mlflow.log_params(logging_data["params"])
                    mlflow.log_metrics(logging_data["metrics"])
                    with tempfile.TemporaryDirectory() as tmp:
                        for arr_name, arr in [
                            ("oof", logging_data["oof"]),
                            ("holdout", logging_data["holdout"]),
                            ("test", logging_data["test"]),
                        ]:
                            path = os.path.join(tmp, f"{arr_name}.npy")
                            np.save(path, arr)
                        mlflow.log_artifacts(tmp, artifact_path="predictions")
            except Exception as e:
                logger.warning(f"MLflow logging failed for {model_name}: {e}")

    # Average across seeds and log per-model results
    for name in all_oof_preds:
        all_oof_preds[name] /= len(seeds)
        all_holdout_preds[name] /= len(seeds)
        all_test_preds[name] /= len(seeds)

        auc = roc_auc_score(holdout_labels, all_holdout_preds[name])
        acc = accuracy_score(
            holdout_labels, (all_holdout_preds[name] >= 0.5).astype(int)
        )
        logger.info(
            f"{name} (avg {len(seeds)} seeds) — "
            f"Holdout AUC: {auc:.4f}, Accuracy: {acc:.4f}"
        )

    # Build stacking matrices (only models that succeeded)
    model_names = [n for n in models.keys() if n in all_oof_preds]
    logger.info(f"Models with predictions: {len(model_names)}/{len(models)}")

    best_test, best_name, auc, all_aucs = _ensemble_predictions(
        model_names,
        all_oof_preds,
        all_holdout_preds,
        all_test_preds,
        train_labels,
        holdout_labels,
        tag,
    )

    # --- Log ensemble to MLflow ---
    ensemble_run_id = None
    if mlflow_ready:
        try:
            mlflow.set_experiment("ensemble")
            with mlflow.start_run(run_name=f"ensemble_{tag}") as run:
                ensemble_run_id = run.info.run_id
                for method, method_auc in all_aucs.items():
                    mlflow.log_metric(f"auc_{method}", method_auc)
                mlflow.log_metric("auc_best", auc)
                mlflow.log_param("best_method", best_name)
                mlflow.log_param("n_models", len(model_names))
                mlflow.log_param("n_seeds", len(seeds))
                mlflow.log_param("source", f"training_{tag}")
            logger.info("MLflow: logged ensemble run to 'ensemble'")
        except Exception as e:
            logger.warning(f"MLflow ensemble logging failed (non-fatal): {e}")

    return best_test, best_name, auc, ensemble_run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true", help="Quick run with small sample"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast iteration: 5 folds, 1 seed, core models only (~3-5 min)",
    )
    parser.add_argument(
        "--neural",
        action="store_true",
        help="Neural models only: 5 folds, 1 seed (resnet, ft_transformer, realmlp)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Custom tag for MLflow experiment name (e.g. 'gbdt-v2')",
    )
    parser.add_argument(
        "--from-experiment",
        nargs="+",
        metavar="EXPERIMENT",
        help="Load predictions from MLflow experiments instead of training",
    )
    parser.add_argument(
        "--from-ensemble",
        type=str,
        metavar="ENSEMBLE",
        help="Load predictions from a curated ensemble (tagged runs) instead of training",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit to Kaggle after generating submission.csv and log LB score to MLflow",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="ablation-pruned",
        choices=list(FEATURE_SETS.keys()),
        help="Feature set to use (default: ablation-pruned)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help="Only train these models (e.g. --models catboost realmlp)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        metavar="SEED",
        help="Override seeds (e.g. --seeds 777)",
    )
    args = parser.parse_args()

    if not args.from_experiment and not args.from_ensemble:
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

    # Impute Cholesterol=0 (missing values in original UCI data)
    train = _impute_cholesterol(train)
    holdout = _impute_cholesterol(holdout)
    test = _impute_cholesterol(test)

    # Engineer features
    train = _engineer_features(train)
    holdout = _engineer_features(holdout)
    test = _engineer_features(test)

    # Select feature set
    if args.features == "all":
        features = [c for c in train.columns if c not in ["id", TARGET]]
    else:
        features = FEATURE_SETS[args.features]
    logger.info(f"Feature set: {args.features} ({len(features)} features)")

    n_features = len(features)
    models = get_models(n_features, fast=args.fast, neural=args.neural)

    # Filter to specific models if requested
    if args.models:
        missing = [m for m in args.models if m not in models]
        if missing:
            logger.error(f"Unknown models: {missing}")
            logger.info(f"Available: {list(models.keys())}")
            sys.exit(1)
        models = {k: v for k, v in models.items() if k in args.models}

    # Filter cat_features in model configs to match the active feature set
    active_set = set(features)
    for config in models.values():
        if "cat_features" in config.get("kwargs", {}):
            config["kwargs"]["cat_features"] = [
                c for c in config["kwargs"]["cat_features"] if c in active_set
            ]
        if "categorical_feature" in config.get("kwargs_fit", {}):
            config["kwargs_fit"]["categorical_feature"] = [
                c
                for c in config["kwargs_fit"]["categorical_feature"]
                if c in active_set
            ]

    # Configure seeds and folds based on mode
    if args.debug:
        seeds, folds_n = SEEDS_FAST, 2
    elif args.fast or args.neural:
        seeds, folds_n = SEEDS_FAST, 5
    else:
        seeds, folds_n = SEEDS_FULL, 10

    # Override seeds if requested
    if args.seeds:
        seeds = args.seeds

    mode_name = (
        "debug"
        if args.debug
        else "fast" if args.fast else "neural" if args.neural else "full"
    )
    tag = args.tag or mode_name

    train_labels = train[TARGET].values
    holdout_labels = holdout[TARGET].values
    ensemble_run_id = None

    if args.from_experiment or args.from_ensemble:
        # Load predictions from MLflow and re-ensemble (no training)
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        if not tracking_uri:
            logger.error(
                "MLFLOW_TRACKING_URI must be set for --from-experiment/--from-ensemble"
            )
            sys.exit(1)

        if args.from_ensemble:
            logger.info(f"Loading predictions from ensemble: {args.from_ensemble}")
            model_names, all_oof, all_holdout, all_test = (
                _load_predictions_from_ensemble(args.from_ensemble, tracking_uri)
            )
        else:
            logger.info(f"Loading predictions from experiments: {args.from_experiment}")
            model_names, all_oof, all_holdout, all_test = _load_predictions_from_mlflow(
                args.from_experiment, tracking_uri
            )

        if not model_names:
            logger.error("No predictions loaded, exiting")
            sys.exit(1)

        best_test, best_method, best_auc, all_aucs = _ensemble_predictions(
            model_names,
            all_oof,
            all_holdout,
            all_test,
            train_labels,
            holdout_labels,
        )

        # Log ensemble to MLflow
        try:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("ensemble")
            run_name = (
                args.from_ensemble
                if args.from_ensemble
                else "_".join(args.from_experiment)
            )
            with mlflow.start_run(run_name=run_name) as run:
                ensemble_run_id = run.info.run_id
                for method, method_auc in all_aucs.items():
                    mlflow.log_metric(f"auc_{method}", method_auc)
                mlflow.log_metric("auc_best", best_auc)
                mlflow.log_param("best_method", best_method)
                mlflow.log_param("n_models", len(model_names))
                mlflow.log_param(
                    "source",
                    args.from_ensemble or ",".join(args.from_experiment),
                )
            logger.info(f"MLflow: logged ensemble run '{run_name}' to 'ensemble'")
        except Exception as e:
            logger.warning(f"MLflow ensemble logging failed (non-fatal): {e}")
    else:
        n_tasks = len(models) * len(seeds)
        logger.info(
            f"Mode: {mode_name} "
            f"— {len(models)} models × {len(seeds)} seeds × {folds_n} folds "
            f"= {n_tasks} tasks"
        )

        # Train ensemble with multi-seed averaging via Ray
        best_test, best_method, best_auc, ensemble_run_id = _train_ensemble(
            train,
            holdout,
            test,
            features,
            models,
            seeds=seeds,
            folds_n=folds_n,
            tag=tag,
        )

    # Generate submission
    submission = sample_submission.copy()
    submission[TARGET] = best_test
    output_path = Path(__file__).parent / "submission.csv"
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")
    logger.info(f"Mean prediction: {np.mean(best_test):.3f}")

    # --- Kaggle submit + poll + log LB score ---
    if args.submit:
        import csv
        import io
        import time

        competition = "playground-series-s6e2"
        message = (
            args.from_ensemble
            if args.from_ensemble
            else "_".join(args.from_experiment) if args.from_experiment else tag
        )

        logger.info(f"Submitting to Kaggle competition: {competition}")
        try:
            subprocess.run(
                [
                    "kaggle",
                    "competitions",
                    "submit",
                    "-c",
                    competition,
                    "-f",
                    str(output_path),
                    "-m",
                    message,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("Kaggle submission uploaded, polling for score...")
        except subprocess.CalledProcessError as e:
            logger.error(f"Kaggle submit failed: {e.stderr}")
            if not args.from_experiment and not args.from_ensemble:
                ray.shutdown()
            sys.exit(1)

        # Poll for completion
        public_score = None
        timeout = 300  # 5 minutes
        poll_interval = 10
        start_time = time.time()

        while time.time() - start_time < timeout:
            time.sleep(poll_interval)
            result = subprocess.run(
                [
                    "kaggle",
                    "competitions",
                    "submissions",
                    "-c",
                    competition,
                    "--csv",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.warning(f"Poll failed: {result.stderr}")
                continue

            # Skip warning lines before CSV header
            csv_lines = [
                l for l in result.stdout.splitlines() if not l.startswith("Warning:")
            ]
            reader = csv.DictReader(csv_lines)
            for row in reader:
                # Check the most recent submission (first row)
                status = row.get("status", "").lower()
                if "complete" in status:
                    public_score = float(row.get("publicScore", 0))
                    logger.info(f"Kaggle public LB score: {public_score}")
                elif "error" in status:
                    logger.error(f"Kaggle submission errored: {row}")
                else:
                    logger.info(f"Submission status: {status}, waiting...")
                break  # only check the latest submission

            if public_score is not None:
                break
        else:
            logger.warning("Timed out waiting for Kaggle score")

        # Log LB score to the ensemble MLflow run
        if public_score is not None:
            try:
                import mlflow

                tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
                mlflow.set_tracking_uri(tracking_uri)
                client = mlflow.tracking.MlflowClient()

                if ensemble_run_id:
                    client.log_metric(ensemble_run_id, "public_lb_score", public_score)
                    logger.info(
                        f"MLflow: logged public_lb_score={public_score} "
                        f"to run {ensemble_run_id}"
                    )
                else:
                    logger.warning(
                        "No ensemble MLflow run ID available, skipping LB score logging"
                    )
            except Exception as e:
                logger.warning(f"MLflow LB score logging failed: {e}")

    if not args.from_experiment and not args.from_ensemble:
        ray.shutdown()


if __name__ == "__main__":
    main()
