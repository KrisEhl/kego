import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import lightgbm as lgbm
import numpy as np
import pandas as pd
import ray
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from kego.datasets.split import build_xy, split_dataset  # noqa: E402
from kego.ensemble import compute_ensemble  # noqa: E402
from kego.ensemble.stacking import l2_stacking  # noqa: E402
from kego.ensemble.weights import hill_climbing  # noqa: E402
from kego.models.neural.ft_transformer import (  # noqa: E402
    FTTransformerModule,
    SkorchFTTransformer,
)
from kego.models.neural.noise import GaussianNoise  # noqa: E402
from kego.models.neural.resnet import ResNetModule, SkorchResNet  # noqa: E402
from kego.models.wrappers import (  # noqa: E402
    GPUXGBClassifier,
    ScaledLogisticRegression,
    ScaledRealMLP,
    SubsampledTabPFN,
)
from kego.preprocessing import make_te_preprocess  # noqa: E402
from kego.tracking import (  # noqa: E402
    get_completed_fingerprints,
    load_predictions_from_ensemble,
    load_predictions_from_mlflow,
    load_predictions_from_runs,
)
from kego.train import train_model_split  # noqa: E402
from kego.utils import (  # noqa: E402
    filter_model_config,
    get_seeds_for_learner,
    make_learner_id,
    task_fingerprint,
)

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
    "tabpfn",
}
NEURAL_ONLY_MODELS = {
    "resnet",
    "ft_transformer",
    "resnet_ple",
    "ft_transformer_ple",
    "realmlp",
}
GPU_MODEL_PREFIXES = {
    "xgboost",
    "catboost",
    "realmlp",
    "resnet",
    "ft_transformer",
    "tabpfn",
}
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


def get_models(
    n_features: int,
    fast: bool = False,
    fast_full: bool = False,
    neural: bool = False,
) -> dict:
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
        "tabpfn": {
            "model": SubsampledTabPFN,
            "kwargs": {
                "cat_features": CAT_FEATURES,
                "max_train_rows": 10000,
                "n_estimators": 4,
                "device": "cuda",
                "random_state": 42,
            },
            "seed_key": "random_state",
            "use_eval_set": False,
        },
        "random_forest": {
            "model": RandomForestClassifier,
            "kwargs": {
                "n_estimators": 500,
                "max_depth": 20,
                "min_samples_leaf": 5,
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
                "max_depth": 20,
                "min_samples_leaf": 5,
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
                "max_depth": 3,
                "learning_rate": 0.061,
                "eval_metric": "auc",
                "early_stopping_rounds": 100,
                "tree_method": "hist",
                "device": "cuda",
                "enable_categorical": True,
                "subsample": 0.856,
                "colsample_bytree": 0.402,
                "min_child_weight": 24,
                "reg_alpha": 6.87,
                "reg_lambda": 3.0,
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
        "resnet_ple": {
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
                "embedding_type": "ple",
                "n_bins": 48,
            },
            "seed_key": "random_state",
            "use_eval_set": False,
            "fold_preprocess": make_te_preprocess(TE_FEATURES, drop_original=True),
        },
        "ft_transformer_ple": {
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
                "embedding_type": "ple",
                "n_bins": 48,
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
            elif name.startswith(("resnet", "ft_transformer")):
                all_models[name]["kwargs"]["max_epochs"] = 50
    elif fast_full:
        all_models = {k: v for k, v in all_models.items() if k in FAST_MODELS}
    elif neural:
        all_models = {k: v for k, v in all_models.items() if k in NEURAL_ONLY_MODELS}
        for name in all_models:
            if name.startswith("realmlp"):
                all_models[name]["kwargs"]["n_epochs"] = 64
            elif name.startswith(("resnet", "ft_transformer")):
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


# ---------------------------------------------------------------------------
# Optuna search spaces
# ---------------------------------------------------------------------------


def _suggest_catboost(trial):
    bootstrap_type = trial.suggest_categorical(
        "bootstrap_type", ["Bayesian", "Bernoulli"]
    )
    params = {
        "iterations": 5000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "bootstrap_type": bootstrap_type,
        "early_stopping_rounds": 100,
        "eval_metric": "AUC",
        "task_type": "GPU",
        "gpu_ram_part": 0.5,
        "verbose": 0,
    }
    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature", 0.0, 10.0
        )
    else:
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    return params


def _suggest_lightgbm(trial):
    return {
        "n_estimators": 2000,
        "num_leaves": trial.suggest_int("num_leaves", 7, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "path_smooth": trial.suggest_float("path_smooth", 0.0, 100.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "metric": "auc",
        "verbosity": -1,
    }


def _suggest_ft_transformer(trial):
    return {
        "n_blocks": trial.suggest_int("n_blocks", 1, 4),
        "d_block": trial.suggest_int("d_block", 64, 256, step=8),
        "attention_n_heads": 8,
        "attention_dropout": trial.suggest_float("attention_dropout", 0.0, 0.5),
        "ffn_dropout": trial.suggest_float("ffn_dropout", 0.0, 0.5),
        "ffn_d_hidden_multiplier": trial.suggest_float(
            "ffn_d_hidden_multiplier", 1.0, 2.667
        ),
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "max_epochs": 50,
        "patience": 5,
    }


def _suggest_xgboost(trial):
    return {
        "n_estimators": 2000,
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "eval_metric": "auc",
        "early_stopping_rounds": 100,
        "tree_method": "hist",
        "device": "cpu",
        "enable_categorical": True,
    }


def _suggest_logistic_regression(trial):
    solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
    params = {
        "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
        "max_iter": 2000,
        "solver": solver,
    }
    if solver == "saga":
        params["penalty"] = trial.suggest_categorical(
            "penalty", ["l1", "l2", "elasticnet"]
        )
        if params["penalty"] == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
    else:
        params["penalty"] = "l2"
    return params


TUNE_SEARCH_SPACES = {
    "catboost": _suggest_catboost,
    "lightgbm": _suggest_lightgbm,
    "xgboost": _suggest_xgboost,
    "ft_transformer": _suggest_ft_transformer,
    "logistic_regression": _suggest_logistic_regression,
}


@ray.remote
def _train_single_model(
    train,
    test,
    holdout,
    features,
    target,
    model_name,
    model_config,
    seed,
    folds_n=10,
    feature_set="all",
    config_fingerprint="",
):
    """Train one model with one seed on a Ray worker."""
    import os
    import sys
    import time

    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    t0 = time.time()
    learner_id = make_learner_id(model_name, feature_set, folds_n)
    print(f"[{learner_id}] Starting seed={seed}", flush=True)

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

    oof_auc = roc_auc_score(train[target].values, oof)
    if target in holdout.columns:
        holdout_auc = roc_auc_score(holdout[target].values, holdout_pred)
    else:
        holdout_auc = None

    metrics = {"oof_auc": oof_auc}
    if holdout_auc is not None:
        metrics["holdout_auc"] = holdout_auc

    params = {
        "model": model_name,
        "seed": seed,
        "folds_n": folds_n,
        "feature_set": feature_set,
        **{
            k: v
            for k, v in model_config["kwargs"].items()
            if isinstance(v, (int, float, str, bool))
        },
    }
    if config_fingerprint:
        params["config_fingerprint"] = config_fingerprint

    logging_data = {
        "params": params,
        "metrics": metrics,
        "oof": oof,
        "holdout": holdout_pred,
        "test": test_pred,
    }

    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    if holdout_auc is not None:
        print(
            f"[{learner_id}] Finished seed={seed} "
            f"— OOF AUC: {oof_auc:.4f}, Holdout AUC: {holdout_auc:.4f} "
            f"({mins}m{secs:02d}s)",
            flush=True,
        )
    else:
        print(
            f"[{learner_id}] Finished seed={seed} "
            f"— OOF AUC: {oof_auc:.4f} (retrain-full, no holdout) "
            f"({mins}m{secs:02d}s)",
            flush=True,
        )
    logging_data["metrics"]["duration_seconds"] = elapsed
    return (
        model_name,
        seed,
        oof,
        holdout_pred,
        test_pred,
        logging_data,
        feature_set,
        folds_n,
    )


def _run_optuna_study(
    model_name,
    base_config,
    suggest_fn,
    train,
    holdout,
    test,
    features,
    feature_set,
    folds_n,
    seed,
    n_trials,
    tag,
):
    """Run Optuna HP tuning for a single model with MLflow logging."""
    import copy
    import time

    import mlflow
    import optuna

    # Determine Ray resource options for tuning.
    # XGBoost tuning uses CPU mode (device: "cpu" in search space).
    if model_name.startswith("xgboost"):
        resource_opts = {"num_cpus": 8}
    elif model_name.startswith("catboost"):
        resource_opts = {"num_gpus": 0.5, "num_cpus": 1}
    elif model_name.startswith("tabpfn"):
        resource_opts = {
            "num_gpus": 1,
            "num_cpus": 1,
            "resources": {"large_gpu": 1},
        }
    elif model_name.startswith(("ft_transformer", "realmlp")):
        resource_opts = {
            "num_gpus": 1,
            "num_cpus": 2,
            "resources": {"large_gpu": 1},
        }
    elif any(model_name.startswith(p) for p in NEURAL_MODEL_PREFIXES):
        resource_opts = {
            "num_gpus": 1,
            "num_cpus": 2,
            "resources": {"large_gpu": 1},
        }
    elif any(model_name.startswith(p) for p in GPU_MODEL_PREFIXES):
        resource_opts = {"num_gpus": 0.5, "num_cpus": 1}
    else:
        resource_opts = {"num_cpus": 8}

    # Put data into Ray object store once
    train_ref = ray.put(train)
    test_ref = ray.put(test)
    holdout_ref = ray.put(holdout)

    # Setup MLflow experiment
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    mlflow_ready = False
    experiment_name = (
        f"playground-s6e2-tune-{tag}-{model_name}"
        if tag
        else f"playground-s6e2-tune-{model_name}"
    )
    if tracking_uri:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            mlflow_ready = True
            logger.info(f"MLflow: logging to experiment '{experiment_name}'")
        except Exception as e:
            logger.warning(f"MLflow setup failed (non-fatal): {e}")

    # Filter config for active features
    active_set = set(features)

    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    logger.info(
        f"Starting Optuna study for '{model_name}': "
        f"{n_trials} trials, {folds_n} folds, feature_set={feature_set}"
    )

    # Determine max parallelism based on resource type.
    max_parallel = 1
    logger.info(f"Optuna parallelism: max_parallel={max_parallel}")

    completed = 0
    failed = 0
    running = {}  # {ray_future: (trial, t0)}

    while completed < n_trials:
        # Launch new trials up to max_parallel
        while (
            len(running) < max_parallel and completed + failed + len(running) < n_trials
        ):
            trial = study.ask()
            t0 = time.time()

            # Get suggested params
            suggested_params = suggest_fn(trial)

            # Build trial config from base config
            trial_config = copy.deepcopy(base_config)
            trial_config["kwargs"].update(suggested_params)

            # CatBoost: Bayesian bootstrap doesn't support subsample
            if model_name.startswith("catboost"):
                if trial_config["kwargs"].get("bootstrap_type") == "Bayesian":
                    trial_config["kwargs"].pop("subsample", None)

            # Prune invalid combos for LightGBM
            if model_name.startswith("lightgbm"):
                num_leaves = trial_config["kwargs"].get("num_leaves", 31)
                max_depth = trial_config["kwargs"].get("max_depth", -1)
                if max_depth > 0 and num_leaves >= 2**max_depth:
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                    completed += 1
                    continue

            # Filter cat_features to match active feature set
            trial_config = filter_model_config(trial_config, active_set)

            # Submit Ray task (non-blocking)
            future = _train_single_model.options(**resource_opts).remote(
                train_ref,
                test_ref,
                holdout_ref,
                features,
                TARGET,
                model_name,
                trial_config,
                seed,
                folds_n,
                feature_set,
            )
            running[future] = (trial, t0)

        if not running:
            break

        # Wait for any trial to complete
        done, _ = ray.wait(list(running.keys()), num_returns=1)

        for future in done:
            trial, t0 = running.pop(future)
            try:
                result = ray.get(future)
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                failed += 1
                continue

            _, _, _, _, _, logging_data, _, _ = result

            oof_auc = logging_data["metrics"]["oof_auc"]
            holdout_auc = logging_data["metrics"].get("holdout_auc")
            elapsed = time.time() - t0

            # Report result to Optuna
            study.tell(trial, oof_auc)
            completed += 1

            # Log to MLflow
            if mlflow_ready:
                try:
                    with mlflow.start_run(run_name=f"trial_{trial.number}"):
                        mlflow.log_params(
                            {
                                "model": model_name,
                                "feature_set": feature_set,
                                "folds_n": folds_n,
                                "seed": seed,
                                "trial_number": trial.number,
                                **{
                                    k: v
                                    for k, v in trial.params.items()
                                    if isinstance(v, (int, float, str, bool))
                                },
                            }
                        )
                        mlflow_metrics = {
                            "oof_auc": oof_auc,
                            "duration_seconds": elapsed,
                        }
                        if holdout_auc is not None:
                            mlflow_metrics["holdout_auc"] = holdout_auc
                        mlflow.log_metrics(mlflow_metrics)
                except Exception as e:
                    logger.warning(
                        f"MLflow logging failed for trial {trial.number}: {e}"
                    )

            try:
                best_so_far = study.best_value
            except ValueError:
                best_so_far = oof_auc
            holdout_str = (
                f" holdout_auc={holdout_auc:.4f}" if holdout_auc is not None else ""
            )
            print(
                f"Trial {completed}/{n_trials}: "
                f"oof_auc={oof_auc:.4f}{holdout_str} "
                f"(best={best_so_far:.4f}) [{elapsed:.0f}s]",
                flush=True,
            )

    # Print best params
    logger.info(f"\n{'='*50}")
    logger.info(f"Best trial for {model_name}: #{study.best_trial.number}")
    logger.info(f"  OOF AUC: {study.best_value:.4f}")
    logger.info(f"  Params: {study.best_trial.params}")
    logger.info(f"{'='*50}")

    # Log best trial summary to MLflow
    if mlflow_ready:
        try:
            with mlflow.start_run(run_name=f"best_trial_{model_name}"):
                mlflow.log_params(
                    {
                        "model": model_name,
                        "feature_set": feature_set,
                        "folds_n": folds_n,
                        "seed": seed,
                        "best_trial_number": study.best_trial.number,
                        **{
                            k: v
                            for k, v in study.best_trial.params.items()
                            if isinstance(v, (int, float, str, bool))
                        },
                    }
                )
                mlflow.log_metric("oof_auc", study.best_value)
                mlflow.set_tag("best_trial", "true")
        except Exception as e:
            logger.warning(f"MLflow best-trial logging failed: {e}")


def _ensemble_predictions(
    model_names,
    all_oof_preds,
    all_holdout_preds,
    all_test_preds,
    train_labels,
    holdout_labels,
    tag="",
    train_df=None,
    holdout_df=None,
    test_df=None,
):
    """Run ensemble methods on collected predictions and return best test preds."""
    eval_label = "Holdout" if holdout_labels is not None else "OOF"

    # Build L2 feature configs from competition-specific feature sets
    l2_feature_configs = None
    if train_df is not None:
        l2_feature_configs = []
        for fs_name, fs_features in [
            ("raw", RAW_FEATURES),
            ("ablation-pruned", FEATURES_ABLATION_PRUNED),
            ("forward-selected", FEATURES_FORWARD_SELECTED),
        ]:
            train_feat = train_df[fs_features].values
            holdout_feat = holdout_df[fs_features].values
            test_feat = test_df[fs_features].values if test_df is not None else None
            l2_feature_configs.append((fs_name, train_feat, holdout_feat, test_feat))

    result = compute_ensemble(
        model_names,
        all_oof_preds,
        all_holdout_preds,
        all_test_preds,
        train_labels,
        holdout_labels,
        l2_feature_configs=l2_feature_configs,
    )

    # --- Display results ---
    logger.info(f"\n{'='*50}")
    for m in result.methods:
        extra = ""
        if "weights" in m.metadata:
            extra = f"\n  Weights: {m.metadata['weights']}"
        if "alpha" in m.metadata:
            extra = f" (alpha={m.metadata['alpha']:.2f})" + extra
        logger.info(f"{tag}{m.name}{extra} — {eval_label} AUC: {m.auc:.4f}")
    logger.info(f"{'='*50}")

    if holdout_labels is None:
        logger.info(f"{tag}(retrain-full: method selection based on OOF AUC)")

    logger.info(
        f"{tag}Best method: {result.best_method} ({eval_label} AUC: {result.best_auc:.4f})"
    )
    if result.calibrated:
        logger.info(f"{tag}Calibration improved AUC, using calibrated predictions")
    elif holdout_labels is not None:
        logger.info(f"{tag}Calibration did not improve AUC, using raw predictions")
    else:
        logger.info(f"{tag}Skipping calibration comparison (no holdout for evaluation)")
    logger.info(f"{'='*50}")

    return result.best_test_preds, result.best_method, result.best_auc, result.all_aucs


def _log_run_to_mlflow(mlflow, learner_id, seed, logging_data, description):
    """Log a single model run to MLflow (params, metrics, prediction artifacts)."""
    import tempfile

    run_name = f"{learner_id}_seed{seed}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(logging_data["params"])
        if description:
            mlflow.log_param("description", description)
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


def _train_ensemble(
    train,
    holdout,
    test,
    feature_sets_map,
    models,
    seed_pool,
    seeds_per_learner,
    folds_list,
    tag="full",
    skip_fingerprints=None,
    preloaded=None,
    description="",
):
    """Train all learners (model x feature_set x folds) with rotating seeds via Ray."""
    # Share data via Ray object store (stored once, shared across all tasks)
    train_ref = ray.put(train)
    test_ref = ray.put(test)
    holdout_ref = ray.put(holdout)

    train_labels = train[TARGET].values
    holdout_labels = holdout[TARGET].values if TARGET in holdout.columns else None

    # Launch tasks: enumerate learners and assign rotating seeds
    futures = []
    task_info = []
    seeds_per_lid = {}  # learner_id -> number of seeds assigned
    skipped = 0

    learner_index = 0
    for fs_name, fs_features in feature_sets_map.items():
        active_set = set(fs_features)
        for folds_n in folds_list:
            for model_name, config in models.items():
                filtered_config = filter_model_config(config, active_set)
                learner_id = make_learner_id(model_name, fs_name, folds_n)
                learner_seeds = get_seeds_for_learner(
                    learner_index, seed_pool, seeds_per_learner
                )
                seeds_per_lid[learner_id] = len(learner_seeds)
                learner_index += 1

                is_gpu = any(model_name.startswith(p) for p in GPU_MODEL_PREFIXES)
                is_neural = any(model_name.startswith(p) for p in NEURAL_MODEL_PREFIXES)

                for seed in learner_seeds:
                    fp = task_fingerprint(
                        model_name, seed, folds_n, fs_name, fs_features, filtered_config
                    )
                    if skip_fingerprints and fp in skip_fingerprints:
                        skipped += 1
                        continue

                    if model_name.startswith("catboost"):
                        opts = {"num_gpus": 0.5, "num_cpus": 1}
                    elif model_name.startswith("tabpfn"):
                        opts = {
                            "num_gpus": 1,
                            "num_cpus": 1,
                            "resources": {"large_gpu": 1},
                        }
                    elif model_name.startswith("realmlp"):
                        opts = {
                            "num_gpus": 1,
                            "num_cpus": 2,
                            "resources": {"large_gpu": 1},
                        }
                    elif model_name.startswith("ft_transformer"):
                        opts = {
                            "num_gpus": 1,
                            "num_cpus": 2,
                            "resources": {"large_gpu": 1},
                        }
                    elif is_neural:
                        opts = {
                            "num_gpus": 1,
                            "num_cpus": 2,
                            "resources": {"large_gpu": 1},
                        }
                    elif is_gpu:
                        opts = {"num_gpus": 0.5, "num_cpus": 1}
                    else:
                        opts = {"num_cpus": 2}
                    future = _train_single_model.options(**opts).remote(
                        train_ref,
                        test_ref,
                        holdout_ref,
                        fs_features,
                        TARGET,
                        model_name,
                        filtered_config,
                        seed,
                        folds_n,
                        fs_name,
                        fp,
                    )
                    futures.append(future)
                    device = f"GPU {opts['num_gpus']}" if is_gpu else "CPU"
                    task_info.append(f"{learner_id} seed={seed} ({device})")

    if skipped:
        logger.info(
            f"Resuming: skipped {skipped} completed tasks (config match), "
            f"launching {len(futures)} new"
        )
    logger.info(f"Launched {len(futures)} Ray tasks ({learner_index} learners):")
    for info in task_info:
        logger.info(f"  - {info}")

    # Setup MLflow for incremental logging
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    mlflow_ready = False
    if tracking_uri:
        try:
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
    actual_seed_counts = {}  # learner_id -> number of seeds actually received
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
            (
                model_name,
                seed,
                oof,
                holdout_pred,
                test_pred,
                logging_data,
                fs_name,
                folds_n_val,
            ) = ray.get(done[0])
        except Exception as e:
            completed += 1
            logger.error(f"[{completed}/{len(futures)}] Task failed: {e}")
            continue
        completed += 1

        learner_id = make_learner_id(model_name, fs_name, folds_n_val)

        if learner_id not in all_oof_preds:
            all_oof_preds[learner_id] = np.zeros(len(train))
            all_holdout_preds[learner_id] = np.zeros(len(holdout))
            all_test_preds[learner_id] = np.zeros(len(test))

        all_oof_preds[learner_id] += oof
        all_holdout_preds[learner_id] += holdout_pred
        all_test_preds[learner_id] += test_pred
        actual_seed_counts[learner_id] = actual_seed_counts.get(learner_id, 0) + 1

        if holdout_labels is not None:
            auc = roc_auc_score(holdout_labels, holdout_pred)
            logger.info(
                f"[{completed}/{len(futures)}] {learner_id} seed={seed} "
                f"— Holdout AUC: {auc:.4f}"
            )
        else:
            logger.info(
                f"[{completed}/{len(futures)}] {learner_id} seed={seed} "
                f"— OOF AUC: {logging_data['metrics']['oof_auc']:.4f}"
            )

        # Log to MLflow immediately
        if mlflow_ready:
            try:
                _log_run_to_mlflow(mlflow, learner_id, seed, logging_data, description)
            except Exception as e:
                logger.warning(f"MLflow logging failed for {learner_id}: {e}")

    # Average across seeds per learner and log results
    for lid in all_oof_preds:
        n = actual_seed_counts.get(lid, seeds_per_lid[lid])
        all_oof_preds[lid] /= n
        all_holdout_preds[lid] /= n
        all_test_preds[lid] /= n

        if holdout_labels is not None:
            auc = roc_auc_score(holdout_labels, all_holdout_preds[lid])
            acc = accuracy_score(
                holdout_labels, (all_holdout_preds[lid] >= 0.5).astype(int)
            )
            logger.info(
                f"{lid} (avg {n} seeds) — "
                f"Holdout AUC: {auc:.4f}, Accuracy: {acc:.4f}"
            )
        else:
            oof_auc = roc_auc_score(train_labels, all_oof_preds[lid])
            logger.info(f"{lid} (avg {n} seeds) — OOF AUC: {oof_auc:.4f}")

    # Merge preloaded predictions from resumed experiment
    if preloaded:
        pre_oof, pre_holdout, pre_test = preloaded
        preloaded_count = 0
        for lid in pre_oof:
            if lid not in all_oof_preds:
                # Fully preloaded learner (all seeds completed previously)
                all_oof_preds[lid] = pre_oof[lid]
                all_holdout_preds[lid] = pre_holdout[lid]
                all_test_preds[lid] = pre_test[lid]
                preloaded_count += 1
                if holdout_labels is not None:
                    auc = roc_auc_score(holdout_labels, pre_holdout[lid])
                    logger.info(f"{lid} (preloaded) — Holdout AUC: {auc:.4f}")
                else:
                    oof_auc = roc_auc_score(train_labels, pre_oof[lid])
                    logger.info(f"{lid} (preloaded) — OOF AUC: {oof_auc:.4f}")
        if preloaded_count:
            logger.info(f"Merged {preloaded_count} preloaded learners from resume")

    # Build ensemble from all learners that succeeded
    learner_names = list(all_oof_preds.keys())
    logger.info(f"Learners with predictions: {len(learner_names)}/{learner_index}")

    best_test, best_name, auc, all_aucs = _ensemble_predictions(
        learner_names,
        all_oof_preds,
        all_holdout_preds,
        all_test_preds,
        train_labels,
        holdout_labels,
        tag,
        train_df=train,
        holdout_df=holdout,
        test_df=test,
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
                mlflow.log_param("n_learners", len(learner_names))
                mlflow.log_param("n_seed_pool", len(seed_pool))
                mlflow.log_param(
                    "seeds_per_learner",
                    seeds_per_learner if seeds_per_learner else len(seed_pool),
                )
                mlflow.log_param("source", f"training_{tag}")
                if description:
                    mlflow.log_param("description", description)
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
        "--fast-full",
        action="store_true",
        help="Core models only with full CV: 10 folds, 3 seeds (~15-20 min)",
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
        nargs="+",
        default=["ablation-pruned"],
        choices=list(FEATURE_SETS.keys()),
        help="Feature set(s) to use (default: ablation-pruned)",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        help="Number of CV folds (e.g. --folds 5 10). Default depends on mode.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help="Only train these models (e.g. --models catboost realmlp)",
    )
    parser.add_argument(
        "--seed-pool",
        "--seeds",
        nargs="+",
        type=int,
        dest="seed_pool",
        metavar="SEED",
        help="Seed pool (e.g. --seed-pool 42 123 777)",
    )
    parser.add_argument(
        "--seeds-per-learner",
        type=int,
        default=None,
        help="Seeds per learner (rotating from pool). Default: all seeds.",
    )
    parser.add_argument(
        "--tune",
        nargs="+",
        metavar="MODEL",
        help="Run Optuna HP tuning for these models (e.g. --tune catboost lightgbm)",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=50,
        help="Number of Optuna trials per model (default: 50)",
    )
    parser.add_argument(
        "--tune-sample",
        type=int,
        default=None,
        help="Subsample training data to N rows for tuning (useful for neural models)",
    )
    parser.add_argument(
        "--retrain-full",
        action="store_true",
        help="Retrain all models on train+holdout combined (no holdout evaluation)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        metavar="EXPERIMENT",
        help="Resume from previous experiment: skip completed tasks, retrain failed/missing",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Free-text description logged to MLflow (e.g. 'tuned catboost LR')",
    )
    args = parser.parse_args()

    if args.retrain_full and (args.from_ensemble or args.from_experiment):
        logger.error(
            "--retrain-full cannot be combined with --from-ensemble/--from-experiment"
        )
        sys.exit(1)

    if args.resume and (args.from_ensemble or args.from_experiment):
        logger.error(
            "--resume cannot be combined with --from-ensemble/--from-experiment"
        )
        sys.exit(1)

    if not args.from_experiment and not args.from_ensemble:
        os.environ.setdefault("RAY_DEDUP_LOGS", "0")
        runtime_env = {}
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            runtime_env["env_vars"] = {"HF_TOKEN": hf_token}
        ray.init(runtime_env=runtime_env if runtime_env else None)
        optuna_logging = __import__("logging").getLogger("optuna")
        optuna_logging.setLevel(__import__("logging").WARNING)

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

    if args.retrain_full:
        logger.info(
            f"Retrain-full: combining train ({len(train)}) + holdout ({len(holdout)}) "
            f"= {len(train) + len(holdout)} rows"
        )
        train = pd.concat([train, holdout]).reset_index(drop=True)
        holdout = test  # holdout has no TARGET — signals retrain-full mode

    # Build feature sets map from args
    feature_sets_map = {}
    for fs_name in args.features:
        if fs_name == "all":
            feature_sets_map["all"] = [
                c for c in train.columns if c not in ["id", TARGET]
            ]
        else:
            feature_sets_map[fs_name] = FEATURE_SETS[fs_name]
    first_fs = next(iter(feature_sets_map.values()))
    for fs_name, fs_features in feature_sets_map.items():
        logger.info(f"Feature set '{fs_name}': {len(fs_features)} features")

    n_features = len(first_fs)
    models = get_models(
        n_features, fast=args.fast, fast_full=args.fast_full, neural=args.neural
    )

    # Filter to specific models if requested
    if args.models:
        missing = [m for m in args.models if m not in models]
        if missing:
            logger.error(f"Unknown models: {missing}")
            logger.info(f"Available: {list(models.keys())}")
            sys.exit(1)
        models = {k: v for k, v in models.items() if k in args.models}

    # Cat feature filtering now happens inside _train_ensemble via filter_model_config

    # Configure seed pool and folds list based on mode
    if args.debug:
        default_seed_pool, default_folds = SEEDS_FAST, [2]
    elif args.fast or args.neural:
        default_seed_pool, default_folds = SEEDS_FAST, [5]
    elif args.fast_full:
        default_seed_pool, default_folds = SEEDS_FULL, [10]
    else:
        default_seed_pool, default_folds = SEEDS_FULL, [10]

    seed_pool = args.seed_pool if args.seed_pool else default_seed_pool
    folds_list = args.folds if args.folds else default_folds

    mode_name = (
        "debug"
        if args.debug
        else (
            "fast"
            if args.fast
            else "fast-full" if args.fast_full else "neural" if args.neural else "full"
        )
    )
    tag = args.tag or mode_name

    train_labels = train[TARGET].values
    holdout_labels = holdout[TARGET].values if TARGET in holdout.columns else None
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
                load_predictions_from_ensemble(args.from_ensemble, tracking_uri)
            )
        else:
            logger.info(f"Loading predictions from experiments: {args.from_experiment}")
            model_names, all_oof, all_holdout, all_test = load_predictions_from_mlflow(
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
            train_df=train,
            holdout_df=holdout,
            test_df=test,
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
    elif args.tune:
        # Optuna hyperparameter tuning mode
        for m in args.tune:
            if m not in TUNE_SEARCH_SPACES:
                logger.error(
                    f"No search space for '{m}'. "
                    f"Available: {list(TUNE_SEARCH_SPACES.keys())}"
                )
                sys.exit(1)

        # Subsample training data if requested
        train_tune = train
        if args.tune_sample and args.tune_sample < len(train):
            train_tune = train.sample(n=args.tune_sample, random_state=42).reset_index(
                drop=True
            )
            logger.info(f"Tuning on {len(train_tune)}/{len(train)} subsampled rows")

        feature_set_name = args.features[0]
        features = feature_sets_map[feature_set_name]
        folds_n = folds_list[0]
        tune_seed = seed_pool[0]

        for model_name in args.tune:
            base_config = models[model_name]
            _run_optuna_study(
                model_name=model_name,
                base_config=base_config,
                suggest_fn=TUNE_SEARCH_SPACES[model_name],
                train=train_tune,
                holdout=holdout,
                test=test,
                features=features,
                feature_set=feature_set_name,
                folds_n=folds_n,
                seed=tune_seed,
                n_trials=args.tune_trials,
                tag=tag,
            )

        # No submission generated in tune mode
        if not args.from_experiment and not args.from_ensemble:
            ray.shutdown()
        return
    else:
        n_learners = len(models) * len(feature_sets_map) * len(folds_list)
        logger.info(
            f"Mode: {mode_name} "
            f"— {len(models)} models × {len(feature_sets_map)} feature sets "
            f"× {len(folds_list)} folds = {n_learners} learners"
        )
        if args.seeds_per_learner:
            logger.info(
                f"  Rotating seeds: {args.seeds_per_learner} per learner "
                f"from pool of {len(seed_pool)}"
            )
        else:
            logger.info(f"  All learners share {len(seed_pool)} seed(s)")

        # Resume: load completed fingerprints and predictions from previous run
        skip_fingerprints = None
        preloaded = None
        if args.resume:
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
            if not tracking_uri:
                logger.error("MLFLOW_TRACKING_URI must be set for --resume")
                sys.exit(1)
            skip_fingerprints, runs_df = get_completed_fingerprints(
                args.resume, tracking_uri
            )
            logger.info(
                f"Resuming '{args.resume}': {len(skip_fingerprints)} completed tasks"
            )
            if not runs_df.empty:
                _, pre_oof, pre_holdout, pre_test = load_predictions_from_runs(
                    runs_df, tracking_uri
                )
                preloaded = (pre_oof, pre_holdout, pre_test)

        # Train ensemble with multi-seed averaging via Ray
        best_test, best_method, best_auc, ensemble_run_id = _train_ensemble(
            train,
            holdout,
            test,
            feature_sets_map,
            models,
            seed_pool=seed_pool,
            seeds_per_learner=args.seeds_per_learner,
            folds_list=folds_list,
            tag=tag,
            skip_fingerprints=skip_fingerprints,
            preloaded=preloaded,
            description=args.description,
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
