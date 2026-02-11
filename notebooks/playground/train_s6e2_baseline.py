import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from kego.datasets.split import build_xy, split_dataset
from kego.train import train_model_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = project_root / "data" / "playground" / "playground-series-s6e2"
TARGET = "Heart Disease"

MODELS = {
    # Deep trees, low learning rate, heavy regularization
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
    # Shallow trees, higher learning rate, light regularization
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
    # Medium depth, moderate learning rate, CatBoost-native regularization
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
        "kwargs_fit": {},
    },
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

    holdout_labels = holdout[TARGET].values
    all_holdout_preds = {}
    all_test_preds = {}

    for name, config in MODELS.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {name}")
        logger.info(f"{'='*50}")

        _, _, holdout_preds, test_preds = train_model_split(
            model=config["model"],
            train=train,
            test=test,
            holdout=holdout,
            features=features,
            target=TARGET,
            kwargs_model=config["kwargs"],
            kwargs_fit=config.get("kwargs_fit", {}),
            folds_n=10,
            use_probability=True,
        )

        auc = roc_auc_score(holdout_labels, holdout_preds)
        acc = accuracy_score(holdout_labels, (holdout_preds >= 0.5).astype(int))
        logger.info(f"{name} — Holdout AUC: {auc:.4f}, Accuracy: {acc:.4f}")

        all_holdout_preds[name] = holdout_preds
        all_test_preds[name] = test_preds

    # Ensemble: simple average
    ensemble_holdout = np.mean(list(all_holdout_preds.values()), axis=0)
    ensemble_test = np.mean(list(all_test_preds.values()), axis=0)

    auc = roc_auc_score(holdout_labels, ensemble_holdout)
    acc = accuracy_score(holdout_labels, (ensemble_holdout >= 0.5).astype(int))
    logger.info(f"\n{'='*50}")
    logger.info(f"Ensemble — Holdout AUC: {auc:.4f}, Accuracy: {acc:.4f}")
    logger.info(f"{'='*50}")

    # Generate submission
    submission = sample_submission.copy()
    submission[TARGET] = ensemble_test
    output_path = Path(__file__).parent / "submission.csv"
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")
    logger.info(f"Mean prediction: {np.mean(ensemble_test):.3f}")


if __name__ == "__main__":
    main()
