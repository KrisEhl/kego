import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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


def main():
    # Load data
    train_full = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

    # Map target to 0/1
    train_full[TARGET] = train_full[TARGET].map({"Presence": 1, "Absence": 0})

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

    # XGBoost hyperparameters
    kwargs_model = {
        "n_estimators": 1000,
        "max_depth": 5,
        "learning_rate": 0.05,
        "eval_metric": "auc",
        "early_stopping_rounds": 50,
        "tree_method": "hist",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
    }

    # Train with K-Fold CV
    model, oof_preds, holdout_preds, test_preds = train_model_split(
        model=XGBClassifier,
        train=train,
        test=test,
        holdout=holdout,
        features=features,
        target=TARGET,
        kwargs_model=kwargs_model,
        folds_n=10,
        use_probability=True,
    )

    # Evaluate on holdout
    holdout_labels = holdout[TARGET].values
    holdout_binary = (holdout_preds >= 0.5).astype(int)
    auc = roc_auc_score(holdout_labels, holdout_preds)
    acc = accuracy_score(holdout_labels, holdout_binary)
    logger.info(f"Holdout AUC: {auc:.4f}")
    logger.info(f"Holdout Accuracy: {acc:.4f}")

    # Generate submission with probabilities
    submission = sample_submission.copy()
    submission[TARGET] = test_preds
    output_path = Path(__file__).parent / "submission.csv"
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")
    logger.info(f"Submission shape: {submission.shape}")
    logger.info(f"Mean prediction: {np.mean(test_preds):.3f}")


if __name__ == "__main__":
    main()
