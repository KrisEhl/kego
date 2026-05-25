"""Baseline LightGBM model for AI Masters Fraud Detection.

Time-based holdout: last 20% of training data (by TransactionDT) is held out
for evaluation, mimicking the temporal train/test split. The holdout set is
NOT used for training or early stopping.

Usage:
    uv run python competitions/ai-masters-fraud-detection/train_baseline.py
"""

import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = project_root / "data" / "ai" / "ai-masters-fraud-detection"
OUTPUT_DIR = Path(__file__).resolve().parent
TARGET = "isFraud"

# ---------------------------------------------------------------------------
# Load & merge
# ---------------------------------------------------------------------------
log.info("Loading data...")
train_full = pd.read_csv(DATA_DIR / "train_transaction.csv").merge(
    pd.read_csv(DATA_DIR / "train_identity.csv"), on="TransactionID", how="left"
)
test = pd.read_csv(DATA_DIR / "test_transaction.csv").merge(
    pd.read_csv(DATA_DIR / "test_identity.csv"), on="TransactionID", how="left"
)
log.info(f"Full train: {train_full.shape}, Test: {test.shape}")

# ---------------------------------------------------------------------------
# Time-based holdout split (last 20% by TransactionDT)
# ---------------------------------------------------------------------------
HOLDOUT_FRAC = 0.2
cutoff = train_full["TransactionDT"].quantile(1 - HOLDOUT_FRAC)
train = train_full[train_full["TransactionDT"] <= cutoff].reset_index(drop=True)
holdout = train_full[train_full["TransactionDT"] > cutoff].reset_index(drop=True)

log.info(f"Train: {len(train):,} rows (DT <= {cutoff:.0f}), Holdout: {len(holdout):,} rows (DT > {cutoff:.0f})")
log.info(f"Fraud rate — train: {train[TARGET].mean():.4f}, holdout: {holdout[TARGET].mean():.4f}")

# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------
drop_cols = ["TransactionID", TARGET]
features = [c for c in train.columns if c not in drop_cols]

# Label-encode object/categorical columns
cat_cols = []
for col in features:
    if train[col].dtype == "object":
        cat_cols.append(col)
        combined = pd.concat([train[col], holdout[col], test[col]], axis=0).astype("category").cat.codes
        train[col] = combined.iloc[: len(train)].values
        holdout[col] = combined.iloc[len(train) : len(train) + len(holdout)].values
        test[col] = combined.iloc[len(train) + len(holdout) :].values

log.info(f"Features: {len(features)} ({len(cat_cols)} label-encoded)")

# ---------------------------------------------------------------------------
# Train with 5-fold CV on train portion only
# ---------------------------------------------------------------------------
LGB_PARAMS = {
    "n_estimators": 2000,
    "max_depth": 7,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_samples": 50,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
holdout_preds = np.zeros(len(holdout))
test_preds = np.zeros(len(test))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train, train[TARGET])):
    log.info(f"--- Fold {fold + 1}/{N_FOLDS} ---")

    X_tr = train.loc[train_idx, features]
    y_tr = train.loc[train_idx, TARGET]
    X_val = train.loc[val_idx, features]
    y_val = train.loc[val_idx, TARGET]

    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred
    fold_auc = roc_auc_score(y_val, val_pred)
    fold_scores.append(fold_auc)
    log.info(f"Fold {fold + 1} AUC: {fold_auc:.6f}")

    holdout_preds += model.predict_proba(holdout[features])[:, 1] / N_FOLDS
    test_preds += model.predict_proba(test[features])[:, 1] / N_FOLDS

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
oof_auc = roc_auc_score(train[TARGET], oof_preds)
holdout_auc = roc_auc_score(holdout[TARGET], holdout_preds)

log.info(f"OOF AUC:     {oof_auc:.6f}  (folds: {[f'{s:.6f}' for s in fold_scores]})")
log.info(f"Mean fold:   {np.mean(fold_scores):.6f} +/- {np.std(fold_scores):.6f}")
log.info(f"HOLDOUT AUC: {holdout_auc:.6f}  <-- best proxy for leaderboard")

# Feature importance
importance = model.feature_importances_
top_idx = np.argsort(importance)[::-1][:30]
log.info("Top 30 features by importance:")
for i in top_idx:
    log.info(f"  {features[i]:<20s}  {importance[i]:>6d}")

# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------
submission = pd.DataFrame({"TransactionID": test["TransactionID"], TARGET: test_preds})
sub_path = OUTPUT_DIR / "submission_baseline.csv"
submission.to_csv(sub_path, index=False)
log.info(f"Submission saved to {sub_path}")
log.info(f"Prediction stats: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")
