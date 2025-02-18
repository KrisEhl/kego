import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


def train_model(
    model,
    X_train,
    y_train,
    ignore_columns: None | list[str] = None,
    feature_selector=None,
    feature_space=None,
):
    X_train = X_train.copy()
    y_train = y_train.copy()
    if ignore_columns is not None:
        X_train = X_train.drop(columns=ignore_columns)
    if feature_selector is not None:
        if feature_space is None:
            raise ValueError(f"Define `feature_space` when using {feature_selector=}.")
        model = feature_selector(model, feature_space)
    model = model.fit(X_train, y_train)
    return model


def train_model_split(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    holdout: pd.DataFrame,
    features: list[str],
    target: str,
    kwargs_model: dict = {},
    folds_n=10,
):
    FOLDS = 10
    kf = KFold(n_splits=folds_n, shuffle=True, random_state=42)

    oof_xgb = np.zeros(len(train))
    pred_xgb = np.zeros(len(test))
    holdout_xgb = np.zeros(len(holdout))

    for i, (train_index, test_index) in enumerate(kf.split(train)):

        logger.info("#" * 25)
        logger.info(f"### Fold {i+1}")
        logger.info("#" * 25)
        x_train = train.loc[train_index, features].copy()
        y_train = train.loc[train_index, target]
        x_valid = train.loc[test_index, features].copy()
        y_valid = train.loc[test_index, target]
        x_test = test[features].copy()
        x_holdout = holdout[features].copy()

        model_trained = model(
            **kwargs_model
            # early_stopping_rounds=25,
        )
        model_trained.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=500)

        # INFER OOF
        oof_xgb[test_index] = model_trained.predict(x_valid)
        # INFER TEST
        pred_xgb += model_trained.predict(x_test)
        holdout_xgb += model_trained.predict(x_holdout)

    # COMPUTE AVERAGE TEST PREDS
    pred_xgb /= FOLDS
    holdout_xgb /= FOLDS
    return model_trained, oof_xgb, holdout_xgb
