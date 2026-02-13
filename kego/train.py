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
    kwargs_fit: dict = {},
    folds_n=10,
    use_probability: bool = True,
    use_eval_set: bool = True,
    kfold_seed: int = 42,
    fold_preprocess=None,
):
    kf = KFold(n_splits=folds_n, shuffle=True, random_state=kfold_seed)

    oof_xgb = np.zeros(len(train))
    pred_xgb = np.zeros(len(test))
    holdout_xgb = np.zeros(len(holdout))
    model_trained = None
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

        if fold_preprocess is not None:
            x_train, x_valid, x_test, x_holdout = fold_preprocess(
                x_train, y_train, x_valid, x_test, x_holdout
            )

        model_trained = model(**kwargs_model)
        if use_eval_set:
            model_trained.fit(
                x_train, y_train, eval_set=[(x_valid, y_valid)], **kwargs_fit
            )
        else:
            model_trained.fit(x_train, y_train, **kwargs_fit)

        if use_probability:
            predict = lambda x: model_trained.predict_proba(x)[:, 1]
        else:
            predict = model_trained.predict

        # INFER OOF
        oof_xgb[test_index] = predict(x_valid)
        # INFER TEST
        pred_xgb += predict(x_test)
        holdout_xgb += predict(x_holdout)

    # COMPUTE AVERAGE TEST PREDS
    pred_xgb /= folds_n
    holdout_xgb /= folds_n
    return model_trained, oof_xgb, holdout_xgb, pred_xgb
