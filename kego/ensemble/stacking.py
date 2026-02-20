import numpy as np
from sklearn.model_selection import StratifiedKFold


def l2_stacking(
    oof_matrix,
    holdout_matrix,
    test_matrix,
    train_labels,
    train_features=None,
    holdout_features=None,
    test_features=None,
    n_splits=5,
    seed=42,
):
    """L2 stacking: train LightGBM meta-model with K-fold CV on L1 predictions."""
    import lightgbm as lgbm
    from lightgbm import LGBMClassifier

    if train_features is not None:
        X_train = np.hstack([oof_matrix, train_features])
        X_holdout = np.hstack([holdout_matrix, holdout_features])
        X_test = np.hstack([test_matrix, test_features])
    else:
        X_train = oof_matrix
        X_holdout = holdout_matrix
        X_test = test_matrix

    l2_oof = np.zeros(len(X_train))
    l2_holdout = np.zeros(len(X_holdout))
    l2_test = np.zeros(len(X_test))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_idx, (in_idx, out_idx) in enumerate(skf.split(X_train, train_labels)):
        lgb = LGBMClassifier(
            n_estimators=500,
            num_leaves=15,
            learning_rate=0.05,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed + fold_idx,
            verbose=-1,
        )
        lgb.fit(
            X_train[in_idx],
            train_labels[in_idx],
            eval_set=[(X_train[out_idx], train_labels[out_idx])],
            callbacks=[lgbm.early_stopping(50, verbose=False)],
        )
        l2_oof[out_idx] = lgb.predict_proba(X_train[out_idx])[:, 1]
        l2_holdout += lgb.predict_proba(X_holdout)[:, 1] / n_splits
        l2_test += lgb.predict_proba(X_test)[:, 1] / n_splits

    return l2_oof, l2_holdout, l2_test
