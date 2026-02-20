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
