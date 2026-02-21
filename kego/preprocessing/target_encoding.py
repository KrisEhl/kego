def make_te_preprocess(te_features, drop_original=False, loo_features=None):
    """Create a fold_preprocess callback that applies target encoding per CV fold.

    Args:
        te_features: Columns to target-encode.
        drop_original: If True, drop the raw categorical columns after adding
            TE versions. Use for models that can't handle categoricals natively
            (LogReg, ResNet) to fix the ordinal fallacy.
        loo_features: Columns to leave-one-out encode. Training set uses
            (group_sum - y_i) / (group_count - 1); val/test/holdout use
            simple group mean.
    """

    def preprocess(x_train, y_train, x_valid, x_test, x_holdout):
        for col in te_features:
            if col not in x_train.columns:
                continue
            means = y_train.groupby(x_train[col]).mean()
            global_mean = y_train.mean()
            for df in [x_train, x_valid, x_test, x_holdout]:
                df[f"{col}_te"] = df[col].map(means).fillna(global_mean)

        for col in loo_features or []:
            if col not in x_train.columns:
                continue
            group_sum = y_train.groupby(x_train[col]).transform("sum")
            group_count = y_train.groupby(x_train[col]).transform("count")
            global_mean = y_train.mean()
            # Training: leave-one-out mean
            x_train[f"{col}_loo"] = (group_sum - y_train) / (group_count - 1).clip(
                lower=1
            )
            # Val/test/holdout: simple group mean
            means = y_train.groupby(x_train[col]).mean()
            for df in [x_valid, x_test, x_holdout]:
                df[f"{col}_loo"] = df[col].map(means).fillna(global_mean)

        if drop_original:
            cols_to_drop = [c for c in te_features if c in x_train.columns]
            for df in [x_train, x_valid, x_test, x_holdout]:
                df.drop(columns=cols_to_drop, inplace=True)
        return x_train, x_valid, x_test, x_holdout

    return preprocess
