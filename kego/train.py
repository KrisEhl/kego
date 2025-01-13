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
