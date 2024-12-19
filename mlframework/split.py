import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    df: pd.DataFrame,
    train_size: float | None = 0.6,
    validate_size: float | None = 0.2,
    test_size: float | None = None,
    shuffle: bool = True,
    random_state: int = 42,
    stratify_column: str | None = None,
):
    if validate_size is None and test_size is not None and train_size is not None:
        validate_size = 1.0 - test_size - train_size
    elif test_size is None and validate_size is not None and train_size is not None:
        test_size = 1.0 - validate_size - train_size
    elif train_size is None and validate_size is not None and test_size is not None:
        train_size = 1.0 - test_size - validate_size
    if validate_size is None or test_size is None or train_size is None:
        raise ValueError(
            f"Only one of {validate_size=} or {test_size=} or {train_size=} can be None!"
        )
    if validate_size + test_size + train_size != 1:
        raise ValueError(
            f"{validate_size=} + {test_size=} + {train_size=} != {validate_size + test_size + train_size}"
        )

    _train_size, _validate_size, _test_size = (
        round(train_size, 6),
        round(validate_size, 6),
        round(test_size, 6),
    )
    if stratify_column is not None:
        stratify = df[stratify_column]
    train, validate_test = train_test_split(
        df,
        train_size=_train_size,
        test_size=round(1 - _train_size, 6),
        random_state=random_state,
        stratify=stratify,
        shuffle=shuffle,
    )
    if _test_size != 0:
        _validate_size = _validate_size / (_validate_size + _test_size)
        _test_size = 1 - _validate_size
        _validate_size, _test_size = round(_validate_size, 6), round(_test_size, 6)
        if stratify_column is not None:
            stratify = validate_test[stratify_column]
        validate, test = train_test_split(
            validate_test,
            train_size=_validate_size,
            test_size=_test_size,
            random_state=random_state,
            stratify=stratify,
            shuffle=shuffle,
        )
    else:
        validate = validate_test
        test = None
    print(
        f"Split as train: {train.shape[0]} ({train_size*100}%), "
        f"validate: {validate.shape[0]} ({validate_size*100}%), "
        f"test: {test.shape[0] if test is not None else None} ({test_size*100 if test_size is not None else None}%)"
    )
    return train, validate, test


def flatten(l):
    return [item for sublist in l for item in sublist]


def build_xy(
    *dfs,
    y_columns: list[str] | str,
    xy_columns: list[str] | None = None,
    drop_columns: list[str] | None = None,
):
    XX = []
    yy = []
    if isinstance(y_columns, str):
        y_columns = [y_columns]
    if drop_columns is not None and y_columns is not None:
        drop_columns = list(drop_columns) + y_columns
    else:
        drop_columns = y_columns
    for df in dfs:
        df = df.copy()
        y = df[y_columns].values.ravel()
        X = df.drop(columns=drop_columns)
        if xy_columns is not None:
            for column in xy_columns:
                y[column] = X[column]

        XX.append(X)
        yy.append(y)
    return flatten([(X, y) for X, y in zip(XX, yy)])
