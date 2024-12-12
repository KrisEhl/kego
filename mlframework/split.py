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

    train_size, validate_size, test_size = (
        round(train_size, 6),
        round(validate_size, 6),
        round(test_size, 6),
    )
    if stratify_column is not None:
        stratify = df[stratify_column]
    train, validate_test = train_test_split(
        df,
        train_size=train_size,
        test_size=test_size + validate_size,
        random_state=random_state,
        stratify=stratify,
        shuffle=shuffle,
    )
    if test_size != 0:
        validate_size, test_size = validate_size / (
            validate_size + test_size
        ), test_size / (validate_size + test_size)
        validate_size, test_size = round(validate_size, 6), round(test_size, 6)
        if stratify_column is not None:
            stratify = validate_test[stratify_column]
        validate, test = train_test_split(
            validate_test,
            train_size=validate_size,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
            shuffle=shuffle,
        )
    else:
        validate = validate_test
        test = None
    return train, validate, test
