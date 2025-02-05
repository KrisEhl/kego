from typing import Any

import numpy as np
import pandas as pd

from kego.columns import assert_missing_columns
from kego.dataframes import retain_df


def from_prediction_to_solution_submission(
    y_true: pd.DataFrame,
    y_prediction: pd.DataFrame | None = None,
    X_prediction: pd.DataFrame | None = None,
    row_id_column_name: str = "id",
    model=None,
    prediction_column: str = "prediction",
    target_column: str | None = None,
    required_columns: list[str] | None = None,
    dummy_column_name_value: dict[str, Any] | None = None,
):
    y_true, submission = create_submission(
        y_true=y_true,
        y_prediction=y_prediction,
        X_prediction=X_prediction,
        model=model,
        target_column=target_column,
    )
    solution, submission, X_prediction = (
        retain_df(y_true),
        retain_df(submission),
        retain_df(X_prediction),
    )
    if required_columns is not None:
        for column in required_columns:
            solution[column] = X_prediction[column]

    add_dummy_columns(dummy_column_name_value, submission, solution)

    submission = submission.rename(columns={target_column: prediction_column})
    assert len(solution) == len(submission)
    submission.insert(0, row_id_column_name, range(len(submission)))
    solution.insert(0, row_id_column_name, range(len(solution)))
    return solution, submission


def add_dummy_columns(dummy_column_name_value, submission, solution):
    if dummy_column_name_value is not None:
        for dummy_column_key, dummy_column_value in dummy_column_name_value.items():
            solution[dummy_column_key], submission[dummy_column_key] = (
                dummy_column_value,
                dummy_column_value,
            )


def create_submission(y_true, y_prediction, X_prediction, model, target_column):
    if y_prediction is None and X_prediction is None:
        raise ValueError(f"Need to provide {X_prediction} or {y_prediction}")
    if y_prediction is None and X_prediction is not None:
        if model is None or target_column is None:
            raise ValueError(
                f"Need to provide `model` and `target_column` when `X_prediction` given."
            )
        else:
            prediction_values = model.predict(X_prediction.copy())
            y_prediction = pd.DataFrame(data={target_column: prediction_values})
    elif y_prediction is not None:
        assert (
            target_column in y_prediction
        ), f"{target_column=} not found in `y_prediction`."
    if not isinstance(y_true, pd.DataFrame):
        y_true = pd.DataFrame(data={target_column: np.array(y_true)})
    return y_true, y_prediction
