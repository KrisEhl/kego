import numpy as np
import pandas as pd
import pandas.api.types
import scipy.stats


class ParticipantVisibleError(Exception):
    pass


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str | None,
    naive_mean: float,
    naive_sigma: float,
    sigma_true: float,
) -> float:
    """
    This is a Gaussian Log Likelihood based metric. For a submission, which contains the predicted mean (x_hat) and variance (x_hat_std),
    we calculate the Gaussian Log-likelihood (GLL) value to the provided ground truth (x). We treat each pair of x_hat,
    x_hat_std as a 1D gaussian, meaning there will be 283 1D gaussian distributions, hence 283 values for each test spectrum,
    the GLL value for one spectrum is the sum of all of them.

    Inputs:
        - solution: Ground Truth spectra (from test set)
            - shape: (nsamples, n_wavelengths)
        - submission: Predicted spectra and errors (from participants)
            - shape: (nsamples, n_wavelengths*2)
        naive_mean: (float) mean from the train set.
        naive_sigma: (float) standard deviation from the train set.
        sigma_true: (float) essentially sets the scale of the outputs.
    """
    solution = solution.copy()
    submission = submission.copy()
    if row_id_column_name is not None:
        del solution[row_id_column_name]
        del submission[row_id_column_name]

    if submission.min().min() < 0:
        raise ParticipantVisibleError("Negative values in the submission")
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f"Submission column {col} must be a number")

    n_wavelengths = len(solution.columns)
    if len(submission.columns) != n_wavelengths * 2:
        raise ParticipantVisibleError("Wrong number of columns in the submission")

    y_pred = submission.iloc[:, :n_wavelengths].values
    # Set a non-zero minimum sigma pred to prevent division by zero errors.
    sigma_pred = np.clip(
        submission.iloc[:, n_wavelengths:].values, a_min=10**-15, a_max=None
    )
    y_true = solution.values

    GLL_pred = np.sum(scipy.stats.norm.logpdf(y_true, loc=y_pred, scale=sigma_pred))
    GLL_true = np.sum(
        scipy.stats.norm.logpdf(
            y_true, loc=y_true, scale=sigma_true * np.ones_like(y_true)
        )
    )
    GLL_mean = np.sum(
        scipy.stats.norm.logpdf(
            y_true,
            loc=naive_mean * np.ones_like(y_true),
            scale=naive_sigma * np.ones_like(y_true),
        )
    )

    submit_score = (GLL_pred - GLL_mean) / (GLL_true - GLL_mean)
    return float(np.clip(submit_score, 0.0, 1.0))


def compute_metrics(solution, submission_values, submission_sigma):
    submission = np.concat((submission_values, submission_sigma), axis=1)
    return score(
        solution=solution,
        submission=submission,
        row_id_column_name="planet_id",
        naive_mean=np.mean(submission_values),
        naive_sigma=np.std(submission_values),
        sigma_true=np.std(solution),
    )
