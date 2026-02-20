import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def binary_labels():
    """200-sample random binary labels (deterministic)."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 2, size=200)


@pytest.fixture
def oof_matrix(binary_labels):
    """200x5 prediction matrix correlated with labels + noise."""
    rng = np.random.RandomState(42)
    n_models = 5
    n_samples = len(binary_labels)
    matrix = np.empty((n_samples, n_models))
    for i in range(n_models):
        noise = rng.normal(0, 0.3, size=n_samples)
        matrix[:, i] = np.clip(binary_labels + noise, 0, 1)
    return matrix


@pytest.fixture
def sample_train_df():
    """Small DataFrame with s6e2-like columns."""
    rng = np.random.RandomState(42)
    n = 100
    return pd.DataFrame(
        {
            "id": range(n),
            "Age": rng.randint(25, 80, n),
            "Sex": rng.randint(0, 2, n),
            "Chest pain type": rng.randint(1, 5, n),
            "BP": rng.randint(90, 200, n),
            "Cholesterol": rng.randint(100, 400, n),
            "FBS over 120": rng.randint(0, 2, n),
            "EKG results": rng.randint(0, 3, n),
            "Max HR": rng.randint(70, 210, n),
            "Exercise angina": rng.randint(0, 2, n),
            "ST depression": rng.uniform(0, 6, n).round(1),
            "Slope of ST": rng.randint(1, 4, n),
            "Number of vessels fluro": rng.randint(0, 4, n),
            "Thallium": rng.choice([3, 6, 7], n),
            "Heart Disease": rng.randint(0, 2, n),
        }
    )
