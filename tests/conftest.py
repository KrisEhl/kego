import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root (contains kego.toml and .git)."""
    return Path(__file__).parent.parent


# --- MLflow schema migration runs ONCE per session, not per test ----------------
# Connecting MLflow to a fresh sqlite DB runs the full alembic migration chain
# (~1-2s). We migrate one template DB once, then copy it (a file copy, ~ms) for
# every test that needs a fresh MLflow store — in-process and for e2e subprocesses.

_TEMPLATE_DB: Path | None = None


@pytest.fixture(scope="session", autouse=True)
def _migrated_mlflow_template(tmp_path_factory):
    global _TEMPLATE_DB
    import mlflow
    from mlflow.tracking import MlflowClient

    db = tmp_path_factory.mktemp("mlflow_template") / "template.db"
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    MlflowClient().search_experiments()  # triggers the full alembic migration once
    mlflow.set_tracking_uri("")
    _TEMPLATE_DB = db
    yield db


def seed_mlflow_db(sqlite_uri: str) -> None:
    """Copy the pre-migrated schema to a sqlite:/// path so it skips alembic."""
    if _TEMPLATE_DB is None or not sqlite_uri.startswith("sqlite:///"):
        return
    dest = Path(sqlite_uri[len("sqlite:///") :])
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy(_TEMPLATE_DB, dest)


@pytest.fixture
def mlflow_db(tmp_path, monkeypatch, _migrated_mlflow_template):
    """Isolated, pre-migrated SQLite MLflow store; resets global tracking URI after."""
    import mlflow

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    seed_mlflow_db(uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri("")


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
