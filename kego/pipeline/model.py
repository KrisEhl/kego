"""Model protocol and registry.

Concrete implementations (CatBoost, LightGBM, XGBoost, neural nets) live in
``kego.models.wrappers`` and the ``kego.models.neural`` package; this module
defines the uniform interface the trainer drives and the name -> class registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from kego.pipeline.config import ModelConfig


@runtime_checkable
class Model(Protocol):
    """Uniform fit/predict surface. The CV loop lives in the trainer, not here."""

    def __init__(self, config: ModelConfig, seed: int) -> None: ...

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        x_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        cat_features: list[str] | None = None,
    ) -> None: ...

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray: ...

    def feature_importance(self) -> dict[str, float] | None: ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> Model: ...


_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    def _wrap(cls: type) -> type:
        _REGISTRY[name] = cls
        return cls

    return _wrap


def get_model_class(name: str) -> type[Model]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
