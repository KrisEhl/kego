"""Ensembling: combine stored predictions into a final test prediction.

Strategies operate purely on the predictions (OOF matrix + holdout + test), so
ensembling is cheap and runs without retraining. ``AutoEnsembler`` tries several
methods and keeps the best by the configured selection metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from kego.pipeline.config import EnsembleConfig
from kego.pipeline.predictions import Predictions
from kego.pipeline.task import Task


@dataclass(frozen=True)
class EnsembleResult:
    method: str
    test: np.ndarray
    score: float
    weights: dict[str, float] | None = None
    per_method_scores: dict[str, float] | None = None


class EnsembleStrategy(Protocol):
    name: str

    def fit_predict(
        self,
        preds: list[Predictions],
        y_train: np.ndarray,
        y_holdout: np.ndarray | None,
        task: Task,
    ) -> EnsembleResult: ...


_REGISTRY: dict[str, type] = {}


def register_strategy(name: str):
    def _wrap(cls: type) -> type:
        _REGISTRY[name] = cls
        return cls

    return _wrap


def get_strategy(name: str) -> EnsembleStrategy:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown ensemble strategy {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


class AutoEnsembler:
    """Try each configured strategy; return the best by ``config.select_by``."""

    def __init__(self, config: EnsembleConfig) -> None:
        self.config = config

    def run(
        self,
        preds: list[Predictions],
        y_train: np.ndarray,
        y_holdout: np.ndarray | None,
        task: Task,
    ) -> EnsembleResult:
        raise NotImplementedError
