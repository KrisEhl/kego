"""Evaluation: scores, feature importance, fold stability.

Operates on stored predictions (and optional fitted-model handles), decoupled
from training so it can run after a cache-only, no-train pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kego.pipeline.config import EvaluatorConfig
from kego.pipeline.predictions import Predictions
from kego.pipeline.task import Task


@dataclass
class EvalReport:
    scores: dict[str, float] = field(default_factory=dict)
    importance: dict[str, dict[str, float]] = field(default_factory=dict)
    fold_stability: dict[str, float] = field(default_factory=dict)


class Evaluator:
    def __init__(self, config: EvaluatorConfig, task: Task) -> None:
        self.config = config
        self.task = task

    def evaluate(
        self,
        preds: list[Predictions],
        y_train: np.ndarray,
        y_holdout: np.ndarray | None = None,
    ) -> EvalReport:
        raise NotImplementedError
