"""Trainer: turn the learner grid into stored predictions.

The grid (model x feature_set x fold x seed) is expanded into
:class:`LearnerSpec`s. For each, the trainer consults the
:class:`~kego.pipeline.predictions.PredictionStore` *first* — any learner whose
fingerprint is already present is loaded, not retrained. Only the misses are
dispatched to the :class:`~kego.pipeline.executor.Executor`. This is the
implicit caching: the user never asks to resume; identical learners are simply
never recomputed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from kego.pipeline.config import LearnerSpec
from kego.pipeline.executor import Executor
from kego.pipeline.predictions import Predictions, PredictionStore, RunMeta
from kego.pipeline.task import RawData, Task

logger = logging.getLogger(__name__)


@dataclass
class TrainContext:
    """Frozen data + metadata handed to each (possibly remote) training task."""

    task: Task
    data: RawData
    feature_columns: dict[str, list[str]]
    experiment: str
    tag: str = ""


class Trainer:
    def __init__(
        self,
        task: Task,
        store: PredictionStore,
        executor: Executor,
        *,
        force: bool = False,
    ) -> None:
        self.task = task
        self.store = store
        self.executor = executor
        self.force = force

    def train_grid(self, specs: list[LearnerSpec], ctx: TrainContext) -> list[Predictions]:
        """Train (or load from cache) every spec; return predictions in order."""
        todo = [s for s in specs if self.force or not self.store.has(s.fingerprint)]
        logger.info("%d cache hits, %d to train", len(specs) - len(todo), len(todo))

        trained = self.executor.map(lambda s: self._train_one(s, ctx), todo)
        for spec, preds in zip(todo, trained, strict=False):
            self.store.save(
                preds,
                RunMeta(fingerprint=spec.fingerprint, experiment=ctx.experiment, tag=ctx.tag),
            )

        out: list[Predictions] = []
        for spec in specs:
            loaded = self.store.load(spec.fingerprint)
            if loaded is None:
                raise RuntimeError(f"prediction missing after train: {spec.label}")
            out.append(loaded)
        return out

    def _train_one(self, spec: LearnerSpec, ctx: TrainContext) -> Predictions:
        """Run k-fold CV for a single learner -> OOF/holdout/test predictions.

        Built here (not on Model) so models stay simple fit/predict objects and
        the CV strategy is shared across all of them.
        """
        raise NotImplementedError
