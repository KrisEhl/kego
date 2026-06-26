"""Prediction store — the seam that makes experimentation fast.

Base learners are expensive (GPU, Ray); ensembling is cheap. By persisting
every learner's out-of-fold / holdout / test predictions keyed by fingerprint,
we can re-ensemble, re-evaluate and re-submit *without retraining*, and skip any
learner that has already been computed.

Caching is **implicit**: the trainer always consults the store before training,
so the user never has to pass a ``--resume`` flag. A local parquet cache sits in
front of MLflow (the source of truth) for fast fingerprint lookups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Predictions:
    """Everything one learner produces. The unit stored and ensembled."""

    fingerprint: str
    model_name: str
    feature_set: str
    seed: int
    oof: np.ndarray  # aligned to the train rows
    holdout: np.ndarray
    test: np.ndarray
    fold_scores: list[float] = field(default_factory=list)
    feature_importance: dict[str, float] | None = None

    @property
    def cv_mean(self) -> float:
        return float(np.mean(self.fold_scores)) if self.fold_scores else float("nan")


@dataclass(frozen=True)
class RunMeta:
    """Bookkeeping attached when a prediction set is saved."""

    fingerprint: str
    experiment: str
    tag: str = ""
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class PredictionStore(Protocol):
    """Persist and retrieve learner predictions by fingerprint."""

    def has(self, fingerprint: str) -> bool: ...

    def load(self, fingerprint: str) -> Predictions | None: ...

    def save(self, predictions: Predictions, meta: RunMeta) -> None: ...

    def query(
        self,
        *,
        experiments: list[str] | None = None,
        ensemble_tag: str | None = None,
    ) -> list[Predictions]:
        """Fetch sets of predictions for re-ensembling (``--from-experiment``)."""
        ...


class LocalCacheStore:
    """Fast on-disk cache keyed by fingerprint (npz/parquet under ``root``)."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(".kego_cache")

    def has(self, fingerprint: str) -> bool:
        raise NotImplementedError

    def load(self, fingerprint: str) -> Predictions | None:
        raise NotImplementedError

    def save(self, predictions: Predictions, meta: RunMeta) -> None:
        raise NotImplementedError

    def query(self, **kwargs: Any) -> list[Predictions]:
        raise NotImplementedError


class MlflowPredictionStore:
    """Source-of-truth store backed by MLflow runs + prediction artifacts.

    ``mlflow`` is imported lazily so the package stays importable without it.
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        self.tracking_uri = tracking_uri

    def has(self, fingerprint: str) -> bool:
        raise NotImplementedError

    def load(self, fingerprint: str) -> Predictions | None:
        raise NotImplementedError

    def save(self, predictions: Predictions, meta: RunMeta) -> None:
        raise NotImplementedError

    def query(self, **kwargs: Any) -> list[Predictions]:
        raise NotImplementedError


class CachingPredictionStore:
    """Local cache in front of a remote (MLflow) store.

    ``has``/``load`` check the local cache first, falling back to remote and
    warming the local cache on a remote hit. ``save`` writes through to both.
    This is what the trainer uses, so caching is automatic.
    """

    def __init__(self, local: LocalCacheStore, remote: MlflowPredictionStore) -> None:
        self.local = local
        self.remote = remote

    def has(self, fingerprint: str) -> bool:
        return self.local.has(fingerprint) or self.remote.has(fingerprint)

    def load(self, fingerprint: str) -> Predictions | None:
        hit = self.local.load(fingerprint)
        if hit is not None:
            return hit
        remote = self.remote.load(fingerprint)
        if remote is not None:
            self.local.save(remote, RunMeta(fingerprint=fingerprint, experiment="_cache"))
        return remote

    def save(self, predictions: Predictions, meta: RunMeta) -> None:
        self.local.save(predictions, meta)
        self.remote.save(predictions, meta)

    def query(self, **kwargs: Any) -> list[Predictions]:
        return self.remote.query(**kwargs)
