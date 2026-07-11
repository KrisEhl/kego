"""kego pipeline — config-driven training / ensembling / submission for Kaggle.

Architecture (data flows left to right; the prediction store is the seam that
makes re-ensembling cheap and caching automatic)::

    FeatureStore -> Trainer -> PredictionStore -> Ensembler -> Submitter
                  (model x fold x seed)   ^               |
                                          +---------------+
                                  re-ensemble without retraining

Competition specifics live behind :class:`~kego.pipeline.task.Task`; execution
behind :class:`~kego.pipeline.executor.Executor`; persistence behind
:class:`~kego.pipeline.predictions.PredictionStore`.
"""

from __future__ import annotations

import importlib
from typing import Any

from kego.pipeline.config import (
    EnsembleConfig,
    EvaluatorConfig,
    FoldScheme,
    GridConfig,
    LearnerSpec,
    ModelConfig,
    PipelineConfig,
    SubmitConfig,
    expand_grid,
    load_config,
)

# Map of public API names to their submodules for lazy loading.
_SUBMODULES = {
    "AutoEnsembler": "ensemble",
    "EnsembleResult": "ensemble",
    "Evaluator": "evaluate",
    "Executor": "executor",
    "RayExecutor": "executor",
    "SerialExecutor": "executor",
    "get_executor": "executor",
    "FeatureSets": "features",
    "FeatureStore": "features",
    "CachingPredictionStore": "predictions",
    "LocalCacheStore": "predictions",
    "MlflowPredictionStore": "predictions",
    "Predictions": "predictions",
    "PredictionStore": "predictions",
    "Pipeline": "runner",
    "RunOutcome": "runner",
    "Submitter": "submit",
    "RawData": "task",
    "Task": "task",
    "get_task": "task",
    "register_task": "task",
    "TrainContext": "train",
    "Trainer": "train",
    "Tuner": "tune",
    "HPSpace": "tune",
}

__all__ = [
    "AutoEnsembler",
    "CachingPredictionStore",
    "EnsembleConfig",
    "EnsembleResult",
    "Evaluator",
    "EvaluatorConfig",
    "Executor",
    "FeatureSets",
    "FeatureStore",
    "FoldScheme",
    "GridConfig",
    "HPSpace",
    "LearnerSpec",
    "LocalCacheStore",
    "MlflowPredictionStore",
    "ModelConfig",
    "Pipeline",
    "PipelineConfig",
    "PredictionStore",
    "Predictions",
    "RawData",
    "RayExecutor",
    "RunOutcome",
    "SerialExecutor",
    "SubmitConfig",
    "Submitter",
    "Task",
    "TrainContext",
    "Trainer",
    "Tuner",
    "expand_grid",
    "get_executor",
    "get_task",
    "load_config",
    "register_task",
]


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        submodule = _SUBMODULES[name]
        module = importlib.import_module(f"kego.pipeline.{submodule}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
