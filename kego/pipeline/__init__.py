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
from kego.pipeline.ensemble import AutoEnsembler, EnsembleResult
from kego.pipeline.evaluate import Evaluator
from kego.pipeline.executor import Executor, RayExecutor, SerialExecutor, get_executor
from kego.pipeline.features import FeatureSets, FeatureStore
from kego.pipeline.model import Model, get_model_class, register_model
from kego.pipeline.predictions import (
    CachingPredictionStore,
    LocalCacheStore,
    MlflowPredictionStore,
    Predictions,
    PredictionStore,
)
from kego.pipeline.runner import Pipeline, RunOutcome
from kego.pipeline.submit import Submitter
from kego.pipeline.task import RawData, Task, get_task, register_task
from kego.pipeline.train import TrainContext, Trainer
from kego.pipeline.tune import HPSpace, Tuner

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
    "Model",
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
    "get_model_class",
    "get_task",
    "load_config",
    "register_model",
    "register_task",
]
