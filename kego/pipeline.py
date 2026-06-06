import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


class FeatureStore:
    def __init__(
        self,
        path: Path | None = None,
        user: str | None = os.environ["USER"],
        hostname: str | None = os.environ["HOSTNAME"],
    ) -> None:
        pass

    def load(self) -> pl.DataFrame: ...


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hyper_params: dict[str, Any]


class EnsemblerConfig:
    name: str
    params: dict[str, Any]


class EvaluatorConfig:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class PipelineConfig:
    model_configs: list[ModelConfig]
    ensembler_config: EnsemblerConfig
    evaluator_config: EvaluatorConfig
    tune_hp: bool
    ensemble: bool


class Model:
    def __init__(self, config: ModelConfig) -> None: ...
    def train(self) -> None: ...
    def tune(self) -> None: ...
    def train_fold(self) -> None: ...
    def predict(self) -> np.ndarray: ...
    def predict_proba(self) -> np.ndarray: ...
    @classmethod
    def load(cls, path) -> "Model": ...
    def save(self, path=None) -> None: ...


def get_model_class(name: str) -> type[Model]: ...


class Ensembler:
    def __init__(self, config: EnsemblerConfig) -> None: ...
    def ensemble(self) -> None: ...


def get_ensembler_class(name: str) -> type[Ensembler]: ...


class Evaluator:
    def __init__(self, config: EvaluatorConfig) -> None: ...
    def eval_importance(self, model: Model | Ensembler) -> None: ...
    def eval_predictions(self, model: Model | Ensembler) -> None: ...


class Pipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.evaluator = Evaluator(config=self.config.evaluator_config)
        self.fs = FeatureStore()
        self.df = self.fs.load()
        self.train, self.holdout = ..., ...
        self.train_fold = ...

    def run(self) -> None:

        models_trained = []
        for model_config in self.config.model_configs:
            model_class = get_model_class(model_config.name)
            model = model_class(config=model_config)
            if self.config.tune_hp:
                model.tune()
            else:
                model.train()
            model.save()
            models_trained.append(model)
            self.evaluator.eval_importance(model=model)
            self.evaluator.eval_predictions(model=model)

        if self.config.ensemble:
            ensembler_class = get_ensembler_class(self.config.ensembler_config.name)
            ensembler = ensembler_class(config=self.config.ensembler_config)
            ensembler.ensemble()
            self.evaluator.eval_importance(ensembler)
            self.evaluator.eval_predictions(ensembler)
