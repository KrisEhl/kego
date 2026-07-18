"""Competition-specific extension point.

Everything that differs between Kaggle competitions — data loading, target
encoding, feature engineering, the metric, submission formatting — lives behind
the :class:`Task` protocol. The generic pipeline (trainer, ensembler, tuner,
submitter) is written entirely against this interface, so adding a competition
means implementing one ``Task`` and registering it; no pipeline changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from kego.pipeline.config import MetricDirection


@dataclass
class RawData:
    """Loaded competition data, before preprocessing/feature engineering."""

    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame
    extra: dict[str, pd.DataFrame] = field(default_factory=dict)


@runtime_checkable
class Task(Protocol):
    """One Kaggle competition's specifics."""

    name: str
    kaggle_slug: str
    target: str
    id_col: str
    metric_direction: MetricDirection

    def load_raw(self) -> RawData:
        """Read raw files and map the target to a model-ready form."""
        ...

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputation + feature engineering. Applied to train/holdout/test alike."""
        ...

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Competition metric (e.g. ROC AUC)."""
        ...

    def make_submission(self, ids: np.ndarray, preds: np.ndarray) -> pd.DataFrame:
        """Format predictions into the competition's submission schema."""
        ...


_REGISTRY: dict[str, type] = {}


def register_task(name: str):
    """Class decorator: ``@register_task("rogii")`` registers a Task impl."""

    def _wrap(cls: type) -> type:
        _REGISTRY[name] = cls
        return cls

    return _wrap


def get_task(name: str) -> Task:
    """Instantiate a registered task by name."""
    if name not in _REGISTRY:
        import importlib.util
        import sys
        from pathlib import Path

        paths_to_try = [
            Path("competitions") / name / "task.py",
            Path("competitions") / name.replace("-", "_") / "task.py",
            Path("task.py"),
            Path.cwd() / "task.py",
        ]

        for path in paths_to_try:
            if path.exists():
                try:
                    spec = importlib.util.spec_from_file_location(f"dynamic_task_{name}", path)
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        sys.modules[f"dynamic_task_{name}"] = mod
                        spec.loader.exec_module(mod)
                        if name in _REGISTRY:
                            break
                except Exception as e:
                    print(f"Warning: failed to load dynamic task from {path}: {e}")

    if name not in _REGISTRY:
        if not name:
            raise KeyError(
                "No task configured (empty task name). Run from a competition directory "
                "containing kego.toml, pass --task <name>, or set 'task' in the config file. "
                f"Registered: {sorted(_REGISTRY)}"
            )
        raise KeyError(f"Unknown task {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()
