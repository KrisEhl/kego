"""Feature store and named feature sets.

``FeatureStore`` resolves where engineered features live (local path, shared
host) and loads them. ``FeatureSet``s are named column subsets declared in the
config so ablations are config-driven, not code changes.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


class FeatureStore:
    """Locate and load a cached engineered feature frame.

    ``user``/``hostname`` are resolved in the body (not as default args) so the
    module imports cleanly even when ``$HOSTNAME`` is unset in the environment.
    """

    def __init__(
        self,
        path: Path | None = None,
        user: str | None = None,
        hostname: str | None = None,
    ) -> None:
        self.path = path
        self.user = user or os.environ.get("USER")
        self.hostname = hostname or os.environ.get("HOSTNAME")

    def load(self) -> pd.DataFrame:
        raise NotImplementedError


class FeatureSets:
    """Registry mapping feature-set name -> list of columns (from config)."""

    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self._mapping = mapping

    def columns(self, name: str) -> list[str]:
        if name not in self._mapping:
            raise KeyError(f"Unknown feature set {name!r}. Known: {sorted(self._mapping)}")
        return self._mapping[name]

    def names(self) -> list[str]:
        return list(self._mapping)
