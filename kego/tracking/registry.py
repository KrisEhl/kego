"""MLflow Model Registry helpers — the cross-fleet leaderboard (see the spec, §5.3)."""

from __future__ import annotations

import math


def leaderboard(uri: str, name: str, sort_by: str = "elo", desc: bool = True) -> list[dict]:
    """Return every registered version of ``name`` as ``{"version", **tags}`` dicts, sorted
    by the numeric ``sort_by`` tag (versions missing/unparsable rank last)."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    versions = client.search_model_versions(f"name='{name}'")
    rows = [{"version": str(v.version), **dict(v.tags)} for v in versions]

    def key(row: dict) -> float:
        try:
            return float(row[sort_by])
        except (KeyError, TypeError, ValueError):
            return -math.inf

    return sorted(rows, key=key, reverse=desc)
