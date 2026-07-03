"""Tracking-URI resolution for the fleet fabric: prefer a reachable central/hub MLflow,
else fall back to a local sqlite store (reconciled later with ``kego sync``)."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path


def default_tracking_uri(fleet_path: str | Path | None = None) -> str:
    """The MLflow URI a CLI command should use: ``KEGO_MLFLOW``/``MLFLOW_TRACKING_URI`` env,
    else the fleet ``[hub].mlflow``, else the local offline sqlite store."""
    explicit = os.environ.get("KEGO_MLFLOW") or os.environ.get("MLFLOW_TRACKING_URI")
    if explicit:
        return explicit
    fp = Path(fleet_path) if fleet_path else Path.cwd() / "fleet.toml"
    if fp.exists():
        from kego.fleet import load_fleet

        return load_fleet(fp).hub.mlflow
    return f"sqlite:///{Path.home() / '.kego' / 'offline.db'}"


def resolve_tracking_uri(
    explicit: str | None,
    hub: str | None,
    reachable: Callable[[str], bool],
    offline: str,
) -> str:
    """Return the MLflow tracking URI to use.

    Prefer ``explicit`` (e.g. ``KEGO_MLFLOW``), then the fleet ``hub`` MLflow, using the
    first that ``reachable(uri)`` accepts; if none are reachable, fall back to the local
    ``sqlite:///<offline>`` store.
    """
    for uri in (explicit, hub):
        if uri and reachable(uri):
            return uri
    return f"sqlite:///{offline}"
