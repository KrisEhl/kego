"""Tracking-URI resolution for the fleet fabric: prefer a reachable central/hub MLflow,
else fall back to a local sqlite store (reconciled later with ``kego sync``)."""

from __future__ import annotations

from collections.abc import Callable


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
