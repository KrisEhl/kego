from __future__ import annotations

import secrets
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlflow.entities import Run
    from mlflow.tracking import MlflowClient

_INFRA_PARAMS = frozenset({"debug", "gpu", "target", "folds", "fold"})


def resolve_runs(
    query: str,
    client: MlflowClient,
    experiment_ids: list[str],
    max_results: int = 10,
) -> list[Run]:
    """Return MLflow runs matching query by kego_id prefix or run name substring.

    Runs two searches (MLflow doesn't support OR) and merges by run_id.
    Caller is responsible for setting mlflow.set_tracking_uri() first.
    """
    by_id = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string=f"tags.kego_id LIKE '{query}%'",
        max_results=max_results,
    )
    by_name = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string=f"tags.`mlflow.runName` LIKE '%{query}%'",
        max_results=max_results,
    )
    seen: set[str] = set()
    runs: list[Run] = []
    for r in [*by_id, *by_name]:
        if r.info.run_id not in seen:
            seen.add(r.info.run_id)
            runs.append(r)
    return runs


def generate_id() -> str:
    """Generate a 6-character hex experiment ID."""
    return secrets.token_hex(3)


def build_experiment_name(
    script: str,
    name: str | None,
    cli_params: dict[str, str],
) -> str:
    """Build a human-readable experiment name.

    If --name is given, use it directly. Otherwise derive from script stem
    plus up to 3 non-infrastructure CLI params.
    """
    if name:
        return name

    stem = Path(script).stem
    key_params = {k: v for k, v in cli_params.items() if k not in _INFRA_PARAMS}
    if not key_params:
        return stem

    suffix = "--".join(f"{k}={v}" for k, v in list(key_params.items())[:3])
    return f"{stem}--{suffix}"
