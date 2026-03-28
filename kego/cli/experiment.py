from __future__ import annotations

import secrets
from pathlib import Path

_INFRA_PARAMS = frozenset({"debug", "gpu", "target", "folds", "fold"})


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
