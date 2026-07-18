"""Stable compatibility identity for resumable model training."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def training_fingerprint(config: Mapping[str, Any], source_paths: Iterable[str | Path]) -> str:
    """Hash effective parameters plus the contents of relevant source/data files."""
    encoded_config = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
    digest = hashlib.sha256(len(encoded_config).to_bytes(8, "big") + encoded_config)
    for raw_path in source_paths:
        path = Path(raw_path)
        label = "/".join(path.parts[-4:]).encode()
        content = path.read_bytes()
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()[:24]
