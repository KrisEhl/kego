"""Write-behind outbox for checkpoint registrations (fleet fabric).

A registry outage must not lose a trained model: when ``register_checkpoint`` fails,
the checkpoint file(s) and tags are snapshotted into ``~/.kego/outbox/`` and replayed
later with ``kego sync``. Files are copied, not referenced — training reuses output
paths (``outputs/mcts.pth``) across runs, so a queued path could otherwise upload a
different model's bytes by the time the sync runs.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path


def default_outbox() -> Path:
    return Path(os.environ.get("KEGO_OUTBOX") or Path.home() / ".kego" / "outbox")


def register_checkpoint_or_queue(
    uri: str,
    name: str,
    checkpoint_path: str,
    tags: dict,
    *,
    training_state_path: str | None = None,
    outbox_dir: str | Path | None = None,
) -> str | None:
    """``register_checkpoint``, but a failure queues the registration for ``kego sync``
    instead of raising. Returns the new version string, or ``None`` when queued (or,
    if even queueing fails, dropped — training must never break on telemetry)."""
    from kego.tracking.registry import register_checkpoint

    try:
        return register_checkpoint(uri, name, checkpoint_path, tags, training_state_path=training_state_path)
    except Exception:
        # inner suppress: even queueing can fail (e.g. disk full) — nothing left to do
        with contextlib.suppress(Exception):
            queue_registration(
                uri, name, checkpoint_path, tags, training_state_path=training_state_path, outbox_dir=outbox_dir
            )
        return None


def queue_registration(
    uri: str,
    name: str,
    checkpoint_path: str,
    tags: dict,
    *,
    training_state_path: str | None = None,
    outbox_dir: str | Path | None = None,
) -> Path:
    """Snapshot a registration (checkpoint bytes + tags) into the outbox; returns the entry dir."""
    outbox = Path(outbox_dir) if outbox_dir else default_outbox()
    entry_dir = outbox / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    entry_dir.mkdir(parents=True)
    shutil.copy2(checkpoint_path, entry_dir / Path(checkpoint_path).name)
    state_name = None
    if training_state_path and Path(training_state_path).exists():
        state_name = Path(training_state_path).name
        shutil.copy2(training_state_path, entry_dir / state_name)
    meta = {
        "uri": uri,  # informational; sync re-resolves the target so a misresolved URI can't stick
        "name": name,
        "checkpoint": Path(checkpoint_path).name,
        "training_state": state_name,
        "tags": {k: str(v) for k, v in tags.items()},
        "queued_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (entry_dir / "entry.json").write_text(json.dumps(meta, indent=2))
    return entry_dir


def pending_registrations(outbox_dir: str | Path | None = None) -> list[Path]:
    """Queued entry dirs, oldest first (dir names start with a timestamp)."""
    outbox = Path(outbox_dir) if outbox_dir else default_outbox()
    if not outbox.is_dir():
        return []
    return sorted(d for d in outbox.iterdir() if (d / "entry.json").exists())


def pending_for(name: str, outbox_dir: str | Path | None = None) -> list[Path]:
    """Queued entry dirs whose registration targets model ``name``."""
    entries = []
    for entry_dir in pending_registrations(outbox_dir):
        with contextlib.suppress(Exception):
            if json.loads((entry_dir / "entry.json").read_text()).get("name") == name:
                entries.append(entry_dir)
    return entries


def sync_outbox(
    uri: str | None = None, outbox_dir: str | Path | None = None
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Replay queued registrations to ``uri`` (default: the resolved hub).

    Entries are deleted on success, kept on failure. Returns
    ``([(entry, model_name, version), ...], [error, ...])``.
    """
    from kego.tracking.registry import register_checkpoint
    from kego.tracking.resolve import default_tracking_uri

    target = uri or default_tracking_uri()
    synced: list[tuple[str, str, str]] = []
    errors: list[str] = []
    for entry_dir in pending_registrations(outbox_dir):
        meta = json.loads((entry_dir / "entry.json").read_text())
        state = str(entry_dir / meta["training_state"]) if meta.get("training_state") else None
        try:
            version = register_checkpoint(
                target, meta["name"], str(entry_dir / meta["checkpoint"]), meta["tags"], training_state_path=state
            )
            shutil.rmtree(entry_dir)
            synced.append((entry_dir.name, meta["name"], version))
        except Exception as exc:
            errors.append(f"{entry_dir.name}: {exc}")
    return synced, errors
