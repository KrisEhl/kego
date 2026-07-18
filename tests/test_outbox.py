"""Write-behind outbox: a registry outage must queue checkpoint registrations for
``kego sync`` instead of dropping them (the omarchyd large256 incident)."""

import json
from pathlib import Path

import pytest

from kego.tracking.outbox import (
    pending_registrations,
    queue_registration,
    register_checkpoint_or_queue,
    sync_outbox,
)


@pytest.fixture
def checkpoint(tmp_path):
    ckpt = tmp_path / "outputs" / "mcts.pth"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"weights-v1")
    return ckpt


def test_queue_snapshots_checkpoint_bytes(tmp_path, checkpoint):
    """The outbox must copy bytes at queue time: outputs/mcts.pth is overwritten by the
    next run, so a stored path would sync the wrong model."""
    outbox = tmp_path / "outbox"
    entry = queue_registration(
        "http://hub:5000", "some-task", str(checkpoint), {"gauntlet_avg": 63.75}, outbox_dir=outbox
    )
    checkpoint.write_bytes(b"weights-v2-different-run")

    assert (entry / "mcts.pth").read_bytes() == b"weights-v1"
    meta = json.loads((entry / "entry.json").read_text())
    assert meta["name"] == "some-task"
    assert meta["tags"] == {"gauntlet_avg": "63.75"}
    assert pending_registrations(outbox) == [entry]


def test_register_or_queue_returns_version_when_registry_up(tmp_path, checkpoint, monkeypatch):
    import kego.tracking.registry as registry

    monkeypatch.setattr(registry, "register_checkpoint", lambda *a, **k: "42")
    version = register_checkpoint_or_queue(
        "http://hub:5000", "some-task", str(checkpoint), {}, outbox_dir=tmp_path / "outbox"
    )
    assert version == "42"
    assert pending_registrations(tmp_path / "outbox") == []


def test_register_or_queue_queues_on_failure_then_syncs(tmp_path, checkpoint, monkeypatch):
    import kego.tracking.registry as registry

    def down(*a, **k):
        raise ConnectionError("hub unreachable")

    monkeypatch.setattr(registry, "register_checkpoint", down)
    outbox = tmp_path / "outbox"
    version = register_checkpoint_or_queue(
        "http://hub:5000",
        "some-task",
        str(checkpoint),
        {"epoch": 50},
        outbox_dir=outbox,
    )
    assert version is None
    assert len(pending_registrations(outbox)) == 1

    # Hub back up: sync replays with the snapshotted file and original tags, then clears.
    seen = {}

    def up(uri, name, ckpt_path, tags, *, training_state_path=None):
        seen.update(uri=uri, name=name, bytes=Path(ckpt_path).read_bytes(), tags=tags)
        return "7"

    monkeypatch.setattr(registry, "register_checkpoint", up)
    synced, errors = sync_outbox(uri="http://hub:5000", outbox_dir=outbox)
    assert errors == []
    assert [(s[1], s[2]) for s in synced] == [("some-task", "7")]
    assert seen["bytes"] == b"weights-v1"
    assert seen["tags"] == {"epoch": "50"}
    assert pending_registrations(outbox) == []


def test_sync_keeps_entry_on_failure(tmp_path, checkpoint, monkeypatch):
    import kego.tracking.registry as registry

    outbox = tmp_path / "outbox"
    queue_registration("http://hub:5000", "some-task", str(checkpoint), {}, outbox_dir=outbox)

    def still_down(*a, **k):
        raise ConnectionError("nope")

    monkeypatch.setattr(registry, "register_checkpoint", still_down)
    synced, errors = sync_outbox(uri="http://hub:5000", outbox_dir=outbox)
    assert synced == []
    assert len(errors) == 1
    assert len(pending_registrations(outbox)) == 1


def test_train_agent_blocks_on_unsynced_outbox(tmp_path, checkpoint, monkeypatch):
    """A new run must not start over un-synced checkpoints of the same task: resume reads
    the registry, so stale registry state silently forks training from the wrong parent."""
    import kego.tracking.registry as registry
    from kego.pipeline.config import PipelineConfig
    from kego.pipeline.runner import Pipeline
    from kego.pipeline.task import register_task

    @register_task("outbox-guard-comp")
    class GuardTask:
        name = "outbox-guard-comp"
        kaggle_slug = "outbox-guard-comp"
        target = "t"
        id_col = "id"
        metric_direction = "maximize"
        is_simulation = True

        def __init__(self):
            self.trained = False

        def train(self, config, epochs=None, output_path=None, **kwargs):
            self.trained = True

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KEGO_OUTBOX", str(tmp_path / "outbox"))
    monkeypatch.delenv("KEGO_IGNORE_OUTBOX", raising=False)
    queue_registration("http://hub:5000", "outbox-guard-comp", str(checkpoint), {}, outbox_dir=tmp_path / "outbox")

    task = GuardTask()
    pipeline = Pipeline(PipelineConfig(task="outbox-guard-comp"))
    pipeline.task = task

    # Registry still down: the auto-sync fails and training must refuse to start.
    def down(*a, **k):
        raise ConnectionError("hub unreachable")

    monkeypatch.setattr(registry, "register_checkpoint", down)
    with pytest.raises(RuntimeError, match="kego sync"):
        pipeline.train_agent(epochs=1)
    assert not task.trained

    # Escape hatch for deliberate offline training.
    monkeypatch.setenv("KEGO_IGNORE_OUTBOX", "1")
    pipeline.train_agent(epochs=1)
    assert task.trained

    # Registry back: the guard auto-syncs and training proceeds.
    monkeypatch.delenv("KEGO_IGNORE_OUTBOX")
    monkeypatch.setattr(registry, "register_checkpoint", lambda *a, **k: "9")
    task.trained = False
    pipeline.train_agent(epochs=1)
    assert task.trained
    assert pending_registrations(tmp_path / "outbox") == []


def test_pending_for_filters_by_model_name(tmp_path, checkpoint):
    outbox = tmp_path / "outbox"
    queue_registration("http://hub:5000", "task-a", str(checkpoint), {}, outbox_dir=outbox)
    queue_registration("http://hub:5000", "task-b", str(checkpoint), {}, outbox_dir=outbox)
    from kego.tracking.outbox import pending_for

    assert len(pending_for("task-a", outbox)) == 1
    assert len(pending_for("nope", outbox)) == 0


def test_queue_includes_training_state(tmp_path, checkpoint):
    state = checkpoint.parent / "mcts_iter50.train.pt"
    state.write_bytes(b"optimizer-state")
    entry = queue_registration(
        "http://hub:5000",
        "some-task",
        str(checkpoint),
        {},
        training_state_path=str(state),
        outbox_dir=tmp_path / "outbox",
    )
    assert (entry / "mcts_iter50.train.pt").read_bytes() == b"optimizer-state"
    assert json.loads((entry / "entry.json").read_text())["training_state"] == "mcts_iter50.train.pt"
