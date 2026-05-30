from pathlib import Path
from unittest.mock import patch

from kego.cli.commands.cluster import sync_repo
from kego.cli.config import ClusterConfig, KegoConfig


def _config(worker_hosts: list[str]) -> KegoConfig:
    return KegoConfig(
        cluster=ClusterConfig(
            ray_address="http://x:8265",
            mlflow_uri="http://x:5000",
            ssh_host="head",
            worker_hosts=worker_hosts,
        ),
        competition=None,
        repo_root=Path("/repo"),
        competition_dir=None,
    )


def test_sync_repo_pulls_head_and_every_worker():
    """A worker left un-synced runs stale code — sync must pull all nodes."""
    pulled: list[str] = []
    with patch("kego.cli.commands.cluster._ssh_run", side_effect=lambda host, cmd: pulled.append(host) or 0):
        rc = sync_repo(_config(["worker1", "worker2"]))
    assert rc == 0
    assert pulled == ["head", "worker1", "worker2"]


def test_sync_repo_propagates_failure_when_a_worker_fails():
    """If a worker pull fails, sync returns non-zero so the caller knows a node is stale."""
    with patch("kego.cli.commands.cluster._ssh_run", side_effect=lambda host, cmd: 1 if host == "worker1" else 0):
        rc = sync_repo(_config(["worker1"]))
    assert rc != 0


def test_sync_repo_head_only_when_no_workers():
    pulled: list[str] = []
    with patch("kego.cli.commands.cluster._ssh_run", side_effect=lambda host, cmd: pulled.append(host) or 0):
        sync_repo(_config([]))
    assert pulled == ["head"]
