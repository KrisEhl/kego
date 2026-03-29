"""kego cluster — manage the Ray cluster head node."""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.error
import urllib.request
from urllib.parse import urlparse

from kego.cli import config as cfg_module

_RAY_PORT = 6379
_RAY_CLIENT_PORT = 10001


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("cluster", help="Manage the Ray cluster")
    sub = p.add_subparsers(dest="cluster_cmd")
    sub.required = True

    sub.add_parser("start", help="Start Ray on the head node via SSH")
    sub.add_parser("stop", help="Stop Ray on the head node via SSH")
    sub.add_parser("status", help="Show cluster node and resource status")

    p.set_defaults(func=_cluster)


def _cluster(args: argparse.Namespace, extra_args: list[str]) -> int:
    config = cfg_module.load_config()

    if args.cluster_cmd == "start":
        return _start(config)
    if args.cluster_cmd == "stop":
        return _stop(config)
    if args.cluster_cmd == "status":
        return _status(config)
    return 1


def _ssh_run(ssh_host: str, cmd: str) -> int:
    result = subprocess.run(  # noqa: S603
        ["ssh", ssh_host, cmd],  # noqa: S607
        text=True,
    )
    return result.returncode


def sync_repo(config: cfg_module.KegoConfig) -> int:
    """Git pull on the cluster head node. Returns 0 on success."""
    if not config.cluster.ssh_host:
        print("Warning: ssh_host not set in kego.toml [cluster] — skipping repo sync")
        return 0
    print(f"Syncing repo on {config.cluster.ssh_host}...", flush=True)
    return _ssh_run(
        config.cluster.ssh_host, f"cd {config.cluster.repo_path} && git pull"
    )


def _start(config: cfg_module.KegoConfig) -> int:
    if not config.cluster.ssh_host:
        print("Error: ssh_host not set in kego.toml [cluster]")
        return 1

    parsed = urlparse(config.cluster.ray_address)
    head_ip = parsed.hostname or "127.0.0.1"
    dashboard_port = parsed.port or 8265
    uv_dir = config.cluster.uv_project_dir

    cmd = (
        f"cd {uv_dir} && "
        f"RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 "
        f"RAY_JOB_START_TIMEOUT_SECONDS=1800 "
        f"uv run ray start --head "
        f"--port={_RAY_PORT} "
        f"--node-ip-address {head_ip} "
        f"--dashboard-host=0.0.0.0 "
        f"--dashboard-port={dashboard_port} "
        f"--ray-client-server-port={_RAY_CLIENT_PORT} "
        f"--num-cpus=$(expr $(nproc --all) - 2) "
        f"--resources '{{\"heavy_gpu\": 1}}'"
    )

    rc = sync_repo(config)
    if rc != 0:
        return rc

    print(f"Starting Ray head on {config.cluster.ssh_host}...")
    return _ssh_run(config.cluster.ssh_host, cmd)


def _stop(config: cfg_module.KegoConfig) -> int:
    if not config.cluster.ssh_host:
        print("Error: ssh_host not set in kego.toml [cluster]")
        return 1

    uv_dir = config.cluster.uv_project_dir
    cmd = f"cd {uv_dir} && uv run ray stop --force"

    print(f"Stopping Ray on {config.cluster.ssh_host}...")
    return _ssh_run(config.cluster.ssh_host, cmd)


def _status(config: cfg_module.KegoConfig) -> int:
    base = config.cluster.ray_address.rstrip("/")

    try:
        req = urllib.request.Request(f"{base}/nodes?view=summary")  # noqa: S310
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            data = json.loads(resp.read())
    except urllib.error.URLError:
        print(f"Cluster unreachable at {base}")
        return 1

    nodes = data.get("data", {}).get("summary", [])
    alive = [n for n in nodes if n.get("raylet", {}).get("state") == "ALIVE"]

    print(f"Nodes: {len(alive)} alive / {len(nodes)} total")
    for node in alive:
        raylet = node.get("raylet", {})
        resources = raylet.get("resourcesTotal", {})
        hostname = node.get("hostname", "?")
        gpus = resources.get("GPU", 0)
        cpus = resources.get("CPU", 0)
        heavy = resources.get("heavy_gpu", 0)
        print(
            f"  {hostname:<20} CPU={cpus:.0f}  GPU={gpus:.0f}"
            + (f"  heavy_gpu={heavy:.0f}" if heavy else "")
        )

    return 0
