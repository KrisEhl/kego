"""Fleet dispatcher: ship the local working tree to a machine and SSH-launch training
detached, logging to the central MLflow. See §5.4 of the fleet-training-fabric spec.

The command builders here are pure (easy to test); ``dispatch()`` wires them to a runner
(``subprocess.run`` by default, injectable for tests). No commit is required — this mirrors
the ship-local-code model the Ray path already uses.
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from kego.fleet import Machine

# Keep the upload light — same intent as the Ray working_dir excludes (runner.py): drop
# VCS/venv/data/caches. Other competitions are excluded dynamically (other_competition_excludes).
DEFAULT_EXCLUDES = [
    ".git",
    ".git/",
    ".venv",
    ".venv/",
    "__pycache__",
    "__pycache__/",
    "*.tar.gz",
    "data",
    "data/",
    "model_data",
    "model_data/",
    "outputs",
    "outputs/",
    "tmp",
    "tmp/",
    "mlruns",
    "mlruns/",
    # Explicitly exclude nested outputs inside competition dirs (e.g. large .pth checkpoints)
    "competitions/*/outputs/",
    "competitions/*/mlruns/",
]


def other_competition_excludes(repo_root: str | Path, keep: str) -> list[str]:
    """Exclude every ``competitions/<name>`` except the active task ``keep`` (cg/ is kept)."""
    comps = Path(repo_root) / "competitions"
    if not comps.is_dir():
        return []
    return [f"competitions/{p.name}" for p in sorted(comps.iterdir()) if p.is_dir() and p.name != keep]


def rsync_command(local_dir: str | Path, machine: Machine, excludes: Sequence[str]) -> list[str]:
    """rsync the *contents* of ``local_dir`` into ``machine.repo`` (trailing slashes), --delete."""
    cmd = ["rsync", "-az", "--delete"]
    cmd += [f"--exclude={e}" for e in excludes]
    cmd += [f"{local_dir}/", f"{machine.ssh}:{machine.repo}/"]
    return cmd


def remote_launch_command(machine: Machine, cmd_args: Sequence[str], run_id: str, log_dir: str = "~/.kego/logs") -> str:
    """A detached remote command: cd into the repo, pin the MLflow run, nohup the kego
    command into a per-run log. Progress is monitored via the MLflow dashboard or by
    tailing the remote log file."""
    task_dir = ""
    if "--task" in cmd_args:
        try:
            idx = list(cmd_args).index("--task")
            if idx + 1 < len(cmd_args):
                task_dir = f"/competitions/{cmd_args[idx + 1]}"
        except ValueError:
            pass

    kego_cmd = "uv run kego " + " ".join(shlex.quote(a) for a in cmd_args)
    log = f"{log_dir}/{run_id}.log"
    return (
        f"mkdir -p {log_dir} && cd {shlex.quote(machine.repo)}{task_dir} && "
        f"KEGO_MLFLOW_RUN_ID={run_id} nohup {kego_cmd} > {log} 2>&1 &"
    )


def ssh_command(machine: Machine, remote_cmd: str) -> list[str]:
    """Wrap a remote command in a login shell over SSH (login shell => uv/PATH available)."""
    return ["ssh", machine.ssh, f"bash -lc {shlex.quote(remote_cmd)}"]


def dispatch(
    machine: Machine,
    cmd_args: Sequence[str],
    run_id: str,
    local_dir: str | Path,
    excludes: Sequence[str],
    runner: Callable[..., object] = subprocess.run,
) -> None:
    """Ship ``local_dir`` to ``machine`` then SSH-launch ``kego <cmd_args>`` detached.

    Raises ``RuntimeError`` if the rsync ship fails (we don't launch against a stale tree).
    """
    ship = runner(rsync_command(local_dir, machine, excludes))
    if getattr(ship, "returncode", 0) != 0:
        raise RuntimeError(f"rsync to {machine.name} ({machine.ssh}) failed")
    remote = remote_launch_command(machine, cmd_args, run_id)
    runner(ssh_command(machine, remote))
