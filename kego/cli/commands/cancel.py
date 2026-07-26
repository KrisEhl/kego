"""``kego cancel`` — terminate an active local, Ray, or fleet run.

Mirrors ``kego status``'s three sources (local PID files under
``.kego/active_runs``, Ray jobs, fleet processes) and ``kego logs``'s target
resolution (machine name vs. MLflow run ID prefix).
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from kego.cli.commands.logs import resolve_run_machine
from kego.fleet import Machine, load_fleet, machine_name
from kego.fleet import repo_root as find_repo_root
from kego.pipeline.runner import list_active_ray_jobs, stop_ray_job

_REMOTE_PROCESS_PATTERN = "train-agent|train_agent|kego league|run_league"
_KILL_GRACE_SECONDS = 10.0
_KILL_POLL_SECONDS = 0.3


def _terminate_pid(pid: int, *, grace: float = _KILL_GRACE_SECONDS) -> bool:
    """SIGTERM ``pid`` — its whole process group when it's the group leader, matching
    Ctrl+C's blast radius so multiprocessing rollout workers go down too — escalating
    to SIGKILL if it outlives ``grace`` seconds. Returns whether it was running at all."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False

    is_leader = False
    with contextlib.suppress(OSError):
        is_leader = os.getpgid(pid) == pid

    def _send(sig: int) -> None:
        with contextlib.suppress(OSError):
            os.killpg(pid, sig) if is_leader else os.kill(pid, sig)

    _send(signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        time.sleep(_KILL_POLL_SECONDS)
    _send(signal.SIGKILL)
    return True


def cancel_local() -> int:
    """Kill every locally active run tracked under ``.kego/active_runs``."""
    active_runs_dir = Path(".kego/active_runs")
    if not active_runs_dir.exists():
        return 0
    killed = 0
    for run_file in active_runs_dir.glob("*.json"):
        with contextlib.suppress(Exception):
            data = json.loads(run_file.read_text())
            pid = data.get("pid")
            if isinstance(pid, int) and _terminate_pid(pid):
                task = data.get("task", "unknown")
                config = data.get("config", "unknown")
                print(f"Killed local run [PID {pid}] task={task} config={config}")
                killed += 1
        run_file.unlink(missing_ok=True)
    return killed


def cancel_ray(job_id: str | None) -> int:
    """Stop active Ray job(s). With no ``job_id`` and more than one active job, lists
    them and asks the caller to disambiguate rather than guessing which to kill."""
    active = list_active_ray_jobs()
    if job_id:
        active = [j for j in active if j.job_id == job_id]
    if not active:
        return 0
    if not job_id and len(active) > 1:
        print("Multiple active Ray jobs; specify one: `kego cancel ray <job_id>`")
        for j in active:
            print(f"  {j.job_id}  {j.entrypoint}")
        return 0

    stopped = 0
    for j in active:
        if stop_ray_job(j.job_id):
            print(f"Stopped Ray job [{j.job_id}] {j.entrypoint}")
            stopped += 1
    return stopped


def _remote_cancel_script() -> str:
    return f"""
if [ "$(uname)" = "Darwin" ]; then
    PS_CMD="ps -eo pid,command"
else
    PS_CMD="ps -eo pid,cmd"
fi
$PS_CMD 2>/dev/null | grep -E '{_REMOTE_PROCESS_PATTERN}' | grep -v -E 'grep|ssh' | while read -r pid rest; do
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    if [ -n "$pgid" ] && [ "$pgid" = "$pid" ]; then
        kill -TERM -"$pid" 2>/dev/null && echo "Killed PID $pid (group): $rest"
    else
        kill -TERM "$pid" 2>/dev/null && echo "Killed PID $pid: $rest"
    fi
done
"""


def cancel_remote(machine: Machine, runner: Callable[..., object] = subprocess.run) -> int:
    """SIGTERM every ``train-agent``/``league`` process found on a fleet machine."""
    is_local = False
    with contextlib.suppress(Exception):
        is_local = machine.name == machine_name()

    script = _remote_cancel_script()
    cmd = ["bash", "-lc", script] if is_local else ["ssh", machine.ssh, f"bash -lc {shlex.quote(script)}"]
    try:
        res = runner(cmd, capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        print(f"Timed out reaching {machine.name}.")
        return 0

    killed = 0
    for line in getattr(res, "stdout", "").splitlines():
        if line.strip():
            print(f"[{machine.name}] {line.strip()}")
            killed += 1
    stderr = getattr(res, "stderr", "")
    if getattr(res, "returncode", 0) != 0 and stderr.strip():
        print(f"[{machine.name}] {stderr.strip()}")
    return killed


def cancel_run(target_or_run_id: str | None, job_id: str | None, task_name: str) -> int:
    if target_or_run_id and target_or_run_id.lower() == "ray":
        stopped = cancel_ray(job_id)
        if not stopped:
            print("No active Ray job found." if job_id else "No active Ray jobs found.")
        return 0

    if not target_or_run_id:
        if cancel_local():
            return 0
        if cancel_ray(None):
            return 0
        print("No active local or Ray runs found. To cancel on a fleet machine, run `kego cancel <machine>`.")
        return 0

    repo_root = find_repo_root()
    fleet_path = repo_root / "fleet.toml"
    target = target_or_run_id

    fleet_machines: list[str] = []
    if fleet_path.exists():
        with contextlib.suppress(Exception):
            fleet_machines = [m.name for m in load_fleet(fleet_path).machines]

    is_machine = target.lower() in ("local", "self", "localhost") or target in fleet_machines

    if not is_machine:
        print(f"Searching MLflow for run '{target}'...")
        resolved = resolve_run_machine(task_name, target)
        if not resolved:
            print(f"Error: Could not resolve machine for run ID prefix: {target}")
            return 1
        full_run_id, resolved_machine = resolved
        print(f"Found run {full_run_id} on machine {resolved_machine}")
        target = resolved_machine

    if target.lower() in ("local", "self", "localhost"):
        if not cancel_local():
            print("No active local run found.")
        return 0

    if not fleet_path.exists():
        print(f"No fleet.toml at {fleet_path}; cannot cancel on {target}.")
        return 1
    try:
        machine = load_fleet(fleet_path).machine(target)
    except KeyError as e:
        print(f"Error: {e}")
        return 1

    if not cancel_remote(machine):
        print(f"No matching active run found on {machine.name}.")
    return 0
