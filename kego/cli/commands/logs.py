import contextlib
import shlex
import subprocess
from pathlib import Path

from kego.fleet import load_fleet, machine_name
from kego.fleet import repo_root as find_repo_root
from kego.tracking import default_tracking_uri


def resolve_run_machine(task_name: str, run_id_prefix: str) -> tuple[str, str] | None:
    from mlflow.tracking import MlflowClient

    with contextlib.suppress(Exception):
        uri = default_tracking_uri()
        client = MlflowClient(tracking_uri=uri)

        # Try exact match first
        if len(run_id_prefix) == 32:
            with contextlib.suppress(Exception):
                run = client.get_run(run_id_prefix)
                machine = run.data.tags.get("machine")
                if machine:
                    return run_id_prefix, machine

        # Search runs in the experiment
        exp = client.get_experiment_by_name(task_name)
        if exp:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=200,
            )
            for run in runs:
                if run.info.run_id.startswith(run_id_prefix):
                    machine = run.data.tags.get("machine")
                    if machine:
                        return run.info.run_id, machine

        # Fallback: search all experiments
        runs = client.search_runs(
            experiment_ids=[e.experiment_id for e in client.search_experiments()],
            max_results=200,
        )
        for run in runs:
            if run.info.run_id.startswith(run_id_prefix):
                machine = run.data.tags.get("machine")
                if machine:
                    return run.info.run_id, machine
    return None


def run_logs(target_or_run_id: str | None, run_id_prefix: str | None, tail: bool, lines: int, task_name: str) -> int:
    # 1. Resolve target machine
    repo_root = find_repo_root()
    fleet_path = repo_root / "fleet.toml"

    target = target_or_run_id
    run_id = run_id_prefix

    if not target:
        target = "local"
        run_id = None
    else:
        # Check if the target is a valid machine name
        fleet_machines = []
        if fleet_path.exists():
            with contextlib.suppress(Exception):
                fleet_machines = [m.name for m in load_fleet(fleet_path).machines]

        is_machine = target.lower() in ("local", "self", "localhost") or target in fleet_machines

        if not is_machine:
            run_id = target
            target = None

    if not target:
        if not run_id:
            print("Error: no run id or machine target given.")
            return 1
        # Resolve machine from run_id via MLflow lookup
        print(f"Searching MLflow for run '{run_id}'...")
        resolved = resolve_run_machine(task_name, run_id)
        if not resolved:
            print(f"Error: Could not resolve machine for run ID prefix: {run_id}")
            return 1
        full_run_id, resolved_machine = resolved
        print(f"Found run {full_run_id} on machine {resolved_machine}")
        target = resolved_machine
        run_id = full_run_id

    is_local = target.lower() in ("local", "self", "localhost")
    if not is_local:
        with contextlib.suppress(Exception):
            if target == machine_name():
                is_local = True

    if is_local:
        log_dir = Path("~/.kego/logs").expanduser()
        if not log_dir.exists():
            print(f"Log directory {log_dir} does not exist locally.")
            return 1

        # Resolve target log file
        if run_id:
            matches = list(log_dir.glob(f"{run_id}*.log"))
            if not matches:
                print(f"No local log file matches prefix: {run_id}")
                return 1
            # Sort by modification time to get the newest match
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            log_path = matches[0]
        else:
            logs = list(log_dir.glob("*.log"))
            if not logs:
                print(f"No local log files found in {log_dir}")
                return 1
            logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            log_path = logs[0]

        print(f"Viewing local log: {log_path}")
        tail_cmd = ["tail", "-n", str(lines)]
        if tail:
            tail_cmd.append("-f")
        tail_cmd.append(str(log_path))
        try:
            res = subprocess.run(tail_cmd)
        except KeyboardInterrupt:
            return 0
        return res.returncode

    # Remote target
    if not fleet_path.exists():
        print(f"No fleet.toml at {fleet_path}; cannot view remote logs for {target}.")
        return 1
    try:
        machine = load_fleet(fleet_path).machine(target)
    except KeyError as e:
        print(f"Error: {e}")
        return 1

    # Find the correct log file on the remote machine
    # We run a remote shell script to find the log path and execute tail
    tail_flag = "-f" if tail else ""

    if run_id:
        # Match by prefix
        remote_script = f"""
        log_dir=~/.kego/logs
        log_path=$(ls -dt $log_dir/{shlex.quote(run_id)}*.log 2>/dev/null | head -n 1)
        if [ -z "$log_path" ]; then
            echo "Error: No remote log file matches prefix: {run_id}" >&2
            exit 1
        fi
        echo "Viewing log on {machine.name}: $log_path"
        tail -n {lines} {tail_flag} "$log_path"
        """
    else:
        # Latest log file
        remote_script = f"""
        log_dir=~/.kego/logs
        log_path=$(ls -dt $log_dir/*.log 2>/dev/null | head -n 1)
        if [ -z "$log_path" ]; then
            echo "Error: No remote log files found in $log_dir on {machine.name}" >&2
            exit 1
        fi
        echo "Viewing latest log on {machine.name}: $log_path"
        tail -n {lines} {tail_flag} "$log_path"
        """

    # Run the tail command over ssh. We want ssh to be interactive so we don't pass -f or -n.
    # We can pass -t if we want a pseudo-tty (helpful for tail -f, propagating signals/Ctrl+C).
    ssh_cmd = ["ssh"]
    if tail:
        ssh_cmd.append("-t")
    ssh_cmd += [machine.ssh, f"bash -lc {shlex.quote(remote_script)}"]

    try:
        res = subprocess.run(ssh_cmd)
    except KeyboardInterrupt:
        return 0
    return res.returncode
