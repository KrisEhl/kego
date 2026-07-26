from kego.dispatch import DEFAULT_EXCLUDES, dispatch, other_competition_excludes
from kego.fleet import git_sha, load_fleet, machine_name
from kego.fleet import repo_root as find_repo_root
from kego.tracking import create_run, default_tracking_uri


def train_agent_overrides(args) -> dict:
    keys = [
        "agent",
        "init_checkpoint",
        "num_workers",
        "variant",
    ]
    return {k: getattr(args, k, None) for k in keys if getattr(args, k, None) is not None}


def dispatch_train_agent(task_name: str, target: str, epochs: int | None, output: str | None, overrides: dict) -> int:
    """Ship the local tree to fleet machine ``target`` and SSH-launch training there (§5.4)."""
    repo_root = find_repo_root()
    fleet_path = repo_root / "fleet.toml"
    if not fleet_path.exists():
        print(f"No fleet.toml at {fleet_path}; cannot dispatch --target {target}.")
        return 1
    try:
        machine = load_fleet(fleet_path).machine(target)
    except KeyError as e:
        print(f"Error: {e}")
        return 1

    uri = default_tracking_uri(fleet_path)
    tags = {
        "machine": machine.name,
        "task": task_name,
        "git_sha": git_sha(repo_root),
        "dispatched_from": machine_name(),
        "target": target,
    }
    run_id = create_run(uri, experiment=task_name, run_name=f"{machine.name}-{task_name}", tags=tags)
    if not run_id:
        print(
            f"Could not create an MLflow run at {uri} (hub unreachable). Offline dispatch is a later phase; aborting."
        )
        return 1

    cmd_args = ["train-agent", "--task", task_name]
    if epochs is not None:
        cmd_args += ["--epochs", str(epochs)]
    if output:
        cmd_args += ["--output", output]
    for key, flag in [
        ("agent", "--agent"),
        ("init_checkpoint", "--init-checkpoint"),
        ("num_workers", "--num-workers"),
        ("variant", "--variant"),
    ]:
        if key in overrides:
            cmd_args += [flag, str(overrides[key])]

    excludes = DEFAULT_EXCLUDES + other_competition_excludes(repo_root, keep=task_name)
    print(f"Dispatching {task_name} to {machine.name} ({machine.ssh}) — run {run_id}")
    try:
        dispatch(machine, cmd_args, run_id=run_id, local_dir=str(repo_root), excludes=excludes)
    except Exception as e:
        print(f"Dispatch failed: {e}")
        return 1
    print(f"Launched on {machine.name}. Track metrics in MLflow at: {uri}")
    print(f"To view remote logs, run:\n  kego logs {machine.name} {run_id}")
    return 0
