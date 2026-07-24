"""``kego`` command-line entry point.

Mirrors the usage sketched in the repo README::

    kego run   --config catboostv1 [--fast] [--params models.0.params.lr=0.01]
    kego tune  --config catboostv1 --tune catboost --trials 50
    kego ensemble --from-experiment expA expB
    kego submit   --from-ensemble my_best --message "cat+lgbm"

``--config`` names a YAML under the configs dir; ``--params`` are OmegaConf
dotlist overrides applied on top. Wire as a console script in pyproject:
``[project.scripts] kego = "kego.pipeline.cli:main"``.
"""

from __future__ import annotations

import argparse


def finite_nonnegative_float(raw: str) -> float:
    import math

    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kego")
    sub = parser.add_subparsers(dest="command", required=True)

    common_base = argparse.ArgumentParser(add_help=False)
    common_base.add_argument("--config", help="config name/path (YAML)")
    common_base.add_argument(
        "--executor", choices=["serial", "ray"], default="serial", help="execution backend (default: %(default)s)"
    )
    common_base.add_argument("--force", action="store_true", help="ignore cache; retrain everything")
    common_base.add_argument("--tag", default="", help="experiment tag")
    common_base.add_argument("--task", help="task name (competition slug)")

    common = argparse.ArgumentParser(add_help=False, parents=[common_base])
    common.add_argument("--params", nargs="*", default=[], help="OmegaConf config overrides")

    run = sub.add_parser("run", parents=[common], help="train grid (cached) + ensemble")
    run.add_argument("--fast", action="store_true")
    run.add_argument("--no-ensemble", dest="do_ensemble", action="store_false")
    run.add_argument("--submit", action="store_true")
    run.add_argument("--model", help="ad-hoc model name to run (e.g. catboost)")
    run.add_argument("--hp-tune", action="store_true", help="enable hyperparameter tuning")
    run.add_argument("--hp-params", nargs="*", default=[], help="hyperparameter search space specs")

    ens = sub.add_parser("ensemble", parents=[common], help="re-ensemble stored preds")
    ens.add_argument("--from-experiment", nargs="+")
    ens.add_argument("--from-ensemble")

    tune = sub.add_parser("tune", parents=[common], help="Optuna HP tuning")
    tune.add_argument("--tune", nargs="+", metavar="MODEL")
    tune.add_argument("--trials", type=int, default=50)

    submit = sub.add_parser("submit", parents=[common], help="submit to Kaggle")
    submit.add_argument("version", nargs="?", help="registry model version for simulation-agent submissions")
    submit.add_argument("--from-ensemble")
    submit.add_argument("--message", default="")

    sub.add_parser("status", parents=[common], help="check current training runs")

    cache = sub.add_parser("cache", parents=[common], help="manage cache")
    cache.add_argument("action", choices=["status", "prune"], help="cache action")

    sub.add_parser("submissions", parents=[common], help="list submissions")

    sync = sub.add_parser("sync", help="replay queued checkpoint registrations from the outbox to the hub")
    sync.add_argument("--uri", help="target MLflow URI (default: resolved hub from fleet.toml)")
    sync.add_argument("--list", action="store_true", help="show pending outbox entries without syncing")

    league = sub.add_parser("league", parents=[common], help="run a task-specific model league")
    league.add_argument(
        "--games", type=int, default=200, help="games per pairing in the round-robin (default: %(default)s)"
    )
    league.add_argument("--search-count", type=int, default=10, help="MCTS simulations per move (default: %(default)s)")
    league.add_argument("--workers", type=int, help="parallel game workers (default: CPU count - 2)")
    league.add_argument("--debug", action="store_true", help="show per-game engine output (default: suppressed)")
    league.add_argument(
        "--cache-dir",
        help="cache directory for downloaded registry checkpoints "
        "(default: competitions/<task>/outputs/cached_registry)",
    )
    league.add_argument(
        "--write-ratings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write Elo ratings back to the model registry (default: enabled)",
    )
    league.add_argument(
        "--include-local-mcts",
        action="store_true",
        help="also enter the local outputs/mcts.pth checkpoint as a participant",
    )
    league.add_argument(
        "--partial-save-every",
        type=int,
        default=5000,
        help="save partial league artifacts to MLflow every N games; 0 disables (default: %(default)s)",
    )
    league.add_argument(
        "--stall-timeout",
        type=finite_nonnegative_float,
        default=300,
        help="fail if no game completes for N seconds; 0 disables (default: %(default)s)",
    )
    league.add_argument("--target", help="fleet machine name to dispatch to (rsync + SSH-launch); omit locally")
    league_sub = league.add_subparsers(dest="league_cmd")

    league_matrix = league_sub.add_parser("matrix", parents=[common], help="show a saved task league matrix")
    league_matrix.add_argument("--run-id", help="MLflow run id to show; defaults to latest league run")

    league_merge = league_sub.add_parser("merge", parents=[common], help="merge saved league matrices")
    league_merge.add_argument(
        "--run-ids",
        nargs="*",
        default=[],
        help="MLflow league run ids to merge; defaults to finished unmerged league runs",
    )
    league_merge.add_argument(
        "--latest",
        type=int,
        help="limit automatic selection to the N latest finished unmerged league runs",
    )
    league_merge.add_argument(
        "--write-ratings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write merged Elo ratings back to the model registry (default: enabled)",
    )

    battle = sub.add_parser("battle", parents=[common], help="battle local pokemon agents against each other")
    battle.add_argument("--agent1", help="path to first agent python file")
    battle.add_argument("--agent2", help="path to second agent python file")
    battle.add_argument("--games", type=int, help="number of games to play")
    battle.add_argument("--deck1", help="optional path to deck CSV for agent 1")
    battle.add_argument("--deck2", help="optional path to deck CSV for agent 2")

    train_parser = sub.add_parser(
        "train-agent", parents=[common_base], help="run task-specific simulation agent/policy training"
    )
    train_parser.add_argument(
        "--agent", required=True, help="agent implementation to train (directory under agents/, e.g. mcts)"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        help="target total training iterations; automatically resumes a compatible shorter run",
    )
    train_parser.add_argument("--output", help="path to save the trained model/weights")
    train_parser.add_argument("--init-checkpoint", help="warm-start from a local .pth path or registry:<version>")
    train_parser.add_argument(
        "--num-workers",
        type=int,
        help="parallel rollout/eval workers (default: [train] num_workers in the competition's "
        "kego.toml, else CPU count - 2, capped at the per-phase game count)",
    )
    train_parser.add_argument(
        "--variant", required=True, help="name of the variant configuration under configs/variants/"
    )
    train_parser.add_argument(
        "--target", help="fleet machine name to dispatch to (rsync + SSH-launch); omit to run locally"
    )

    models = sub.add_parser("models", parents=[common], help="show the model-registry Elo standings for a task")
    models.add_argument("--sort-by", default="elo", help="metric tag to rank agents by")
    models.add_argument(
        "--breakdown",
        "-b",
        action="store_true",
        help="show per-opponent win-rate columns (wr_*) instead of the metadata columns",
    )
    models.add_argument(
        "--color", action=argparse.BooleanOptionalAction, default=None, help="color Elo by rating uncertainty"
    )
    models.add_argument(
        "--mlflow",
        action="store_true",
        help="show a direct link to each registry version's originating MLflow run",
    )
    models.add_argument(
        "--extended",
        "-e",
        action="store_true",
        help="show extended metadata columns (machine, git_sha)",
    )
    models_sub = models.add_subparsers(dest="models_cmd")

    models_prune = models_sub.add_parser(
        "prune", parents=[common], help="retire model versions from standings and league runs"
    )
    models_prune.add_argument(
        "versions",
        nargs="*",
        metavar="VERSION",
        help="version numbers to retire — the 'version' column of `kego models` "
        "(e.g. `kego models prune 31 34`); omit when using --drop-worse",
    )
    models_prune.add_argument(
        "--drop-worse",
        action="store_true",
        help="retire statistically worse variants of the same model architecture",
    )
    models_prune.add_argument(
        "--drop-worse-min-games",
        type=int,
        default=20,
        help="minimum games required for a version to be considered rated and subject to drop-worse",
    )
    models_prune.add_argument(
        "--drop-worse-k",
        type=float,
        default=1.96,
        help="confidence multiplier for drop-worse (default %(default)s for a 95%% confidence interval)",
    )

    models_unprune = models_sub.add_parser("unprune", parents=[common], help="restore retired model versions")
    models_unprune.add_argument(
        "versions",
        nargs="+",
        metavar="VERSION",
        help="version numbers to restore — the 'version' column of `kego models` (pruned versions "
        "are hidden from standings; find them in the MLflow model registry under archived versions)",
    )

    logs = sub.add_parser("logs", parents=[common], help="view or tail the logs of an active/past fleet run")
    logs.add_argument(
        "target_or_run_id",
        nargs="?",
        help="fleet machine name or MLflow run ID prefix; if omitted, defaults to local latest log",
    )
    logs.add_argument(
        "run_id",
        nargs="?",
        help="optional run ID prefix (only needed if target_or_run_id is specified as a machine name)",
    )
    logs.add_argument(
        "--tail",
        "-f",
        action="store_true",
        help="tail the log file continuously (like tail -f)",
    )
    logs.add_argument(
        "--lines",
        "-n",
        type=int,
        default=100,
        help="number of lines to print from the end of the log (default: %(default)s)",
    )

    return parser


def detect_task() -> str:
    from pathlib import Path

    import tomllib

    curr = Path.cwd()
    for parent in [curr, *curr.parents]:
        toml_path = parent / "kego.toml"
        if toml_path.exists():
            try:
                with open(toml_path, "rb") as f:
                    data = tomllib.load(f)
                if "competition" in data:
                    comp = data["competition"]
                    if isinstance(comp, dict):
                        return comp.get("slug") or comp.get("name") or "default"
                    elif isinstance(comp, list):
                        for item in comp:
                            if isinstance(item, dict):
                                if "name" in item:
                                    return item["name"]
                                if "slug" in item:
                                    return item["slug"]
            except Exception:  # noqa: S110
                pass
    return "default"


def _train_agent_overrides(args) -> dict:
    keys = [
        "agent",
        "init_checkpoint",
        "num_workers",
        "variant",
    ]
    return {k: getattr(args, k, None) for k in keys if getattr(args, k, None) is not None}


def _dispatch_train_agent(task_name: str, target: str, epochs: int | None, output: str | None, overrides: dict) -> int:
    """Ship the local tree to fleet machine ``target`` and SSH-launch training there (§5.4)."""

    from kego.dispatch import DEFAULT_EXCLUDES, dispatch, other_competition_excludes
    from kego.fleet import git_sha, load_fleet, machine_name
    from kego.fleet import repo_root as find_repo_root
    from kego.tracking import create_run, default_tracking_uri

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


def _dispatch_league(task_name: str, target: str, args) -> int:
    import contextlib

    from mlflow.tracking import MlflowClient

    from kego.dispatch import DEFAULT_EXCLUDES, dispatch, other_competition_excludes
    from kego.fleet import git_sha, load_fleet, machine_name
    from kego.fleet import repo_root as find_repo_root
    from kego.tracking import create_run, default_tracking_uri

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
        "job": "leaderboard",
        "write_ratings": str(args.write_ratings),
    }
    run_id = create_run(uri, experiment=task_name, run_name=f"{machine.name}-{task_name}-leaderboard", tags=tags)
    if not run_id:
        print(f"Could not create an MLflow run at {uri}; aborting league dispatch.")
        return 1

    cmd_args = [
        "league",
        "--task",
        task_name,
        "--games",
        str(args.games),
        "--search-count",
        str(args.search_count),
    ]
    if args.workers is not None:
        cmd_args += ["--workers", str(args.workers)]
    if args.debug:
        cmd_args.append("--debug")
    if args.cache_dir:
        cmd_args += ["--cache-dir", args.cache_dir]
    if args.include_local_mcts:
        cmd_args.append("--include-local-mcts")
    cmd_args += ["--partial-save-every", str(args.partial_save_every)]
    cmd_args += ["--stall-timeout", str(args.stall_timeout)]
    cmd_args.append("--write-ratings" if args.write_ratings else "--no-write-ratings")

    excludes = DEFAULT_EXCLUDES + other_competition_excludes(repo_root, keep=task_name)
    print(f"Dispatching league for {task_name} to {machine.name} ({machine.ssh}) — run {run_id}")
    try:
        dispatch(machine, cmd_args, run_id=run_id, local_dir=str(repo_root), excludes=excludes)
    except Exception as e:
        print(f"Dispatch failed: {e}")
        with contextlib.suppress(Exception):
            MlflowClient(tracking_uri=uri).set_terminated(run_id, status="FAILED")
        return 1
    print(f"Launched league on {machine.name}. Track run/log metadata in MLflow at: {uri}")
    print(f"To view remote logs, run:\n  kego logs {machine.name} {run_id}")
    return 0


def _resolve_run_machine(task_name: str, run_id_prefix: str) -> tuple[str, str] | None:
    import contextlib

    from mlflow.tracking import MlflowClient

    from kego.tracking import default_tracking_uri

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


def _run_logs(target_or_run_id: str | None, run_id_prefix: str | None, tail: bool, lines: int, task_name: str) -> int:
    import contextlib
    import shlex
    import subprocess
    from pathlib import Path

    from kego.fleet import load_fleet
    from kego.fleet import repo_root as find_repo_root

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
        resolved = _resolve_run_machine(task_name, run_id)
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
            from kego.fleet import machine_name as get_machine_name

            if target == get_machine_name():
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


def _run_league_locally(task_name: str, args) -> int:
    import runpy
    import sys

    from kego.fleet import repo_root as find_repo_root

    if task_name != "pokemon-tcg-ai-battle":
        print(f"Task '{task_name}' does not implement a league runner.")
        return 1

    repo_root = find_repo_root()
    script = repo_root / "competitions" / task_name / "run_league.py"
    argv = [
        str(script),
        "--task",
        task_name,
        "--games",
        str(args.games),
        "--search-count",
        str(args.search_count),
        "--write-ratings" if args.write_ratings else "--no-write-ratings",
    ]
    if args.workers is not None:
        argv += ["--workers", str(args.workers)]
    if args.debug:
        argv.append("--debug")
    if args.cache_dir:
        argv += ["--cache-dir", args.cache_dir]
    if args.include_local_mcts:
        argv.append("--include-local-mcts")
    argv += ["--partial-save-every", str(args.partial_save_every)]
    argv += ["--stall-timeout", str(args.stall_timeout)]

    old_argv = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv
    return 0


def _show_league_matrix(task_name: str, run_id: str | None = None) -> int:
    import tempfile
    from pathlib import Path

    from mlflow.tracking import MlflowClient

    from kego.tracking import default_tracking_uri

    uri = default_tracking_uri()
    client = MlflowClient(tracking_uri=uri)
    if not run_id:
        exp = client.get_experiment_by_name(task_name)
        if not exp:
            print(f"No MLflow experiment named {task_name!r}.")
            return 1
        runs = client.search_runs(
            [exp.experiment_id],
            "tags.job = 'leaderboard' and attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=100,
        )
        if not runs:
            print(f"No saved league runs found for {task_name}.")
            return 1
        run_id = next((run.info.run_id for run in runs if _has_league_artifacts(client, run.info.run_id)), None)
        if not run_id:
            print(f"No saved league runs with league artifacts found for {task_name}.")
            return 1

    with tempfile.TemporaryDirectory() as tmp:
        last_error = None
        for artifact_path in ("league/matrix.md", "matrix.md"):
            try:
                path = Path(client.download_artifacts(run_id, artifact_path, dst_path=tmp))
            except Exception as e:
                last_error = f"{artifact_path}: {e}"
                continue
            print(f"{task_name} league matrix from MLflow run {run_id}")
            print(path.read_text().rstrip())
            return 0
        print(f"Could not download league matrix for run {run_id}: {last_error or 'no matching artifact path'}")
        return 1


def _load_league_module(task_name: str):
    import importlib.util

    from kego.fleet import repo_root as find_repo_root

    repo_root = find_repo_root()
    script = repo_root / "competitions" / task_name / "run_league.py"
    spec = importlib.util.spec_from_file_location(f"{task_name}_run_league", script)
    if not spec or not spec.loader:
        raise RuntimeError(f"Could not load league runner at {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_submission_module(task_name: str):
    import importlib.util

    from kego.fleet import repo_root as find_repo_root

    script = find_repo_root() / "competitions" / task_name / "submit_leader.py"
    spec = importlib.util.spec_from_file_location(f"{task_name}_submit", script)
    if not spec or not spec.loader:
        raise RuntimeError(f"Could not load submission helper at {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_league_csv(path):
    import csv

    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    names = rows[0][1:]
    values = {}
    for row in rows[1:]:
        player = row[0]
        for opponent, value in zip(names, row[1:], strict=True):
            values[(player, opponent)] = int(float(value))
    return names, values


def _has_league_artifacts(client, run_id: str) -> bool:
    from pathlib import Path

    try:
        names = {a.path for a in client.list_artifacts(run_id, "league")}
    except Exception:
        names = set()
    if {
        "league/matrix.md",
        "league/wins.csv",
        "league/games.csv",
        "league/results.json",
        "league/participants.json",
    } <= names:
        return True
    try:
        names = {a.path for a in client.list_artifacts(run_id, "")}
    except Exception:
        local_dir = Path.cwd() / "outputs" / "league_runs" / run_id
        return all(
            (local_dir / name).exists()
            for name in ["matrix.md", "wins.csv", "games.csv", "results.json", "participants.json"]
        )
    else:
        return {"matrix.md", "wins.csv", "games.csv", "results.json", "participants.json"} <= names


def _auto_league_run_ids(task_name: str, latest: int | None = None) -> list[str]:
    from mlflow.tracking import MlflowClient

    from kego.tracking import default_tracking_uri

    client = MlflowClient(tracking_uri=default_tracking_uri())
    exp = client.get_experiment_by_name(task_name)
    if not exp:
        return []

    merge_runs = client.search_runs(
        [exp.experiment_id],
        "tags.job = 'leaderboard_merge'",
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )
    consumed: set[str] = set()
    for run in merge_runs:
        consumed.update(rid for rid in run.data.tags.get("source_run_ids", "").split(",") if rid)

    league_runs = client.search_runs(
        [exp.experiment_id],
        "tags.job = 'leaderboard' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )
    selected = [
        run.info.run_id
        for run in league_runs
        if run.info.run_id not in consumed and _has_league_artifacts(client, run.info.run_id)
    ]
    return selected[:latest] if latest is not None else selected


def _merge_leagues(task_name: str, run_ids: list[str], write_ratings: bool, latest: int | None = None) -> int:
    import json
    import tempfile
    from pathlib import Path

    import numpy as np
    from mlflow.tracking import MlflowClient

    from kego.tracking import create_run, default_tracking_uri
    from kego.tracking.league import Rating, rate_round, results_from_winmatrix
    from kego.tracking.registry import read_ratings
    from kego.tracking.registry import write_ratings as write_registry_ratings

    uri = default_tracking_uri()
    client = MlflowClient(tracking_uri=uri)
    league_module = _load_league_module(task_name)
    if not run_ids:
        print("No league run ids provided; selecting finished unmerged league runs automatically.")
        run_ids = _auto_league_run_ids(task_name, latest=latest)
    if not run_ids:
        print("No finished unmerged league runs with league artifacts found.")
        return 1
    print(f"Merging {len(run_ids)} league run(s): {' '.join(run_ids)}")

    names: list[str] = []
    wins_by_pair: dict[tuple[str, str], int] = {}
    games_by_pair: dict[tuple[str, str], int] = {}
    name_to_version: dict[str, str] = {}
    anchor_elos: dict[str, float] = {}

    with tempfile.TemporaryDirectory() as tmp:
        for run_id in run_ids:
            try:
                league_dir = Path(client.download_artifacts(run_id, "league", dst_path=tmp))
            except Exception:
                local_dir = Path.cwd() / "outputs" / "league_runs" / run_id
                if not (local_dir / "participants.json").exists():
                    raise
                league_dir = local_dir
            meta = json.loads((league_dir / "participants.json").read_text())
            for name in meta["participants"]:
                if name not in names:
                    names.append(name)
            name_to_version.update(meta.get("name_to_version", {}))
            anchor_elos.update({k: float(v) for k, v in meta.get("anchor_elos", {}).items()})

            _, wins = _read_league_csv(league_dir / "wins.csv")
            _, games = _read_league_csv(league_dir / "games.csv")
            for key, value in wins.items():
                wins_by_pair[key] = wins_by_pair.get(key, 0) + value
            for key, value in games.items():
                games_by_pair[key] = games_by_pair.get(key, 0) + value

    n = len(names)
    wins_matrix = np.zeros((n, n))
    games_matrix = np.zeros((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            wins_matrix[i][j] = wins_by_pair.get((a, b), 0)
            games_matrix[i][j] = games_by_pair.get((a, b), 0)

    results = []
    for i, name in enumerate(names):
        total_wins = sum(wins_matrix[i][j] for j in range(n) if i != j)
        total_games = sum(games_matrix[i][j] for j in range(n) if i != j)
        avg_wr = (total_wins / total_games) * 100 if total_games > 0 else 0
        results.append({"name": name, "avg_wr": avg_wr, "wins": total_wins, "games": total_games})
    results.sort(key=lambda x: x["avg_wr"], reverse=True)

    matrix_md = league_module._format_league_matrix(names, wins_matrix, games_matrix, results)
    print(matrix_md)

    merged_run_id = create_run(
        uri,
        experiment=task_name,
        run_name=f"{task_name}-leaderboard-merge",
        tags={
            "job": "leaderboard_merge",
            "task": task_name,
            "source_run_ids": ",".join(run_ids),
            "write_ratings": str(write_ratings),
        },
    )
    if merged_run_id:
        args = __import__("types").SimpleNamespace(
            task=task_name, games="merged", search_count="merged", workers=0, write_ratings=write_ratings
        )
        artifact_dir = league_module._persist_league_artifacts(
            client,
            merged_run_id,
            names,
            wins_matrix,
            games_matrix,
            results,
            name_to_version,
            anchor_elos,
            args,
        )
        client.log_metric(merged_run_id, "league_participants", n)
        client.log_metric(merged_run_id, "league_games_total", int(games_matrix.sum() / 2))
        client.set_terminated(merged_run_id, status="FINISHED")
        print(f"\nMerged league artifacts written to {artifact_dir} and MLflow run {merged_run_id}")

    if write_ratings:
        round_results = results_from_winmatrix(names, wins_matrix.tolist(), games_matrix.tolist())
        version_ratings = read_ratings(uri, task_name)
        prior = {
            name: Rating(version_ratings[v]["elo"], version_ratings[v]["elo_rd"])
            for name, v in name_to_version.items()
            if v in version_ratings
        }
        updated = rate_round(prior, round_results, anchor_elos)
        round_games = {name: len(res) for name, res in round_results.items()}
        prior_games = {name: version_ratings.get(v, {}).get("games", 0) for name, v in name_to_version.items()}
        ratings_by_version = {
            name_to_version[name]: {
                "elo": r.elo,
                "elo_rd": r.rd,
                "games": prior_games.get(name, 0) + round_games.get(name, 0),
            }
            for name, r in updated.items()
            if name in name_to_version
        }
        write_registry_ratings(uri, task_name, ratings_by_version)
        print(f"Updated Elo ratings for {len(ratings_by_version)} registered agent(s).")
    return 0


def _rule_agent_rows(task_name: str) -> list[dict]:
    if task_name != "pokemon-tcg-ai-battle":
        return []
    return [
        {
            "agent": name,
            "version": "rule",
            "type": "rule",
            "elo": str(elo),
            "elo_rd": "30.0",
            "games": "-",
            "deck": deck,
            "rating_status": "anchor",
        }
        for name, elo, deck in [
            ("Mega Abomasnow ex", 1650.0, "abomasnow"),
            ("Dragapult ex", 1520.0, "dragapult"),
            ("Mega Lucario ex", 1450.0, "lucario"),
            ("Zacian ex", 1350.0, "zacian"),
            ("Random", 1200.0, "sample"),
        ]
    ]


def _registry_agent_name(row: dict) -> str:
    if row.get("variant"):
        version = row.get("version", "-")
        epoch = row.get("epoch")
        suffix = f" @{epoch}" if epoch and str(epoch) != "-" else ""
        return f"v{version} {row['variant']}{suffix}"
    if row.get("agent_name"):
        return str(row["agent_name"])

    version = row.get("version", "-")
    deck = row.get("deck")
    model_args = row.get("model_args")
    epoch = row.get("epoch")

    if deck or model_args:
        parts = [f"v{version}", f"mcts-{deck or 'unknown'}"]
        if model_args:
            parts.append(str(model_args).strip("()").replace(", ", "/"))
        if epoch:
            parts.append(f"@{epoch}")
        return " ".join(parts)

    return f"Registry v{version}"


def _numeric_row_value(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("-inf")


def _mlflow_run_link(tracking_uri: str, experiment_id: str, run_id: str) -> str:
    """Build an MLflow UI link when the tracking URI is browser-accessible."""
    from urllib.parse import quote, urlparse

    if urlparse(tracking_uri).scheme not in {"http", "https"}:
        return "-"
    return (
        f"{tracking_uri.rstrip('/')}/#/experiments/{quote(str(experiment_id), safe='')}"
        f"/runs/{quote(str(run_id), safe='')}"
    )


def _use_color(force: bool | None = None) -> bool:
    import os
    import sys

    if force is not None:
        return force
    return "NO_COLOR" not in os.environ and (
        os.environ.get("FORCE_COLOR") not in (None, "", "0") or sys.stdout.isatty()
    )


def main(argv: list[str] | None = None) -> int:
    import os

    os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")

    parser = build_parser()
    args = parser.parse_args(argv)

    import dataclasses

    from kego.pipeline.config import ModelConfig, PipelineConfig, load_config
    from kego.pipeline.executor import get_executor
    from kego.pipeline.runner import Pipeline

    task_name = getattr(args, "task", None) or detect_task()

    if args.command == "logs":
        return _run_logs(args.target_or_run_id, args.run_id, args.tail, args.lines, task_name)

    if args.command == "sync":
        import json

        from kego.tracking import pending_registrations, sync_outbox

        pending = pending_registrations()
        if args.list:
            if not pending:
                print("Outbox empty — nothing queued.")
                return 0
            for entry_dir in pending:
                meta = json.loads((entry_dir / "entry.json").read_text())
                print(
                    f"{entry_dir.name}  model={meta['name']}  checkpoint={meta['checkpoint']}  "
                    f"queued={meta['queued_at']}"
                )
            return 0
        if not pending:
            print("Outbox empty — nothing to sync.")
            return 0
        synced, errors = sync_outbox(uri=getattr(args, "uri", None))
        for entry, model, version in synced:
            print(f"Synced {entry} -> {model} v{version}")
        for err in errors:
            print(f"Failed: {err}")
        return 1 if errors else 0

    if args.command == "models":
        from mlflow.tracking import MlflowClient

        from kego.tracking import (
            archive_version,
            default_tracking_uri,
            filter_worse_versions,
            format_leaderboard,
            leaderboard,
            unarchive_version,
        )
        from kego.tracking.prune import active_model_versions

        uri = default_tracking_uri()
        models_cmd = getattr(args, "models_cmd", None)

        if models_cmd == "prune":
            client = MlflowClient(tracking_uri=uri)
            versions = list(getattr(args, "versions", []) or [])
            drop_worse = getattr(args, "drop_worse", False)
            if bool(versions) == bool(drop_worse):
                print("Specify either version numbers or --drop-worse (not both, not neither).")
                return 1
            to_archive: list[str] = []
            if drop_worse:
                active = active_model_versions(client, task_name)
                _, dropped = filter_worse_versions(
                    active,
                    min_games=args.drop_worse_min_games,
                    k=args.drop_worse_k,
                )
                to_archive = [str(v.version) for v in dropped]
                if not to_archive:
                    print(f"No worse variants to prune for '{task_name}'.")
                    return 0
                print(f"Drop-worse selected {len(to_archive)} version(s): {', '.join(f'v{v}' for v in to_archive)}")
            else:
                to_archive = [str(v) for v in versions]

            for version in to_archive:
                try:
                    archive_version(client, task_name, version, dropped=drop_worse)
                    print(f"Pruned version {version} of model '{task_name}'.")
                except Exception as e:
                    print(f"Error pruning version {version}: {e}")
                    return 1
            print("Pruned versions are excluded from standings and league runs.")
            return 0

        if models_cmd == "unprune":
            client = MlflowClient(tracking_uri=uri)
            for version in args.versions:
                try:
                    unarchive_version(client, task_name, str(version))
                    print(f"Unpruned version {version} of model '{task_name}'.")
                except Exception as e:
                    print(f"Error unpruning version {version}: {e}")
                    return 1
            print("Restored versions are included back on standings and league runs.")
            return 0

        rows = leaderboard(uri, task_name, sort_by=args.sort_by)
        submission_stats = Pipeline(PipelineConfig(task=task_name)).model_submission_stats()
        mlflow_client = MlflowClient(tracking_uri=uri) if args.mlflow else None
        for row in rows:
            row.setdefault("trained", row.get("completed_iterations") or row.get("epoch") or "-")
            row.setdefault("agent", _registry_agent_name(row))
            row.setdefault("type", "registry")
            row.update(submission_stats.get(str(row.get("version")), {"submitted": "-", "public_rank": "-"}))

            # Combine submitted + public_rank into "kaggle"
            submitted = row.get("submitted", "-")
            public_rank = row.get("public_rank", "-")
            if submitted == "-" or submitted == "no":
                row["kaggle"] = "-"
            elif public_rank == "-":
                row["kaggle"] = submitted
            else:
                row["kaggle"] = f"{submitted} #{public_rank}"

            # Combine epoch + trained into "iters"
            epoch = row.get("epoch")
            trained = row.get("trained")
            if str(epoch) == str(trained) or epoch == "-" or trained == "-":
                row["iters"] = str(trained) if trained else "-"
            else:
                row["iters"] = f"{trained}/{epoch}"

            training_run_id = row.get("training_run_id") or row.get("run_id")
            if mlflow_client and training_run_id:
                try:
                    run = mlflow_client.get_run(training_run_id)
                    # Older registry versions only reference their artifact-registration
                    # run. Do not present that empty run as the metrics-bearing trainer.
                    if not row.get("training_run_id") and not run.data.metrics:
                        row["mlflow"] = "-"
                    else:
                        row["mlflow"] = _mlflow_run_link(uri, run.info.experiment_id, training_run_id)
                except Exception:
                    row["mlflow"] = "-"
        rows = [*rows, *_rule_agent_rows(task_name)]
        rows = sorted(rows, key=lambda row: _numeric_row_value(row, args.sort_by), reverse=True)
        seen: set[str] = set()
        if args.breakdown:
            wr_cols = sorted({k for r in rows for k in r if k.startswith("wr_")})
            base = ["agent", "type", args.sort_by, "gauntlet_avg", *wr_cols]
            if args.extended:
                base.extend(["machine", "git_sha"])
            base.extend(["created", "version"])
        else:
            base = [
                "agent",
                "type",
                args.sort_by,
                "elo_rd",
                "games",
                "gauntlet_avg",
                "kaggle",
                "deck",
                "iters",
            ]
            if args.extended:
                base.extend(["machine", "git_sha"])
            base.extend(["created", "version"])
        cols = [c for c in base if not (c in seen or seen.add(c))]
        if args.mlflow:
            cols.append("mlflow")
        print(f"{task_name} — {len(rows)} agents · tracking {uri}")
        use_color = _use_color(args.color)
        if "elo" in cols and use_color:
            print("elo color: green <= +/-1 RD, yellow <= +/-5 RD, red > +/-5 RD")
        print(format_leaderboard(rows, cols, max_widths={"git_sha": 8}, color_elo=use_color))
        return 0

    if args.command == "league":
        league_cmd = getattr(args, "league_cmd", None)
        if league_cmd == "matrix":
            return _show_league_matrix(task_name, getattr(args, "run_id", None))
        if league_cmd == "merge":
            return _merge_leagues(task_name, args.run_ids, args.write_ratings, latest=args.latest)
        target = getattr(args, "target", None)
        if target:
            return _dispatch_league(task_name, target, args)
        return _run_league_locally(task_name, args)

    if args.command == "run" and not args.config and not getattr(args, "model", None):
        parser.error("Either --config or --model must be specified.")

    if args.config:
        try:
            config = load_config(args.config, getattr(args, "params", []), task_name=task_name)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            return 1
    else:
        config = PipelineConfig(task=task_name)

    if getattr(args, "model", None):
        model_params = {}
        if args.params:
            for p in args.params:
                if ":" in p:
                    k, v = p.split(":", 1)
                elif "=" in p:
                    k, v = p.split("=", 1)
                else:
                    k, v = p, "true"

                try:
                    if v.lower() == "true":
                        val = True
                    elif v.lower() == "false":
                        val = False
                    elif "." in v:
                        val = float(v)
                    else:
                        val = int(v)
                except ValueError:
                    val = v
                model_params[k] = val

        hp_space_dict = {}
        if getattr(args, "hp_params", None):
            from kego.pipeline.tune import HPSpace

            for hp_spec in args.hp_params:
                hp = HPSpace.parse(hp_spec)
                hp_space_dict[hp.name] = {
                    "type": hp.type,
                    "low": hp.low,
                    "high": hp.high,
                    "log": hp.log,
                    "choices": hp.choices,
                }

        model_config = ModelConfig(
            name=args.model,
            hyper_params=model_params,
            hp_space=hp_space_dict,
        )
        config = dataclasses.replace(config, models=[model_config])

    if getattr(args, "hp_tune", False):
        config = dataclasses.replace(config, tune=True)

    if getattr(args, "do_ensemble", None) is not None:
        config = dataclasses.replace(config, do_ensemble=args.do_ensemble)
    if getattr(args, "force", False):
        config = dataclasses.replace(config, force=args.force)

    executor_kind = getattr(args, "executor", "serial")
    executor = get_executor(executor_kind)

    pipeline = Pipeline(config, executor=executor)

    if args.command == "run":
        try:
            outcome = pipeline.run()
            if getattr(args, "submit", False):
                pipeline.submit(outcome, message=getattr(args, "tag", ""))
        except NotImplementedError as e:
            print(f"Error: {e}")
            return 1
        return 0
    elif args.command == "tune":
        models_to_tune = args.tune if getattr(args, "tune", None) else [m.name for m in config.models]
        try:
            pipeline.tune(models=models_to_tune)
        except NotImplementedError as e:
            print(f"Error: {e}")
            return 1
        return 0
    elif args.command == "ensemble":
        try:
            pipeline.ensemble(
                experiments=args.from_experiment,
                ensemble_tag=args.tag,
            )
        except NotImplementedError as e:
            print(f"Error: {e}")
            return 1
        return 0
    elif args.command == "submit":
        from kego.pipeline.runner import RunOutcome

        message = args.message
        if task_name == "pokemon-tcg-ai-battle":
            prepared = _load_submission_module(task_name).prepare_submission(args.version)
            message = message or prepared["message"]
        elif args.version:
            print(f"Error: registry version arguments are not supported for task {task_name!r}.")
            return 1
        outcome = RunOutcome(predictions=[])
        try:
            pipeline.submit(outcome, message=message)
        except NotImplementedError as e:
            print(f"Error: {e}")
            return 1
        return 0
    elif args.command == "status":
        pipeline.status()
        return 0
    elif args.command == "cache":
        pipeline.cache(args.action)
        return 0
    elif args.command == "submissions":
        pipeline.submissions()
        return 0
    elif args.command == "battle":
        # Resolve battle parameters: CLI arguments take precedence over config file values
        agent1 = getattr(args, "agent1", None)
        agent2 = getattr(args, "agent2", None)
        games = getattr(args, "games", None)
        deck1 = getattr(args, "deck1", None)
        deck2 = getattr(args, "deck2", None)

        if args.config:
            battle_cfg = config.battle
            if agent1 is None:
                agent1 = battle_cfg.agent1
            if agent2 is None:
                agent2 = battle_cfg.agent2
            if games is None:
                games = battle_cfg.games
            if deck1 is None:
                deck1 = battle_cfg.deck1
            if deck2 is None:
                deck2 = battle_cfg.deck2

        # Defaults
        if games is None:
            games = 10

        if not agent1 or not agent2:
            parser.error(
                "Both --agent1 and --agent2 must be specified (either via CLI arguments or in the --config file)."
            )

        try:
            from kego.pipeline.battle import run_battle_benchmark

            run_battle_benchmark(
                agent1_path=agent1,
                agent2_path=agent2,
                num_games=games,
                deck1_path=deck1,
                deck2_path=deck2,
            )
        except Exception as e:
            print(f"Error running battle benchmark: {e}")
            return 1
        return 0

    elif args.command == "train-agent":
        epochs = getattr(args, "epochs", None)
        output = getattr(args, "output", None)
        overrides = _train_agent_overrides(args)
        target = getattr(args, "target", None)
        if target and target not in ("local", "cluster"):
            return _dispatch_train_agent(task_name, target, epochs, output, overrides)
        try:
            pipeline.train_agent(epochs=epochs, output_path=output, **overrides)
        except NotImplementedError as e:
            print(f"Error: {e}")
            return 1
        except Exception as e:
            print(f"Error during training: {e}")
            return 1
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
