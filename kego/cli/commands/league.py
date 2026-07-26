import contextlib
import csv
import importlib.util
import json
import runpy
import sys
import tempfile
from pathlib import Path

import numpy as np

from kego.dispatch import DEFAULT_EXCLUDES, dispatch, other_competition_excludes
from kego.fleet import git_sha, load_fleet, machine_name
from kego.fleet import repo_root as find_repo_root
from kego.tracking import create_run, default_tracking_uri, read_ratings
from kego.tracking import write_ratings as write_registry_ratings
from kego.tracking.league import Rating, rate_round, results_from_winmatrix


def load_league_module(task_name: str):
    repo_root = find_repo_root()
    script = repo_root / "competitions" / task_name / "run_league.py"
    spec = importlib.util.spec_from_file_location(f"{task_name}_run_league", script)
    if not spec or not spec.loader:
        raise RuntimeError(f"Could not load league runner at {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_league_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    names = rows[0][1:]
    values = {}
    for row in rows[1:]:
        player = row[0]
        for opponent, value in zip(names, row[1:], strict=True):
            values[(player, opponent)] = int(float(value))
    return names, values


def has_league_artifacts(client, run_id: str) -> bool:
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


def auto_league_run_ids(task_name: str, latest: int | None = None) -> list[str]:
    from mlflow.tracking import MlflowClient

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
        if run.info.run_id not in consumed and has_league_artifacts(client, run.info.run_id)
    ]
    return selected[:latest] if latest is not None else selected


def dispatch_league(task_name: str, target: str, args) -> int:
    from mlflow.tracking import MlflowClient

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


def run_league_locally(task_name: str, args) -> int:
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


def show_league_matrix(task_name: str, run_id: str | None = None) -> int:
    from mlflow.tracking import MlflowClient

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
        run_id = next((run.info.run_id for run in runs if has_league_artifacts(client, run.info.run_id)), None)
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


def merge_leagues(task_name: str, run_ids: list[str], write_ratings: bool, latest: int | None = None) -> int:
    from mlflow.tracking import MlflowClient

    uri = default_tracking_uri()
    client = MlflowClient(tracking_uri=uri)
    league_module = load_league_module(task_name)
    if not run_ids:
        print("No league run ids provided; selecting finished unmerged league runs automatically.")
        run_ids = auto_league_run_ids(task_name, latest=latest)
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

            _, wins = read_league_csv(league_dir / "wins.csv")
            _, games = read_league_csv(league_dir / "games.csv")
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
