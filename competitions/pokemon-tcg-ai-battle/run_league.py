import argparse
import ast
import contextlib
import csv
import json
import multiprocessing as mp
import os
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch

# Resolve competition directory and add repository root to path
comp_dir = Path(__file__).parent.resolve()
repo_root = comp_dir.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(comp_dir))

from mlflow.tracking import MlflowClient

from kego.pipeline.battle import load_agent, load_deck, locate_cg_dir, run_game
from kego.tracking import default_tracking_uri

# 1. Setup CG path
cg_parent = locate_cg_dir()
sys.path.insert(0, str(cg_parent))


class ProgressBar:
    def __init__(self, total, width=40):
        self.total = total
        self.width = width
        self.start_time = time.time()
        self.completed = 0

    def update(self, count=1):
        self.completed += count
        elapsed = time.time() - self.start_time

        if self.completed > 0:
            rate = self.completed / elapsed
            eta_seconds = (self.total - self.completed) / rate
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "--:--"

        elapsed_str = self._format_time(elapsed)

        progress = self.completed / self.total
        filled_length = int(self.width * progress)
        bar = "█" * filled_length + "-" * (self.width - filled_length)
        percent = int(progress * 100)

        sys.stdout.write(
            f"\r|{bar}| {percent}% ({self.completed}/{self.total}) [Elapsed: {elapsed_str} | ETA: {eta_str}]"
        )
        sys.stdout.flush()

        if self.completed >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _format_time(self, seconds):
        if seconds < 0:
            return "00:00"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"


def _worker_init():
    # Avoid core oversubscription
    torch.set_num_threads(1)


def instantiate_agent(cfg):
    mod = load_agent(cfg["file"])
    deck = load_deck(cfg["deck"])
    if cfg["type"] == "mcts":
        agent_obj = mod.MCTSTransformerAgent(
            deck=cfg["deck"], model_path=cfg["model_path"], model_args=cfg.get("model_args")
        )
        mod._agent_instance = agent_obj
    return {"mod": mod, "deck": deck}


def _parse_model_args(raw: str | None):
    if not raw:
        return None
    try:
        parsed = ast.literal_eval(raw)
        return tuple(parsed) if isinstance(parsed, (list, tuple)) else None
    except (SyntaxError, ValueError):
        return None


def default_workers() -> int:
    cores = os.cpu_count() or 4
    return max(1, cores - 2)


def _finish_mlflow_run(client: MlflowClient, run_id: str | None, status: str = "FINISHED") -> None:
    if not run_id:
        return
    try:
        client.set_terminated(run_id, status=status)
    except Exception:
        pass


def _ensure_league_run(client: MlflowClient, uri: str, task: str, args) -> tuple[str | None, bool]:
    run_id = os.environ.get("KEGO_MLFLOW_RUN_ID")
    if run_id:
        return run_id, False
    try:
        from kego.tracking import create_run

        run_id = create_run(
            uri,
            experiment=task,
            run_name=f"{task}-leaderboard",
            tags={"job": "leaderboard", "task": task, "write_ratings": str(args.write_ratings)},
        )
        return run_id, bool(run_id)
    except Exception:
        return None, False


def _format_league_matrix(participant_names, wins_matrix, games_matrix, results) -> str:
    sorted_names = [r["name"] for r in results]
    headers = ["Participant"] + sorted_names + ["Average WR"]
    table_rows = [headers]

    for idx, r in enumerate(results):
        prefix = ""
        if idx == 0:
            prefix = "🥇 "
        elif idx == 1:
            prefix = "🥈 "
        elif idx == 2:
            prefix = "🥉 "

        row = [f"{prefix}**{r['name']}**"]
        idx_i = participant_names.index(r["name"])
        for name_j in sorted_names:
            if r["name"] == name_j:
                row.append("-")
            else:
                idx_j = participant_names.index(name_j)
                wr = (
                    (wins_matrix[idx_i][idx_j] / games_matrix[idx_i][idx_j]) * 100
                    if games_matrix[idx_i][idx_j] > 0
                    else 0
                )
                row.append(f"{int(wr)}%" if wr.is_integer() else f"{wr:.1f}%")
        avg_str = f"**{int(r['avg_wr'])}%**" if r["avg_wr"].is_integer() else f"**{r['avg_wr']:.1f}%**"
        row.append(avg_str)
        table_rows.append(row)

    col_widths = [max(len(row[col_idx]) for row in table_rows) for col_idx in range(len(headers))]
    sep_row = []
    for col_idx, w in enumerate(col_widths):
        sep_row.append(":" + "-" * (w - 1) if col_idx == 0 else ":" + "-" * (w - 2) + ":")

    lines = ["League Results Matrix (Row vs Column Win Rate %):"]
    header_cells = [
        table_rows[0][c].ljust(col_widths[c]) if c == 0 else table_rows[0][c].center(col_widths[c])
        for c in range(len(headers))
    ]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| " + " | ".join(sep_row) + " |")
    for row in table_rows[1:]:
        row_cells = [
            row[c].ljust(col_widths[c]) if c == 0 else row[c].center(col_widths[c]) for c in range(len(headers))
        ]
        lines.append("| " + " | ".join(row_cells) + " |")
    return "\n".join(lines)


def _write_matrix_csv(path: Path, participant_names, matrix) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["participant", *participant_names])
        for name, row in zip(participant_names, matrix, strict=True):
            writer.writerow([name, *row])


def _persist_league_artifacts(
    client: MlflowClient,
    run_id: str | None,
    participant_names,
    wins_matrix,
    games_matrix,
    results,
    name_to_version,
    anchor_elos,
    args,
    dropped_variants=None,
) -> Path:
    artifact_id = run_id or f"local_{uuid.uuid4().hex[:12]}"
    out_dir = comp_dir / "outputs" / "league_runs" / artifact_id
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_md = _format_league_matrix(participant_names, wins_matrix, games_matrix, results)
    (out_dir / "matrix.md").write_text(matrix_md + "\n")
    _write_matrix_csv(out_dir / "wins.csv", participant_names, wins_matrix.astype(int).tolist())
    _write_matrix_csv(out_dir / "games.csv", participant_names, games_matrix.astype(int).tolist())
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "participants.json").write_text(
        json.dumps(
            {
                "participants": participant_names,
                "name_to_version": name_to_version,
                "anchor_elos": anchor_elos,
                "dropped_variants": dropped_variants or [],
                "args": {
                    "task": args.task,
                    "games": args.games,
                    "search_count": args.search_count,
                    "workers": args.workers,
                    "write_ratings": args.write_ratings,
                    "include_local_mcts": getattr(args, "include_local_mcts", False),
                    "partial_save_every": getattr(args, "partial_save_every", 0),
                },
            },
            indent=2,
        )
    )
    if run_id:
        try:
            client.log_artifacts(run_id, str(out_dir), artifact_path="league")
        except Exception as e:
            print(f"\nWarning: could not log league artifacts to MLflow ({e}). Local copy: {out_dir}")
    return out_dir


def _aggregate_completed_results(results_list, task_map, n_participants):
    wins_matrix = np.zeros((n_participants, n_participants))
    games_matrix = np.zeros((n_participants, n_participants))
    completed = 0
    for task_idx, winner_val in enumerate(results_list):
        if winner_val is None:
            continue
        completed += 1
        if winner_val == -1:
            continue
        i, j = task_map[task_idx]
        games_matrix[i][j] += 1
        games_matrix[j][i] += 1
        if winner_val == 0:
            wins_matrix[i][j] += 1
        elif winner_val == 1:
            wins_matrix[j][i] += 1
    return wins_matrix, games_matrix, completed


def _build_league_results(participant_names, wins_matrix, games_matrix):
    results = []
    for i, name in enumerate(participant_names):
        total_wins = sum(wins_matrix[i][j] for j in range(len(participant_names)) if i != j)
        total_games = sum(games_matrix[i][j] for j in range(len(participant_names)) if i != j)
        avg_wr = (total_wins / total_games) * 100 if total_games > 0 else 0
        results.append({"name": name, "avg_wr": avg_wr, "wins": total_wins, "games": total_games})
    results.sort(key=lambda x: x["avg_wr"], reverse=True)
    return results


def _run_single_game(payload):
    cfg_a, cfg_b, p0_is_a, game_id, debug = payload

    # Redirect stdout and stderr to devnull to suppress subprocess logging unless debug is enabled
    if debug:
        cm = contextlib.nullcontext()
    else:
        cm = contextlib.ExitStack()
        devnull = open(os.devnull, "w")
        cm.enter_context(contextlib.redirect_stdout(devnull))
        cm.enter_context(contextlib.redirect_stderr(devnull))

    with cm:
        try:
            a_loaded = instantiate_agent(cfg_a)
            b_loaded = instantiate_agent(cfg_b)

            if p0_is_a:
                d0, d1 = a_loaded["deck"], b_loaded["deck"]
                a0_mod, a1_mod = a_loaded["mod"], b_loaded["mod"]
            else:
                d0, d1 = b_loaded["deck"], a_loaded["deck"]
                a0_mod, a1_mod = b_loaded["mod"], a_loaded["mod"]

            winner = run_game(a0_mod, a1_mod, d0, d1)
            if p0_is_a:
                return 0 if winner == 0 else 1
            else:
                return 0 if winner == 1 else 1
        except Exception as e:
            if debug:
                sys.stderr.write(f"\nGame {game_id} failed with error: {e}\n")
                sys.stderr.flush()
            return -1


def _run_single_game_indexed(payload_and_idx):
    payload, idx = payload_and_idx
    res = _run_single_game(payload)
    return idx, res


def _select_checkpoint(path: str, wanted: str | None) -> str | None:
    p = Path(path)
    if p.is_file() and p.suffix == ".pth" and (wanted is None or p.name == wanted):
        return str(p)
    if not p.is_dir():
        return None
    pths = sorted(p.rglob("*.pth"))
    if wanted:
        return next((str(candidate) for candidate in pths if candidate.name == wanted), None)
    if len(pths) == 1:
        return str(pths[0])
    return next((str(candidate) for candidate in pths if candidate.name == "mcts.pth"), None)


def _registry_cache_root(cache_dir: str | None) -> Path:
    return Path(cache_dir) if cache_dir else comp_dir / "outputs" / "cached_registry"


def _registry_cache_dir(cache_root: Path, run_id: str) -> Path:
    return cache_root / f"run_{run_id}"


def _fleet_ssh_targets(machine_tag: str) -> list[str]:
    fallback = {
        "omarchyd": "kristian@omarchyd",
        "omarchyl": "kristian@omarchyl",
        "DESKTOP-68OIS2S": "kristian@DESKTOP-68OIS2S",
        "mn-exjk9p93n75h": "kristian.ehlert@mn-exjk9p93n75h",
    }
    try:
        from kego.fleet import load_fleet

        machines = {m.name: m.ssh for m in load_fleet(repo_root / "fleet.toml").machines}
    except Exception:
        machines = fallback
    ordered = []
    if machine_tag and machine_tag in machines:
        ordered.append(machines[machine_tag])
    if "omarchyd" in machines and "omarchyd" != machine_tag:
        ordered.append(machines["omarchyd"])
    ordered.extend(ssh for name, ssh in machines.items() if name not in {machine_tag, "omarchyd"})
    return list(dict.fromkeys(ordered))


def _remote_checkpoint_path(ssh_target: str, source: str, wanted: str | None, debug: bool) -> str | None:
    remote_root = source[7:] if source.startswith("file://") else source
    if remote_root.endswith(".pth"):
        check = f"test -f {shlex.quote(remote_root)} && printf '%s\\n' {shlex.quote(remote_root)}"
    elif wanted:
        candidate = f"{remote_root.rstrip('/')}/{wanted}"
        check = f"test -f {shlex.quote(candidate)} && printf '%s\\n' {shlex.quote(candidate)}"
    else:
        check = (
            f"find {shlex.quote(remote_root)} -maxdepth 2 -type f -name '*.pth' "
            "-printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-"
        )
    res = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", ssh_target, check],
        capture_output=True,
        text=True,
    )
    if debug and res.returncode != 0:
        print(f"  SSH find on {ssh_target} failed: {res.stderr.strip()}")
    remote_path = res.stdout.strip().splitlines()[0] if res.returncode == 0 and res.stdout.strip() else ""
    return remote_path or None


def download_checkpoint(client, v, local_dir, debug):
    wanted = v.tags.get("checkpoint_filename") if v.tags else None
    cached = _select_checkpoint(local_dir, wanted)
    if cached:
        return cached

    os.makedirs(local_dir, exist_ok=True)

    try:
        if debug:
            downloaded = client.download_artifacts(v.run_id, "checkpoint", dst_path=local_dir)
        else:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                downloaded = client.download_artifacts(v.run_id, "checkpoint", dst_path=local_dir)

        selected = _select_checkpoint(downloaded, wanted)
        if selected:
            return selected
    except Exception as e:
        if debug:
            print(f"  MLflow client download failed: {e}. Trying SCP fallback...")

    machine_tag = (v.tags or {}).get("machine", "")
    for ssh_target in _fleet_ssh_targets(machine_tag):
        remote_path = _remote_checkpoint_path(ssh_target, v.source, wanted, debug)
        if not remote_path:
            if debug:
                print(f"  No checkpoint found under {v.source} on {ssh_target}; trying next...")
            continue
        checkpoint_path = os.path.join(local_dir, os.path.basename(remote_path))
        if debug:
            print(f"  SCP downloading from {ssh_target}:{remote_path} ...")
        cmd = ["scp", f"{ssh_target}:{remote_path}", checkpoint_path]
        res = subprocess.run(cmd, capture_output=not debug)
        if res.returncode == 0 and os.path.exists(checkpoint_path):
            return checkpoint_path
        elif debug:
            print(f"  SCP from {ssh_target} failed, trying next...")

    raise RuntimeError("Failed to download checkpoint via MLflow and SCP from any fleet machine.")


def main():
    parser = argparse.ArgumentParser(
        description="Run a round-robin league between registered models and heuristic baseline agents."
    )
    parser.add_argument("--games", type=int, default=4, help="Number of games per matchup (alternating starts)")
    parser.add_argument("--task", type=str, default="pokemon-tcg-ai-battle", help="Task name to query in registry")
    parser.add_argument("--search-count", type=int, default=10, help="Inference search count for MCTS agents")
    parser.add_argument(
        "--workers", type=int, default=default_workers(), help="Number of parallel workers (defaults to CPU count - 2)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory root for downloaded registry checkpoints (default: outputs/cached_registry)",
    )
    parser.add_argument(
        "--write-ratings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write updated Elo ratings back to MLflow model-version tags",
    )
    parser.add_argument(
        "--include-local-mcts",
        action="store_true",
        help="Include local outputs/mcts.pth as a participant (not merge-safe across fleet machines)",
    )
    parser.add_argument(
        "--partial-save-every",
        type=int,
        default=5000,
        help="Persist partial league artifacts every N completed games; use 0 to disable",
    )
    args = parser.parse_args()

    # Set MCTS search count for all workers
    os.environ["MCTS_SEARCH_COUNT"] = str(args.search_count)

    # Pin league agents to CPU. This runner spawns one worker per core; each loads the model
    # onto self.device, so MPS/CUDA would put N processes on one GPU. On Apple Silicon (unified
    # memory) that exhausts system RAM and hangs the laptop. Inference here is tiny + CPU-parallel
    # is already fast. Override with MCTS_DEVICE=mps/cuda if you really want it.
    os.environ.setdefault("MCTS_DEVICE", "cpu")

    # 2. Get registered models from MLflow
    uri = default_tracking_uri()
    if args.debug:
        print(f"Connecting to MLflow Tracking Server at: {uri}")
    else:
        print(f"Connecting to MLflow at {uri}...")

    client = MlflowClient(tracking_uri=uri)
    mlflow_run_id, _owns_mlflow_run = _ensure_league_run(client, uri, args.task, args)
    if mlflow_run_id:
        try:
            client.set_tag(mlflow_run_id, "job", "leaderboard")
            client.set_tag(mlflow_run_id, "task", args.task)
            client.log_param(mlflow_run_id, "job", "leaderboard")
            client.log_param(mlflow_run_id, "games", args.games)
            client.log_param(mlflow_run_id, "search_count", args.search_count)
            client.log_param(mlflow_run_id, "workers", args.workers)
            client.log_param(mlflow_run_id, "write_ratings", args.write_ratings)
        except Exception:
            pass

    try:
        from kego.tracking.prune import active_model_versions

        versions = active_model_versions(client, args.task)
    except Exception as e:
        if args.debug:
            print(f"Warning: Could not fetch models from registry ({e}).")
        versions = []

    dropped_variants: list[dict] = []

    model_checkpoints = {}
    cache_base_dir = _registry_cache_root(args.cache_dir)

    if not args.debug and len(versions) > 0:
        print(f"Caching registered checkpoints from hub ({len(versions)} versions)...")

    for i, v in enumerate(versions, 1):
        v_name = f"Registry v{v.version}"
        if not args.debug:
            print(f"  [{i}/{len(versions)}] Caching {v_name}... ", end="", flush=True)
        else:
            print(f"Loading checkpoint for {v_name} (run {v.run_id})...")
        try:
            local_dir = _registry_cache_dir(cache_base_dir, v.run_id)
            wanted = v.tags.get("checkpoint_filename") if v.tags else None
            cached = _select_checkpoint(str(local_dir), wanted)

            checkpoint_path = download_checkpoint(client, v, str(local_dir), args.debug)
            model_checkpoints[v_name] = checkpoint_path

            if args.debug:
                print(f"  Checkpoint ready: {checkpoint_path}")
            else:
                if cached:
                    print("already cached.")
                else:
                    print("done.")
        except Exception as e:
            if args.debug:
                print(f"  Error obtaining checkpoint: {e}")
            else:
                print(f"failed: {e}")

    # Check for local checkpoints as well. Disabled by default so fleet runs do not
    # depend on machine-local output files.
    local_mcts = comp_dir / "outputs" / "mcts.pth"
    if args.include_local_mcts and local_mcts.exists():
        model_checkpoints["Local (outputs/mcts.pth)"] = str(local_mcts)

    # 3. Define the participants
    participants = {}

    for v in versions:
        v_name = f"Registry v{v.version}"
        if v_name in model_checkpoints:
            deck_name = v.tags.get("deck") if v.tags else None
            if not deck_name:
                raise ValueError(
                    f"{v_name} is missing required model-version tag 'deck'. "
                    "Refusing to guess a deck for registry-backed league evaluation."
                )
            participants[v_name] = {
                "type": "mcts",
                "file": "competitions/pokemon-tcg-ai-battle/agents/mcts",
                "deck": f"competitions/pokemon-tcg-ai-battle/decks/{deck_name}.csv",
                "model_path": model_checkpoints[v_name],
                "model_args": _parse_model_args(v.tags.get("model_args") if v.tags else None),
            }

    # Check for local checkpoints as well
    local_mcts = comp_dir / "outputs" / "mcts.pth"
    if args.include_local_mcts and local_mcts.exists():
        local_deck = "decks/abomasnow.csv"
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore
        try:
            with open(comp_dir / "kego.toml", "rb") as f:
                toml_data = tomllib.load(f)
            local_deck = toml_data.get("competition", {}).get("deck_file", local_deck)
        except Exception:
            pass
        participants["Local (outputs/mcts.pth)"] = {
            "type": "mcts",
            "file": "competitions/pokemon-tcg-ai-battle/agents/mcts",
            "deck": str(comp_dir / local_deck),
            "model_path": str(local_mcts),
        }

    rule_agents = {
        "Zacian ex": {
            "type": "rule",
            "file": "competitions/pokemon-tcg-ai-battle/agents/zacian.py",
            "deck": "competitions/pokemon-tcg-ai-battle/decks/zacian.csv",
        },
        "Mega Abomasnow ex": {
            "type": "rule",
            "file": "competitions/pokemon-tcg-ai-battle/agents/abomasnow.py",
            "deck": "competitions/pokemon-tcg-ai-battle/decks/abomasnow.csv",
        },
        "Dragapult ex": {
            "type": "rule",
            "file": "competitions/pokemon-tcg-ai-battle/agents/dragapult.py",
            "deck": "competitions/pokemon-tcg-ai-battle/decks/dragapult.csv",
        },
        "Mega Lucario ex": {
            "type": "rule",
            "file": "competitions/pokemon-tcg-ai-battle/agents/lucario.py",
            "deck": "competitions/pokemon-tcg-ai-battle/decks/lucario.csv",
        },
        "Random": {
            "type": "rule",
            "file": "competitions/pokemon-tcg-ai-battle/agents/random_agent.py",
            "deck": "data/pokemon/pokemon-tcg-ai-battle/sample_submission/sample_submission/deck.csv",
        },
    }

    for name, cfg in rule_agents.items():
        full_agent_path = repo_root / cfg["file"]
        if full_agent_path.exists():
            participants[name] = cfg

    # Resolve all deck and file paths to absolute paths
    for name, cfg in participants.items():
        cfg["deck"] = str(repo_root / cfg["deck"]) if not Path(cfg["deck"]).is_absolute() else cfg["deck"]
        cfg["file"] = str(repo_root / cfg["file"]) if not Path(cfg["file"]).is_absolute() else cfg["file"]

    # Map participant display name -> registry version (only registry-backed agents get rated).
    name_to_version = {f"Registry v{v.version}": str(v.version) for v in versions}

    # Pinned anchor Elos (spec Appendix A.4), keyed by participant display name.
    anchor_elos = {
        "Random": 1200.0,
        "Zacian ex": 1350.0,
        "Mega Lucario ex": 1450.0,
        "Dragapult ex": 1520.0,
        "Mega Abomasnow ex": 1650.0,
    }

    participant_names = list(participants.keys())
    n_participants = len(participant_names)

    if n_participants < 2:
        print("Not enough participants to run a league.")
        _finish_mlflow_run(client, mlflow_run_id)
        return

    # Wins and games matrices
    wins_matrix = np.zeros((n_participants, n_participants))
    games_matrix = np.zeros((n_participants, n_participants))

    # Prepare list of tasks
    tasks = []
    task_map = []

    for i in range(n_participants):
        for j in range(i + 1, n_participants):
            name_a = participant_names[i]
            name_b = participant_names[j]
            cfg_a = participants[name_a]
            cfg_b = participants[name_b]

            for game_idx in range(args.games):
                p0_is_a = game_idx % 2 == 0
                tasks.append(((cfg_a, cfg_b, p0_is_a, len(tasks), args.debug), len(tasks)))
                task_map.append((i, j))

    print(f"\nStarting round-robin league between {n_participants} participants ({len(tasks)} games total)...")
    print(f"Running games in parallel across {args.workers} workers (MCTS SEARCH_COUNT={args.search_count})...")
    if not tasks:
        print("No games requested; participant dry-run only. No Elo ratings written.")
        for name in participant_names:
            print(f"  - {name}")
        empty_results = [{"name": name, "avg_wr": 0.0, "wins": 0, "games": 0} for name in participant_names]
        artifact_dir = _persist_league_artifacts(
            client,
            mlflow_run_id,
            participant_names,
            wins_matrix,
            games_matrix,
            empty_results,
            name_to_version,
            anchor_elos,
            args,
            dropped_variants=dropped_variants,
        )
        print(f"League artifacts written to {artifact_dir}")
        if mlflow_run_id:
            try:
                client.log_metric(mlflow_run_id, "league_participants", n_participants)
                client.log_metric(mlflow_run_id, "league_games_total", 0)
            except Exception:
                pass
        _finish_mlflow_run(client, mlflow_run_id)
        return

    # Set up child process environments
    os.environ["PYTHONPATH"] = os.pathsep.join(
        p for p in [str(comp_dir), str(repo_root), os.environ.get("PYTHONPATH", "")] if p
    )
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Run in parallel
    results_list = [None] * len(tasks)
    partial_save_every = max(0, args.partial_save_every)
    next_partial_save = partial_save_every
    pool = mp.get_context("spawn").Pool(args.workers, initializer=_worker_init)
    try:
        pbar = ProgressBar(total=len(tasks))
        for task_idx, winner_val in pool.imap_unordered(_run_single_game_indexed, tasks):
            results_list[task_idx] = winner_val
            pbar.update(1)
            if partial_save_every and pbar.completed >= next_partial_save:
                wins_matrix, games_matrix, completed = _aggregate_completed_results(
                    results_list, task_map, n_participants
                )
                partial_results = _build_league_results(participant_names, wins_matrix, games_matrix)
                artifact_dir = _persist_league_artifacts(
                    client,
                    mlflow_run_id,
                    participant_names,
                    wins_matrix,
                    games_matrix,
                    partial_results,
                    name_to_version,
                    anchor_elos,
                    args,
                    dropped_variants=dropped_variants,
                )
                print(f"\nPartial league artifacts written to {artifact_dir} ({completed}/{len(tasks)} games)")
                next_partial_save += partial_save_every
    except KeyboardInterrupt:
        print("\nInterrupted; saving completed league games before shutdown...")
        pool.terminate()
        wins_matrix, games_matrix, completed = _aggregate_completed_results(results_list, task_map, n_participants)
        partial_results = _build_league_results(participant_names, wins_matrix, games_matrix)
        artifact_dir = _persist_league_artifacts(
            client,
            mlflow_run_id,
            participant_names,
            wins_matrix,
            games_matrix,
            partial_results,
            name_to_version,
            anchor_elos,
            args,
            dropped_variants=dropped_variants,
        )
        print(f"Partial league artifacts written to {artifact_dir} ({completed}/{len(tasks)} games)")
        _finish_mlflow_run(client, mlflow_run_id, status="KILLED")
        return
    except Exception:
        print("\nLeague failed; saving completed games before raising the error...")
        pool.terminate()
        wins_matrix, games_matrix, completed = _aggregate_completed_results(results_list, task_map, n_participants)
        partial_results = _build_league_results(participant_names, wins_matrix, games_matrix)
        artifact_dir = _persist_league_artifacts(
            client,
            mlflow_run_id,
            participant_names,
            wins_matrix,
            games_matrix,
            partial_results,
            name_to_version,
            anchor_elos,
            args,
            dropped_variants=dropped_variants,
        )
        print(f"Partial league artifacts written to {artifact_dir} ({completed}/{len(tasks)} games)")
        _finish_mlflow_run(client, mlflow_run_id, status="FAILED")
        raise
    else:
        pool.close()
    finally:
        pool.join()

    wins_matrix, games_matrix, _ = _aggregate_completed_results(results_list, task_map, n_participants)

    # --- League Elo: update ratings from this round and optionally write back to the registry ---
    if not args.write_ratings:
        print("\nRead-only league: not writing Elo ratings to the registry.")
    else:
        try:
            from kego.tracking import read_ratings, write_ratings
            from kego.tracking.league import Rating, rate_round, results_from_winmatrix

            round_results = results_from_winmatrix(participant_names, wins_matrix.tolist(), games_matrix.tolist())

            # Prior ratings keyed by display name (from registry tags); unrated players default later.
            version_ratings = read_ratings(uri, args.task)
            prior = {
                name: Rating(version_ratings[v]["elo"], version_ratings[v]["elo_rd"])
                for name, v in name_to_version.items()
                if v in version_ratings
            }
            updated = rate_round(prior, round_results, anchor_elos)

            # Games this round per player = number of outcomes emitted for it.
            round_games = {name: len(res) for name, res in round_results.items()}
            prior_games = {name: version_ratings.get(v, {}).get("games", 0) for name, v in name_to_version.items()}

            ratings_by_version = {
                name_to_version[name]: {
                    "elo": r.elo,
                    "elo_rd": r.rd,
                    "games": prior_games.get(name, 0) + round_games.get(name, 0),
                }
                for name, r in updated.items()
                if name in name_to_version  # skip "Local (...)" and anything not in the registry
            }
            if ratings_by_version:
                write_ratings(uri, args.task, ratings_by_version)
                print(
                    f"\nUpdated Elo ratings for {len(ratings_by_version)} registered agent(s): {sorted(ratings_by_version)}"
                )
            else:
                print("\nNo registry-backed agents in this league — no ratings written.")
        except Exception as e:
            print(f"\nWarning: could not update league Elo ratings ({e}); showing the round-robin matrix below.")

    results = _build_league_results(participant_names, wins_matrix, games_matrix)
    matrix_md = _format_league_matrix(participant_names, wins_matrix, games_matrix, results)
    print("\n" + matrix_md)
    artifact_dir = _persist_league_artifacts(
        client,
        mlflow_run_id,
        participant_names,
        wins_matrix,
        games_matrix,
        results,
        name_to_version,
        anchor_elos,
        args,
        dropped_variants=dropped_variants,
    )
    print(f"\nLeague artifacts written to {artifact_dir}")

    if mlflow_run_id:
        try:
            client.log_metric(mlflow_run_id, "league_participants", n_participants)
            client.log_metric(mlflow_run_id, "league_games_total", int(games_matrix.sum() / 2))
            if results:
                client.log_metric(mlflow_run_id, "league_top_avg_wr", float(results[0]["avg_wr"]))
                client.set_tag(mlflow_run_id, "league_top", results[0]["name"])
        except Exception:
            pass
    _finish_mlflow_run(client, mlflow_run_id)


if __name__ == "__main__":
    main()
