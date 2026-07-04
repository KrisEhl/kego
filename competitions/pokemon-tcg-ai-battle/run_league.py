import argparse
import ast
import contextlib
import multiprocessing as mp
import os
import shlex
import subprocess
import sys
import time
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
        "--workers", type=int, default=os.cpu_count(), help="Number of parallel workers (defaults to CPU count)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    args = parser.parse_args()

    # Set MCTS search count for all workers
    os.environ["MCTS_SEARCH_COUNT"] = str(args.search_count)

    # 2. Get registered models from MLflow
    uri = default_tracking_uri()
    if args.debug:
        print(f"Connecting to MLflow Tracking Server at: {uri}")
    else:
        print(f"Connecting to MLflow at {uri}...")

    client = MlflowClient(tracking_uri=uri)

    try:
        versions = client.search_model_versions(f"name='{args.task}'")
    except Exception as e:
        if args.debug:
            print(f"Warning: Could not fetch models from registry ({e}).")
        versions = []

    model_checkpoints = {}
    cache_base_dir = comp_dir / "outputs" / "cached_registry"

    if not args.debug and len(versions) > 0:
        print("Caching registered checkpoints from hub...")

    for v in versions:
        v_name = f"Registry v{v.version}"
        if args.debug:
            print(f"Loading checkpoint for {v_name} (run {v.run_id})...")
        try:
            local_dir = cache_base_dir / f"v{v.version}"
            checkpoint_path = download_checkpoint(client, v, str(local_dir), args.debug)
            model_checkpoints[v_name] = checkpoint_path
            if args.debug:
                print(f"  Checkpoint ready: {checkpoint_path}")
        except Exception as e:
            if args.debug:
                print(f"  Error obtaining checkpoint: {e}")

    # Check for local checkpoints as well
    local_mcts = comp_dir / "outputs" / "mcts.pth"
    if local_mcts.exists():
        model_checkpoints["Local (outputs/mcts.pth)"] = str(local_mcts)

    # 3. Define the participants
    participants = {}

    for v in versions:
        v_name = f"Registry v{v.version}"
        if v_name in model_checkpoints:
            deck_name = v.tags.get("deck", "abomasnow") if v.tags else "abomasnow"
            participants[v_name] = {
                "type": "mcts",
                "file": "competitions/pokemon-tcg-ai-battle/agents/mcts.py",
                "deck": f"competitions/pokemon-tcg-ai-battle/decks/{deck_name}.csv",
                "model_path": model_checkpoints[v_name],
                "model_args": _parse_model_args(v.tags.get("model_args") if v.tags else None),
            }

    # Check for local checkpoints as well
    local_mcts = comp_dir / "outputs" / "mcts.pth"
    if local_mcts.exists():
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
            "file": "competitions/pokemon-tcg-ai-battle/agents/mcts.py",
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
        return

    # Set up child process environments
    os.environ["PYTHONPATH"] = os.pathsep.join(
        p for p in [str(comp_dir), str(repo_root), os.environ.get("PYTHONPATH", "")] if p
    )
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Run in parallel
    results_list = [None] * len(tasks)
    pool = mp.get_context("spawn").Pool(args.workers, initializer=_worker_init)
    try:
        pbar = ProgressBar(total=len(tasks))
        for task_idx, winner_val in pool.imap_unordered(_run_single_game_indexed, tasks):
            results_list[task_idx] = winner_val
            pbar.update(1)
    finally:
        pool.close()
        pool.join()

    # Aggregate wins
    for task_idx, winner_val in enumerate(results_list):
        if winner_val == -1:
            continue
        i, j = task_map[task_idx]
        games_matrix[i][j] += 1
        games_matrix[j][i] += 1
        if winner_val == 0:
            wins_matrix[i][j] += 1
        elif winner_val == 1:
            wins_matrix[j][i] += 1

    # --- League Elo: update ratings from this round and write back to the registry ---
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

    # Generate Markdown Table sorted by average win rate
    results = []
    for i in range(n_participants):
        total_wins = 0
        total_games = 0
        for j in range(n_participants):
            if i != j:
                total_wins += wins_matrix[i][j]
                total_games += games_matrix[i][j]
        avg_wr = (total_wins / total_games) * 100 if total_games > 0 else 0
        results.append({"name": participant_names[i], "avg_wr": avg_wr, "wins": total_wins, "games": total_games})

    # Sort descending by average win rate
    results.sort(key=lambda x: x["avg_wr"], reverse=True)
    sorted_names = [r["name"] for r in results]

    # Build the full table structure for column width calculations
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
                val_str = f"{int(wr)}%" if wr.is_integer() else f"{wr:.1f}%"
                row.append(val_str)
        avg_str = f"**{int(r['avg_wr'])}%**" if r["avg_wr"].is_integer() else f"**{r['avg_wr']:.1f}%**"
        row.append(avg_str)
        table_rows.append(row)

    # Compute column widths
    col_widths = []
    for col_idx in range(len(headers)):
        max_w = max(len(row[col_idx]) for row in table_rows)
        col_widths.append(max_w)

    # Build the separator row
    sep_row = []
    for col_idx in range(len(headers)):
        w = col_widths[col_idx]
        if col_idx == 0:
            sep_row.append(":" + "-" * (w - 1))
        else:
            sep_row.append(":" + "-" * (w - 2) + ":")

    print("\nLeague Results Matrix (Row vs Column Win Rate %):")
    # Print headers
    header_cells = [
        table_rows[0][c].ljust(col_widths[c]) if c == 0 else table_rows[0][c].center(col_widths[c])
        for c in range(len(headers))
    ]
    print("| " + " | ".join(header_cells) + " |")
    # Print separator
    print("| " + " | ".join(sep_row) + " |")
    # Print body
    for row in table_rows[1:]:
        row_cells = [
            row[c].ljust(col_widths[c]) if c == 0 else row[c].center(col_widths[c]) for c in range(len(headers))
        ]
        print("| " + " | ".join(row_cells) + " |")


if __name__ == "__main__":
    main()
