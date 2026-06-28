"""Subcommand for battling local Pokemon TCG agents against each other to benchmark them."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def locate_cg_dir() -> Path:
    """Find the 'cg' game engine library parent directory."""
    repo_root = Path(__file__).resolve().parents[2]
    dirs_to_try = [
        repo_root / "data/pokemon/pokemon-tcg-ai-battle-challenge-strategy/sample_submission/sample_submission",
        repo_root / "data/pokemon/pokemon-tcg-ai-battle/sample_submission/sample_submission",
        Path(
            "/home/kristian/projects/kego/data/pokemon/pokemon-tcg-ai-battle-challenge-strategy/sample_submission/sample_submission"
        ),
        Path("/home/kristian/projects/kego/data/pokemon/pokemon-tcg-ai-battle/sample_submission/sample_submission"),
        repo_root / "data/pokemon/pokemon-tcg-ai-battle-challenge-strategy/sample_submission",
        repo_root / "data/pokemon/pokemon-tcg-ai-battle/sample_submission",
        repo_root / "competitions/pokemon-tcg-ai-battle/sample_submission",
        repo_root / "competitions/pokemon-tcg-ai-battle",
        repo_root / "data/pokemon",
        repo_root,
        Path("."),
    ]
    for d in dirs_to_try:
        if (d / "cg").exists():
            return d.resolve()

    raise ImportError(
        "Could not find the 'cg' game engine library.\n"
        "Please ensure you have accepted the Kaggle competition rules and downloaded the "
        "competition data, or placed the 'cg' directory inside the workspace."
    )


def load_agent(filepath: str) -> Any:
    """Dynamically load an agent module from a file path."""
    p = Path(filepath)
    if not p.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        p = repo_root / filepath
    p = p.resolve()

    if not p.exists():
        raise FileNotFoundError(f"Agent file not found at {p}")

    # Ensure parent directory is in sys.path so relative imports in main.py work
    sys.path.insert(0, str(p.parent))

    spec = importlib.util.spec_from_file_location(p.stem, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load agent spec for {p}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "agent"):
        raise AttributeError(f"Module at {p} does not contain an 'agent' function")

    return module


def load_deck(deck_path_or_module: str | Any) -> list[int]:
    """Load a deck list of 60 card IDs from a CSV or dynamically from agent module."""
    if isinstance(deck_path_or_module, str):
        p = Path(deck_path_or_module)
        if not p.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            p = repo_root / deck_path_or_module
        p = p.resolve()

        if not p.exists():
            raise FileNotFoundError(f"Deck CSV not found at {p}")
        with open(p) as f:
            lines = f.read().splitlines()
        deck = [int(line.strip()) for line in lines if line.strip() and not line.strip().startswith("#")]
        if len(deck) < 60:
            raise ValueError(f"Deck CSV at {p} has only {len(deck)} cards (need at least 60)")
        return deck[:60]

    # Try calling read_deck_csv on the module
    if hasattr(deck_path_or_module, "read_deck_csv"):
        try:
            return deck_path_or_module.read_deck_csv()
        except Exception:  # noqa: S110
            pass

    # Default fallback: check if there's a deck.csv in the same directory as the module
    module_path = getattr(deck_path_or_module, "__file__", None)
    if module_path:
        module_dir = Path(module_path).parent
        deck_csv = module_dir / "deck.csv"
        if deck_csv.exists():
            with open(deck_csv) as f:
                lines = f.read().splitlines()
            deck = [int(line.strip()) for line in lines if line.strip() and not line.strip().startswith("#")]
            if len(deck) >= 60:
                return deck[:60]

    raise ValueError("Could not load deck for agent. Please provide a path to a deck CSV file via --deck1/--deck2.")


def run_game(agent1_mod: Any, agent2_mod: Any, deck1: list[int], deck2: list[int]) -> int:
    """Runs a single game between agent1 and agent2.

    Returns:
        int: 0 if agent1 wins, 1 if agent2 wins.
    """
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    obs_dict, start_data = battle_start(deck1, deck2)
    if not obs_dict:
        raise RuntimeError(f"Failed to start battle. Error type: {start_data.errorType if start_data else 'unknown'}")

    try:
        while True:
            obs = to_observation_class(obs_dict)

            # Check if game is finished
            if obs.current and obs.current.result != -1:
                return obs.current.result

            if obs.select is None:
                action = deck1
            else:
                active_player = obs_dict.get("current", {}).get("yourIndex", 0)
                if active_player == 0:
                    action = agent1_mod.agent(obs_dict)
                else:
                    action = agent2_mod.agent(obs_dict)

            obs_dict = battle_select(action)
    finally:
        battle_finish()


def run_battle_benchmark(
    agent1_path: str,
    agent2_path: str,
    num_games: int = 10,
    deck1_path: str | None = None,
    deck2_path: str | None = None,
) -> None:
    """Alternates playing Agent 1 and Agent 2 against each other and prints wins statistics."""
    # 1. Locate and insert cg directory in sys.path
    cg_parent_dir = locate_cg_dir()
    sys.path.insert(0, str(cg_parent_dir))

    # 2. Dynamically load the agent modules
    print(f"Loading Agent 1 from: {agent1_path}")
    agent1_mod = load_agent(agent1_path)

    print(f"Loading Agent 2 from: {agent2_path}")
    agent2_mod = load_agent(agent2_path)

    # 3. Load the decks
    deck1 = load_deck(deck1_path or agent1_mod)
    deck2 = load_deck(deck2_path or agent2_mod)

    # 4. Run the benchmark games
    print(f"\nStarting benchmark: {num_games} games...")
    agent1_wins = 0
    agent2_wins = 0

    for game_idx in range(num_games):
        p0_agent = 1 if game_idx % 2 == 0 else 2

        if p0_agent == 1:
            d0, d1 = deck1, deck2
            a0_mod, a1_mod = agent1_mod, agent2_mod
        else:
            d0, d1 = deck2, deck1
            a0_mod, a1_mod = agent2_mod, agent1_mod

        print(f"Game {game_idx + 1}/{num_games}: Agent {p0_agent} goes first... ", end="", flush=True)

        try:
            winner = run_game(a0_mod, a1_mod, d0, d1)
            if winner == 0:
                game_winner = p0_agent
            else:
                game_winner = 2 if p0_agent == 1 else 1

            if game_winner == 1:
                agent1_wins += 1
            else:
                agent2_wins += 1

            print(f"Winner: Agent {game_winner}")
        except Exception as e:
            print(f"Failed! Error: {e}")

    print("\n" + "=" * 40)
    print("BENCHMARK RESULTS")
    print("=" * 40)
    print(f"Agent 1 ({agent1_path}): {agent1_wins} wins ({agent1_wins / num_games * 100:.1f}%)")
    print(f"Agent 2 ({agent2_path}): {agent2_wins} wins ({agent2_wins / num_games * 100:.1f}%)")
    print("=" * 40)
