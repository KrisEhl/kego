"""Local Pokémon TCG AI Battle Simulation Runner.

Runs a local match between the configured agent and itself
using the underlying compiled game engine library.
"""

import sys
from pathlib import Path

# Resolve competition directory and add repository root to path
comp_dir = Path(__file__).parent.resolve()
repo_root = comp_dir.parent.parent
sys.path.insert(0, str(repo_root))

from kego.pipeline.battle import load_agent, load_deck, locate_cg_dir

# Setup CG path
cg_parent = locate_cg_dir()
sys.path.insert(0, str(cg_parent))

from cg.api import to_observation_class
from cg.game import battle_finish, battle_select, battle_start


def run_match():
    # Load settings from kego.toml
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore

    config_path = comp_dir / "kego.toml"
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    comp_cfg = cfg.get("competition", {})
    agent_file = comp_cfg.get("agent_file")
    deck_file = comp_cfg.get("deck_file")

    if not agent_file or not deck_file:
        raise ValueError("agent_file and deck_file must be explicitly set in kego.toml")

    agent_path = comp_dir / agent_file
    deck_path = comp_dir / deck_file

    print(f"Loading agent {agent_path.name}...")
    agent_mod = load_agent(str(agent_path))
    print(f"Loading deck {deck_path.name}...")
    deck0 = load_deck(str(deck_path))
    deck1 = load_deck(str(deck_path))

    print("Starting local simulation battle...")
    obs_dict, start_data = battle_start(deck0, deck1)

    if not obs_dict:
        print(f"Failed to start battle. Error code: {start_data.errorType}")
        return

    turn_count = 0
    while True:
        # Wrap dictionary in Observation class to check results
        obs = to_observation_class(obs_dict)

        # Check if game is finished
        if obs.current and obs.current.result != -1:
            winner = obs.current.result
            print(f"\nBattle Finished! Winner: Player {winner}")
            break

        if obs.select is None:
            # Game is starting or in deck selection phase
            action = deck0
        else:
            # Query the agent for selected option indexes
            action = agent_mod.agent(obs_dict)

        # Step the game simulator
        obs_dict = battle_select(action)
        turn_count += 1

        if turn_count % 10 == 0:
            current_turn = obs_dict.get("current", {}).get("turn", 0) if obs_dict.get("current") else 0
            print(f"Turn {current_turn} in progress...")

    battle_finish()
    print("Simulation complete.")


if __name__ == "__main__":
    run_match()
