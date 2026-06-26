"""Local Pokémon TCG AI Battle Simulation Runner.

Runs a local match between two agents (or the same agent against itself)
using the underlying compiled game engine library.
"""

import sys
from pathlib import Path

# Add competition directory and sample submission directory to path
comp_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(comp_dir))

# Find the cg library (either in local folder or data folder)
cg_dir = Path("data/pokemon/pokemon-tcg-ai-battle/sample_submission")
if not cg_dir.exists():
    cg_dir = Path("/home/kristian/projects/kego/data/pokemon/pokemon-tcg-ai-battle/sample_submission")
sys.path.insert(0, str(cg_dir))

import main as agent_mod  # imports main.py from the competition folder
from cg.api import to_observation_class
from cg.game import battle_finish, battle_select, battle_start


def run_match():
    print("Loading decks...")
    deck0 = agent_mod.read_deck_csv()
    deck1 = agent_mod.read_deck_csv()

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
