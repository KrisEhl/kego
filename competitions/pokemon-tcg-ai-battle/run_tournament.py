import sys
from pathlib import Path

import numpy as np

# Resolve competition directory and add repository root to path
comp_dir = Path(__file__).parent.resolve()
repo_root = comp_dir.parent.parent
sys.path.insert(0, str(repo_root))

from kego.pipeline.battle import load_agent, load_deck, locate_cg_dir, run_game

# 1. Setup CG path
cg_parent = locate_cg_dir()
sys.path.insert(0, str(cg_parent))

# 2. Define the agents and their decks
agents = {
    "Random": {
        "file": "competitions/pokemon-tcg-ai-battle/agents/random_agent.py",
        "deck": "data/pokemon/pokemon-tcg-ai-battle/sample_submission/sample_submission/deck.csv",
    },
    "Zacian ex": {
        "file": "competitions/pokemon-tcg-ai-battle/agents/zacian.py",
        "deck": "competitions/pokemon-tcg-ai-battle/decks/zacian.csv",
    },
    "Mega Abomasnow ex": {
        "file": "competitions/pokemon-tcg-ai-battle/agents/abomasnow.py",
        "deck": "competitions/pokemon-tcg-ai-battle/decks/abomasnow.csv",
    },
    "Dragapult ex": {
        "file": "competitions/pokemon-tcg-ai-battle/agents/dragapult.py",
        "deck": "competitions/pokemon-tcg-ai-battle/decks/dragapult.csv",
    },
    "Mega Lucario ex": {
        "file": "competitions/pokemon-tcg-ai-battle/agents/lucario.py",
        "deck": "competitions/pokemon-tcg-ai-battle/decks/lucario.csv",
    },
}


def run_tournament(games_per_match=20):
    # Load all agents and decks
    loaded_agents = {}
    for name, cfg in agents.items():
        print(f"Loading {name}...")
        mod = load_agent(cfg["file"])
        deck = load_deck(cfg["deck"])
        loaded_agents[name] = {"mod": mod, "deck": deck}

    agent_names = list(agents.keys())
    n_agents = len(agent_names)

    # Matrix to store win counts: matrix[A][B] = wins of A against B
    wins_matrix = np.zeros((n_agents, n_agents))
    games_matrix = np.zeros((n_agents, n_agents))

    print("\nStarting round-robin tournament...")
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            name_a = agent_names[i]
            name_b = agent_names[j]
            a_cfg = loaded_agents[name_a]
            b_cfg = loaded_agents[name_b]

            print(f"Matchup: {name_a} vs {name_b} ({games_per_match} games) ... ", end="", flush=True)

            a_wins = 0
            b_wins = 0

            for game_idx in range(games_per_match):
                # Alternate starting player
                p0_is_a = game_idx % 2 == 0

                if p0_is_a:
                    d0, d1 = a_cfg["deck"], b_cfg["deck"]
                    a0_mod, a1_mod = a_cfg["mod"], b_cfg["mod"]
                else:
                    d0, d1 = b_cfg["deck"], a_cfg["deck"]
                    a0_mod, a1_mod = b_cfg["mod"], a_cfg["mod"]

                try:
                    winner = run_game(a0_mod, a1_mod, d0, d1)
                    if winner == 0:
                        if p0_is_a:
                            a_wins += 1
                        else:
                            b_wins += 1
                    else:
                        if p0_is_a:
                            b_wins += 1
                        else:
                            a_wins += 1
                except Exception as e:
                    print(f"\nGame failed with error: {e}")

            wins_matrix[i][j] = a_wins
            wins_matrix[j][i] = b_wins
            games_matrix[i][j] = games_per_match
            games_matrix[j][i] = games_per_match
            print(f"Results: {name_a} won {a_wins}, {name_b} won {b_wins}")

    # Generate Markdown Table
    print("\nTournament Results Matrix (Row vs Column Win Rate %):")
    header = "| Agent | " + " | ".join(agent_names) + " | Average WR |"
    separator = "| --- | " + " | ".join(["---"] * n_agents) + " | --- |"
    print(header)
    print(separator)

    for i in range(n_agents):
        row_winrates = []
        total_wins = 0
        total_games = 0
        for j in range(n_agents):
            if i == j:
                row_winrates.append("-")
            else:
                wr = (wins_matrix[i][j] / games_matrix[i][j]) * 100
                row_winrates.append(f"{wr:.1f}%")
                total_wins += wins_matrix[i][j]
                total_games += games_matrix[i][j]
        avg_wr = (total_wins / total_games) * 100 if total_games > 0 else 0
        row_str = f"| **{agent_names[i]}** | " + " | ".join(row_winrates) + f" | **{avg_wr:.1f}%** |"
        print(row_str)


if __name__ == "__main__":
    run_tournament(games_per_match=20)
