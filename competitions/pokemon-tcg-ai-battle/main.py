import os
import random

from cg.api import Observation, to_observation_class


def read_deck_csv() -> list[int]:
    """Read deck.csv.

    Returns:
        list[int]: A list of card IDs in the deck.
    """
    # Look for deck.csv in potential local and remote locations
    file_paths = [
        "deck.csv",
        "/kaggle_simulations/agent/deck.csv",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv"),
        "data/pokemon/pokemon-tcg-ai-battle/sample_submission/deck.csv",
    ]

    for path in file_paths:
        if os.path.exists(path):
            with open(path) as file:
                csv = file.read().split("\n")
            deck = []
            for i in range(60):
                deck.append(int(csv[i]))
            return deck

    raise FileNotFoundError("deck.csv not found in any of the expected paths.")


def agent(obs_dict: dict) -> list[int]:
    """Implement Your Pokémon Trading Card Game Agent.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.

    Returns:
        list[int]: A list of option index.
    """
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        # In the initial selection, the obs.select is None, and it is necessary to return the deck.
        # The deck is a list of 60 card IDs.
        # The deck must comply with the Pokémon Trading Card Game rules.
        return read_deck_csv()

    return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)  # select randomly
