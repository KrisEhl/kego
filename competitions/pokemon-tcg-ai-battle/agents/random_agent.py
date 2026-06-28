import random

try:
    from agents.base import BaseAgent
except ImportError:
    from base_agent import BaseAgent
from cg.api import to_observation_class


class RandomAgent(BaseAgent):
    def __init__(self, deck=None):
        # Default fallback deck
        self.deck = [1071] * 60

    def get_deck(self) -> list[int]:
        return self.deck

    def act(self, obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return self.get_deck()

        n = len(obs.select.option)
        if n == 0:
            return []

        min_c = max(1, obs.select.minCount)
        k = min(obs.select.maxCount, n)
        k = max(k, min(min_c, n))

        opts = list(range(n))
        return random.sample(opts, k)


# Define agent instance wrapper
_agent_instance = None


def agent(obs_dict: dict) -> list[int]:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = RandomAgent()
    return _agent_instance.act(obs_dict)
