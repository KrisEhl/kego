try:
    from agents.base import RuleScoringAgent, get_int_value
except ImportError:
    from base_agent import RuleScoringAgent, get_int_value
from cg.api import Observation


class ZacianAgent(RuleScoringAgent):
    def __init__(self, deck="zacian.csv"):
        super().__init__(deck)

    def score_option(self, option, select_type: int, select_context: int, obs: Observation, option_index: int) -> float:
        opt_type_val = get_int_value(option.type)

        # MAIN select type is 0
        if select_type == 0:
            if opt_type_val == 13:  # ATTACK
                # Scale slightly by index so the last attack option (usually main/stronger) scores highest
                return 1000.0 + option_index
            elif opt_type_val == 8:  # ATTACH
                if obs.current and not obs.current.energyAttached:
                    return 500.0
                return -100.0
            elif opt_type_val == 7:  # PLAY
                return 100.0
            elif opt_type_val == 14:  # END
                return 0.0
            elif opt_type_val == 12:  # RETREAT
                return -50.0

        # Fallback to general base class behavior for YES_NO or other contexts
        return super().score_option(option, select_type, select_context, obs, option_index)


# Define submission agent wrapper for Kaggle and pipeline execution
_agent_instance = None


def agent(obs_dict: dict) -> list[int]:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ZacianAgent()
    return _agent_instance.act(obs_dict)
