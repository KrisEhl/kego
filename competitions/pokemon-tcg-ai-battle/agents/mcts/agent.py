import os
import random

import torch

try:
    from base_agent import BaseAgent
except ImportError:
    from agents.base import BaseAgent
from cg.api import search_begin, search_end, search_step, to_observation_class

from .encoding import get_decoder_input, get_encoder_input
from .model import MODEL_ARGS, MyModel, model_args_from_state_dict
from .search import Evaluator, create_node, eval_nn, select_child


def _nn_evaluator(model: MyModel, your_deck: list[int]) -> Evaluator:
    """Build the inference-time `evaluate` closure for `create_node`."""

    def evaluate(obs, actions) -> tuple[float, list[float]]:
        sv_enc = get_encoder_input(obs, your_deck)
        sv_dec = get_decoder_input(obs, actions)
        return eval_nn(sv_enc, sv_dec, model)

    return evaluate


class MCTSTransformerAgent(BaseAgent):
    def __init__(self, deck="abomasnow.csv", model_path=None, model_args=None):
        self.deck = self._load_deck(deck)
        forced = os.environ.get("MCTS_DEVICE")
        self.device = (
            torch.device(forced)
            if forced
            else torch.device(
                "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        )
        state = None
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"[MCTSTransformerAgent] model_path not found: {model_path}")
            state = torch.load(model_path, map_location=self.device)
        self.model_args = tuple(model_args or (model_args_from_state_dict(state) if state is not None else MODEL_ARGS))
        self.model = MyModel(*self.model_args).to(self.device)
        self.model.eval()

        if state is not None:
            self.model.load_state_dict(state)
            print(f"[MCTSTransformerAgent] loaded weights from {model_path}", flush=True)

        self.SEARCH_COUNT = int(os.environ.get("MCTS_SEARCH_COUNT", "10"))
        print(f"[MCTSTransformerAgent] SEARCH_COUNT={self.SEARCH_COUNT}", flush=True)

    def get_deck(self) -> list[int]:
        return self.deck

    def act(self, obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return self.get_deck()

        your_index = obs.current.yourIndex
        state = obs.current
        active = state.players[1 - your_index].active
        search_state = search_begin(
            obs,
            your_deck=random.sample(self.deck, state.players[your_index].deckCount),
            your_prize=random.sample(self.deck, len(state.players[your_index].prize)),
            opponent_deck=[1072] * state.players[1 - your_index].deckCount,
            opponent_prize=[1] * len(state.players[1 - your_index].prize),
            opponent_hand=[1] * state.players[1 - your_index].handCount,
            opponent_active=[1072] if len(active) > 0 and active[0] is None else [],
        )

        evaluate = _nn_evaluator(self.model, self.deck)
        root = create_node(None, search_state, your_index, evaluate)
        for _ in range(self.SEARCH_COUNT):
            current = root
            while True:
                next_child = select_child(current, your_index)
                if next_child is None:
                    break
                if next_child.node is None:
                    s_state = search_step(current.state.searchId, next_child.select)
                    next_child.node = create_node(current, s_state, your_index, evaluate)
                    break
                current = next_child.node
                if current.state.observation.current.result >= 0:
                    current.backprop(current.value)
                    break

        max_child = None
        max_visit = -1
        for child in root.children:
            if child.node is not None and max_visit < child.node.visit:
                max_child = child
                max_visit = child.node.visit

        search_end()
        if max_child is not None:
            return max_child.select
        return root.children[0].select if root.children else [0]


_agent_instance = None


def _agent_dir() -> str:
    module_file = globals().get("__file__")
    if module_file:
        return os.path.dirname(os.path.abspath(module_file))
    return "/kaggle_simulations/agent"


def _default_model_path() -> str | None:
    explicit = os.environ.get("MCTS_MODEL_PATH")
    if explicit:
        return explicit
    base_dir = _agent_dir()
    candidates = [
        os.path.join(base_dir, "mcts.pth"),
        os.path.join(os.getcwd(), "mcts.pth"),
        "/kaggle_simulations/agent/mcts.pth",
    ]
    return next((p for p in candidates if os.path.exists(p)), None)


def agent(obs_dict: dict) -> list[int]:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = MCTSTransformerAgent(
            deck=os.environ.get("MCTS_DECK", "deck.csv"),
            model_path=_default_model_path(),
        )
    return _agent_instance.act(obs_dict)
