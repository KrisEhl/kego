"""Comprehensive tests for easy/nominal behavior AND edge cases across all core MCTS functions.

Covers:
- search.py: enumerate_action_combinations, create_node, select_child, build_children
- opponent.py: infer_opponent_hidden_cards, _revealed_cards
- encoding.py: SparseVector, encode_state, encode_actions
- agent.py: MCTSTransformerAgent (deck query prompt, SEARCH_COUNT env override)
"""

import random
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COMP = REPO_ROOT / "competitions" / "pokemon-tcg-ai-battle"


def _import_mcts():
    from kego.pipeline.battle import locate_cg_dir

    for p in (str(locate_cg_dir()), str(COMP)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import agents.mcts as mcts

    return mcts


@pytest.fixture(scope="module")
def env():
    try:
        return _import_mcts()
    except Exception as e:
        pytest.skip(f"pokemon MCTS env unavailable: {e}")


# -----------------------------------------------------------------------------
# 1. search.py Edge Cases & Nominal Behaviors
# -----------------------------------------------------------------------------


class TestActionCombinationsEdgeCases:
    def test_nominal_behavior(self, env):
        # Nominal: 2 choices from 4 options
        res = env.enumerate_action_combinations(2, 4, cap=64)
        assert res == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

    def test_cap_zero(self, env):
        # Edge case: cap = 0
        assert env.enumerate_action_combinations(2, 4, cap=0) == []

    def test_zero_num_options(self, env):
        # Edge case: num_options = 0 with max_count = 0
        assert env.enumerate_action_combinations(0, 0) == [[]]

    def test_max_count_greater_than_num_options(self, env):
        # Edge case: max_count > num_options -> clamped to max available num_options
        assert env.enumerate_action_combinations(3, 2) == [[0, 1]]


class TestCreateNodeTerminalAndEdgeCases:
    def test_terminal_win(self, env):
        # Edge case: state.result == your_index -> win (+1.0)
        obs = types.SimpleNamespace(current=types.SimpleNamespace(result=0, yourIndex=0))
        state = types.SimpleNamespace(observation=obs)
        node = env.create_node(None, state, your_index=0, evaluate=None)
        assert node.value == 1.0
        assert node.visit == 1
        assert node.total == 1.0

    def test_terminal_loss(self, env):
        # Edge case: state.result != your_index -> loss (-1.0)
        obs = types.SimpleNamespace(current=types.SimpleNamespace(result=1, yourIndex=0))
        state = types.SimpleNamespace(observation=obs)
        node = env.create_node(None, state, your_index=0, evaluate=None)
        assert node.value == -1.0
        assert node.visit == 1
        assert node.total == -1.0

    def test_terminal_draw(self, env):
        # Edge case: state.result == RESULT_DRAW (2) -> draw (0.0)
        obs = types.SimpleNamespace(current=types.SimpleNamespace(result=env.RESULT_DRAW, yourIndex=0))
        state = types.SimpleNamespace(observation=obs)
        node = env.create_node(None, state, your_index=0, evaluate=None)
        assert node.value == 0.0
        assert node.visit == 1
        assert node.total == 0.0

    def test_non_terminal_opponent_turn_value_negation(self, env):
        # Edge case: Opponent turn (state.yourIndex = 1 != your_index = 0) -> value negated (-v)
        obs = types.SimpleNamespace(
            current=types.SimpleNamespace(result=-1, yourIndex=1),
            select=types.SimpleNamespace(maxCount=1, option=[types.SimpleNamespace(type=1)]),
        )
        state = types.SimpleNamespace(observation=obs)

        def dummy_eval(obs, actions):
            return 0.8, [1.0]

        node = env.create_node(None, state, your_index=0, evaluate=dummy_eval)
        assert node.value == -0.8
        assert node.total == -0.8


class TestSelectChildEdgeCases:
    def test_no_children_returns_none(self, env):
        # Edge case: Node with no children -> returns None
        state = types.SimpleNamespace(observation=types.SimpleNamespace(current=types.SimpleNamespace(yourIndex=0)))
        node = env.Node(None, state)
        assert env.select_child(node, your_index=0) is None

    def test_unvisited_child_uses_parent_q(self, env):
        # Edge case: Child node is None -> q = current.total / current.visit
        state = types.SimpleNamespace(observation=types.SimpleNamespace(current=types.SimpleNamespace(yourIndex=0)))
        node = env.Node(None, state)
        node.visit = 5
        node.total = 2.5  # parent mean q = 0.5
        child = env.Child([0], 0.5)
        node.children = [child]

        chosen = env.select_child(node, your_index=0, c_puct=1.0)
        assert chosen is child


# -----------------------------------------------------------------------------
# 2. opponent.py Edge Cases & Archetype Inferences
# -----------------------------------------------------------------------------


class TestOpponentInferenceEdgeCases:
    def test_no_revealed_signature_cards_fallback(self, env):
        # Edge case: No revealed cards match any archetype signature
        # Should pick randomly from candidates without error
        opponent = types.SimpleNamespace(active=[], bench=[], discard=[], deckCount=40, prize=[None] * 6, handCount=5)
        state = types.SimpleNamespace(players=[types.SimpleNamespace(), opponent], stadium=[])

        hidden = env.infer_opponent_hidden_cards(state, your_index=0, rng=random.Random(42))  # noqa: S311
        assert hidden.archetype in env.ARCHETYPE_DECKS
        assert len(hidden.deck) == 40
        assert len(hidden.prize) == 6
        assert len(hidden.hand) == 5

    def test_overfull_revealed_cards_padded_with_choices(self, env):
        # Edge case: Opponent played more cards than archetype contains
        # Truncation / padding logic must handle without out-of-index error
        opponent = types.SimpleNamespace(active=[], bench=[], discard=[], deckCount=55, prize=[None] * 6, handCount=5)
        state = types.SimpleNamespace(players=[types.SimpleNamespace(), opponent], stadium=[])

        hidden = env.infer_opponent_hidden_cards(state, your_index=0, rng=random.Random(10))  # noqa: S311
        assert len(hidden.deck) == 55
        assert len(hidden.prize) == 6
        assert len(hidden.hand) == 5


# -----------------------------------------------------------------------------
# 3. encoding.py Edge Cases & Action Options
# -----------------------------------------------------------------------------


class TestEncodingActionOptionEdgeCases:
    def test_empty_action_list(self, env):
        # Edge case: Action list contains empty selection []
        obs = types.SimpleNamespace(
            current=types.SimpleNamespace(yourIndex=0, players=[types.SimpleNamespace(hand=[])]),
            select=types.SimpleNamespace(context=0, option=[]),
        )
        sv = env.encode_actions(obs, [[]])
        assert sv.index == [0]
        assert sv.value == [1.0]

    def test_out_of_bounds_tool_and_energy_indices_handled_safely(self, env):
        # Edge case: o.toolIndex or o.energyIndex out of bounds on card
        from cg.api import OptionType

        opt_tool = types.SimpleNamespace(type=OptionType.TOOL_CARD, area=0, index=0, playerIndex=0, toolIndex=5)
        obs = types.SimpleNamespace(
            current=types.SimpleNamespace(yourIndex=0, players=[types.SimpleNamespace(hand=[])]),
            select=types.SimpleNamespace(context=1, option=[opt_tool]),
        )
        # Should execute safely without IndexError
        sv = env.encode_actions(obs, [[0]])
        assert isinstance(sv.index, list)


# -----------------------------------------------------------------------------
# 4. agent.py Deck Prompt & Env Overrides
# -----------------------------------------------------------------------------


class TestAgentEdgeCases:
    def test_deck_query_prompt_returns_deck(self, env, monkeypatch):
        # Edge case: obs.select is None -> act() returns deck list directly
        agent = env.MCTSTransformerAgent(deck=[10, 20, 30])
        obs_dict = {"current": {"yourIndex": 0}, "select": None}
        res = agent.act(obs_dict)
        assert res == [10, 20, 30]

    def test_search_count_env_override(self, env, monkeypatch):
        # Edge case: MCTS_SEARCH_COUNT environment variable overrides variant default
        monkeypatch.setenv("MCTS_SEARCH_COUNT", "42")
        agent = env.MCTSTransformerAgent(deck=[1] * 60)
        assert agent.SEARCH_COUNT == 42
