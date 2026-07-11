"""Tests for shared, pure MCTS search helpers extracted from `agents.mcts`.

Part of the behavior-preserving refactor of the pokemon MCTS agent (see
tests/test_pokemon_mcts_golden.py for the feature-encoding goldens). This file holds
unit tests for the pure functions promoted out of `agents/mcts.py` and
`train_agent.py` as the refactor proceeds — later tasks add more tests here.
"""

import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COMP = REPO_ROOT / "competitions" / "pokemon-tcg-ai-battle"


def _import_mcts():
    """Set up sys.path like train_agent.py does and import agents.mcts.

    Raises if the cg engine data isn't available (callers turn that into a skip).
    """
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
    except Exception as e:  # cg engine unavailable, missing competition data, etc.
        pytest.skip(f"pokemon MCTS env unavailable: {e}")


class TestEnumerateActionCombinations:
    def test_single_pick(self, env):
        assert env.enumerate_action_combinations(1, 3) == [[0], [1], [2]]

    def test_two_picks_all_lexicographic_pairs(self, env):
        assert env.enumerate_action_combinations(2, 4) == [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3],
        ]

    def test_caps_at_64(self, env):
        # C(20, 2) == 190 possible combinations; generation must stop at the cap.
        result = env.enumerate_action_combinations(2, 20)
        assert len(result) == 64

    def test_zero_picks_yields_single_empty_combination(self, env):
        assert env.enumerate_action_combinations(0, 5) == [[]]

    def test_full_selection_yields_single_combination(self, env):
        assert env.enumerate_action_combinations(3, 3) == [[0, 1, 2]]


def _state_with_your_index(your_index: int):
    """Minimal stand-in for SearchState: select_child only touches
    ``.state.observation.current.yourIndex`` off the Node it's given."""
    return types.SimpleNamespace(observation=types.SimpleNamespace(current=types.SimpleNamespace(yourIndex=your_index)))


class TestBuildChildren:
    def test_probabilities_sum_to_one_and_preserve_order_and_argmax(self, env):
        node = env.Node(None, None)
        actions = [[0], [1], [2], [3]]
        policy = [0.1, 0.5, -0.3, 0.9]

        env.build_children(node, actions, policy)

        assert [c.select for c in node.children] == actions
        probs = [c.prob for c in node.children]
        assert sum(probs) == pytest.approx(1.0)
        # Softmax is monotonic in the policy logits, so the argmax (and full order)
        # of the resulting probabilities must match the argmax (order) of the policy.
        assert probs.index(max(probs)) == policy.index(max(policy))
        assert sorted(range(len(policy)), key=lambda i: policy[i]) == sorted(range(len(probs)), key=lambda i: probs[i])


class TestSelectChild:
    def test_prefers_higher_prior_unvisited_child_on_fresh_node(self, env):
        # "Fresh" node: it has been backpropped once (visit=1, giving it a mean value)
        # but none of its children have been expanded yet (child.node is None), which
        # is exactly the state a just-created node is in during search.
        current = env.Node(None, _state_with_your_index(0))
        current.visit = 1
        current.total = 0.3
        low_prior = env.Child([0], 0.2)
        high_prior = env.Child([1], 0.5)
        mid_prior = env.Child([2], 0.3)
        current.children = [low_prior, high_prior, mid_prior]

        chosen = env.select_child(current, your_index=0)

        assert chosen is high_prior

    def test_opponent_to_move_selects_child_minimizing_q(self, env):
        # The node's state says the opponent (yourIndex=1) is on the move while we are
        # your_index=0, so select_child must flip the sign of q before comparing —
        # i.e. it should pick the child whose own q is the *smallest* (best for us).
        state = _state_with_your_index(1)
        current = env.Node(None, state)
        current.visit = 0
        current.total = 0.0

        high_q_child = env.Child([0], 0.9)
        high_q_child.node = env.Node(current, state)
        high_q_child.node.total = 0.8
        high_q_child.node.visit = 1

        low_q_child = env.Child([1], 0.9)
        low_q_child.node = env.Node(current, state)
        low_q_child.node.total = -0.5
        low_q_child.node.visit = 1

        current.children = [high_q_child, low_q_child]

        chosen = env.select_child(current, your_index=0)

        assert chosen is low_q_child
