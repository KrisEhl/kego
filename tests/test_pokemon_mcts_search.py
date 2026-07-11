"""Tests for shared, pure MCTS search helpers extracted from `agents.mcts`.

Part of the behavior-preserving refactor of the pokemon MCTS agent (see
tests/test_pokemon_mcts_golden.py for the feature-encoding goldens). This file holds
unit tests for the pure functions promoted out of `agents/mcts.py` and
`train_agent.py` as the refactor proceeds — later tasks add more tests here.
"""

import sys
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
