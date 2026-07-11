from collections import Counter
from pathlib import Path

import pytest

from cg.api import all_card_data
from cg.game import battle_finish, battle_start

DECK_DIR = Path(__file__).resolve().parents[1] / "competitions/pokemon-tcg-ai-battle/decks"
DECK_PATHS = sorted(DECK_DIR.glob("*.csv"))
CARDS_BY_ID = {card.cardId: card for card in all_card_data()}


def load_deck(path: Path) -> list[int]:
    return [int(line) for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]


@pytest.mark.parametrize("path", DECK_PATHS, ids=lambda path: path.stem)
def test_saved_deck_has_exactly_60_cards(path):
    assert len(load_deck(path)) == 60


@pytest.mark.parametrize("path", DECK_PATHS, ids=lambda path: path.stem)
def test_saved_deck_uses_known_simulator_card_ids(path):
    unknown_ids = set(load_deck(path)) - CARDS_BY_ID.keys()
    assert unknown_ids == set()


@pytest.mark.parametrize("path", DECK_PATHS, ids=lambda path: path.stem)
def test_saved_deck_respects_four_copy_limit(path):
    names = Counter(CARDS_BY_ID[card_id].name for card_id in load_deck(path) if CARDS_BY_ID[card_id].cardType != 5)
    assert {name: count for name, count in names.items() if count > 4} == {}


@pytest.mark.parametrize("path", DECK_PATHS, ids=lambda path: path.stem)
def test_saved_deck_has_at_most_one_ace_spec(path):
    ace_specs = [CARDS_BY_ID[card_id].name for card_id in load_deck(path) if CARDS_BY_ID[card_id].aceSpec]
    assert len(ace_specs) <= 1


@pytest.mark.parametrize("path", DECK_PATHS, ids=lambda path: path.stem)
def test_saved_deck_contains_a_basic_pokemon(path):
    assert any(CARDS_BY_ID[card_id].basic for card_id in load_deck(path))


@pytest.mark.parametrize("path", DECK_PATHS, ids=lambda path: path.stem)
def test_saved_deck_is_accepted_by_battle_engine(path):
    deck = load_deck(path)
    try:
        battle_start(deck, deck)
    finally:
        battle_finish()
