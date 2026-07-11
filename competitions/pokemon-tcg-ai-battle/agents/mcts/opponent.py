"""Opponent archetype detection and hidden-card determinization."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol, TypeVar

from cg.api import State

ARCHETYPE_DECKS: dict[str, tuple[int, ...]] = {
    "abomasnow": (
        721,
        721,
        722,
        722,
        722,
        722,
        723,
        723,
        723,
        723,
        1121,
        1121,
        1121,
        1121,
        1126,
        1192,
        1192,
        1192,
        1192,
        1227,
        1227,
        1227,
        1227,
        1262,
        1262,
        1262,
        *([3] * 34),
    ),
    "dragapult": (
        119,
        119,
        119,
        119,
        120,
        120,
        120,
        120,
        121,
        121,
        121,
        140,
        184,
        235,
        235,
        1071,
        1079,
        1079,
        1080,
        1086,
        1086,
        1086,
        1086,
        1097,
        1097,
        1120,
        1120,
        1120,
        1120,
        1121,
        1121,
        1121,
        1121,
        1152,
        1152,
        1152,
        1156,
        1182,
        1182,
        1182,
        1198,
        1198,
        1198,
        1198,
        1210,
        1210,
        1227,
        1227,
        1227,
        1227,
        1256,
        1256,
        2,
        2,
        2,
        2,
        5,
        5,
        5,
        5,
    ),
    "dragapult_blaziken": (
        119,
        119,
        119,
        119,
        120,
        120,
        120,
        120,
        121,
        121,
        324,
        324,
        325,
        326,
        326,
        235,
        235,
        112,
        112,
        31,
        140,
        272,
        1071,
        1227,
        1227,
        1227,
        1227,
        1182,
        1182,
        1182,
        1198,
        1198,
        1231,
        1086,
        1086,
        1086,
        1086,
        1121,
        1121,
        1121,
        1121,
        1152,
        1152,
        1152,
        1079,
        1079,
        1079,
        1097,
        1097,
        1080,
        1256,
        1260,
        5,
        5,
        5,
        2,
        2,
        2,
        7,
        7,
    ),
    "zacian": (
        336,
        336,
        336,
        336,
        1205,
        1205,
        1205,
        1205,
        1227,
        1227,
        1227,
        1227,
        1235,
        1235,
        1235,
        1235,
        1182,
        1182,
        1198,
        1198,
        1121,
        1121,
        1121,
        1121,
        1123,
        1123,
        1123,
        1123,
        1122,
        1122,
        1122,
        1122,
        1158,
        1097,
        1097,
        1118,
        1118,
        1140,
        1140,
        1140,
        1116,
        1116,
        1116,
        1139,
        1139,
        1139,
        *([8] * 14),
    ),
    "lucario": (
        673,
        673,
        673,
        674,
        674,
        674,
        675,
        675,
        676,
        676,
        676,
        677,
        677,
        677,
        677,
        678,
        678,
        678,
        678,
        1102,
        1102,
        1102,
        1102,
        1123,
        1123,
        1123,
        1141,
        1141,
        1142,
        1142,
        1142,
        1152,
        1152,
        1152,
        1152,
        1159,
        1182,
        1182,
        1192,
        1192,
        1192,
        1192,
        1227,
        1227,
        1227,
        1252,
        1252,
        *([6] * 13),
    ),
}

SIGNATURE_MAP = {
    **dict.fromkeys((3, 721, 722, 723, 1126, 1262), "abomasnow"),
    **dict.fromkeys((8, 336, 1116, 1118, 1122, 1139, 1140, 1158, 1205, 1235), "zacian"),
    **dict.fromkeys((6, 673, 674, 675, 676, 677, 678, 1102, 1141, 1142, 1159, 1252), "lucario"),
    **dict.fromkeys((184, 1120, 1156, 1210), "dragapult"),
    **dict.fromkeys((7, 31, 112, 272, 324, 325, 326, 1231, 1260), "dragapult_blaziken"),
    **dict.fromkeys((2, 5, 119, 120, 121, 140, 235, 1071, 1079, 1080, 1086, 1256), "dragapult_any"),
}


T = TypeVar("T")


class RandomSource(Protocol):
    def choice(self, sequence: list[T]) -> T: ...
    def choices(self, population: tuple[T, ...], *, k: int) -> list[T]: ...
    def shuffle(self, values: list[T]) -> None: ...


@dataclass(frozen=True)
class OpponentHiddenCards:
    archetype: str
    deck: list[int]
    prize: list[int]
    hand: list[int]


def _revealed_cards(state: State, opponent_index: int) -> tuple[list[int], list[int]]:
    opponent = state.players[opponent_index]
    visible: list[int] = []
    for pokemon in [*(opponent.active or []), *(opponent.bench or [])]:
        if pokemon is not None:
            visible.append(pokemon.id)
            visible.extend(card.id for card in (pokemon.tools or []) if card is not None)
            visible.extend(card.id for card in (pokemon.energyCards or []) if card is not None)
    visible.extend(card.id for card in (opponent.discard or []) if card is not None)
    voting = [*visible, *(card.id for card in (state.stadium or []) if card is not None)]
    return visible, voting


def infer_opponent_hidden_cards(state: State, your_index: int, rng: RandomSource) -> OpponentHiddenCards:
    """Infer an opponent archetype and sample its unseen hand, prizes, and deck."""
    opponent = state.players[1 - your_index]
    visible, voting = _revealed_cards(state, 1 - your_index)
    votes = dict.fromkeys(ARCHETYPE_DECKS, 0)
    for card_id in voting:
        if (archetype := SIGNATURE_MAP.get(card_id)) == "dragapult_any":
            votes["dragapult"] += 1
            votes["dragapult_blaziken"] += 1
        elif archetype is not None:
            votes[archetype] += 1

    max_votes = max(votes.values())
    candidates = [name for name, count in votes.items() if count == max_votes] if max_votes else list(votes)
    archetype = rng.choice(candidates)
    full_deck = ARCHETYPE_DECKS[archetype]
    remaining = list((Counter(full_deck) - Counter(visible)).elements())
    needed = opponent.deckCount + len(opponent.prize) + opponent.handCount
    if len(remaining) < needed:
        remaining.extend(rng.choices(full_deck, k=needed - len(remaining)))
    elif len(remaining) > needed:
        remaining = remaining[:needed]
    rng.shuffle(remaining)
    hand_end, prize_end = opponent.handCount, opponent.handCount + len(opponent.prize)
    return OpponentHiddenCards(archetype, remaining[prize_end:], remaining[hand_end:prize_end], remaining[:hand_end])
