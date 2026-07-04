"""Anchored Glicko-1 league rating — the relevance-preserving ranking metric (spec §5.6)."""

from __future__ import annotations

import math
from dataclasses import dataclass

Q = math.log(10.0) / 400.0
DEFAULT_RATING = 1500.0
DEFAULT_RD = 350.0
ANCHOR_RD = 30.0  # anchors are well-established, so they inform strongly


@dataclass(frozen=True)
class Rating:
    elo: float
    rd: float


def _g(rd: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * Q**2 * rd**2 / math.pi**2)


def expected_score(rating: Rating, opp: Rating) -> float:
    return 1.0 / (1.0 + 10.0 ** (-_g(opp.rd) * (rating.elo - opp.elo) / 400.0))


def update_player(rating: Rating, results: list[tuple[Rating, float]]) -> Rating:
    """One Glicko-1 rating period. ``results`` = [(opponent_rating, score in {0,0.5,1}), ...]."""
    if not results:
        return rating
    d2_inv = 0.0
    delta = 0.0
    for opp, score in results:
        g = _g(opp.rd)
        e = expected_score(rating, opp)
        d2_inv += Q**2 * g**2 * e * (1.0 - e)
        delta += g * (score - e)
    denom = 1.0 / rating.rd**2 + d2_inv
    return Rating(rating.elo + (Q / denom) * delta, math.sqrt(1.0 / denom))
