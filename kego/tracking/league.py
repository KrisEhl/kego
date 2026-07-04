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


def results_from_winmatrix(
    names: list[str], wins: list[list[float]], games: list[list[float]]
) -> dict[str, list[tuple[str, float]]]:
    """Expand a wins/games matrix into per-player ``(opponent, score)`` game outcomes."""
    out: dict[str, list[tuple[str, float]]] = {n: [] for n in names}
    n = len(names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            played = int(games[i][j])
            won = int(wins[i][j])
            for k in range(played):
                out[names[i]].append((names[j], 1.0 if k < won else 0.0))
    return out


_DEFAULT_INITIAL_RATING = Rating(DEFAULT_RATING, DEFAULT_RD)  # module-level singleton (Rating is frozen/immutable)


def rate_round(
    prior: dict[str, Rating],
    results: dict[str, list[tuple[str, float]]],
    anchors: dict[str, float],
    *,
    initial: Rating = _DEFAULT_INITIAL_RATING,
    anchor_rd: float = ANCHOR_RD,
) -> dict[str, Rating]:
    """Update every non-anchor player from this round's results (pre-round opponent ratings)."""

    def rating_of(name: str) -> Rating:
        if name in anchors:
            return Rating(anchors[name], anchor_rd)
        return prior.get(name, initial)

    updated: dict[str, Rating] = {}
    for player, games_played in results.items():
        if player in anchors:
            continue
        updated[player] = update_player(rating_of(player), [(rating_of(o), s) for o, s in games_played])
    return updated
