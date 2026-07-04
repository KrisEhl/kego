# League Elo Rating (Minimal Slice) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `make leaderboard` continuously update `kego models`: each round of games updates an anchored Glicko Elo rating written back to the registry, so the leaderboard gets sharper as more games are played and ranks agents that have already saturated the fixed-opponent `gauntlet_avg`.

**Architecture:** Pure, unit-testable Glicko-1 rating math + tournament-matrix→results expansion live in a new core module `kego/tracking/league.py`. Persistence (read/write `elo`/`elo_rd`/`games` tags) lives in `kego/tracking/registry.py`. The pokemon `run_league.py` (which already reads the registry, downloads checkpoints, and plays the round-robin in parallel) is the thin orchestrator: it maps its win matrix to rated players + pinned anchors, calls the core rating functions, and writes results back. `kego models` ranks by `elo` (falling back gracefully when unrated). Ratings accumulate **incrementally** through the tags (read prior → update with this round → write back); auditable match-history artifacts and the bounded ladder/cron/dedup/offline machinery from §5.6 are deferred.

**Tech Stack:** Python 3.10+, MLflow Model Registry (sqlite/http), stdlib `math` for Glicko-1 (no new dependencies), pytest with `sqlite:///` MLflow for tests.

## Global Constraints

- Python 3.10+ type hints throughout (`from __future__ import annotations` where helpful).
- **No new dependencies** — Glicko-1 is pure `math`.
- MLflow is imported **lazily inside functions**, never at module top level (matches `kego/tracking/registry.py`).
- All registry tag values are **strings**; readers coerce (`float(...)`) and rank missing/unparsable last (matches existing `leaderboard()`).
- Pinned anchor Elos (from spec Appendix A.4), keyed by the `run_league.py` participant display names:
  `Random=1200`, `Zacian ex=1350`, `Mega Lucario ex=1450`, `Dragapult ex=1520`, `Mega Abomasnow ex=1650`.
- Glicko-1 constants: `Q = ln(10)/400`, new-agent rating `1500`, new-agent RD `350`, anchor RD `30`.
- `gauntlet_avg`/`wr_*` are written by training only and MUST NOT be overwritten by the league. The league writes only `elo`, `elo_rd`, `games`, `rating_status`.
- Run tests with `uv run pytest`.

---

### Task 1: Glicko-1 rating primitives (core, pure)

**Files:**
- Create: `kego/tracking/league.py`
- Test: `tests/test_league.py`

**Interfaces:**
- Consumes: nothing (pure stdlib).
- Produces:
  - `Rating` — `@dataclass(frozen=True)` with fields `elo: float`, `rd: float`.
  - `expected_score(rating: Rating, opp: Rating) -> float`
  - `update_player(rating: Rating, results: list[tuple[Rating, float]]) -> Rating` — one Glicko-1 rating period; `results` is `[(opponent_rating, score), ...]` with `score ∈ {0.0, 0.5, 1.0}`; empty `results` returns `rating` unchanged.
  - Module constants: `Q`, `DEFAULT_RATING = 1500.0`, `DEFAULT_RD = 350.0`, `ANCHOR_RD = 30.0`.

- [ ] **Step 1: Write the failing test** (canonical Glickman worked example)

`tests/test_league.py`:
```python
import math

from kego.tracking.league import Rating, expected_score, update_player


def test_update_player_matches_glickman_example():
    # Glickman's canonical example: player 1500/200 vs three opponents.
    player = Rating(1500.0, 200.0)
    results = [
        (Rating(1400.0, 30.0), 1.0),
        (Rating(1550.0, 100.0), 0.0),
        (Rating(1700.0, 300.0), 0.0),
    ]
    updated = update_player(player, results)
    assert math.isclose(updated.elo, 1464.1, abs_tol=1.0)
    assert math.isclose(updated.rd, 151.4, abs_tol=1.0)


def test_update_player_no_results_is_unchanged():
    player = Rating(1500.0, 350.0)
    assert update_player(player, []) == player


def test_expected_score_even_when_equal():
    assert math.isclose(expected_score(Rating(1500.0, 0.0), Rating(1500.0, 0.0)), 0.5, abs_tol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_league.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kego.tracking.league'`

- [ ] **Step 3: Write minimal implementation**

`kego/tracking/league.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_league.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add kego/tracking/league.py tests/test_league.py
git commit -m "feat(league): Glicko-1 rating primitives"
```

---

### Task 2: Round rating update + win-matrix expansion (core, pure)

**Files:**
- Modify: `kego/tracking/league.py`
- Test: `tests/test_league.py`

**Interfaces:**
- Consumes: `Rating`, `update_player`, `DEFAULT_RATING`, `DEFAULT_RD`, `ANCHOR_RD` from Task 1.
- Produces:
  - `results_from_winmatrix(names: list[str], wins: list[list[float]], games: list[list[float]]) -> dict[str, list[tuple[str, float]]]` — expands a wins/games matrix into per-player `(opponent_name, score)` game outcomes (one entry per game; `1.0` for a win, `0.0` for a loss). `wins[i][j]` = games `i` won vs `j`; `games[i][j]` = games played between `i` and `j`.
  - `rate_round(prior: dict[str, Rating], results: dict[str, list[tuple[str, float]]], anchors: dict[str, float], *, initial: Rating = Rating(DEFAULT_RATING, DEFAULT_RD), anchor_rd: float = ANCHOR_RD) -> dict[str, Rating]` — returns updated ratings for every **non-anchor** player named in `results`, using **pre-round** ratings for all opponents. Anchor names resolve to `Rating(anchors[name], anchor_rd)` and are never updated. Unknown non-anchor players enter at `initial`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_league.py`:
```python
from kego.tracking.league import rate_round, results_from_winmatrix


def test_results_from_winmatrix_expands_per_game():
    names = ["v1", "random"]
    wins = [[0, 3], [1, 0]]   # v1 beat random 3, random beat v1 1
    games = [[0, 4], [4, 0]]
    res = results_from_winmatrix(names, wins, games)
    assert sorted(s for _, s in res["v1"]) == [0.0, 1.0, 1.0, 1.0]
    assert all(opp == "random" for opp, _ in res["v1"])
    assert sorted(s for _, s in res["random"]) == [0.0, 0.0, 0.0, 1.0]


def test_rate_round_beating_anchor_raises_and_sharpens():
    anchors = {"random": 1200.0}
    results = {"v1": [("random", 1.0)] * 4, "random": [("v1", 0.0)] * 4}
    out = rate_round({}, results, anchors)
    assert "random" not in out              # anchors never updated
    assert out["v1"].elo > 1500.0           # a new player that wins climbs
    assert out["v1"].rd < 350.0             # and its uncertainty shrinks


def test_rate_round_uses_prior_rating_for_known_player():
    anchors = {"zacian": 1350.0}
    prior = {"v1": Rating(1700.0, 60.0)}
    results = {"v1": [("zacian", 0.0)] * 2}  # v1 unexpectedly loses to a weaker anchor
    out = rate_round(prior, results, anchors)
    assert out["v1"].elo < 1700.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_league.py -k "winmatrix or rate_round" -v`
Expected: FAIL with `ImportError: cannot import name 'rate_round'`

- [ ] **Step 3: Write minimal implementation**

Append to `kego/tracking/league.py`:
```python
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


def rate_round(
    prior: dict[str, Rating],
    results: dict[str, list[tuple[str, float]]],
    anchors: dict[str, float],
    *,
    initial: Rating = Rating(DEFAULT_RATING, DEFAULT_RD),
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_league.py -v`
Expected: PASS (6 tests total)

- [ ] **Step 5: Commit**

```bash
git add kego/tracking/league.py tests/test_league.py
git commit -m "feat(league): round rating update + win-matrix expansion"
```

---

### Task 3: Read/write ratings on the registry (core)

**Files:**
- Modify: `kego/tracking/registry.py`
- Modify: `kego/tracking/__init__.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: existing `leaderboard(uri, name, sort_by=...)` from `registry.py`.
- Produces:
  - `read_ratings(uri: str, name: str) -> dict[str, dict]` — maps `version -> {"elo": float, "elo_rd": float, "games": int}` for versions that already carry an `elo` tag; versions never rated are omitted (caller supplies defaults). `elo_rd` defaults to `350.0` and `games` to `0` if those tags are missing.
  - `write_ratings(uri: str, name: str, ratings: dict[str, dict]) -> None` — for each `version -> {"elo", "elo_rd", "games"}`, sets string tags `elo` (1 dp), `elo_rd` (1 dp), `games` (int), and `rating_status="rated"` via `MlflowClient.set_model_version_tag`.
- Both are re-exported from `kego.tracking`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_registry.py`:
```python
def test_write_and_read_ratings_round_trip(tmp_path):
    pytest.importorskip("mlflow")
    from kego.tracking import read_ratings, register_checkpoint, write_ratings

    uri = f"sqlite:///{tmp_path / 'ml.db'}"
    ckpt = tmp_path / "m.pth"
    ckpt.write_bytes(b"w")
    register_checkpoint(uri, "pokemon", str(ckpt), tags={"gauntlet_avg": 90.2})

    write_ratings(uri, "pokemon", {"1": {"elo": 1748.04, "elo_rd": 41.2, "games": 140}})

    ratings = read_ratings(uri, "pokemon")
    assert ratings["1"]["elo"] == 1748.0
    assert ratings["1"]["elo_rd"] == 41.2
    assert ratings["1"]["games"] == 140


def test_write_ratings_preserves_training_tags(tmp_path):
    pytest.importorskip("mlflow")
    from kego.tracking import leaderboard, register_checkpoint, write_ratings

    uri = f"sqlite:///{tmp_path / 'ml.db'}"
    ckpt = tmp_path / "m.pth"
    ckpt.write_bytes(b"w")
    register_checkpoint(uri, "pokemon", str(ckpt), tags={"gauntlet_avg": 90.2, "wr_random": 100})

    write_ratings(uri, "pokemon", {"1": {"elo": 1600.0, "elo_rd": 200.0, "games": 8}})

    row = leaderboard(uri, "pokemon", sort_by="elo")[0]
    assert row["gauntlet_avg"] == "90.2"   # training tag untouched
    assert row["wr_random"] == "100"
    assert row["rating_status"] == "rated"


def test_read_ratings_skips_unrated(tmp_path):
    pytest.importorskip("mlflow")
    from kego.tracking import read_ratings, register_checkpoint

    uri = f"sqlite:///{tmp_path / 'ml.db'}"
    ckpt = tmp_path / "m.pth"
    ckpt.write_bytes(b"w")
    register_checkpoint(uri, "pokemon", str(ckpt), tags={"gauntlet_avg": 90.2})
    assert read_ratings(uri, "pokemon") == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry.py -k ratings -v`
Expected: FAIL with `ImportError: cannot import name 'read_ratings'`

- [ ] **Step 3: Write minimal implementation**

Append to `kego/tracking/registry.py`:
```python
def read_ratings(uri: str, name: str) -> dict[str, dict]:
    """version -> {"elo", "elo_rd", "games"} for versions already carrying an ``elo`` tag."""
    out: dict[str, dict] = {}
    for row in leaderboard(uri, name, sort_by="version"):
        if "elo" not in row:
            continue
        out[row["version"]] = {
            "elo": float(row["elo"]),
            "elo_rd": float(row.get("elo_rd", 350.0)),
            "games": int(float(row.get("games", 0))),
        }
    return out


def write_ratings(uri: str, name: str, ratings: dict[str, dict]) -> None:
    """Write ``elo``/``elo_rd``/``games``/``rating_status`` tags onto each registry version."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    for version, r in ratings.items():
        client.set_model_version_tag(name, str(version), "elo", str(round(r["elo"], 1)))
        client.set_model_version_tag(name, str(version), "elo_rd", str(round(r["elo_rd"], 1)))
        client.set_model_version_tag(name, str(version), "games", str(int(r["games"])))
        client.set_model_version_tag(name, str(version), "rating_status", "rated")
```

Then extend the exports in `kego/tracking/__init__.py`. Change the registry import line and `__all__`:
```python
from .registry import format_leaderboard, leaderboard, read_ratings, register_checkpoint, write_ratings
```
and add `"read_ratings",` and `"write_ratings",` to the `__all__` list (keep it alphabetically sorted).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry.py -v`
Expected: PASS (all existing + 3 new)

- [ ] **Step 5: Commit**

```bash
git add kego/tracking/registry.py kego/tracking/__init__.py tests/test_registry.py
git commit -m "feat(league): read/write elo ratings on the registry"
```

---

### Task 4: `kego models` ranks by Elo

**Files:**
- Modify: `kego/pipeline/cli.py` (parser default at ~line 76; `models` handler at ~line 178)
- Test: `tests/test_new_commands.py` (`test_models_parser` at ~line 444)

**Interfaces:**
- Consumes: `leaderboard`, `format_leaderboard`, `default_tracking_uri` (already imported in the handler).
- Produces: `kego models` defaults to `--sort-by elo`; the non-breakdown column set becomes `[sort_by, "elo_rd", "games", "gauntlet_avg", "machine", "git_sha", "version"]`. The `--breakdown` behaviour from the current code is unchanged.

- [ ] **Step 1: Write the failing test** (update the existing parser test)

Replace `test_models_parser` in `tests/test_new_commands.py` with:
```python
def test_models_parser():
    parser = build_parser()

    args = parser.parse_args(["models", "--task", "pokemon-tcg-ai-battle"])
    assert args.command == "models"
    assert args.sort_by == "elo"          # default now ranks by league Elo
    assert args.breakdown is False

    args = parser.parse_args(["models", "--task", "x", "--sort-by", "gauntlet_avg", "-b"])
    assert args.sort_by == "gauntlet_avg"
    assert args.breakdown is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_new_commands.py::test_models_parser -v`
Expected: FAIL — `assert 'gauntlet_avg' == 'elo'` (current default is `gauntlet_avg`)

- [ ] **Step 3: Write minimal implementation**

In `kego/pipeline/cli.py`, change the `models` parser default:
```python
    models.add_argument("--sort-by", default="elo", help="metric tag to rank agents by")
```

In the `models` handler, change the non-breakdown `base` column list:
```python
        if args.breakdown:
            wr_cols = sorted({k for r in rows for k in r if k.startswith("wr_")})
            base = [args.sort_by, "gauntlet_avg", *wr_cols, "version"]
        else:
            base = [args.sort_by, "elo_rd", "games", "gauntlet_avg", "machine", "git_sha", "version"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_new_commands.py::test_models_parser -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kego/pipeline/cli.py tests/test_new_commands.py
git commit -m "feat(league): kego models ranks by Elo, shows elo_rd/games"
```

---

### Task 5: Wire the pokemon league runner to rate + write back; retire the duplicate

**Files:**
- Modify: `competitions/pokemon-tcg-ai-battle/run_league.py` (participant construction ~line 230-273; after aggregation ~line 338)
- Delete: `competitions/pokemon-tcg-ai-battle/run_tournament.py` (stale duplicate; not registry-aware, no Elo)
- Modify: `competitions/pokemon-tcg-ai-battle/Makefile` (comment on the `leaderboard` target ~line 48)

**Interfaces:**
- Consumes: `results_from_winmatrix`, `rate_round`, `Rating` from `kego.tracking.league`; `read_ratings`, `write_ratings` from `kego.tracking`; `default_tracking_uri`.
- Produces: after each `make leaderboard`, every registry-backed participant's `elo`/`elo_rd`/`games`/`rating_status` tags are updated (incrementally: prior tags + this round). No new interface consumed by other tasks.

This task is integration glue over the game engine, so it is verified by a live run, not a unit test — the pure pieces it calls are already covered by Tasks 1-3.

- [ ] **Step 1: Track registry versions alongside participant names**

In `run_league.py`, when building `model_checkpoints`, the registered participants are named `f"Registry v{v.version}"`. Add a name→version map so results can be written back. After the `participants` dict is fully built (right before `participant_names = list(participants.keys())` at ~line 279), add:
```python
    # Map participant display name -> registry version (only registry-backed agents get rated).
    name_to_version = {f"Registry v{v.version}": str(v.version) for v in versions}

    # Pinned anchor Elos (spec Appendix A.4), keyed by participant display name.
    anchor_elos = {
        "Random": 1200.0,
        "Zacian ex": 1350.0,
        "Mega Lucario ex": 1450.0,
        "Dragapult ex": 1520.0,
        "Mega Abomasnow ex": 1650.0,
    }
```

- [ ] **Step 2: After aggregation, compute and persist ratings**

At the end of `main()`, after the wins/games matrices are filled and **before** the markdown table is printed (after the `wins_matrix[j][i] += 1` aggregation loop, ~line 338), insert:
```python
    # --- League Elo: update ratings from this round and write back to the registry ---
    from kego.tracking import read_ratings, write_ratings
    from kego.tracking.league import DEFAULT_RATING, DEFAULT_RD, Rating, rate_round, results_from_winmatrix

    round_results = results_from_winmatrix(participant_names, wins_matrix.tolist(), games_matrix.tolist())

    # Prior ratings keyed by display name (from registry tags); unrated players default later.
    version_ratings = read_ratings(uri, args.task)
    prior = {
        name: Rating(version_ratings[v]["elo"], version_ratings[v]["elo_rd"])
        for name, v in name_to_version.items()
        if v in version_ratings
    }
    updated = rate_round(prior, round_results, anchor_elos)

    # Games this round per player = number of outcomes emitted for it.
    round_games = {name: len(res) for name, res in round_results.items()}
    prior_games = {name: version_ratings.get(v, {}).get("games", 0) for name, v in name_to_version.items()}

    ratings_by_version = {
        name_to_version[name]: {
            "elo": r.elo,
            "elo_rd": r.rd,
            "games": prior_games.get(name, 0) + round_games.get(name, 0),
        }
        for name, r in updated.items()
        if name in name_to_version  # skip "Local (...)" and anything not in the registry
    }
    if ratings_by_version:
        write_ratings(uri, args.task, ratings_by_version)
        print(f"\nUpdated Elo ratings for {len(ratings_by_version)} registered agent(s): "
              f"{sorted(ratings_by_version)}")
    else:
        print("\nNo registry-backed agents in this league — no ratings written.")
```

- [ ] **Step 3: Delete the stale duplicate script**

```bash
git rm competitions/pokemon-tcg-ai-battle/run_tournament.py
```

- [ ] **Step 4: Update the Makefile comment**

In `competitions/pokemon-tcg-ai-battle/Makefile`, update the comment above the `leaderboard` target (~line 48) to reflect that it now persists Elo:
```makefile
# Run a round-robin round, update each registered agent's league Elo, and write it back
# to the registry (so `make models` reflects the new games). More rounds = sharper ratings.
#   make leaderboard GAMES=4 SEARCH_COUNT=10 WORKERS=8 DEBUG=1
```

- [ ] **Step 5: Verify the pure glue compiles and the module imports**

Run: `uv run python -c "import ast; ast.parse(open('competitions/pokemon-tcg-ai-battle/run_league.py').read()); print('run_league.py parses')"`
Expected: `run_league.py parses`

Run: `uv run pytest tests/test_league.py tests/test_registry.py -v`
Expected: PASS (all rating tests still green)

- [ ] **Step 6: Live verification (requires the hub up + a checkpoint registered)**

Run: `cd competitions/pokemon-tcg-ai-battle && make leaderboard GAMES=2`
Expected: prints the round-robin matrix, then `Updated Elo ratings for N registered agent(s): [...]`.

Run: `uv run kego models`
Expected: ranked by `elo`, with `elo`/`elo_rd`/`games` populated (no longer `-`).

Run: `cd competitions/pokemon-tcg-ai-battle && make leaderboard GAMES=2` (again)
Expected: `games` increases and `elo_rd` shrinks vs the previous run — confirming accumulation.

- [ ] **Step 7: Commit**

```bash
git add competitions/pokemon-tcg-ai-battle/run_league.py competitions/pokemon-tcg-ai-battle/Makefile
git commit -m "feat(league): persist Elo from make leaderboard; retire run_tournament.py"
```

---

## Deferred to full §5.6 (explicitly out of scope here)

- Bounded ladder / opponent sampling (this slice is a full round-robin; fine at 2-5 agents).
- Auditable match-history CSV artifacts in a `league-<task>` experiment (this slice accumulates via tags only).
- Hub cron + `rating_status=pending` placement queue + single-writer serialization.
- Checkpoint dedup (git_sha + weight hash) and `active=false` retirement.
- Offline reconciliation on `kego fleet sync`.
- Generic `kego league run --task` CLI with a competition player-instantiation plug-in (this slice keeps orchestration in `run_league.py`).
```
