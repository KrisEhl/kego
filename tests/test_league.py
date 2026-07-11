import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

from kego.tracking.league import Rating, expected_score, rate_round, results_from_winmatrix, update_player


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


def test_results_from_winmatrix_expands_per_game():
    names = ["v1", "random"]
    wins = [[0, 3], [1, 0]]  # v1 beat random 3, random beat v1 1
    games = [[0, 4], [4, 0]]
    res = results_from_winmatrix(names, wins, games)
    assert sorted(s for _, s in res["v1"]) == [0.0, 1.0, 1.0, 1.0]
    assert all(opp == "random" for opp, _ in res["v1"])
    assert sorted(s for _, s in res["random"]) == [0.0, 0.0, 0.0, 1.0]


def test_rate_round_beating_anchor_raises_and_sharpens():
    anchors = {"random": 1200.0}
    results = {"v1": [("random", 1.0)] * 4, "random": [("v1", 0.0)] * 4}
    out = rate_round({}, results, anchors)
    assert "random" not in out  # anchors never updated
    assert out["v1"].elo > 1500.0  # a new player that wins climbs
    assert out["v1"].rd < 350.0  # and its uncertainty shrinks


def test_rate_round_uses_prior_rating_for_known_player():
    anchors = {"zacian": 1350.0}
    prior = {"v1": Rating(1700.0, 60.0)}
    results = {"v1": [("zacian", 0.0)] * 2}  # v1 unexpectedly loses to a weaker anchor
    out = rate_round(prior, results, anchors)
    assert 1500.0 < out["v1"].elo < 1700.0  # used prior 1700/60 (a default-fallback would crash to ~1132)


def test_download_checkpoint_uses_version_checkpoint_filename(tmp_path):
    league_path = Path(__file__).resolve().parents[1] / "competitions" / "pokemon-tcg-ai-battle" / "run_league.py"
    spec = importlib.util.spec_from_file_location("pokemon_run_league", league_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    artifact_dir = tmp_path / "artifact" / "checkpoint"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "mcts.pth").write_bytes(b"old")
    expected = artifact_dir / "mcts_model_iter50.pth"
    expected.write_bytes(b"new")

    class Client:
        def download_artifacts(self, run_id, artifact_path, dst_path):
            assert run_id == "run1"
            assert artifact_path == "checkpoint"
            return str(artifact_dir)

    version = SimpleNamespace(
        run_id="run1",
        source=f"file://{artifact_dir}",
        tags={"checkpoint_filename": expected.name},
    )

    assert Path(module.download_checkpoint(Client(), version, str(tmp_path / "cache"), debug=True)) == expected


def test_download_checkpoint_uses_cached_single_pth_without_filename_tag(tmp_path):
    league_path = Path(__file__).resolve().parents[1] / "competitions" / "pokemon-tcg-ai-battle" / "run_league.py"
    spec = importlib.util.spec_from_file_location("pokemon_run_league", league_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cache = tmp_path / "cache"
    cache.mkdir()
    expected = cache / "mcts_small192_selfplay48_train100.pth"
    expected.write_bytes(b"w")
    version = SimpleNamespace(run_id="run1", source="/remote/checkpoint", tags={})

    assert Path(module.download_checkpoint(object(), version, str(cache), debug=False)) == expected


def test_download_checkpoint_discovers_remote_pth_before_scp(tmp_path, monkeypatch):
    league_path = Path(__file__).resolve().parents[1] / "competitions" / "pokemon-tcg-ai-battle" / "run_league.py"
    spec = importlib.util.spec_from_file_location("pokemon_run_league", league_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Client:
        def download_artifacts(self, run_id, artifact_path, dst_path):
            raise RuntimeError("mlflow artifact path is stale")

    version = SimpleNamespace(
        run_id="run1",
        source="/home/kristian/mlflow/artifacts/50/run1/artifacts/checkpoint",
        tags={"machine": "omarchyl"},
    )

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "ssh":
            return SimpleNamespace(
                returncode=0,
                stdout="/home/kristian/mlflow/artifacts/50/run1/artifacts/checkpoint/mcts_selfplay96_train100.pth\n",
                stderr="",
            )
        assert cmd[0] == "scp"
        dest = Path(cmd[-1])
        dest.write_bytes(b"weights")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    out = Path(module.download_checkpoint(Client(), version, str(tmp_path / "cache"), debug=False))

    assert out.name == "mcts_selfplay96_train100.pth"
    assert out.read_bytes() == b"weights"
    assert calls[0][:4] == ["ssh", "-o", "BatchMode=yes", "-o"]
    assert calls[1][0] == "scp"


def test_persist_league_artifacts_writes_mergeable_files(tmp_path, monkeypatch):
    league_path = Path(__file__).resolve().parents[1] / "competitions" / "pokemon-tcg-ai-battle" / "run_league.py"
    spec = importlib.util.spec_from_file_location("pokemon_run_league", league_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "comp_dir", tmp_path)

    class Client:
        def __init__(self):
            self.logged = []

        def log_artifacts(self, run_id, local_dir, artifact_path):
            self.logged.append((run_id, Path(local_dir), artifact_path))

    client = Client()
    names = ["Registry v1", "Mega Abomasnow ex"]
    wins = module.np.array([[0, 1], [3, 0]])
    games = module.np.array([[0, 4], [4, 0]])
    results = [
        {"name": "Mega Abomasnow ex", "avg_wr": 75.0, "wins": 3, "games": 4},
        {"name": "Registry v1", "avg_wr": 25.0, "wins": 1, "games": 4},
    ]
    args = SimpleNamespace(
        task="pokemon-tcg-ai-battle",
        games=4,
        search_count=10,
        workers=2,
        write_ratings=False,
        partial_save_every=5000,
    )

    out = module._persist_league_artifacts(
        client,
        "run1",
        names,
        wins,
        games,
        results,
        {"Registry v1": "1"},
        {"Mega Abomasnow ex": 1650.0},
        args,
    )

    assert (out / "matrix.md").read_text().startswith("League Results Matrix")
    assert "Mega Abomasnow ex" in (out / "matrix.md").read_text()
    assert (out / "wins.csv").read_text().splitlines()[0] == "participant,Registry v1,Mega Abomasnow ex"
    meta = json.loads((out / "participants.json").read_text())
    assert meta["name_to_version"] == {"Registry v1": "1"}
    assert meta["args"]["partial_save_every"] == 5000
    assert client.logged == [("run1", out, "league")]


def test_registry_league_refuses_to_guess_missing_deck_tag():
    league_path = Path(__file__).resolve().parents[1] / "competitions" / "pokemon-tcg-ai-battle" / "run_league.py"

    content = league_path.read_text()

    assert 'v.tags.get("deck", "abomasnow")' not in content
    assert "missing required model-version tag 'deck'" in content
    assert "Refusing to guess a deck for registry-backed league evaluation" in content


def test_aggregate_completed_results_skips_pending_and_failed_games():
    league_path = Path(__file__).resolve().parents[1] / "competitions" / "pokemon-tcg-ai-battle" / "run_league.py"
    spec = importlib.util.spec_from_file_location("pokemon_run_league", league_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    wins, games, completed = module._aggregate_completed_results(
        [0, 1, -1, None],
        [(0, 1), (0, 1), (1, 2), (0, 2)],
        3,
    )

    assert completed == 3
    assert wins.tolist() == [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
    assert games.tolist() == [[0, 2, 0], [2, 0, 0], [0, 0, 0]]
