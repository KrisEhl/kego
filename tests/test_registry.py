import pytest

from kego.tracking import leaderboard, register_checkpoint, resolve_training_resume


def test_leaderboard_ranks_by_elo_descending(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    uri = f"sqlite:///{tmp_path / 'ml.db'}"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    client.create_registered_model("pokemon")
    client.create_model_version("pokemon", source="file:///tmp/a", tags={"elo": "1712", "machine": "wsl"})
    client.create_model_version("pokemon", source="file:///tmp/b", tags={"elo": "1748", "machine": "m5"})
    client.create_model_version("pokemon", source="file:///tmp/c", tags={"elo": "1502", "machine": "m5"})

    board = leaderboard(uri, "pokemon", sort_by="elo")

    assert [float(r["elo"]) for r in board] == [1748.0, 1712.0, 1502.0]
    assert board[0]["machine"] == "m5"
    assert board[0]["version"] == "2"  # the elo=1748 version was registered 2nd


def test_leaderboard_missing_metric_sorts_last(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    uri = f"sqlite:///{tmp_path / 'ml.db'}"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    client.create_registered_model("pokemon")
    client.create_model_version("pokemon", source="file:///tmp/a", tags={"elo": "1600"})
    client.create_model_version("pokemon", source="file:///tmp/b", tags={})  # unrated yet

    board = leaderboard(uri, "pokemon", sort_by="elo")

    assert board[0]["elo"] == "1600"
    assert "elo" not in board[-1]  # the unrated version ranks last


def test_leaderboard_empty_for_unknown_model(tmp_path):
    pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'ml.db'}"
    assert leaderboard(uri, "nonexistent") == []


def test_register_checkpoint_readable_via_leaderboard(tmp_path):
    pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'ml.db'}"
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"fake-weights")

    version = register_checkpoint(uri, "pokemon", str(ckpt), tags={"elo": 1748, "machine": "m5"})

    assert version == "1"
    board = leaderboard(uri, "pokemon", sort_by="elo")
    assert len(board) == 1
    assert board[0]["version"] == "1"
    assert board[0]["machine"] == "m5"
    assert board[0]["checkpoint_filename"] == "model.pth"
    assert float(board[0]["elo"]) == 1748.0  # int tag coerced to a parseable string


def test_register_checkpoint_filename_tag_uses_actual_file(tmp_path):
    pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'ml.db'}"
    ckpt = tmp_path / "mcts_model_iter50.pth"
    ckpt.write_bytes(b"w")

    register_checkpoint(uri, "pokemon", str(ckpt), tags={"checkpoint_filename": "wrong.pth"})

    row = leaderboard(uri, "pokemon", sort_by="version")[0]
    assert row["checkpoint_filename"] == "mcts_model_iter50.pth"


def test_register_checkpoint_increments_version(tmp_path):
    pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'ml.db'}"
    ckpt = tmp_path / "m.pth"
    ckpt.write_bytes(b"w")

    register_checkpoint(uri, "pokemon", str(ckpt), tags={"elo": 1600})
    v2 = register_checkpoint(uri, "pokemon", str(ckpt), tags={"elo": 1700})

    assert v2 == "2"


def test_resolve_training_resume_selects_highest_compatible_iteration(tmp_path):
    pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'ml.db'}"
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"weights")

    for iteration, fingerprint in [(50, "same"), (250, "same"), (290, "different"), (350, "same")]:
        state = tmp_path / f"model_iter{iteration}.train.pt"
        state.write_text(f"state-{iteration}")
        register_checkpoint(
            uri,
            "pokemon",
            str(weights),
            tags={"training_fingerprint": fingerprint, "completed_iterations": iteration},
            training_state_path=str(state),
        )

    resume = resolve_training_resume(uri, "pokemon", "same", target_iterations=300, cache_dir=tmp_path / "cache")

    assert resume is not None
    assert resume.completed_iterations == 250
    assert resume.path.read_text() == "state-250"


def test_resolve_training_resume_accepts_exact_completed_target(tmp_path):
    pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'ml.db'}"
    weights = tmp_path / "model.pth"
    state = tmp_path / "model_iter300.train.pt"
    weights.write_bytes(b"weights")
    state.write_bytes(b"state")
    register_checkpoint(
        uri,
        "pokemon",
        str(weights),
        tags={"training_fingerprint": "same", "completed_iterations": 300},
        training_state_path=str(state),
    )

    resume = resolve_training_resume(uri, "pokemon", "same", target_iterations=300, cache_dir=tmp_path / "cache")

    assert resume is not None
    assert resume.completed_iterations == 300


def test_format_leaderboard_has_header_and_ranked_rows():
    from kego.tracking import format_leaderboard

    rows = [
        {"version": "2", "gauntlet_avg": "71.2", "machine": "wsl"},
        {"version": "1", "gauntlet_avg": "58.0", "machine": "m5"},
    ]
    out = format_leaderboard(rows, columns=["gauntlet_avg", "machine", "version"])
    lines = out.splitlines()
    assert "rank" in lines[0] and "gauntlet_avg" in lines[0]
    assert lines[1].split()[0] == "1"  # first data row is rank 1
    assert "wsl" in lines[1]
    assert lines[2].split()[0] == "2"
    assert "m5" in lines[2]


def test_format_leaderboard_missing_key_shows_dash():
    from kego.tracking import format_leaderboard

    out = format_leaderboard([{"version": "1", "machine": "m5"}], columns=["gauntlet_avg", "machine"])
    assert "-" in out.splitlines()[1]


def test_format_leaderboard_caps_display_width():
    from kego.tracking import format_leaderboard

    out = format_leaderboard(
        [{"version": "1", "git_sha": "b4a1117a785bbed997cd4c93ccfddc12136d516f"}],
        columns=["git_sha", "version"],
        max_widths={"git_sha": 8},
    )

    assert "b4a1117a" in out
    assert "785bbed" not in out


def test_format_leaderboard_colors_elo_by_rating_uncertainty():
    from kego.tracking import format_leaderboard

    out = format_leaderboard(
        [
            {"version": "1", "elo": "1700", "elo_rd": "0.9"},
            {"version": "2", "elo": "1600", "elo_rd": "4.9"},
            {"version": "3", "elo": "1500", "elo_rd": "9.9"},
        ],
        columns=["elo", "version"],
        color_elo=True,
    )

    assert "\x1b[32m1700\x1b[0m" in out
    assert "\x1b[33m1600\x1b[0m" in out
    assert "\x1b[31m1500\x1b[0m" in out
    assert out.splitlines()[1].split()[0] == "1"


def test_format_leaderboard_empty():
    from kego.tracking import format_leaderboard

    assert "no models" in format_leaderboard([], columns=["gauntlet_avg"]).lower()


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
    assert row["gauntlet_avg"] == "90.2"  # training tag untouched
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


def test_leaderboard_excludes_archived_or_dropped_versions(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    uri = f"sqlite:///{tmp_path / 'ml.db'}"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    client.create_registered_model("pokemon")
    client.create_model_version("pokemon", source="file:///tmp/a", tags={"elo": "1600"})
    client.create_model_version("pokemon", source="file:///tmp/b", tags={"elo": "1700", "status": "archived"})
    v3 = client.create_model_version("pokemon", source="file:///tmp/c", tags={"elo": "1800"})

    client.transition_model_version_stage("pokemon", v3.version, stage="Archived")

    board = leaderboard(uri, "pokemon", sort_by="elo")

    assert len(board) == 1
    assert board[0]["version"] == "1"
    assert board[0]["elo"] == "1600"
