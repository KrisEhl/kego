import pytest

from kego.tracking import leaderboard, register_checkpoint


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
    assert float(board[0]["elo"]) == 1748.0  # int tag coerced to a parseable string


def test_register_checkpoint_increments_version(tmp_path):
    pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'ml.db'}"
    ckpt = tmp_path / "m.pth"
    ckpt.write_bytes(b"w")

    register_checkpoint(uri, "pokemon", str(ckpt), tags={"elo": 1600})
    v2 = register_checkpoint(uri, "pokemon", str(ckpt), tags={"elo": 1700})

    assert v2 == "2"
