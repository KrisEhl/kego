import pytest

from kego.cli.config import (
    KegoConfig,
    find_competition_dir,
    find_repo_root,
    load_config,
)


@pytest.fixture
def root_toml(tmp_path):
    (tmp_path / ".git").mkdir()
    toml = tmp_path / "kego.toml"
    toml.write_text("""
[cluster]
ray_address = "http://192.168.1.1:8265"
mlflow_uri = "http://192.168.1.1:5000"

[cluster.resources]
default = {num_gpus = 0.5}
heavy = {num_gpus = 1}
""")
    return tmp_path


@pytest.fixture
def competition_toml(root_toml):
    comp_dir = root_toml / "competitions" / "birdclef-2026"
    comp_dir.mkdir(parents=True)
    (comp_dir / "kego.toml").write_text("""
[competition]
slug = "birdclef-2026"
kaggle_user = "aldisued"
enable_gpu = false
submit_file = "submission.csv"
pattern = "inference_only"
inference_notebook = "inference/kernel.py"
checkpoint_dir = "training/outputs"
primary_metric = "cmAP"
""")
    return comp_dir


def test_find_repo_root(root_toml):
    result = find_repo_root(root_toml / "competitions" / "birdclef-2026")
    assert result == root_toml


def test_find_repo_root_raises_when_no_git(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_repo_root(tmp_path)


def test_find_competition_dir(competition_toml):
    result = find_competition_dir(competition_toml)
    assert result == competition_toml


def test_find_competition_dir_returns_none_for_root(root_toml):
    result = find_competition_dir(root_toml)
    assert result is None


def test_load_config_cluster(root_toml, monkeypatch):
    monkeypatch.chdir(root_toml)
    cfg = load_config(repo_root=root_toml, competition_dir=None)
    assert isinstance(cfg, KegoConfig)
    assert cfg.cluster.ray_address == "http://192.168.1.1:8265"
    assert cfg.cluster.mlflow_uri == "http://192.168.1.1:5000"
    assert cfg.competition is None


def test_load_config_with_competition(root_toml, competition_toml, monkeypatch):
    monkeypatch.chdir(competition_toml)
    cfg = load_config(repo_root=root_toml, competition_dir=competition_toml)
    assert cfg.competition is not None
    assert cfg.competition.slug == "birdclef-2026"
    assert cfg.competition.enable_gpu is False
    assert cfg.competition.primary_metric == "cmAP"
    assert cfg.competition.training_notebook is None
