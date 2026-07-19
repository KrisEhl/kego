import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_submit_leader():
    path = Path(__file__).resolve().parents[1] / "competitions" / "pokemon-tcg-ai-battle" / "submit_leader.py"
    spec = importlib.util.spec_from_file_location("pokemon_submit_leader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_elo_leader_uses_first_ranked_model():
    module = load_submit_leader()
    rows = [{"version": "31", "elo": 1700}, {"version": "30", "elo": 1680}]

    assert module.select_elo_leader(rows) == rows[0]


def test_select_elo_leader_rejects_empty_registry():
    module = load_submit_leader()

    with pytest.raises(ValueError, match="No models are registered"):
        module.select_elo_leader([])


def test_set_competition_value_replaces_or_adds_selected_model_metadata():
    module = load_submit_leader()
    content = (
        '[competition]\nagent_file = "agents/random.py"\ndeck_file = "decks/random.csv"\n'
        '\n[train]\ndeck_file = "decks/training.csv"\n'
    )

    content = module.set_competition_value(content, "agent_file", "agents/mcts")
    content = module.set_competition_value(content, "deck_file", "decks/abomasnow.csv")
    content = module.set_competition_value(content, "variant", "large256_abomasnow")

    assert 'agent_file = "agents/mcts"' in content
    assert 'deck_file = "decks/abomasnow.csv"' in content
    assert 'variant = "large256_abomasnow"' in content
    assert '[train]\ndeck_file = "decks/training.csv"' in content


def test_validate_variant_metadata_requires_registry_deck_and_architecture(tmp_path):
    module = load_submit_leader()
    variant = tmp_path / "variant.toml"
    variant.write_text('deck_file = "decks/abomasnow.csv"\nmodel_args = [256, 4, 512, 2, 2]\n')

    assert module.validate_variant_metadata(
        "30", "abomasnow", "(256, 4, 512, 2, 2)", "large256_abomasnow", variant
    ) == (256, 4, 512, 2, 2)

    with pytest.raises(ValueError, match="does not match"):
        module.validate_variant_metadata("30", "zacian", "(256, 4, 512, 2, 2)", "large256_abomasnow", variant)

    with pytest.raises(ValueError, match="model_args"):
        module.validate_variant_metadata("30", "abomasnow", "(192, 4, 384, 2, 2)", "large256_abomasnow", variant)


def test_resolve_registry_model_fetches_exact_version_directly():
    module = load_submit_leader()
    archived = SimpleNamespace(version="30", tags={"deck": "abomasnow"}, current_stage="Archived")

    class Client:
        def get_model_version(self, name, version):
            assert (name, version) == ("pokemon-tcg-ai-battle", "30")
            return archived

    selected, version = module.resolve_registry_model(Client(), "http://mlflow", "30")

    assert selected == {"version": "30", "deck": "abomasnow"}
    assert version is archived
