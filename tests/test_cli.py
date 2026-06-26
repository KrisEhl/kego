from kego.pipeline.cli import build_parser
from kego.pipeline.config import PipelineConfig, expand_grid, load_config
from kego.pipeline.tune import HPSpace


def test_hp_space_parse():
    # Test numeric parsing
    hp = HPSpace.parse("max_trees::0:9:log")
    assert hp.name == "max_trees"
    assert hp.type == "int"
    assert hp.low == 0
    assert hp.high == 9
    assert hp.log is True

    hp = HPSpace.parse("depth::4:10:int")
    assert hp.name == "depth"
    assert hp.type == "int"
    assert hp.low == 4
    assert hp.high == 10
    assert hp.log is False

    # Test categorical parsing
    hp = HPSpace.parse("model_type::xgb,cat,lgbm:categorical")
    assert hp.name == "model_type"
    assert hp.type == "categorical"
    assert hp.choices == ["xgb", "cat", "lgbm"]


def test_cli_parser_structure():
    parser = build_parser()

    # Test a valid run command
    args = parser.parse_args(
        [
            "run",
            "--model",
            "catboost",
            "--params",
            "learning_rate:0.01",
            "--hp-tune",
            "--hp-params",
            "max_trees::0:9:log",
        ]
    )
    assert args.command == "run"
    assert args.model == "catboost"
    assert args.params == ["learning_rate:0.01"]
    assert args.hp_tune is True
    assert args.hp_params == ["max_trees::0:9:log"]


def test_config_load_and_override(tmp_path, monkeypatch):
    # Create a dummy config YAML
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_file = config_dir / "test_config.yaml"

    yaml_content = """
task: my_competition
models:
  - name: xgboost
    hyper_params:
      max_depth: 6
grid:
  feature_sets:
    - baseline
  seeds:
    - 42
"""
    config_file.write_text(yaml_content)

    # Temporarily change working directory to tmp_path to resolve 'configs/'
    monkeypatch.chdir(tmp_path)

    # Load config and apply overrides
    config = load_config("test_config", overrides=["featureset:v2", "models.0.hyper_params.max_depth=8"])

    assert config.task == "my_competition"
    assert config.grid.feature_sets == ["v2"]
    assert config.models[0].name == "xgboost"
    assert config.models[0].hyper_params["max_depth"] == 8


def test_expand_grid():
    from kego.pipeline.config import FoldScheme, GridConfig, ModelConfig

    config = PipelineConfig(
        task="dummy",
        models=[ModelConfig(name="xgb"), ModelConfig(name="cat")],
        grid=GridConfig(feature_sets=["f1", "f2"], folds=[FoldScheme(n=5)], seeds=[42, 43]),
    )

    specs = expand_grid(config)
    # Expected models (2) * feature_sets (2) * folds (1) * seeds (2) = 8 specs
    assert len(specs) == 8
    assert specs[0].model.name == "xgb"
    assert specs[0].feature_set == "f1"
    assert specs[0].seed == 42
