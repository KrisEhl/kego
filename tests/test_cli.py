import pytest

from kego.pipeline.cli import build_parser
from kego.pipeline.config import PipelineConfig, expand_grid, load_config
from kego.pipeline.tune import HPSpace


def test_every_subcommand_help_renders():
    """argparse %-expands ALL help strings (that's how %(default)s works), so an unescaped
    literal % (e.g. "95% confidence") crashes -h with `TypeError: %c requires int or char`."""
    parser = build_parser()

    def walk(p):
        p.format_help()
        for action in p._actions:
            if hasattr(action, "choices") and isinstance(action.choices, dict):
                for sub in action.choices.values():
                    walk(sub)

    walk(parser)


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


def test_train_agent_rejects_config_params():
    """Catch train-agent silently accepting overrides that its training task ignores."""
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["train-agent", "--agent", "mcts", "--variant", "small192_zacian", "--params", "train_steps=500"]
        )


def test_train_agent_requires_agent_and_variant():
    """--agent and --variant are mandatory so training is always fully specified."""
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["train-agent", "--variant", "small192_zacian"])

    with pytest.raises(SystemExit):
        parser.parse_args(["train-agent", "--agent", "mcts"])


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


def test_logs_subcommand_parser():
    parser = build_parser()

    # Parse target and default options
    args = parser.parse_args(["logs", "omarchyl"])
    assert args.command == "logs"
    assert args.target_or_run_id == "omarchyl"
    assert args.run_id is None
    assert args.tail is False
    assert args.lines == 100

    # Parse with run ID, tail, and line count overrides
    args = parser.parse_args(["logs", "omarchyd", "5034b3b0", "--tail", "--lines", "25"])
    assert args.command == "logs"
    assert args.target_or_run_id == "omarchyd"
    assert args.run_id == "5034b3b0"
    assert args.tail is True
    assert args.lines == 25


def test_run_logs_local_resolution(tmp_path, monkeypatch, capsys):
    from kego.pipeline.cli import _run_logs

    # Mock find_repo_root to return our tmp_path
    monkeypatch.setattr("kego.fleet.repo_root", lambda *a: tmp_path)
    monkeypatch.setattr("kego.fleet.machine_name", lambda *a: "localhost")

    # Setup a mock local logs directory
    local_logs_dir = tmp_path / ".kego" / "logs"
    local_logs_dir.mkdir(parents=True)

    # Create two dummy logs with different mtimes
    log1 = local_logs_dir / "abc12345678901234567890123456789.log"
    log1.write_text("log content 1")

    log2 = local_logs_dir / "def12345678901234567890123456789.log"
    log2.write_text("log content 2")

    # Mock subprocess.run to just verify the tail call without actually executing
    called_cmd = []

    class FakeCompletedProcess:
        returncode = 0

    def mock_run(cmd, *args, **kwargs):
        called_cmd.append(cmd)
        return FakeCompletedProcess()

    monkeypatch.setattr("subprocess.run", mock_run)

    # Mock Path.expanduser to resolve to tmp_path
    from pathlib import Path

    orig_expanduser = Path.expanduser
    monkeypatch.setattr(
        Path,
        "expanduser",
        lambda self: tmp_path / ".kego" / "logs" if str(self) == "~/.kego/logs" else orig_expanduser(self),
    )

    # Test latest log file resolution
    ret = _run_logs("local", None, False, 100, "dummy-task")
    assert ret == 0
    assert called_cmd
    # Should resolve to log2 because it is the latest
    assert called_cmd[0] == ["tail", "-n", "100", str(log2)]

    # Test matching log file by prefix
    called_cmd.clear()
    ret = _run_logs("local", "abc", True, 20, "dummy-task")
    assert ret == 0
    assert called_cmd
    assert called_cmd[0] == ["tail", "-n", "20", "-f", str(log1)]


def test_run_logs_mlflow_resolution(tmp_path, monkeypatch):
    from kego.pipeline.cli import _run_logs

    # Mock find_repo_root
    monkeypatch.setattr("kego.fleet.repo_root", lambda *a: tmp_path)
    monkeypatch.setattr("kego.fleet.machine_name", lambda *a: "localhost")

    # Setup mock fleet.toml
    fleet_toml = tmp_path / "fleet.toml"
    fleet_toml.write_text("""
[hub]
name = "omarchyl"
mlflow = "http://omarchyd:5000"

[[machine]]
name = "omarchyd"
ssh = "user@omarchyd-host"
role = "gpu"
repo = "/some/repo"
""")

    # Mock _resolve_run_machine
    resolved_calls = []

    def mock_resolve(task_name, run_id):
        resolved_calls.append((task_name, run_id))
        return "5034b3b0ac8d474d810c4b9fa40cc659", "omarchyd"

    monkeypatch.setattr("kego.pipeline.cli._resolve_run_machine", mock_resolve)

    # Mock subprocess.run for ssh command
    called_cmd = []

    class FakeCompletedProcess:
        returncode = 0

    def mock_run(cmd, *args, **kwargs):
        called_cmd.append(cmd)
        return FakeCompletedProcess()

    monkeypatch.setattr("subprocess.run", mock_run)

    # Call _run_logs with only run_id prefix (target_or_run_id = "5034b3b0")
    ret = _run_logs("5034b3b0", None, False, 10, "pokemon-tcg-ai-battle")
    assert ret == 0
    assert resolved_calls == [("pokemon-tcg-ai-battle", "5034b3b0")]
    assert called_cmd
    # SSH should be invoked on user@omarchyd-host targeting omarchyd machine
    assert "user@omarchyd-host" in called_cmd[0]
    assert "5034b3b0ac8d474d810c4b9fa40cc659" in called_cmd[0][2]


def test_dispatch_train_agent_logs_help_text(tmp_path, monkeypatch, capsys):
    from kego.pipeline.cli import _dispatch_train_agent

    monkeypatch.setattr("kego.fleet.repo_root", lambda *a: tmp_path)
    monkeypatch.setattr("kego.tracking.create_run", lambda *a, **kw: "test_run_123")
    monkeypatch.setattr("kego.tracking.default_tracking_uri", lambda *a, **kw: "http://mlflow:5000")
    monkeypatch.setattr("kego.fleet.git_sha", lambda *a, **kw: "sha123")
    monkeypatch.setattr("kego.fleet.machine_name", lambda *a, **kw: "local")
    monkeypatch.setattr("kego.dispatch.dispatch", lambda *a, **kw: None)

    fleet_toml = tmp_path / "fleet.toml"
    fleet_toml.write_text("""
[hub]
name = "omarchyl"
mlflow = "http://mlflow:5000"

[[machine]]
name = "gpu1"
ssh = "user@gpu1-host"
role = "gpu"
repo = "/repo"
""")

    ret = _dispatch_train_agent("pokemon-tcg-ai-battle", "gpu1", 10, None, {})
    assert ret == 0

    captured = capsys.readouterr().out
    assert "To view remote logs, run:  kego logs gpu1 test_run_123" in captured


def test_dispatch_league_logs_help_text(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    from kego.pipeline.cli import _dispatch_league

    monkeypatch.setattr("kego.fleet.repo_root", lambda *a: tmp_path)
    monkeypatch.setattr("kego.tracking.create_run", lambda *a, **kw: "league_run_456")
    monkeypatch.setattr("kego.tracking.default_tracking_uri", lambda *a, **kw: "http://mlflow:5000")
    monkeypatch.setattr("kego.fleet.git_sha", lambda *a, **kw: "sha123")
    monkeypatch.setattr("kego.fleet.machine_name", lambda *a, **kw: "local")
    monkeypatch.setattr("kego.dispatch.dispatch", lambda *a, **kw: None)

    fleet_toml = tmp_path / "fleet.toml"
    fleet_toml.write_text("""
[hub]
name = "omarchyl"
mlflow = "http://mlflow:5000"

[[machine]]
name = "gpu1"
ssh = "user@gpu1-host"
role = "gpu"
repo = "/repo"
""")

    args = SimpleNamespace(
        games=10,
        search_count=100,
        workers=2,
        debug=False,
        cache_dir=None,
        include_local_mcts=False,
        partial_save_every=10,
        stall_timeout=600,
        write_ratings=True,
    )

    ret = _dispatch_league("pokemon-tcg-ai-battle", "gpu1", args)
    assert ret == 0

    captured = capsys.readouterr().out
    assert "To view remote logs, run:  kego logs gpu1 league_run_456" in captured
