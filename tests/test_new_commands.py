import os
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

from kego.pipeline.cli import build_parser
from kego.pipeline.config import FoldScheme, GridConfig, ModelConfig, PipelineConfig
from kego.pipeline.runner import Pipeline
from kego.pipeline.task import RawData, register_task


@register_task("dummy-comp")
class DummyTask:
    name = "dummy-comp"
    kaggle_slug = "dummy-comp"
    target = "target"
    id_col = "id"
    metric_direction = "maximize"
    is_simulation = False

    def load_raw(self):
        return RawData(None, None, None)

    def preprocess(self, df):
        return df

    def make_submission(self, ids, preds):
        return Path("dummy_submission.csv")


@register_task("pokemon-tcg-ai-battle")
class DummySimTask(DummyTask):
    name = "pokemon-tcg-ai-battle"
    kaggle_slug = "pokemon-tcg-ai-battle"
    is_simulation = True

    def make_submission(self, ids, preds):
        return Path("dummy_submission.tar.gz")


def test_parser_new_commands():
    parser = build_parser()

    # 1. Test status parser
    args = parser.parse_args(["status"])
    assert args.command == "status"

    # 2. Test cache parser
    args = parser.parse_args(["cache", "status"])
    assert args.command == "cache"
    assert args.action == "status"

    args = parser.parse_args(["cache", "prune"])
    assert args.command == "cache"
    assert args.action == "prune"

    # 3. Test submissions parser
    args = parser.parse_args(["submissions"])
    assert args.command == "submissions"

    # 4. Test battle parser
    args = parser.parse_args(["battle", "--agent1", "a1.py", "--agent2", "a2.py", "--games", "5"])
    assert args.command == "battle"
    assert args.agent1 == "a1.py"
    assert args.agent2 == "a2.py"
    assert args.games == 5

    # 5. Test train-agent parser
    args = parser.parse_args(["train-agent", "--epochs", "10", "--output", "my_model.pth"])
    assert args.command == "train-agent"
    assert args.epochs == 10
    assert args.output == "my_model.pth"


def test_status_execution(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    from kego.pipeline import runner

    # Isolate from the real Ray cluster: pretend it is reachable with no jobs.
    class FakeClient:
        def list_jobs(self):
            return []

    monkeypatch.setattr(runner, "_make_ray_job_client", lambda _addr: FakeClient())

    # Mock task and config
    task = DummyTask()
    config = PipelineConfig(task="dummy-comp")

    pipeline = Pipeline(config)
    pipeline.task = task

    # Run status when no active runs directory exists
    pipeline.status()
    captured = capsys.readouterr()
    assert "No active training runs found." in captured.out

    # Create a mock active run file
    active_dir = tmp_path / ".kego" / "active_runs"
    active_dir.mkdir(parents=True)
    run_file = active_dir / "run123.json"
    run_file.write_text(
        '{"task": "dummy-comp", "config": "baseline", "pid": 1234, "progress": "2/5", "active_workers": ["worker-1"]}'
    )

    pipeline.status()
    captured = capsys.readouterr()
    assert "Active Runs:" in captured.out
    assert "[Run run123]" in captured.out
    assert "PID: 1234" in captured.out
    assert "Progress: 2/5" in captured.out
    assert "worker-1" in captured.out


def test_submissions_execution(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    task = DummyTask()
    config = PipelineConfig(task="dummy-comp")

    pipeline = Pipeline(config)
    pipeline.task = task

    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "List of submissions..."
        mock_run.return_value = mock_result

        pipeline.submissions()
        captured = capsys.readouterr()

        assert "List of submissions..." in captured.out

        # Verify subprocess was called with expected arguments
        args, _ = mock_run.call_args
        cmd = args[0]
        assert "competitions" in cmd
        assert "submissions" in cmd
        assert "dummy-comp" in cmd


def test_cache_status_execution(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    task = DummyTask()
    # Define a config with 1 model and 2 seeds = 2 total specs in grid
    config = PipelineConfig(
        task="dummy-comp",
        models=[ModelConfig(name="xgb")],
        grid=GridConfig(seeds=[42, 43], feature_sets=["baseline"], folds=[FoldScheme(n=5)]),
    )

    pipeline = Pipeline(config)
    pipeline.task = task

    # Mock prediction store has check
    with patch.object(pipeline.store, "has", side_effect=lambda fp: fp == "dummy_fp_1" or False):
        # We need to mock expand_grid to return two items with specific fingerprints
        from kego.pipeline.config import LearnerSpec

        # Ensure instances are initialized to expand_grid
        _ = LearnerSpec(model=config.models[0], feature_set="baseline", fold_scheme=config.grid.folds[0], seed=42)
        _ = LearnerSpec(model=config.models[0], feature_set="baseline", fold_scheme=config.grid.folds[0], seed=43)

        # Mock fingerprints
        with patch.object(LearnerSpec, "fingerprint", new_callable=PropertyMock) as mock_fp:
            mock_fp.side_effect = ["dummy_fp_1", "dummy_fp_2"]

            pipeline.cache("status")
            captured = capsys.readouterr()

            assert "Cache Status:" in captured.out
            assert "Total specs in grid: 2" in captured.out
            assert "Cached specs: 1" in captured.out
            assert "Cache coverage: 50.0%" in captured.out


def test_cache_prune_execution(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    task = DummyTask()
    config = PipelineConfig(
        task="dummy-comp",
        models=[ModelConfig(name="xgb")],
        grid=GridConfig(seeds=[42], feature_sets=["baseline"], folds=[FoldScheme(n=5)]),
    )

    pipeline = Pipeline(config)
    pipeline.task = task

    # Mock LocalCacheStore root path
    cache_root = tmp_path / ".kego_cache"
    cache_root.mkdir()
    pipeline.store.local.root = cache_root

    # Create two files in cache: one active, one stale
    active_fp = "dummy_fp_1"
    stale_fp = "dummy_fp_stale"

    active_file = cache_root / f"{active_fp}.npz"
    stale_file = cache_root / f"{stale_fp}.npz"

    active_file.write_text("data")
    stale_file.write_text("data")

    # Mock expand_grid fingerprints
    from kego.pipeline.config import LearnerSpec

    with patch.object(LearnerSpec, "fingerprint", new_callable=PropertyMock) as mock_fp:
        mock_fp.return_value = active_fp
        pipeline.cache("prune")
        captured = capsys.readouterr()

        assert "Cache Pruned:" in captured.out
        assert "Deleted: 1 files" in captured.out

        # Active file should still exist, stale file should be deleted
        assert active_file.exists()
        assert not stale_file.exists()


def test_battle_execution(tmp_path, monkeypatch, capsys):
    agent1_file = tmp_path / "agent1.py"
    agent1_file.write_text("def agent(obs):\n    return [0]\ndef read_deck_csv():\n    return [1]*60\n")

    agent2_file = tmp_path / "agent2.py"
    agent2_file.write_text("def agent(obs):\n    return [0]\ndef read_deck_csv():\n    return [1]*60\n")

    # Mock locate_cg_dir
    with patch("kego.pipeline.battle.locate_cg_dir") as mock_locate:
        mock_locate.return_value = tmp_path

        # We need to mock cg.game functions
        sys_modules_mock = {
            "cg": MagicMock(),
            "cg.api": MagicMock(),
            "cg.game": MagicMock(),
        }

        with patch.dict("sys.modules", sys_modules_mock):
            # Mock the behaviors of battle_start, battle_select, to_observation_class
            from cg.game import battle_select, battle_start

            # battle_start returns (obs_dict, start_data)
            obs_dict_1 = {"current": {"yourIndex": 0, "result": -1}}
            obs_dict_2 = {"current": {"yourIndex": 1, "result": 1}}  # agent 2 wins

            battle_start.return_value = (obs_dict_1, MagicMock())
            battle_select.side_effect = [obs_dict_2]

            # Mock to_observation_class
            from cg.api import to_observation_class

            obs_mock_1 = MagicMock()
            obs_mock_1.current.result = -1
            obs_mock_1.select = MagicMock()

            obs_mock_2 = MagicMock()
            obs_mock_2.current.result = 1  # winner is agent 2 (which is Player 1 in first game)
            obs_mock_2.select = MagicMock()

            to_observation_class.side_effect = [obs_mock_1, obs_mock_2]

            from kego.pipeline.battle import run_battle_benchmark

            run_battle_benchmark(agent1_path=str(agent1_file), agent2_path=str(agent2_file), num_games=1)

            captured = capsys.readouterr()
            assert "Winner: Agent 2" in captured.out
            assert "BENCHMARK RESULTS" in captured.out
            assert "Agent 2" in captured.out


def test_battle_config_load_execution(tmp_path, monkeypatch, capsys):
    config_dir = tmp_path / "competitions" / "pokemon-tcg-ai-battle" / "configs"
    config_dir.mkdir(parents=True)

    agent1_file = tmp_path / "agent1.py"
    agent1_file.write_text("def agent(obs):\n    return [0]\ndef read_deck_csv():\n    return [1]*60\n")

    agent2_file = tmp_path / "agent2.py"
    agent2_file.write_text("def agent(obs):\n    return [0]\ndef read_deck_csv():\n    return [1]*60\n")

    yaml_content = f"""
task: pokemon-tcg-ai-battle
battle:
  agent1: "{agent1_file.as_posix()}"
  agent2: "{agent2_file.as_posix()}"
  games: 2
"""
    config_file = config_dir / "battle_config.yaml"
    config_file.write_text(yaml_content)

    monkeypatch.chdir(tmp_path)

    # Mock locate_cg_dir
    with patch("kego.pipeline.battle.locate_cg_dir") as mock_locate:
        mock_locate.return_value = tmp_path

        # Mock cg.game functions
        sys_modules_mock = {
            "cg": MagicMock(),
            "cg.api": MagicMock(),
            "cg.game": MagicMock(),
        }

        with patch.dict("sys.modules", sys_modules_mock):
            from cg.api import to_observation_class
            from cg.game import battle_select, battle_start

            obs_dict_1 = {"current": {"yourIndex": 0, "result": -1}}
            obs_dict_2 = {"current": {"yourIndex": 1, "result": 0}}  # Agent 1 wins

            battle_start.return_value = (obs_dict_1, MagicMock())
            battle_select.return_value = obs_dict_2

            obs_mock_1 = MagicMock()
            obs_mock_1.current.result = -1
            obs_mock_1.select = MagicMock()

            obs_mock_2 = MagicMock()
            obs_mock_2.current.result = 0  # winner is player 0
            obs_mock_2.select = MagicMock()

            to_observation_class.side_effect = [obs_mock_1, obs_mock_2, obs_mock_1, obs_mock_2]

            from kego.pipeline.cli import main

            main(["battle", "--config", "battle_config", "--task", "pokemon-tcg-ai-battle"])

            captured = capsys.readouterr()
            assert "Winner: Agent 1" in captured.out
            assert "Agent 1" in captured.out
            assert "Agent 2" in captured.out


def test_train_agent_execution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Register a mock task that implements train method
    from kego.pipeline.task import register_task

    @register_task("trainable-comp")
    class TrainableTask:
        name = "trainable-comp"
        kaggle_slug = "trainable-comp"
        target = "target"
        id_col = "id"
        metric_direction = "maximize"
        is_simulation = True

        def __init__(self):
            self.trained = False
            self.epochs = None
            self.output_path = None

        def train(self, config, epochs=None, output_path=None, **kwargs):
            self.trained = True
            self.epochs = epochs
            self.output_path = output_path

    task = TrainableTask()

    from kego.pipeline.config import PipelineConfig
    from kego.pipeline.runner import Pipeline

    config = PipelineConfig(task="trainable-comp")
    pipeline = Pipeline(config)
    pipeline.task = task

    pipeline.train_agent(epochs=5, output_path="out.pth")
    assert task.trained
    assert task.epochs == 5
    assert task.output_path == "out.pth"


# ---------------------------------------------------------------------------
# Ray dashboard address resolution + status robustness
# ---------------------------------------------------------------------------


def test_resolve_dashboard_address():
    from kego.pipeline.runner import _resolve_dashboard_address

    # ray:// client address -> http dashboard on :8265
    assert _resolve_dashboard_address("ray://omarchyd:10001") == "http://omarchyd:8265"
    # explicit http(s) dashboard passes through unchanged
    assert _resolve_dashboard_address("http://host:8265") == "http://host:8265"
    assert _resolve_dashboard_address("https://host:8265") == "https://host:8265"
    # unset -> default head node http dashboard
    assert _resolve_dashboard_address(None) == "http://omarchyd:8265"


def test_ray_job_client_ignores_ray_address_env(monkeypatch):
    """A ray:// RAY_ADDRESS must not override the explicit http dashboard URL.

    Ray's get_address_for_submission_client always lets RAY_ADDRESS override the
    passed address; if it is a ray:// client address the submission client routes
    through the Ray Client port (often unreachable) and times out.
    """
    from kego.pipeline import runner

    seen = {}

    class FakeClient:
        def __init__(self, address):
            seen["address"] = address
            seen["ray_address_env"] = os.environ.get("RAY_ADDRESS")

    monkeypatch.setattr("ray.dashboard.modules.job.sdk.JobSubmissionClient", FakeClient)
    monkeypatch.setenv("RAY_ADDRESS", "ray://omarchyd:10001")

    runner._make_ray_job_client("http://omarchyd:8265")

    assert seen["address"] == "http://omarchyd:8265"
    # During construction, RAY_ADDRESS must not be the ray:// override.
    assert seen["ray_address_env"] != "ray://omarchyd:10001"
    # The env var is restored after construction.
    assert os.environ.get("RAY_ADDRESS") == "ray://omarchyd:10001"


def test_status_surfaces_remote_query_error(tmp_path, monkeypatch, capsys):
    """A failed cluster query must be reported, not silently shown as 'no runs'."""
    monkeypatch.chdir(tmp_path)
    from kego.pipeline import runner

    pipeline = Pipeline(PipelineConfig(task="dummy-comp"))
    pipeline.task = DummyTask()

    def boom(_address):
        raise ConnectionError("ray client connection timeout")

    monkeypatch.setattr(runner, "_make_ray_job_client", boom)

    pipeline.status()
    out = capsys.readouterr().out

    # The real error is surfaced (not swallowed by a bare except).
    assert "ray client connection timeout" in out
    # And it does NOT falsely assert an authoritative "no active runs".
    assert "No active training runs found." not in out
