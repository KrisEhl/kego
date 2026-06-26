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


def test_status_execution(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

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
