import os
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from kego.pipeline.cli import build_parser
from kego.pipeline.config import FoldScheme, GridConfig, ModelConfig, PipelineConfig
from kego.pipeline.runner import Pipeline, format_submissions
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
    args = parser.parse_args(
        [
            "train-agent",
            "--agent",
            "mcts",
            "--epochs",
            "10",
            "--output",
            "my_model.pth",
            "--init-checkpoint",
            "registry:12",
            "--num-workers",
            "12",
            "--variant",
            "small192_zacian",
        ]
    )
    assert args.command == "train-agent"
    assert args.agent == "mcts"
    assert args.epochs == 10
    assert args.output == "my_model.pth"
    assert args.init_checkpoint == "registry:12"
    assert args.num_workers == 12
    assert args.variant == "small192_zacian"

    args = parser.parse_args(["submit", "30", "--task", "pokemon-tcg-ai-battle"])
    assert args.command == "submit"
    assert args.version == "30"


def test_submit_registry_version_prepares_selected_model(monkeypatch):
    from types import SimpleNamespace

    from kego.pipeline import cli

    prepared_versions = []
    submitted_messages = []
    helper = SimpleNamespace(
        prepare_submission=lambda version: (
            prepared_versions.append(version) or {"version": version, "message": f"Registry v{version}"}
        )
    )
    monkeypatch.setattr(cli, "_load_submission_module", lambda _task: helper)
    monkeypatch.setattr(
        Pipeline,
        "submit",
        lambda _pipeline, _outcome, message: (
            submitted_messages.append(message) or SimpleNamespace(status="complete", public_score=None)
        ),
    )

    assert cli.main(["submit", "30", "--task", "pokemon-tcg-ai-battle"]) == 0
    assert prepared_versions == ["30"]
    assert submitted_messages == ["Registry v30"]


def test_status_execution(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    from kego.pipeline import runner

    # Isolate from the real Ray cluster: pretend it is reachable with no jobs.
    class FakeClient:
        def list_jobs(self):
            return []

    monkeypatch.setattr(runner, "_is_port_open", lambda _address, timeout=0.2: True)
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
    import os

    active_dir = tmp_path / ".kego" / "active_runs"
    active_dir.mkdir(parents=True)
    run_file = active_dir / "run123.json"
    run_file.write_text(
        f'{{"task": "dummy-comp", "config": "baseline", "pid": {os.getpid()}, "progress": "2/5", "active_workers": ["worker-1"]}}'
    )

    pipeline.status()
    captured = capsys.readouterr()
    assert "Active Runs:" in captured.out
    assert "[Run run123]" in captured.out
    assert f"PID: {os.getpid()}" in captured.out
    assert "Progress: 2/5" in captured.out
    assert "worker-1" in captured.out


def test_submissions_execution(tmp_path, monkeypatch, capsys):
    import zipfile

    monkeypatch.chdir(tmp_path)

    task = DummyTask()
    config = PipelineConfig(task="dummy-comp")

    pipeline = Pipeline(config)
    pipeline.task = task

    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "ref,fileName,date,description,status,publicScore,privateScore\n"
            "54814340,submission.tar.gz,2026-07-18 18:59:31.567000,"
            "Registry v30 with a deliberately long description that should fit cleanly in one compact row,"
            "SubmissionStatus.COMPLETE,600.0,\n"
            "54366384,submission.tar.gz,2026-07-06 12:00:00.000000,Registry v2,"
            "SubmissionStatus.COMPLETE,500.0,\n"
            "54366385,submission.tar.gz,2026-07-06 13:00:00.000000,Registry v2,"
            "SubmissionStatus.COMPLETE,650.0,\n"
            "54365896,submission.tar.gz,2026-07-05 14:53:53.527000,Registry v2,SubmissionStatus.ERROR,,\n"
            "54324742,submission.tar.gz,2026-07-04 08:48:22.273000,Registry v1,"
            "SubmissionStatus.COMPLETE,999.0,\n"
        )

        def fake_run(cmd, **_kwargs):
            if "submissions" in cmd:
                return mock_result
            leaderboard_dir = Path(cmd[cmd.index("-p") + 1])
            with zipfile.ZipFile(leaderboard_dir / "leaderboard.zip", "w") as archive:
                archive.writestr(
                    "leaderboard.csv",
                    "Rank,TeamId,TeamName,LastSubmissionDate,Score,SubmissionCount,TeamMemberUserNames\n"
                    "1,1,A,2026-07-18,900.0,1,a\n"
                    "2,2,B,2026-07-18,700.0,1,b\n"
                    "3,3,Us,2026-07-18,650.0,2,us\n",
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        from types import SimpleNamespace

        mock_run.side_effect = fake_run

        pipeline.submissions()
        lines = capsys.readouterr().out.splitlines()

        assert lines[0].split() == ["date", "status", "score", "file", "description"]
        assert "2026-07-18 18:59" in lines[2]
        assert "COMPLETE" in lines[2]
        assert "SubmissionStatus" not in "\n".join(lines)
        assert "600.0" in lines[2]
        error_line = next(line for line in lines if "ERROR" in line)
        assert "  -  " in error_line
        assert "..." in lines[2]
        assert "" not in lines
        ranking = lines.index("Current registry models (latest 2; best public score)")
        assert lines[ranking + 3].split() == ["v2", "650.0", "3", "2"]
        assert lines[ranking + 4].split() == ["v30", "600.0", "4", "1"]
        assert all("v1" not in line for line in lines[ranking:])

        # Verify subprocess was called with expected arguments
        args, _ = mock_run.call_args_list[0]
        cmd = args[0]
        assert "competitions" in cmd
        assert "submissions" in cmd
        assert "dummy-comp" in cmd
        assert "--csv" in cmd
        leaderboard_cmd = mock_run.call_args_list[1][0][0]
        assert "leaderboard" in leaderboard_cmd
        assert "--download" in leaderboard_cmd


def test_format_submissions_preserves_multiline_csv_description():
    csv_text = (
        "Warning: upgrade available\n"
        "fileName,date,description,status,publicScore,privateScore\n"
        'submission.tar.gz,2026-07-18 18:59:31.567000,"first line\nWarning: second line",'
        "SubmissionStatus.COMPLETE,600.0,\n"
    )

    output = format_submissions(csv_text)

    assert "first line Warning: second line" in output
    with pytest.raises(ValueError, match="at least 3"):
        format_submissions(csv_text, description_width=2)


def test_format_submissions_ranks_latest_models_by_timestamp_including_unscored():
    csv_text = (
        "fileName,date,description,status,publicScore,privateScore\n"
        "submission.tar.gz,2026-07-01 12:00:00.000000,Registry v2,"
        "SubmissionStatus.COMPLETE,700.0,\n"
        "submission.tar.gz,2026-07-18 12:00:00.000000,Registry v40,"
        "SubmissionStatus.ERROR,,\n"
        "submission.tar.gz,2026-07-17 12:00:00.000000,Registry v30,"
        "SubmissionStatus.COMPLETE,600.0,\n"
    )
    leaderboard_csv = (
        "Rank,TeamId,TeamName,LastSubmissionDate,Score,SubmissionCount,TeamMemberUserNames\n"
        "1,1,A,2026-07-18,900.0,1,a\n"
        "2,2,B,2026-07-18,650.0,1,b\n"
    )

    lines = format_submissions(csv_text, leaderboard_csv=leaderboard_csv).splitlines()
    ranking = lines.index("Current registry models (latest 2; best public score)")

    assert lines[ranking + 3].split() == ["v30", "600.0", "3", "1"]
    assert lines[ranking + 4].split() == ["v40", "-", "-", "0"]
    assert all("v2" not in line for line in lines[ranking:])


def test_model_submission_stats_reports_attempts_and_best_public_rank():
    from kego.pipeline.runner import model_submission_stats

    submissions = (
        "fileName,date,description,status,publicScore,privateScore\n"
        "submission.tar.gz,2026-07-18,Registry v30,SubmissionStatus.COMPLETE,600.0,\n"
        "submission.tar.gz,2026-07-17,Registry v30,SubmissionStatus.ERROR,,\n"
        "submission.tar.gz,2026-07-16,Registry v2,SubmissionStatus.COMPLETE,650.0,\n"
    )
    public_board = "Rank,TeamName,Score\n1,A,900.0\n2,B,700.0\n3,Us,650.0\n"

    assert model_submission_stats(submissions, public_board) == {
        "30": {"submitted": "yes (2)", "public_rank": "4"},
        "2": {"submitted": "yes", "public_rank": "3"},
    }


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
            self.kwargs = None

        def train(self, config, epochs=None, output_path=None, **kwargs):
            self.trained = True
            self.epochs = epochs
            self.output_path = output_path
            self.kwargs = kwargs

    task = TrainableTask()

    from kego.pipeline.config import PipelineConfig
    from kego.pipeline.runner import Pipeline

    config = PipelineConfig(task="trainable-comp")
    pipeline = Pipeline(config)
    pipeline.task = task

    pipeline.train_agent(
        epochs=5,
        output_path="out.pth",
        init_checkpoint="registry:12",
        deck_file="decks/lucario.csv",
        self_play_games=72,
        search_count=32,
        train_steps=80,
        num_workers=10,
        model_args="192,4,384,2,2",
    )
    assert task.trained
    assert task.epochs == 5
    assert task.output_path == "out.pth"
    assert task.kwargs["init_checkpoint"] == "registry:12"
    assert task.kwargs["deck_file"] == "decks/lucario.csv"
    assert task.kwargs["self_play_games"] == 72
    assert task.kwargs["search_count"] == 32
    assert task.kwargs["train_steps"] == 80
    assert task.kwargs["num_workers"] == 10
    assert task.kwargs["model_args"] == "192,4,384,2,2"


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

    monkeypatch.setattr(runner, "_is_port_open", lambda _address, timeout=0.2: True)
    monkeypatch.setattr(runner, "_make_ray_job_client", boom)

    pipeline.status()
    out = capsys.readouterr().out

    # The real error is surfaced (not swallowed by a bare except).
    assert "ray client connection timeout" in out
    # And it does NOT falsely assert an authoritative "no active runs".
    assert "No active training runs found." not in out


def test_status_labels_two_gpus_and_keeps_eta_column_aligned(tmp_path, monkeypatch, capsys):
    """This catches multi-GPU telemetry being ambiguous and overrunning the GPU column."""
    monkeypatch.chdir(tmp_path)
    from kego.fleet import Fleet, Hub, Machine
    from kego.pipeline import runner

    pipeline = Pipeline(PipelineConfig(task="dummy-comp"))
    pipeline.task = DummyTask()
    fleet = Fleet(
        hub=Hub("gpu-box", "http://gpu-box:5000"),
        machines=(Machine("gpu-box", "user@gpu-box", "gpu", "/repo", gpus=("rtx3090", "rtx3090")),),
    )
    monkeypatch.setattr(runner, "_is_port_open", lambda _address, timeout=0.2: False)
    monkeypatch.setattr("kego.fleet.load_fleet", lambda _path: fleet)
    monkeypatch.setattr(
        runner,
        "_poll_machine",
        lambda _machine: {
            "name": "gpu-box",
            "status": "Online",
            "load": "26.4% / 1.0 1.0 1.0 (32c)",
            "gpu": "0%/0% (5.5/48.0 GB)",
            "gpu_count": 2,
            "runs": [("9809", "1569e637" * 4, "Iter 202/250 - Training", "19m")],
        },
    )

    pipeline.status()
    lines = capsys.readouterr().out.splitlines()
    header = next(line for line in lines if "Machine" in line and "GPU" in line and "ETA" in line)
    row = next(line for line in lines if line.startswith("gpu-box"))

    assert "2 GPUs: 0%/0% (5.5/48.0G)" in row
    assert row.index("19m") == header.index("ETA")


def test_poll_machine_parses_darwin_process_commands(monkeypatch):
    from kego.fleet import Machine
    from kego.pipeline import runner

    run_id = "05754c803bf248e29796d66ab75a5689"

    class FakeCompleted:
        returncode = 0
        stderr = ""
        stdout = f"""
CPU_UTIL: 9.41%
LOAD: {{ 1.23 1.96 2.08 }}
CORES: 14
GPU: N/A
PROCS:
15845    07:57 bash -lc mkdir -p ~/.kego/logs && cd /repo/competitions/pokemon-tcg-ai-battle && KEGO_MLFLOW_RUN_ID={run_id} nohup uv run kego train-agent --task pokemon-tcg-ai-battle --epochs 80 --output outputs/model.pth > ~/.kego/logs/{run_id}.log 2>&1 &
15848    07:57 uv run kego train-agent --task pokemon-tcg-ai-battle --epochs 80 --output outputs/model.pth
15859    07:57 /repo/.venv/bin/python3 /repo/.venv/bin/kego train-agent --task pokemon-tcg-ai-battle --epochs 80 --output outputs/model.pth
LOGS:
LOG_PARSED: run_id={run_id} | curr=15 | total=80 | step=Training complete (value=0.0062 policy=0.0073) | done=
"""

    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: FakeCompleted())

    res = runner._poll_machine(Machine("mn", "user@mn", "cpu", "/repo"))

    assert res["status"] == "Online"
    assert res["load"] == "9.41% / 1.23 1.96 2.08 (14c)"
    assert res["runs"] == [("15845", run_id, "Iter 15/80 - Training complete (value=0.0062 policy=0.0073)", "34m")]


def test_poll_machine_parses_league_progress(monkeypatch):
    from kego.fleet import Machine
    from kego.pipeline import runner

    run_id = "a1035113a0af4680bf7fa87a6c9b4500"

    class FakeCompleted:
        returncode = 0
        stderr = ""
        stdout = f"""
CPU_UTIL: 92.4%
LOAD: 18.10 18.06 18.02
CORES: 20
GPU: 0, 0, 6144
PROCS:
26452    02:30:59 bash -lc mkdir -p ~/.kego/logs && cd /repo/competitions/pokemon-tcg-ai-battle && KEGO_MLFLOW_RUN_ID={run_id} nohup uv run kego league --task pokemon-tcg-ai-battle --games 200 --search-count 10 > ~/.kego/logs/{run_id}.log 2>&1 < /dev/null &
26454    02:30:59 uv run kego league --task pokemon-tcg-ai-battle --games 200 --search-count 10
26457    02:30:59 /repo/.venv/bin/python3 /repo/.venv/bin/kego league --task pokemon-tcg-ai-battle --games 200 --search-count 10
LOGS:
LOG_PARSED: run_id={run_id} | curr=35650 | total=65000 | step=ETA 02:04:18 | done= | kind=league
"""

    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: FakeCompleted())

    res = runner._poll_machine(Machine("omarchyl", "user@omarchyl", "gpu", "/repo"))

    assert res["status"] == "Online"
    assert res["gpu"] == "0% (0.0/6.0 GB)"
    assert res["runs"] == [("26452", run_id, "Games 35650/65000", "02:04:18")]


def test_poll_machine_uses_local_shell_for_current_machine(monkeypatch):
    from kego.fleet import Machine
    from kego.pipeline import runner

    seen = {}

    class FakeCompleted:
        returncode = 0
        stderr = ""
        stdout = "CPU_UTIL: 1%\nLOAD: 0.1 0.2 0.3\nCORES: 4\nGPU: N/A\nPROCS:\nLOGS:\n"

    def fake_run(cmd, **_kwargs):
        seen["cmd"] = cmd
        return FakeCompleted()

    monkeypatch.setattr("kego.fleet.machine_name", lambda: "localbox")
    monkeypatch.setattr("subprocess.run", fake_run)

    res = runner._poll_machine(Machine("localbox", "user@localbox", "cpu", "/repo"))

    assert seen["cmd"][:2] == ["bash", "-lc"]
    assert seen["cmd"][0] != "ssh"
    assert res["status"] == "Online"


def test_poll_machine_ignores_local_league_dispatch_ssh(monkeypatch):
    from kego.fleet import Machine
    from kego.pipeline import runner

    run_id = "d38b7ed3f55447d9bcdf34a1cbd0a7bd"

    class FakeCompleted:
        returncode = 0
        stderr = ""
        stdout = f"""
CPU_UTIL: 1%
LOAD: 0.1 0.2 0.3
CORES: 4
GPU: N/A
PROCS:
73274 27:53 ssh kristian@omarchyd bash -lc 'mkdir -p ~/.kego/logs && cd /repo && KEGO_MLFLOW_RUN_ID={run_id} nohup uv run kego league --task pokemon-tcg-ai-battle > ~/.kego/logs/{run_id}.log 2>&1 < /dev/null &'
LOGS:
LOG_PARSED: run_id={run_id} | curr=3771 | total=16250 | step=ETA 01:30:02 | done= | kind=league
"""

    monkeypatch.setattr("kego.fleet.machine_name", lambda: "localbox")
    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: FakeCompleted())

    res = runner._poll_machine(Machine("localbox", "user@localbox", "cpu", "/repo"))

    assert res["status"] == "Online"
    assert res["runs"] == []


def test_poll_machine_uses_longer_configurable_timeout(monkeypatch):
    from kego.fleet import Machine
    from kego.pipeline import runner

    seen = {}

    class FakeCompleted:
        returncode = 0
        stderr = ""
        stdout = "CPU_UTIL: 1%\nLOAD: 0.1 0.2 0.3\nCORES: 4\nGPU: N/A\nPROCS:\nLOGS:\n"

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["timeout"] = kwargs["timeout"]
        return FakeCompleted()

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setenv("KEGO_STATUS_CONNECT_TIMEOUT", "6")
    monkeypatch.setenv("KEGO_STATUS_TIMEOUT", "11")

    res = runner._poll_machine(Machine("remote", "user@remote", "cpu", "/repo"))

    assert res["status"] == "Online"
    assert "ConnectTimeout=6" in seen["cmd"]
    assert seen["timeout"] == 11


def test_models_parser():
    parser = build_parser()

    args = parser.parse_args(["models", "--task", "pokemon-tcg-ai-battle"])
    assert args.command == "models"
    assert args.sort_by == "elo"  # default now ranks by league Elo
    assert args.breakdown is False
    assert args.color is None
    assert args.mlflow is False

    args = parser.parse_args(["models", "--task", "x", "--sort-by", "gauntlet_avg", "-b", "--color", "--mlflow"])
    assert args.sort_by == "gauntlet_avg"
    assert args.breakdown is True
    assert args.color is True
    assert args.mlflow is True


def test_mlflow_run_link_targets_run_page():
    from kego.pipeline.cli import _mlflow_run_link

    assert (
        _mlflow_run_link("http://omarchyd:5000/", "pokemon experiment", "run/123")
        == "http://omarchyd:5000/#/experiments/pokemon%20experiment/runs/run%2F123"
    )
    assert _mlflow_run_link("sqlite:///tracking.db", "1", "abc") == "-"


def test_rule_agent_rows_are_available_for_pokemon_models():
    from kego.pipeline.cli import _numeric_row_value, _rule_agent_rows

    rows = _rule_agent_rows("pokemon-tcg-ai-battle")

    assert [r["agent"] for r in rows] == [
        "Mega Abomasnow ex",
        "Dragapult ex",
        "Mega Lucario ex",
        "Zacian ex",
        "Random",
    ]
    assert rows[0]["type"] == "rule"
    assert rows[0]["version"] == "rule"
    assert _numeric_row_value(rows[0], "elo") == 1650.0
    assert _rule_agent_rows("other-task") == []


def test_registry_agent_name_prefers_logged_tag():
    from kego.pipeline.cli import _registry_agent_name

    assert _registry_agent_name({"version": "2", "agent_name": "mcts-abomasnow"}) == "mcts-abomasnow"


def test_registry_agent_name_derives_legacy_name():
    from kego.pipeline.cli import _registry_agent_name

    row = {"version": "2", "deck": "abomasnow", "model_args": "(256, 4, 512, 2, 2)", "epoch": "100"}

    assert _registry_agent_name(row) == "v2 mcts-abomasnow 256/4/512/2/2 @100"


def test_registry_agent_name_does_not_mislabel_trained_through_as_checkpoint_epoch():
    from kego.pipeline.cli import _registry_agent_name

    row = {"version": "30", "variant": "large256_abomasnow", "completed_iterations": "300"}

    assert _registry_agent_name(row) == "v30 large256_abomasnow"


def test_models_forced_color_shows_elo_trust_legend(monkeypatch, capsys):
    from kego.pipeline import cli

    monkeypatch.setattr(cli, "detect_task", lambda: "pokemon-tcg-ai-battle")
    monkeypatch.setattr("kego.tracking.default_tracking_uri", lambda: "http://mlflow")
    monkeypatch.setattr(
        "kego.tracking.leaderboard",
        lambda uri, task, sort_by: [{"version": "1", "elo": "1700", "elo_rd": "0.9"}],
    )
    monkeypatch.setenv("NO_COLOR", "1")

    assert cli.main(["models", "--task", "pokemon-tcg-ai-battle", "--color"]) == 0
    out = capsys.readouterr().out

    assert "elo color:" in out
    assert "Registry v1" in out
    assert "\x1b[32m1700\x1b[0m" in out
    assert "created" in out


def test_models_shows_submission_status_and_best_public_rank(monkeypatch, capsys):
    from kego.pipeline import cli

    monkeypatch.setattr(cli, "detect_task", lambda: "pokemon-tcg-ai-battle")
    monkeypatch.setattr("kego.tracking.default_tracking_uri", lambda: "http://mlflow")
    monkeypatch.setattr(
        "kego.tracking.leaderboard",
        lambda uri, task, sort_by: [{"version": "30", "elo": "1700"}, {"version": "31", "elo": "1600"}],
    )
    monkeypatch.setattr(
        Pipeline,
        "model_submission_stats",
        lambda _pipeline: {"30": {"submitted": "yes (2)", "public_rank": "4"}},
    )

    assert cli.main(["models", "--task", "pokemon-tcg-ai-battle", "--no-color"]) == 0
    out = capsys.readouterr().out

    assert "submitted" in out
    assert "public_rank" in out
    row30 = next(line for line in out.splitlines() if "Registry v30" in line)
    row31 = next(line for line in out.splitlines() if "Registry v31" in line)
    assert "yes (2)" in row30 and "4" in row30
    assert "-" in row31


def test_models_mlflow_adds_originating_run_links(monkeypatch, capsys):
    from types import SimpleNamespace

    from kego.pipeline import cli

    monkeypatch.setattr(cli, "detect_task", lambda: "pokemon-tcg-ai-battle")
    monkeypatch.setattr("kego.tracking.default_tracking_uri", lambda: "http://omarchyd:5000")
    monkeypatch.setattr(
        "kego.tracking.leaderboard",
        lambda uri, task, sort_by: [
            {"version": "30", "elo": "1700", "run_id": "registration30", "training_run_id": "run30"},
            {"version": "31", "elo": "1600", "run_id": None},
        ],
    )
    monkeypatch.setattr(Pipeline, "model_submission_stats", lambda _pipeline: {})

    class FakeClient:
        def __init__(self, tracking_uri):
            assert tracking_uri == "http://omarchyd:5000"

        def get_run(self, run_id):
            assert run_id == "run30"
            return SimpleNamespace(info=SimpleNamespace(experiment_id="42"))

    monkeypatch.setattr("mlflow.tracking.MlflowClient", FakeClient)

    assert cli.main(["models", "--task", "pokemon-tcg-ai-battle", "--mlflow", "--no-color"]) == 0
    out = capsys.readouterr().out

    assert "mlflow" in out
    assert "http://omarchyd:5000/#/experiments/42/runs/run30" in out


def test_models_no_color_suppresses_elo_trust_colors(monkeypatch, capsys):
    from kego.pipeline import cli

    monkeypatch.setattr(cli, "detect_task", lambda: "pokemon-tcg-ai-battle")
    monkeypatch.setattr("kego.tracking.default_tracking_uri", lambda: "http://mlflow")
    monkeypatch.setattr(
        "kego.tracking.leaderboard",
        lambda uri, task, sort_by: [{"version": "1", "elo": "1700", "elo_rd": "0.9"}],
    )
    assert cli.main(["models", "--task", "pokemon-tcg-ai-battle", "--no-color"]) == 0
    out = capsys.readouterr().out

    assert "elo color:" not in out
    assert "\x1b[" not in out


def test_models_prune(monkeypatch, capsys):
    from kego.pipeline import cli

    monkeypatch.setattr(cli, "detect_task", lambda: "pokemon-tcg-ai-battle")
    monkeypatch.setattr("kego.tracking.default_tracking_uri", lambda: "http://mlflow")

    transitioned = []
    tagged = []

    class FakeMlflowClient:
        def __init__(self, tracking_uri):
            pass

        def transition_model_version_stage(self, name, version, stage):
            transitioned.append((name, version, stage))

        def set_model_version_tag(self, name, version, key, value):
            tagged.append((name, version, key, value))

    import mlflow.tracking

    monkeypatch.setattr(mlflow.tracking, "MlflowClient", FakeMlflowClient)

    assert cli.main(["models", "prune", "--task", "pokemon-tcg-ai-battle", "5"]) == 0
    out = capsys.readouterr().out

    assert transitioned == [("pokemon-tcg-ai-battle", "5", "Archived")]
    assert tagged == [("pokemon-tcg-ai-battle", "5", "status", "archived")]
    assert "Pruned version 5" in out


def test_models_unprune(monkeypatch, capsys):
    from kego.pipeline import cli

    monkeypatch.setattr(cli, "detect_task", lambda: "pokemon-tcg-ai-battle")
    monkeypatch.setattr("kego.tracking.default_tracking_uri", lambda: "http://mlflow")

    transitioned = []
    tagged = []

    class FakeMlflowClient:
        def __init__(self, tracking_uri):
            pass

        def transition_model_version_stage(self, name, version, stage):
            transitioned.append((name, version, stage))

        def set_model_version_tag(self, name, version, key, value):
            tagged.append((name, version, key, value))

    import mlflow.tracking

    monkeypatch.setattr(mlflow.tracking, "MlflowClient", FakeMlflowClient)

    assert cli.main(["models", "unprune", "--task", "pokemon-tcg-ai-battle", "5"]) == 0
    out = capsys.readouterr().out

    assert transitioned == [("pokemon-tcg-ai-battle", "5", "None")]
    assert tagged == [
        ("pokemon-tcg-ai-battle", "5", "status", "active"),
        ("pokemon-tcg-ai-battle", "5", "dropped", "false"),
    ]
    assert "Unpruned version 5" in out


def test_models_prune_parser():
    parser = build_parser()
    args = parser.parse_args(
        [
            "models",
            "prune",
            "--task",
            "pokemon-tcg-ai-battle",
            "--drop-worse",
            "--drop-worse-min-games",
            "40",
            "--drop-worse-k",
            "2.5",
        ]
    )
    assert args.command == "models"
    assert args.models_cmd == "prune"
    assert args.drop_worse is True
    assert args.drop_worse_min_games == 40
    assert args.drop_worse_k == 2.5
    assert args.versions == []

    args = parser.parse_args(["models", "prune", "--task", "pokemon-tcg-ai-battle", "3", "5"])
    assert args.versions == ["3", "5"]
    assert args.drop_worse is False


def test_league_parser():
    parser = build_parser()
    args = parser.parse_args(
        [
            "league",
            "--task",
            "pokemon-tcg-ai-battle",
            "--target",
            "mn",
            "--games",
            "8",
            "--search-count",
            "12",
            "--workers",
            "4",
            "--no-write-ratings",
            "--cache-dir",
            "outputs/cache",
            "--partial-save-every",
            "123",
            "--stall-timeout",
            "45",
        ]
    )

    assert args.command == "league"
    assert args.league_cmd is None
    assert args.target == "mn"
    assert args.games == 8
    assert args.search_count == 12
    assert args.workers == 4
    assert args.write_ratings is False
    assert args.cache_dir == "outputs/cache"
    assert args.partial_save_every == 123
    assert args.stall_timeout == 45


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_league_parser_rejects_invalid_stall_timeout(value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["league", "--stall-timeout", value])


def test_league_matrix_parser():
    parser = build_parser()
    args = parser.parse_args(["league", "matrix", "--task", "pokemon-tcg-ai-battle", "--run-id", "abc123"])

    assert args.command == "league"
    assert args.league_cmd == "matrix"
    assert args.run_id == "abc123"


def test_league_merge_parser():
    parser = build_parser()
    args = parser.parse_args(
        [
            "league",
            "merge",
            "--task",
            "pokemon-tcg-ai-battle",
            "--run-ids",
            "run1",
            "run2",
            "--write-ratings",
        ]
    )

    assert args.command == "league"
    assert args.league_cmd == "merge"
    assert args.run_ids == ["run1", "run2"]
    assert args.write_ratings is True

    args = parser.parse_args(["league", "merge", "--task", "pokemon-tcg-ai-battle"])
    assert args.run_ids == []
    assert args.latest is None
    assert args.write_ratings is True

    args = parser.parse_args(["league", "merge", "--task", "pokemon-tcg-ai-battle", "--latest", "2"])
    assert args.command == "league"
    assert args.league_cmd == "merge"
    assert args.run_ids == []
    assert args.latest == 2

    args = parser.parse_args(["league", "merge", "--task", "pokemon-tcg-ai-battle", "--no-write-ratings"])
    assert args.write_ratings is False


def test_auto_league_run_ids_skips_consumed_and_missing_artifacts(monkeypatch):
    from types import SimpleNamespace

    from kego.pipeline import cli

    class Client:
        def get_experiment_by_name(self, name):
            return SimpleNamespace(experiment_id="50")

        def search_runs(self, _experiment_ids, filter_string, order_by, max_results):
            if "leaderboard_merge" in filter_string:
                return [SimpleNamespace(data=SimpleNamespace(tags={"source_run_ids": "old1,old2"}))]
            return [
                SimpleNamespace(info=SimpleNamespace(run_id="new2")),
                SimpleNamespace(info=SimpleNamespace(run_id="new1")),
                SimpleNamespace(info=SimpleNamespace(run_id="old1")),
                SimpleNamespace(info=SimpleNamespace(run_id="bad")),
            ]

        def list_artifacts(self, run_id, path):
            if run_id == "bad":
                return []
            if path == "league":
                return [
                    SimpleNamespace(path="league/matrix.md"),
                    SimpleNamespace(path="league/wins.csv"),
                    SimpleNamespace(path="league/games.csv"),
                    SimpleNamespace(path="league/results.json"),
                    SimpleNamespace(path="league/participants.json"),
                ]
            return []

    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda tracking_uri=None: Client())
    monkeypatch.setattr("kego.tracking.default_tracking_uri", lambda: "http://mlflow")

    assert cli._auto_league_run_ids("pokemon-tcg-ai-battle") == ["new2", "new1"]
    assert cli._auto_league_run_ids("pokemon-tcg-ai-battle", latest=1) == ["new2"]


def test_show_league_matrix_falls_back_to_root_artifact(monkeypatch, capsys):

    from kego.pipeline import cli

    class Client:
        def download_artifacts(self, run_id, path, dst_path):
            if path == "league/matrix.md":
                raise RuntimeError("missing league/")
            out = Path(dst_path) / "matrix.md"
            out.write_text("# matrix\n")
            return str(out)

    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda tracking_uri=None: Client())
    monkeypatch.setattr("kego.tracking.default_tracking_uri", lambda: "http://mlflow")

    assert cli._show_league_matrix("pokemon-tcg-ai-battle", run_id="run1") == 0
    out = capsys.readouterr().out
    assert "league matrix from MLflow run run1" in out
    assert "# matrix" in out


def test_show_league_matrix_skips_running_runs(monkeypatch, capsys):
    from types import SimpleNamespace

    from kego.pipeline import cli

    class Client:
        def get_experiment_by_name(self, name):
            return SimpleNamespace(experiment_id="1")

        def search_runs(self, *_args, **_kwargs):
            return [
                SimpleNamespace(info=SimpleNamespace(run_id="running")),
                SimpleNamespace(info=SimpleNamespace(run_id="done")),
            ]

        def list_artifacts(self, run_id, path):
            if run_id != "done":
                return []
            if path == "league":
                return [
                    SimpleNamespace(path="league/matrix.md"),
                    SimpleNamespace(path="league/wins.csv"),
                    SimpleNamespace(path="league/games.csv"),
                    SimpleNamespace(path="league/results.json"),
                    SimpleNamespace(path="league/participants.json"),
                ]
            return []

        def download_artifacts(self, run_id, path, dst_path):
            out = Path(dst_path) / "matrix.md"
            out.write_text(f"matrix for {run_id}\n")
            return str(out)

    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda tracking_uri=None: Client())
    monkeypatch.setattr("kego.tracking.default_tracking_uri", lambda: "http://mlflow")
    monkeypatch.setattr(cli, "_has_league_artifacts", lambda client, run_id: run_id == "done")

    assert cli._show_league_matrix("pokemon-tcg-ai-battle") == 0
    out = capsys.readouterr().out
    assert "run done" in out
    assert "matrix for done" in out


def test_has_league_artifacts_accepts_root_layout(monkeypatch):
    from types import SimpleNamespace

    from kego.pipeline import cli

    class Client:
        def list_artifacts(self, run_id, path):
            if path == "league":
                return []
            return [
                SimpleNamespace(path="matrix.md"),
                SimpleNamespace(path="wins.csv"),
                SimpleNamespace(path="games.csv"),
                SimpleNamespace(path="results.json"),
                SimpleNamespace(path="participants.json"),
            ]

    assert cli._has_league_artifacts(Client(), "run1") is True


def test_train_agent_target_parser():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train-agent",
            "--agent",
            "mcts",
            "--task",
            "pkmn",
            "--target",
            "m5",
            "--epochs",
            "200",
            "--init-checkpoint",
            "registry:7",
            "--num-workers",
            "10",
            "--variant",
            "large256_abomasnow",
        ]
    )
    assert args.command == "train-agent"
    assert args.target == "m5"
    assert args.epochs == 200
    assert args.init_checkpoint == "registry:7"
    assert args.num_workers == 10
    assert args.variant == "large256_abomasnow"

    # --target is optional (local run when omitted)
    args = parser.parse_args(["train-agent", "--agent", "mcts", "--variant", "small192_zacian", "--epochs", "5"])
    assert args.target is None
