from pathlib import Path
from unittest.mock import MagicMock, patch

from kego.pipeline.config import SubmitConfig
from kego.pipeline.submit import Submitter
from kego.pipeline.task import RawData


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


class DummySimTask(DummyTask):
    name = "pokemon-tcg-ai-battle"
    kaggle_slug = "pokemon-tcg-ai-battle"
    is_simulation = True

    def make_submission(self, ids, preds):
        return Path("dummy_submission.tar.gz")


def test_submit_standard(tmp_path):
    task = DummyTask()
    config = SubmitConfig(enabled=True, poll_timeout_s=5)
    submitter = Submitter(task, config)

    with patch("subprocess.run") as mock_run:
        # Mock subprocess.run responses:
        # 1. kaggle competitions submit
        # 2. kaggle competitions submissions
        mock_submit = MagicMock()
        mock_submit.returncode = 0

        mock_submissions = MagicMock()
        mock_submissions.returncode = 0
        mock_submissions.stdout = "status,publicScore\ncomplete,0.85\n"

        mock_run.side_effect = [mock_submit, mock_submissions]

        res = submitter.submit(tmp_path / "submission.csv", "my message")

        assert res.public_score == 0.85
        assert res.status == "complete"

        # Verify first subprocess call
        args, _ = mock_run.call_args_list[0]
        cmd = args[0]
        assert "competitions" in cmd
        assert "submit" in cmd
        assert "-c" in cmd
        assert "dummy-comp" in cmd
        assert "-f" in cmd
        assert "-k" not in cmd


def test_submit_simulation(tmp_path, monkeypatch):
    task = DummySimTask()
    config = SubmitConfig(enabled=True, poll_timeout_s=5)
    submitter = Submitter(task, config)

    # Set up dummy kernel folder
    kernel_dir = tmp_path / "competitions" / "pokemon-tcg-ai-battle" / "kernel"
    kernel_dir.mkdir(parents=True)
    metadata_path = kernel_dir / "kernel-metadata.json"
    metadata_path.write_text('{"id": "aldisued/ptcg-ai-battle-submission"}')

    monkeypatch.chdir(tmp_path)

    with patch("subprocess.run") as mock_run:
        # Mock subprocess.run responses:
        # 1. kaggle kernels push
        # 2. kaggle kernels status -> complete
        # 3. kaggle competitions submit
        # 4. kaggle competitions submissions
        mock_push = MagicMock()
        mock_push.returncode = 0
        mock_push.stdout = "Kernel version 5 successfully pushed.\n"

        mock_status = MagicMock()
        mock_status.returncode = 0
        mock_status.stdout = 'aldisued/ptcg-ai-battle-submission has status "complete"\n'

        mock_submit = MagicMock()
        mock_submit.returncode = 0

        mock_submissions = MagicMock()
        mock_submissions.returncode = 0
        mock_submissions.stdout = "status,publicScore\ncomplete,0.92\n"

        mock_run.side_effect = [mock_push, mock_status, mock_submit, mock_submissions]

        res = submitter.submit(tmp_path / "submission.tar.gz", "sim message")

        assert res.public_score == 0.92
        assert res.status == "complete"

        # Verify kernel push call
        call_push = mock_run.call_args_list[0][0][0]
        assert "kernels" in call_push
        assert "push" in call_push

        # Verify status call
        call_status = mock_run.call_args_list[1][0][0]
        assert "kernels" in call_status
        assert "status" in call_status
        assert "aldisued/ptcg-ai-battle-submission" in call_status

        # Verify submit call
        call_submit = mock_run.call_args_list[2][0][0]
        assert "competitions" in call_submit
        assert "submit" in call_submit
        assert "-k" in call_submit
        assert "aldisued/ptcg-ai-battle-submission" in call_submit
        assert "-f" in call_submit
        assert "-v" in call_submit
        assert "5" in call_submit


def test_submit_simulation_from_competition_dir(tmp_path, monkeypatch):
    """The pipeline is normally invoked from the competition directory itself
    (where kego.toml + outputs/ live), so the kernel dir is just ``kernel/`` —
    not ``competitions/<slug>/kernel``. Submitting from there must still find it."""
    task = DummySimTask()
    config = SubmitConfig(enabled=True, poll_timeout_s=5)
    submitter = Submitter(task, config)

    # Layout: cwd IS the competition dir, so kernel/ sits directly under it.
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel-metadata.json").write_text('{"id": "aldisued/ptcg-ai-battle-submission"}')

    monkeypatch.chdir(tmp_path)

    with patch("subprocess.run") as mock_run:
        mock_push = MagicMock()
        mock_push.returncode = 0
        mock_push.stdout = "Kernel version 5 successfully pushed.\n"

        mock_status = MagicMock()
        mock_status.returncode = 0
        mock_status.stdout = 'aldisued/ptcg-ai-battle-submission has status "complete"\n'

        mock_submit = MagicMock()
        mock_submit.returncode = 0

        mock_submissions = MagicMock()
        mock_submissions.returncode = 0
        mock_submissions.stdout = "status,publicScore\ncomplete,0.92\n"

        mock_run.side_effect = [mock_push, mock_status, mock_submit, mock_submissions]

        res = submitter.submit(tmp_path / "submission.tar.gz", "sim message")

        assert res.status == "complete"
        assert res.public_score == 0.92

        # The kernel push must target the kernel/ dir, not a missing nested path.
        call_push = mock_run.call_args_list[0][0][0]
        assert "push" in call_push
        assert str(Path("kernel")) in call_push


def test_pokemon_tcg_ai_battle_make_submission(tmp_path, monkeypatch):
    """Test that the PokemonTCGAIBattleTask.make_submission correctly reads kego.toml,
    checks file existence, and packages agents/base.py as base_agent.py alongside
    renaming the configured agent and deck files to main.py and deck.csv."""
    import importlib.util

    task_path = Path(__file__).resolve().parents[1] / "competitions/pokemon-tcg-ai-battle/task.py"
    spec = importlib.util.spec_from_file_location("pokemon_tcg_task", task_path)
    task_module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(task_module)
    PokemonTCGAIBattleTask = task_module.PokemonTCGAIBattleTask

    task = PokemonTCGAIBattleTask()

    # Set up temporary directories to simulate repository root
    comp_dir = tmp_path / "competitions" / "pokemon-tcg-ai-battle"
    comp_dir.mkdir(parents=True)

    # Create agents and decks folders
    agents_dir = comp_dir / "agents"
    agents_dir.mkdir()
    decks_dir = comp_dir / "decks"
    decks_dir.mkdir()

    # Create mock agent, base, and deck files
    agent_file = agents_dir / "mock_agent.py"
    agent_file.write_text("print('mock agent')")

    base_file = agents_dir / "base.py"
    base_file.write_text("print('base agent')")

    deck_file = decks_dir / "mock_deck.csv"
    deck_file.write_text("1\n2\n3\n")

    # Create mock sample submission files (cg directory)
    src_dir = tmp_path / "data" / "pokemon" / "pokemon-tcg-ai-battle" / "sample_submission" / "sample_submission"
    src_dir.mkdir(parents=True)
    cg_dir = src_dir / "cg"
    cg_dir.mkdir()
    (cg_dir / "api.py").write_text("print('cg api')")

    # Write a kego.toml
    kego_toml = comp_dir / "kego.toml"
    kego_toml.write_text("""
[competition]
agent_file = "agents/mock_agent.py"
deck_file = "decks/mock_deck.csv"
""")

    # Change working directory to tmp_path
    monkeypatch.chdir(tmp_path)

    import tarfile

    import numpy as np

    # Mock Path inside task.py to return tmp_path as repo_root
    with patch.object(task_module, "Path") as mock_path_class:

        def path_mock_factory(*args, **kwargs):
            if args and "__file__" in str(args[0]):
                p = MagicMock()
                # parents[2] should resolve to tmp_path
                p.resolve.return_value.parents = [tmp_path, tmp_path, tmp_path]
                return p
            return Path(*args, **kwargs)

        mock_path_class.side_effect = path_mock_factory

        sub_path = task.make_submission(np.array([]), np.array([]))

        # Verify target paths
        assert sub_path.exists()
        assert sub_path.name == "submission.tar.gz"

        # Verify contents of submission.tar.gz
        with tarfile.open(sub_path, "r:gz") as tar:
            members = {m.name: m for m in tar.getmembers()}
            assert "main.py" in members
            assert "deck.csv" in members
            assert "base_agent.py" in members
            assert "cg" in members
            assert "cg/api.py" in members

            # Read and assert main.py content
            f_main = tar.extractfile("main.py")
            assert f_main.read().decode() == "print('mock agent')"

            # Read and assert base_agent.py content
            f_base = tar.extractfile("base_agent.py")
            assert f_base.read().decode() == "print('base agent')"

            # Read and assert deck.csv content
            f_deck = tar.extractfile("deck.csv")
            assert f_deck.read().decode() == "1\n2\n3\n"

        # Verify submission_notebook.py generation
        notebook_path = comp_dir / "kernel" / "submission_notebook.py"
        assert notebook_path.exists()
        notebook_content = notebook_path.read_text()
        assert "main_py_content =" in notebook_content
        assert "print('mock agent')" in notebook_content
        assert 'os.walk("/kaggle/input")' in notebook_content
        assert (
            "base_agent_content" in notebook_content
            or "base_agent_py_content" in notebook_content
            or "base_py_content" in notebook_content
        )
        assert "print('base agent')" in notebook_content


def test_pokemon_mcts_submission_agent_auto_loads_packaged_weights():
    """Kaggle calls ``agent()`` without our local league's explicit model_path.
    The submitted MCTS agent must therefore discover packaged mcts.pth itself."""
    mcts_path = Path(__file__).resolve().parents[1] / "competitions/pokemon-tcg-ai-battle/agents/mcts.py"
    content = mcts_path.read_text()

    assert "def _default_model_path()" in content
    assert 'os.path.join(base_dir, "mcts.pth")' in content
    assert '"/kaggle_simulations/agent/mcts.pth"' in content
    assert 'deck=os.environ.get("MCTS_DECK", "deck.csv")' in content
    assert "model_path=_default_model_path()" in content


def test_submit_leader_uses_registry_deck_tag():
    submit_leader = Path(__file__).resolve().parents[1] / "competitions/pokemon-tcg-ai-battle/submit_leader.py"
    content = submit_leader.read_text()

    assert 'deck_name = leader.get("deck", "abomasnow")' in content
    assert 'deck_file = "decks/{deck_name}.csv"' in content
    assert 'deck_file = "decks/abomasnow.csv"' not in content
