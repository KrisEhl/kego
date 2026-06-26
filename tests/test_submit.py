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
