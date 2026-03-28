import os
import textwrap
from unittest.mock import patch

import pytest

from kego.cli.runner import parse_kego_lines, run


@pytest.fixture
def dummy_script(tmp_path):
    """A script that emits KEGO_ lines and exits 0."""
    script = tmp_path / "dummy_train.py"
    script.write_text(
        textwrap.dedent("""
        import sys
        print("Training started")
        print("KEGO_METRIC fold_auc 0.8821")
        print("KEGO_METRIC val_loss 0.3142")
        print("KEGO_PARAM backbone efficientnet_b0")
        print("Normal output line")
        sys.exit(0)
    """)
    )
    return script


@pytest.fixture
def failing_script(tmp_path):
    script = tmp_path / "fail_train.py"
    script.write_text("import sys; sys.exit(1)")
    return script


def test_parse_kego_lines_metric():
    lines = [
        "KEGO_METRIC fold_auc 0.8821",
        "Normal line",
        "KEGO_METRIC val_loss 0.3142",
    ]
    metrics, params = parse_kego_lines(lines)
    assert metrics == {"fold_auc": 0.8821, "val_loss": 0.3142}
    assert params == {}


def test_parse_kego_lines_param():
    lines = [
        "KEGO_PARAM backbone efficientnet_b0",
        "KEGO_PARAM n_mels 224",
    ]
    metrics, params = parse_kego_lines(lines)
    assert params == {"backbone": "efficientnet_b0", "n_mels": "224"}
    assert metrics == {}


def test_parse_kego_lines_ignores_non_kego():
    lines = ["just normal output", "KEGO_INVALID x y", ""]
    metrics, params = parse_kego_lines(lines)
    assert metrics == {}
    assert params == {}


def test_run_returns_zero_on_success(dummy_script):
    env_vars = {
        "MLFLOW_TRACKING_URI": "",
        "KEGO_EXPERIMENT_NAME": "test-exp",
        "KEGO_EXPERIMENT_ID": "abc123",
        "KEGO_CLI_PARAMS": "{}",
    }
    with patch.dict(os.environ, env_vars), patch("kego.cli.runner._log_to_mlflow"):
        exit_code = run([str(dummy_script)])
    assert exit_code == 0


def test_run_returns_nonzero_on_failure(failing_script):
    env_vars = {
        "MLFLOW_TRACKING_URI": "",
        "KEGO_EXPERIMENT_NAME": "test-exp",
        "KEGO_EXPERIMENT_ID": "abc123",
        "KEGO_CLI_PARAMS": "{}",
    }
    with patch.dict(os.environ, env_vars), patch("kego.cli.runner._log_to_mlflow"):
        exit_code = run([str(failing_script)])
    assert exit_code == 1


def test_run_captures_metrics(dummy_script):
    captured = {}

    def fake_log(
        tracking_uri,
        experiment_name,
        run_name,
        experiment_id,
        cli_params,
        metrics,
        params,
        extra_tags=None,
    ):
        captured.update({"metrics": metrics, "params": params})

    env_vars = {
        "MLFLOW_TRACKING_URI": "",
        "KEGO_EXPERIMENT_NAME": "test",
        "KEGO_EXPERIMENT_ID": "abc123",
        "KEGO_CLI_PARAMS": "{}",
    }
    with (
        patch.dict(os.environ, env_vars),
        patch("kego.cli.runner._log_to_mlflow", side_effect=fake_log),
    ):
        run([str(dummy_script)])

    assert captured["metrics"]["fold_auc"] == pytest.approx(0.8821)
    assert captured["metrics"]["val_loss"] == pytest.approx(0.3142)
    assert captured["params"]["backbone"] == "efficientnet_b0"
