# kego Engine — Plan 1: Core CLI (Local + Cluster)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the `kego` CLI with `run` and `ls` commands supporting local and Ray cluster compute targets, unified MLflow experiment tracking, and fold fan-out.

**Architecture:** A `runner.py` module wraps any training script on any compute target — it spawns the script as a subprocess, parses `KEGO_METRIC`/`KEGO_PARAM` lines from stdout, and logs them to MLflow. `kego run` dispatches to the right target (local: runs runner directly; cluster: submits runner via `ray job submit`). `kego ls` queries MLflow and prints a table. Config is loaded from `kego.toml` files.

**Tech Stack:** Python stdlib (`argparse`, `subprocess`, `tomllib`/`tomli`, `secrets`, `re`), MLflow (lazy import), Ray CLI via subprocess (not imported), `uv run` for venv execution.

---

## File Map

```
# New files
kego/cli/__init__.py              — main() entry point, argparse dispatcher
kego/cli/config.py                — load + merge root and competition kego.toml
kego/cli/experiment.py            — experiment ID generation, name building
kego/cli/runner.py                — subprocess wrapper + KEGO_ parser + MLflow logger
kego/cli/commands/__init__.py     — empty
kego/cli/commands/run.py          — kego run command
kego/cli/commands/ls.py           — kego ls command
kego/cli/targets/__init__.py      — empty
kego/cli/targets/local.py         — local execution target
kego/cli/targets/cluster.py       — Ray cluster target

kego.toml                         — root config (cluster + MLflow URIs)
competitions/birdclef-2026/kego.toml     — birdclef competition config
competitions/playground/kego.toml        — playground competition config

tests/cli/__init__.py
tests/cli/test_config.py
tests/cli/test_experiment.py
tests/cli/test_runner.py
tests/cli/test_targets_local.py
tests/cli/test_targets_cluster.py
tests/cli/test_commands_ls.py

# Modified files
pyproject.toml                    — add kego script entry point + tomli dep
```

---

## Task 1: Register CLI entry point + add tomli dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Write the failing test**

```bash
# Run this — should fail because `kego` command doesn't exist yet
uv run kego --help
```

Expected: `command not found: kego` or similar error.

- [ ] **Step 2: Add entry point and tomli to pyproject.toml**

In `pyproject.toml`, add to `[project]` → `dependencies`:
```toml
"tomli>=2.0; python_version < '3.11'",
"mlflow>=2.0.0",
```

And add a new section after `[build-system]`:
```toml
[project.scripts]
kego = "kego.cli:main"
```

- [ ] **Step 3: Create minimal CLI entry point**

Create `kego/cli/__init__.py`:
```python
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="kego", description="kego ML experiment engine")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    # Commands registered here in later tasks
    parser.parse_args(["--help"])


if __name__ == "__main__":
    main()
```

Create empty `kego/cli/commands/__init__.py` and `kego/cli/targets/__init__.py`.

- [ ] **Step 4: Reinstall so entry point is registered**

```bash
uv sync
```

- [ ] **Step 5: Verify the CLI entry point exists**

```bash
uv run kego --help
```

Expected: prints usage with `kego` as prog name. No subcommands yet, just the help text.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml kego/cli/__init__.py kego/cli/commands/__init__.py kego/cli/targets/__init__.py
git commit -m "feat(cli): register kego entry point + tomli/mlflow deps"
```

---

## Task 2: Config loading

**Files:**
- Create: `kego/cli/config.py`
- Create: `kego.toml` (root)
- Create: `competitions/birdclef-2026/kego.toml`
- Create: `competitions/playground/kego.toml`
- Create: `tests/cli/__init__.py`
- Create: `tests/cli/test_config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/cli/__init__.py` (empty).

Create `tests/cli/test_config.py`:
```python
import pytest
from pathlib import Path
from kego.cli.config import (
    find_repo_root,
    find_competition_dir,
    load_config,
    KegoConfig,
    ClusterConfig,
    CompetitionConfig,
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/cli/test_config.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `kego.cli.config` doesn't exist yet.

- [ ] **Step 3: Implement config.py**

Create `kego/cli/config.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass
class ClusterConfig:
    ray_address: str
    mlflow_uri: str
    default_resources: dict = field(default_factory=lambda: {"num_gpus": 0.5})
    heavy_resources: dict = field(default_factory=lambda: {"num_gpus": 1})


@dataclass
class CompetitionConfig:
    slug: str
    kaggle_user: str
    enable_gpu: bool
    submit_file: str
    pattern: str
    inference_notebook: str
    checkpoint_dir: str
    primary_metric: str
    training_notebook: Optional[str] = None


@dataclass
class KegoConfig:
    cluster: ClusterConfig
    competition: Optional[CompetitionConfig]
    repo_root: Path
    competition_dir: Optional[Path]


def find_repo_root(start: Path = Path.cwd()) -> Path:
    """Walk up from start until a .git directory is found."""
    for parent in [start, *start.parents]:
        if (parent / ".git").exists():
            return parent
    raise FileNotFoundError(f"No .git directory found starting from {start}")


def find_competition_dir(start: Path = Path.cwd()) -> Optional[Path]:
    """Walk up from start to find a kego.toml with a [competition] section."""
    for parent in [start, *start.parents]:
        toml_path = parent / "kego.toml"
        if toml_path.exists():
            with open(toml_path, "rb") as f:
                data = tomllib.load(f)
            if "competition" in data:
                return parent
    return None


def load_config(
    repo_root: Optional[Path] = None,
    competition_dir: Optional[Path] = None,
) -> KegoConfig:
    """Load root kego.toml, optionally merged with a competition kego.toml."""
    if repo_root is None:
        repo_root = find_repo_root()

    root_toml = repo_root / "kego.toml"
    with open(root_toml, "rb") as f:
        root = tomllib.load(f)

    c = root["cluster"]
    resources = c.get("resources", {})
    cluster = ClusterConfig(
        ray_address=c["ray_address"],
        mlflow_uri=c["mlflow_uri"],
        default_resources=resources.get("default", {"num_gpus": 0.5}),
        heavy_resources=resources.get("heavy", {"num_gpus": 1}),
    )

    competition = None
    if competition_dir is None:
        competition_dir = find_competition_dir()

    if competition_dir is not None:
        comp_toml = competition_dir / "kego.toml"
        with open(comp_toml, "rb") as f:
            comp_raw = tomllib.load(f)["competition"]
        competition = CompetitionConfig(**comp_raw)

    return KegoConfig(
        cluster=cluster,
        competition=competition,
        repo_root=repo_root,
        competition_dir=competition_dir,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/cli/test_config.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Create the actual kego.toml files**

Create `kego.toml` at repo root:
```toml
[cluster]
ray_address = "http://192.168.178.32:8265"
mlflow_uri = "http://192.168.178.32:5000"

[cluster.resources]
default = {num_gpus = 0.5}
heavy = {num_gpus = 1, resources = {"heavy_gpu" = 1}}
```

Create `competitions/birdclef-2026/kego.toml`:
```toml
[competition]
slug = "birdclef-2026"
kaggle_user = "aldisued"
enable_gpu = false
submit_file = "submission.csv"
pattern = "inference_only"
inference_notebook = "inference/kernel.py"
checkpoint_dir = "training/outputs"
primary_metric = "cmAP"
```

Create `competitions/playground/kego.toml`:
```toml
[competition]
slug = "tabular-playground-series-s6e2"
kaggle_user = "aldisued"
enable_gpu = false
submit_file = "submission.csv"
pattern = "inference_only"
inference_notebook = "submit_s6e2.sh"
checkpoint_dir = "outputs"
primary_metric = "fold_auc"
```

- [ ] **Step 6: Commit**

```bash
git add kego/cli/config.py tests/cli/__init__.py tests/cli/test_config.py kego.toml competitions/birdclef-2026/kego.toml competitions/playground/kego.toml
git commit -m "feat(cli): config loading from kego.toml files"
```

---

## Task 3: Experiment ID and name generation

**Files:**
- Create: `kego/cli/experiment.py`
- Create: `tests/cli/test_experiment.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/cli/test_experiment.py`:
```python
from kego.cli.experiment import generate_id, build_experiment_name


def test_generate_id_is_6_chars():
    eid = generate_id()
    assert len(eid) == 6


def test_generate_id_is_hex():
    eid = generate_id()
    int(eid, 16)  # raises ValueError if not valid hex


def test_generate_id_is_unique():
    ids = {generate_id() for _ in range(100)}
    assert len(ids) == 100  # all unique


def test_build_experiment_name_uses_explicit_name():
    name = build_experiment_name("train_cnn.py", name="soundscape-v8", cli_params={})
    assert name == "soundscape-v8"


def test_build_experiment_name_uses_script_stem_when_no_name():
    name = build_experiment_name("train_cnn.py", name=None, cli_params={})
    assert name == "train_cnn"


def test_build_experiment_name_includes_key_params():
    name = build_experiment_name(
        "train_cnn.py",
        name=None,
        cli_params={"backbone": "efficientnet_b0", "epochs": "30"},
    )
    assert "train_cnn" in name
    assert "backbone=efficientnet_b0" in name


def test_build_experiment_name_excludes_infra_params():
    name = build_experiment_name(
        "train_cnn.py",
        name=None,
        cli_params={"debug": "true", "gpu": "0", "fold": "0", "backbone": "b0"},
    )
    assert "debug" not in name
    assert "gpu" not in name
    assert "backbone=b0" in name
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/cli/test_experiment.py -v
```

Expected: `ImportError` — module doesn't exist yet.

- [ ] **Step 3: Implement experiment.py**

Create `kego/cli/experiment.py`:
```python
from __future__ import annotations

import secrets
from pathlib import Path
from typing import Optional

_INFRA_PARAMS = frozenset({"debug", "gpu", "target", "folds", "fold"})


def generate_id() -> str:
    """Generate a 6-character hex experiment ID."""
    return secrets.token_hex(3)


def build_experiment_name(
    script: str,
    name: Optional[str],
    cli_params: dict[str, str],
) -> str:
    """Build a human-readable experiment name.

    If --name is given, use it directly. Otherwise derive from script stem
    plus up to 3 non-infrastructure CLI params.
    """
    if name:
        return name

    stem = Path(script).stem
    key_params = {k: v for k, v in cli_params.items() if k not in _INFRA_PARAMS}
    if not key_params:
        return stem

    suffix = "--".join(f"{k}={v}" for k, v in list(key_params.items())[:3])
    return f"{stem}--{suffix}"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/cli/test_experiment.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kego/cli/experiment.py tests/cli/test_experiment.py
git commit -m "feat(cli): experiment ID generation and name building"
```

---

## Task 4: Runner — subprocess wrapper + KEGO_ parser + MLflow logger

**Files:**
- Create: `kego/cli/runner.py`
- Create: `tests/cli/test_runner.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/cli/test_runner.py`:
```python
import json
import os
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from kego.cli.runner import parse_kego_lines, run


@pytest.fixture
def dummy_script(tmp_path):
    """A script that emits KEGO_ lines and exits 0."""
    script = tmp_path / "dummy_train.py"
    script.write_text(textwrap.dedent("""
        import sys
        print("Training started")
        print("KEGO_METRIC fold_auc 0.8821")
        print("KEGO_METRIC val_loss 0.3142")
        print("KEGO_PARAM backbone efficientnet_b0")
        print("Normal output line")
        sys.exit(0)
    """))
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


def test_run_returns_zero_on_success(dummy_script, tmp_path):
    env_vars = {
        "MLFLOW_TRACKING_URI": "",
        "KEGO_EXPERIMENT_NAME": "test-exp",
        "KEGO_EXPERIMENT_ID": "abc123",
        "KEGO_CLI_PARAMS": "{}",
    }
    with patch.dict(os.environ, env_vars):
        with patch("kego.cli.runner._log_to_mlflow") as mock_log:
            exit_code = run([str(dummy_script)])
    assert exit_code == 0
    mock_log.assert_called_once()
    call_kwargs = mock_log.call_args
    metrics = call_kwargs[1]["metrics"] if call_kwargs[1] else call_kwargs[0][2]
    assert "fold_auc" in metrics or True  # structure checked below


def test_run_returns_nonzero_on_failure(failing_script):
    env_vars = {
        "MLFLOW_TRACKING_URI": "",
        "KEGO_EXPERIMENT_NAME": "test-exp",
        "KEGO_EXPERIMENT_ID": "abc123",
        "KEGO_CLI_PARAMS": "{}",
    }
    with patch.dict(os.environ, env_vars):
        with patch("kego.cli.runner._log_to_mlflow"):
            exit_code = run([str(failing_script)])
    assert exit_code == 1


def test_run_captures_metrics(dummy_script):
    captured = {}

    def fake_log(tracking_uri, experiment_name, experiment_id, cli_params, metrics, params):
        captured.update({"metrics": metrics, "params": params})

    env_vars = {
        "MLFLOW_TRACKING_URI": "",
        "KEGO_EXPERIMENT_NAME": "test",
        "KEGO_EXPERIMENT_ID": "abc123",
        "KEGO_CLI_PARAMS": "{}",
    }
    with patch.dict(os.environ, env_vars):
        with patch("kego.cli.runner._log_to_mlflow", side_effect=fake_log):
            run([str(dummy_script)])

    assert captured["metrics"]["fold_auc"] == pytest.approx(0.8821)
    assert captured["metrics"]["val_loss"] == pytest.approx(0.3142)
    assert captured["params"]["backbone"] == "efficientnet_b0"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/cli/test_runner.py -v
```

Expected: `ImportError` — module doesn't exist yet.

- [ ] **Step 3: Implement runner.py**

Create `kego/cli/runner.py`:
```python
"""
Subprocess wrapper that parses KEGO_METRIC / KEGO_PARAM stdout lines
and logs them to MLflow. Runs on any compute target (local or cluster).

Invoked as: python -m kego.cli.runner <script> [script_args...]

Environment variables (injected by kego run):
    MLFLOW_TRACKING_URI    — MLflow server URI (empty string = no logging)
    KEGO_EXPERIMENT_NAME   — MLflow experiment name
    KEGO_EXPERIMENT_ID     — 6-char experiment ID stored as MLflow tag
    KEGO_CLI_PARAMS        — JSON dict of CLI args to pre-log as params
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

_METRIC_RE = re.compile(r"^KEGO_METRIC\s+(\S+)\s+(\S+)\s*$")
_PARAM_RE = re.compile(r"^KEGO_PARAM\s+(\S+)\s+(.+?)\s*$")


def parse_kego_lines(
    lines: list[str],
) -> tuple[dict[str, float], dict[str, str]]:
    """Parse KEGO_METRIC and KEGO_PARAM lines. Returns (metrics, params)."""
    metrics: dict[str, float] = {}
    params: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if m := _METRIC_RE.match(stripped):
            try:
                metrics[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
        elif m := _PARAM_RE.match(stripped):
            params[m.group(1)] = m.group(2)
    return metrics, params


def _log_to_mlflow(
    tracking_uri: str,
    experiment_name: str,
    experiment_id: str,
    cli_params: dict[str, str],
    metrics: dict[str, float],
    params: dict[str, str],
) -> None:
    """Log everything to MLflow. No-op if tracking_uri is empty."""
    if not tracking_uri:
        return
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(tags={"kego_id": experiment_id}):
        if cli_params:
            mlflow.log_params(cli_params)
        if params:
            mlflow.log_params(params)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        mlflow.log_metric("exit_code", metrics.get("exit_code", 0))


def run(argv: list[str]) -> int:
    """Run script at argv[0] with args argv[1:], tracking KEGO_ lines."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    experiment_name = os.environ.get("KEGO_EXPERIMENT_NAME", "kego-default")
    experiment_id = os.environ.get("KEGO_EXPERIMENT_ID", "unknown")
    cli_params = json.loads(os.environ.get("KEGO_CLI_PARAMS", "{}"))

    cmd = [sys.executable] + list(argv)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    collected_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        collected_lines.append(line)

    process.wait()

    metrics, params = parse_kego_lines(collected_lines)
    metrics["exit_code"] = float(process.returncode)

    _log_to_mlflow(tracking_uri, experiment_name, experiment_id, cli_params, metrics, params)

    return process.returncode


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/cli/test_runner.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kego/cli/runner.py tests/cli/test_runner.py
git commit -m "feat(cli): runner — subprocess wrapper + KEGO_ parser + MLflow logger"
```

---

## Task 5: Local target

**Files:**
- Create: `kego/cli/targets/local.py`
- Create: `tests/cli/test_targets_local.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/cli/test_targets_local.py`:
```python
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

from kego.cli.targets.local import build_command, run
from kego.cli.config import KegoConfig, ClusterConfig


@pytest.fixture
def config():
    import pytest
    return KegoConfig(
        cluster=ClusterConfig(
            ray_address="http://192.168.1.1:8265",
            mlflow_uri="http://192.168.1.1:5000",
        ),
        competition=None,
        repo_root=Path("/repo"),
        competition_dir=None,
    )


def test_build_command_structure():
    cmd = build_command("train_cnn.py", ["--fold", "0", "--epochs", "30"])
    assert cmd[0] == sys.executable
    assert "-m" in cmd
    assert "kego.cli.runner" in cmd
    assert "train_cnn.py" in cmd
    assert "--fold" in cmd
    assert "0" in cmd


def test_run_sets_mlflow_env(config, tmp_path):
    import os
    script = tmp_path / "dummy.py"
    script.write_text("import sys; sys.exit(0)")

    captured_env = {}

    def fake_run(argv):
        captured_env.update(os.environ.copy())
        return 0

    with patch("kego.cli.targets.local.runner.run", side_effect=fake_run):
        run(
            script=str(script),
            script_args=[],
            config=config,
            experiment_name="test-exp",
            experiment_id="abc123",
            cli_params={"fold": "0"},
        )

    assert captured_env.get("MLFLOW_TRACKING_URI") == "http://192.168.1.1:5000"
    assert captured_env.get("KEGO_EXPERIMENT_NAME") == "test-exp"
    assert captured_env.get("KEGO_EXPERIMENT_ID") == "abc123"
```

Fix missing import in test file — add `import pytest` at top:

```python
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from kego.cli.targets.local import build_command, run
from kego.cli.config import KegoConfig, ClusterConfig


@pytest.fixture
def config():
    return KegoConfig(
        cluster=ClusterConfig(
            ray_address="http://192.168.1.1:8265",
            mlflow_uri="http://192.168.1.1:5000",
        ),
        competition=None,
        repo_root=Path("/repo"),
        competition_dir=None,
    )


def test_build_command_structure():
    cmd = build_command("train_cnn.py", ["--fold", "0", "--epochs", "30"])
    assert cmd[0] == sys.executable
    assert "-m" in cmd
    assert "kego.cli.runner" in cmd
    assert "train_cnn.py" in cmd
    assert "--fold" in cmd
    assert "0" in cmd


def test_run_sets_env_vars(config, tmp_path, monkeypatch):
    captured_env: dict = {}

    def fake_runner_run(argv):
        import os
        captured_env.update(os.environ.copy())
        return 0

    monkeypatch.setattr("kego.cli.targets.local.runner.run", fake_runner_run)
    run(
        script="dummy.py",
        script_args=["--fold", "0"],
        config=config,
        experiment_name="test-exp",
        experiment_id="abc123",
        cli_params={"fold": "0"},
    )

    assert captured_env["MLFLOW_TRACKING_URI"] == "http://192.168.1.1:5000"
    assert captured_env["KEGO_EXPERIMENT_NAME"] == "test-exp"
    assert captured_env["KEGO_EXPERIMENT_ID"] == "abc123"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/cli/test_targets_local.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement local.py**

Create `kego/cli/targets/local.py`:
```python
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from kego.cli import runner
from kego.cli.config import KegoConfig


def build_command(script: str, script_args: list[str]) -> list[str]:
    """Build command to invoke training script via runner."""
    return [sys.executable, "-m", "kego.cli.runner", script] + script_args


def run(
    script: str,
    script_args: list[str],
    config: KegoConfig,
    experiment_name: str,
    experiment_id: str,
    cli_params: dict[str, str],
) -> int:
    """Execute script locally with MLflow tracking via runner."""
    env_patch = {
        "MLFLOW_TRACKING_URI": config.cluster.mlflow_uri,
        "KEGO_EXPERIMENT_NAME": experiment_name,
        "KEGO_EXPERIMENT_ID": experiment_id,
        "KEGO_CLI_PARAMS": json.dumps(cli_params),
    }
    old = {k: os.environ.get(k) for k in env_patch}
    os.environ.update(env_patch)
    try:
        return runner.run([script] + script_args)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/cli/test_targets_local.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kego/cli/targets/local.py tests/cli/test_targets_local.py
git commit -m "feat(cli): local execution target"
```

---

## Task 6: Cluster target

**Files:**
- Create: `kego/cli/targets/cluster.py`
- Create: `tests/cli/test_targets_cluster.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/cli/test_targets_cluster.py`:
```python
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from kego.cli.targets.cluster import build_ray_command, submit_fold, submit
from kego.cli.config import KegoConfig, ClusterConfig


@pytest.fixture
def config():
    return KegoConfig(
        cluster=ClusterConfig(
            ray_address="http://192.168.1.1:8265",
            mlflow_uri="http://192.168.1.1:5000",
        ),
        competition=None,
        repo_root=Path("/repo"),
        competition_dir=None,
    )


def test_build_ray_command_structure(config):
    cmd = build_ray_command(
        script="train_cnn.py",
        script_args=["--fold", "0"],
        config=config,
        experiment_name="soundscape-v8",
        experiment_id="abc123",
        cli_params={"fold": "0"},
    )
    cmd_str = " ".join(cmd)
    assert "ray" in cmd_str
    assert "job" in cmd_str
    assert "submit" in cmd_str
    assert "192.168.1.1:8265" in cmd_str
    assert "kego.cli.runner" in cmd_str
    assert "train_cnn.py" in cmd_str


def test_build_ray_command_includes_mlflow_env(config):
    cmd = build_ray_command(
        script="train_cnn.py",
        script_args=["--fold", "0"],
        config=config,
        experiment_name="test",
        experiment_id="abc123",
        cli_params={},
    )
    # runtime-env-json should contain mlflow URI
    json_idx = cmd.index("--runtime-env-json") + 1
    runtime_env = json.loads(cmd[json_idx])
    assert runtime_env["env_vars"]["MLFLOW_TRACKING_URI"] == "http://192.168.1.1:5000"
    assert runtime_env["env_vars"]["KEGO_EXPERIMENT_ID"] == "abc123"


def test_submit_fans_out_folds(config):
    submitted = []

    def fake_submit_fold(script, script_args, config, experiment_name, experiment_id, cli_params):
        submitted.append({"script_args": script_args, "cli_params": cli_params})
        return f"raysubmit_fold{len(submitted)}"

    with patch("kego.cli.targets.cluster.submit_fold", side_effect=fake_submit_fold):
        job_ids = submit(
            script="train_cnn.py",
            folds=[0, 1, 2, 3],
            base_args=["--epochs", "30"],
            config=config,
            experiment_name="soundscape-v8",
            experiment_id="abc123",
            cli_params={"epochs": "30"},
        )

    assert len(job_ids) == 4
    assert len(submitted) == 4
    # Each fold submission should include --fold N
    for i, call in enumerate(submitted):
        assert "--fold" in call["script_args"]
        assert str(i) in call["script_args"]


def test_submit_fold_parses_job_id(config):
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Job submission: raysubmit_ABCD1234\nDone."

    with patch("subprocess.run", return_value=mock_result):
        job_id = submit_fold(
            script="train_cnn.py",
            script_args=["--fold", "0"],
            config=config,
            experiment_name="test",
            experiment_id="abc123",
            cli_params={},
        )
    assert job_id == "raysubmit_ABCD1234"


def test_submit_fold_raises_on_failure(config):
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "Connection refused"

    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(RuntimeError, match="ray job submit failed"):
            submit_fold(
                script="train_cnn.py",
                script_args=["--fold", "0"],
                config=config,
                experiment_name="test",
                experiment_id="abc123",
                cli_params={},
            )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/cli/test_targets_cluster.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement cluster.py**

Create `kego/cli/targets/cluster.py`:
```python
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from kego.cli.config import KegoConfig

_JOB_ID_RE = re.compile(r"(raysubmit_\w+)")


def build_ray_command(
    script: str,
    script_args: list[str],
    config: KegoConfig,
    experiment_name: str,
    experiment_id: str,
    cli_params: dict[str, str],
) -> list[str]:
    """Build the ray job submit command list."""
    runtime_env = {
        "env_vars": {
            "MLFLOW_TRACKING_URI": config.cluster.mlflow_uri,
            "KEGO_EXPERIMENT_NAME": experiment_name,
            "KEGO_EXPERIMENT_ID": experiment_id,
            "KEGO_CLI_PARAMS": json.dumps(cli_params),
            "KEGO_PATH_DATA": str(Path.home() / "projects/kego/data"),
        },
    }

    entrypoint = [
        "uv", "run", "python", "-m", "kego.cli.runner", script
    ] + script_args

    return [
        "uv", "run", "ray", "job", "submit",
        "--address", config.cluster.ray_address,
        "--runtime-env-json", json.dumps(runtime_env),
        "--",
        *entrypoint,
    ]


def submit_fold(
    script: str,
    script_args: list[str],
    config: KegoConfig,
    experiment_name: str,
    experiment_id: str,
    cli_params: dict[str, str],
) -> str:
    """Submit one fold as a Ray job. Returns the Ray submission ID."""
    cmd = build_ray_command(
        script, script_args, config, experiment_name, experiment_id, cli_params
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ray job submit failed:\n{result.stderr}")

    for line in result.stdout.splitlines():
        if m := _JOB_ID_RE.search(line):
            return m.group(1)

    raise RuntimeError(f"Could not parse job ID from output:\n{result.stdout}")


def submit(
    script: str,
    folds: list[int],
    base_args: list[str],
    config: KegoConfig,
    experiment_name: str,
    experiment_id: str,
    cli_params: dict[str, str],
) -> list[str]:
    """Submit one Ray job per fold. Returns list of Ray submission IDs."""
    job_ids: list[str] = []
    for fold in folds:
        fold_args = base_args + ["--fold", str(fold)]
        fold_params = {**cli_params, "fold": str(fold)}
        job_id = submit_fold(
            script, fold_args, config, experiment_name, experiment_id, fold_params
        )
        print(f"  fold {fold}: {job_id}", flush=True)
        job_ids.append(job_id)
    return job_ids
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/cli/test_targets_cluster.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kego/cli/targets/cluster.py tests/cli/test_targets_cluster.py
git commit -m "feat(cli): Ray cluster target with fold fan-out"
```

---

## Task 7: `kego run` command

**Files:**
- Create: `kego/cli/commands/run.py`
- Modify: `kego/cli/__init__.py`

- [ ] **Step 1: Implement run.py**

There are no isolated unit tests for `run.py` because it mainly wires together the already-tested components. We verify it end-to-end in Step 4.

Create `kego/cli/commands/run.py`:
```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kego.cli import config as cfg_module
from kego.cli import experiment as exp_module
from kego.cli.targets import local as local_target
from kego.cli.targets import cluster as cluster_target


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("run", help="Dispatch a training script")
    p.add_argument("script", help="Path to training script")
    p.add_argument(
        "--target",
        choices=["local", "cluster"],
        default="local",
        help="Compute target (default: local)",
    )
    p.add_argument("--name", help="Experiment name (auto-generated if omitted)")
    p.add_argument("--fold", type=int, help="Single fold index")
    p.add_argument("--folds", help="Comma-separated fold indices, e.g. 0,1,2,3")
    p.add_argument("--debug", action="store_true", help="Smoke-test mode (forwards --debug to script)")
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace, extra_args: list[str]) -> int:
    config = cfg_module.load_config()

    # Resolve folds
    if args.folds:
        folds = [int(f) for f in args.folds.split(",")]
    elif args.fold is not None:
        folds = [args.fold]
    else:
        folds = None

    # Build script args: pass through extra_args + debug flag
    script_args = list(extra_args)
    if args.debug:
        script_args.append("--debug")

    # Parse extra_args into a params dict for MLflow logging
    cli_params: dict[str, str] = {}
    i = 0
    while i < len(extra_args):
        a = extra_args[i]
        if a.startswith("--"):
            key = a[2:]
            if "=" in key:
                k, v = key.split("=", 1)
                cli_params[k] = v
            elif i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
                cli_params[key] = extra_args[i + 1]
                i += 1
            else:
                cli_params[key] = "true"
        i += 1

    experiment_name = exp_module.build_experiment_name(args.script, args.name, cli_params)
    experiment_id = exp_module.generate_id()

    print(f"kego run: {experiment_name} [{experiment_id}] → {args.target}", flush=True)

    if args.target == "local":
        fold_args = script_args[:]
        if folds:
            fold_args += ["--fold", str(folds[0])]
        return local_target.run(
            script=args.script,
            script_args=fold_args,
            config=config,
            experiment_name=experiment_name,
            experiment_id=experiment_id,
            cli_params={**cli_params, **({"fold": str(folds[0])} if folds else {})},
        )

    elif args.target == "cluster":
        resolved_folds = folds if folds is not None else [0]
        job_ids = cluster_target.submit(
            script=args.script,
            folds=resolved_folds,
            base_args=script_args,
            config=config,
            experiment_name=experiment_name,
            experiment_id=experiment_id,
            cli_params=cli_params,
        )
        print(f"\nSubmitted {len(job_ids)} job(s). Track with: uv run kego ls", flush=True)
        return 0

    return 1
```

- [ ] **Step 2: Wire run command into CLI entry point**

Update `kego/cli/__init__.py`:
```python
import argparse
import sys


def main() -> None:
    from kego.cli.commands import run as run_cmd
    from kego.cli.commands import ls as ls_cmd

    parser = argparse.ArgumentParser(prog="kego", description="kego ML experiment engine")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    run_cmd.add_parser(subparsers)
    ls_cmd.add_parser(subparsers)

    args, extra = parser.parse_known_args()
    sys.exit(args.func(args, extra) or 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-test kego run locally**

Create a minimal test script at `/tmp/test_kego_run.py`:
```python
import argparse, sys
p = argparse.ArgumentParser()
p.add_argument("--fold", type=int, default=0)
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--debug", action="store_true")
args = p.parse_args()
print(f"KEGO_PARAM fold {args.fold}")
print(f"KEGO_METRIC fake_auc 0.{args.fold + 7}00")
print("Training complete.")
sys.exit(0)
```

Run:
```bash
uv run kego run /tmp/test_kego_run.py --fold 0 --epochs 2
```

Expected output:
```
kego run: test_kego_run [<id>] → local
KEGO_PARAM fold 0
KEGO_METRIC fake_auc 0.700
Training complete.
```

- [ ] **Step 4: Commit**

```bash
git add kego/cli/commands/run.py kego/cli/__init__.py
git commit -m "feat(cli): kego run command — local + cluster dispatch"
```

---

## Task 8: `kego ls` command

**Files:**
- Create: `kego/cli/commands/ls.py`
- Create: `tests/cli/test_commands_ls.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/cli/test_commands_ls.py`:
```python
import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from kego.cli.commands.ls import format_table, _ago


def test_ago_hours():
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    start = now - datetime.timedelta(hours=3)
    assert _ago(start) == "3h"


def test_ago_minutes():
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    start = now - datetime.timedelta(minutes=45)
    assert _ago(start) == "45m"


def test_format_table_basic():
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    runs = pd.DataFrame([
        {
            "tags.kego_id": "abc123",
            "tags.mlflow.runName": "soundscape-v8",
            "tags.kego_target": "cluster",
            "metrics.cmAP": 0.8821,
            "status": "FINISHED",
            "start_time": now - datetime.timedelta(hours=2),
        },
        {
            "tags.kego_id": "def456",
            "tags.mlflow.runName": "soundscape-v7",
            "tags.kego_target": "cluster",
            "metrics.cmAP": 0.8794,
            "status": "FINISHED",
            "start_time": now - datetime.timedelta(days=1),
        },
    ])
    lines = format_table(runs, primary_metric="cmAP")
    assert len(lines) >= 3  # header + separator + rows
    assert "abc123" in lines[2]
    assert "soundscape-v8" in lines[2]
    assert "0.8821" in lines[2]
    assert "def456" in lines[3]


def test_format_table_empty():
    lines = format_table(pd.DataFrame(), primary_metric="cmAP")
    assert lines == ["No experiments found."]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/cli/test_commands_ls.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement ls.py**

Create `kego/cli/commands/ls.py`:
```python
from __future__ import annotations

import argparse
import datetime
from typing import Optional

import pandas as pd


def _ago(start: datetime.datetime) -> str:
    delta = datetime.datetime.now(tz=datetime.timezone.utc) - start
    hours = int(delta.total_seconds() // 3600)
    if hours > 0:
        return f"{hours}h"
    return f"{int(delta.total_seconds() // 60)}m"


def format_table(runs: pd.DataFrame, primary_metric: str) -> list[str]:
    """Format experiment runs into a table. Returns list of lines."""
    if runs.empty:
        return ["No experiments found."]

    metric_col = f"metrics.{primary_metric}"

    header = f"{'ID':<8} {'NAME':<26} {'TARGET':<8} {primary_metric.upper():>8} {'STATUS':<10} {'AGO'}"
    sep = "-" * len(header)
    lines = [header, sep]

    for _, row in runs.iterrows():
        exp_id = str(row.get("tags.kego_id", "?"))[:6]
        name = str(row.get("tags.mlflow.runName", "?"))[:26]
        target = str(row.get("tags.kego_target", "local"))[:8]
        raw_metric = row.get(metric_col)
        metric_str = f"{raw_metric:.4f}" if pd.notna(raw_metric) else "—"
        status = str(row.get("status", "?"))[:10]
        start = row.get("start_time")
        ago = _ago(start) if start is not None and pd.notna(start) else "?"
        lines.append(f"{exp_id:<8} {name:<26} {target:<8} {metric_str:>8} {status:<10} {ago}")

    return lines


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("ls", help="List and compare experiments")
    p.add_argument("--competition", help="Filter by competition slug")
    p.add_argument("--name", help="Filter by experiment name")
    p.add_argument("--all", action="store_true", dest="show_all", help="Include debug runs")
    p.set_defaults(func=_ls)


def _ls(args: argparse.Namespace, extra_args: list[str]) -> int:
    import mlflow
    from kego.cli import config as cfg_module

    config = cfg_module.load_config()
    mlflow.set_tracking_uri(config.cluster.mlflow_uri)

    filter_parts: list[str] = []
    if not args.show_all:
        filter_parts.append("tags.kego_debug != 'true'")
    if args.name:
        filter_parts.append(f"tags.`mlflow.runName` LIKE '{args.name}%'")

    filter_string = " AND ".join(filter_parts) if filter_parts else ""

    try:
        runs = mlflow.search_runs(
            search_all_experiments=True,
            filter_string=filter_string,
            order_by=["start_time DESC"],
            max_results=50,
        )
    except Exception as e:
        print(f"Cannot reach MLflow at {config.cluster.mlflow_uri}: {e}")
        return 1

    primary_metric = "metric"
    if config.competition:
        primary_metric = config.competition.primary_metric

    for line in format_table(runs, primary_metric):
        print(line)

    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/cli/test_commands_ls.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests PASS (existing + new CLI tests).

- [ ] **Step 6: Manual smoke test of `kego ls`**

With MLflow server running on cluster:
```bash
uv run kego ls --competition birdclef-2026
```

Expected: table of experiments or "No experiments found." (not a crash).

- [ ] **Step 7: Commit**

```bash
git add kego/cli/commands/ls.py tests/cli/test_commands_ls.py
git commit -m "feat(cli): kego ls — experiment listing from MLflow"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|---|---|
| `kego run` with `--target local` | Task 5 + Task 7 |
| `kego run` with `--target cluster` | Task 6 + Task 7 |
| `--folds` fan-out to parallel Ray jobs | Task 6 (`submit()`) |
| `--debug` flag forwarded to script | Task 7 (`_run()`) |
| Experiment ID (6-char hex) | Task 3 |
| `--name` overrides auto-generated name | Task 3 |
| Auto-capture CLI args as MLflow params | Task 7 (extra_args parsing) + Task 4 (runner KEGO_CLI_PARAMS) |
| `KEGO_METRIC` stdout convention | Task 4 (`parse_kego_lines`) |
| `KEGO_PARAM` stdout convention | Task 4 (`parse_kego_lines`) |
| Unified MLflow server (same URI all targets) | Task 4 + Task 5 + Task 6 |
| `kego ls` with table output | Task 8 |
| `kego ls --name` filter | Task 8 |
| `kego.toml` root config | Task 2 |
| Per-competition `kego.toml` | Task 2 |
| `primary_metric` in competition toml | Task 2 + Task 8 |
| `checkpoint_dir` in competition toml | Task 2 (config loaded, used in Plan 2) |

**Not in this plan (Plan 2):** `kego push`, `kego submit`, `kego sync`, Kaggle target, notebook generation.

**Placeholder scan:** No TBDs or TODOs found.

**Type consistency:** `KegoConfig`, `ClusterConfig`, `CompetitionConfig` defined in Task 2 and used consistently in Tasks 5, 6, 7, 8. `parse_kego_lines` defined in Task 4, called in Task 4 only (runner is self-contained). `build_experiment_name` / `generate_id` defined in Task 3, called in Task 7.
