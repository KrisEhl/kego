import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from kego.pipeline.task import RawData, register_task


@register_task("pokemon-tcg-ai-battle")
class PokemonTCGAIBattleTask:
    name = "pokemon-tcg-ai-battle"
    kaggle_slug = "pokemon-tcg-ai-battle"
    target = "score"
    id_col = "id"
    metric_direction = "maximize"
    is_simulation = True

    def load_raw(self) -> RawData:
        # Return mock raw data
        return RawData(
            train=pd.DataFrame(),
            test=pd.DataFrame(),
            sample_submission=pd.DataFrame(),
        )

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return 0.0

    def train(self, config, epochs: int | None = None, output_path: str | None = None, **kwargs) -> None:
        import importlib.util
        from pathlib import Path

        comp_dir = Path(__file__).resolve().parent
        train_file = comp_dir / "train_agent.py"

        spec = importlib.util.spec_from_file_location("train_agent", str(train_file))
        train_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_module)

        # Load custom config if present in kego.toml
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore

        eval_games = 5
        self_play_games = 10
        config_path = comp_dir / "kego.toml"
        if config_path.exists():
            try:
                with open(config_path, "rb") as f:
                    toml_data = tomllib.load(f)
                train_cfg = toml_data.get("train", {})
                eval_games = train_cfg.get("eval_games", eval_games)
                self_play_games = train_cfg.get("self_play_games", self_play_games)
            except Exception:
                pass

        iterations = epochs if epochs is not None else 3
        out_path = output_path if output_path is not None else "outputs/mcts_model.pth"

        train_module.run_training_loop(
            iterations=iterations, eval_games=eval_games, self_play_games=self_play_games, output_path=out_path
        )

    def make_submission(self, ids: np.ndarray, preds: np.ndarray) -> Path:
        # Locate sample submission files
        repo_root = Path(__file__).resolve().parents[2]
        src_dir = repo_root / "data/pokemon/pokemon-tcg-ai-battle/sample_submission/sample_submission"
        if not src_dir.exists():
            src_dir = repo_root / "data/pokemon/pokemon-tcg-ai-battle/sample_submission"

        comp_dir = Path("competitions/pokemon-tcg-ai-battle")
        if not comp_dir.exists():
            comp_dir = repo_root / "competitions/pokemon-tcg-ai-battle"

        # Load settings from kego.toml
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore

        config_path = comp_dir / "kego.toml"
        if not config_path.exists():
            raise FileNotFoundError(f"kego.toml not found at {config_path}")

        try:
            with open(config_path, "rb") as f:
                cfg = tomllib.load(f)
            comp_cfg = cfg.get("competition", {})
            agent_file = comp_cfg.get("agent_file")
            deck_file = comp_cfg.get("deck_file")
        except Exception as e:
            raise ValueError(f"Failed to read kego.toml: {e}")

        if not agent_file or not deck_file:
            raise ValueError("agent_file and deck_file must be explicitly set in [competition] block of kego.toml")

        # Determine paths
        agent_path = comp_dir / agent_file
        if not agent_path.exists():
            raise FileNotFoundError(f"Configured agent_file not found at {agent_path.resolve()}")

        deck_path = comp_dir / deck_file
        if not deck_path.exists():
            raise FileNotFoundError(f"Configured deck_file not found at {deck_path.resolve()}")

        # Create target tar.gz inside the competition folder
        target_path = comp_dir / "submission.tar.gz"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect auxiliary helper files in the competition folder that might be imported by the agent
        helpers = {}
        base_path = comp_dir / "agents/base.py"
        if base_path.exists():
            try:
                with open(base_path) as f:
                    helpers["base_agent.py"] = f.read()
            except Exception:
                pass

        # Create local tar.gz
        with tarfile.open(target_path, "w:gz") as tar:
            tar.add(agent_path, arcname="main.py")
            tar.add(deck_path, arcname="deck.csv")
            if base_path.exists():
                tar.add(base_path, arcname="base_agent.py")
            cg_dir = src_dir / "cg"
            if cg_dir.exists():
                tar.add(cg_dir, arcname="cg")

        # Read contents for kernel notebook generation
        with open(agent_path) as f:
            main_py_content = f.read()
        with open(deck_path) as f:
            deck_csv_content = f.read()

        kernel_dir = comp_dir / "kernel"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        notebook_path = kernel_dir / "submission_notebook.py"

        # Helper codes generation for the notebook
        helpers_code = ""
        for h_name, h_content in helpers.items():
            safe_var_name = h_name.replace(".", "_").replace("-", "_")
            helpers_code += f"""
# Write helper: {h_name}
{safe_var_name}_content = {repr(h_content)}
with open(os.path.join(WORKING_DIR, "{h_name}"), "w") as f:
    f.write({safe_var_name}_content)
"""

        notebook_content = f"""\"\"\"PTCG AI Battle — Submission Kernel

Dynamically generated to package the agent.
\"\"\"
import os
import shutil
import tarfile

INPUT_DIR = "/kaggle/input/competitions/pokemon-tcg-ai-battle"
SAMPLE_DIR = os.path.join(INPUT_DIR, "sample_submission")
WORKING_DIR = "/kaggle/working"

# Write custom main.py
main_py_content = {repr(main_py_content)}
with open(os.path.join(WORKING_DIR, "main.py"), "w") as f:
    f.write(main_py_content)

# Write custom deck.csv
deck_csv_content = {repr(deck_csv_content)}
with open(os.path.join(WORKING_DIR, "deck.csv"), "w") as f:
    f.write(deck_csv_content)
{helpers_code}
# Copy cg directory
import glob
def find_cg_dir():
    candidates = [
        "/kaggle/input/pokemon-tcg-ai-battle/sample_submission/cg",
        "/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission/cg",
        "/kaggle/input/**/sample_submission/cg",
        "/kaggle/input/**/cg",
    ]
    for pattern in candidates:
        for path in glob.glob(pattern, recursive=True):
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "api.py")):
                return path
    raise FileNotFoundError("Could not find the official cg SDK directory.")

try:
    cg_src = find_cg_dir()
    cg_dst = os.path.join(WORKING_DIR, "cg")
    if os.path.exists(cg_dst):
        shutil.rmtree(cg_dst)
    shutil.copytree(cg_src, cg_dst)
except Exception as e:
    print(f"Error copying cg: {{e}}")

# Create submission.tar.gz
submission_path = os.path.join(WORKING_DIR, "submission.tar.gz")
with tarfile.open(submission_path, "w:gz") as tar:
    for item in ["main.py", "deck.csv", "cg"] + {list(helpers.keys())}:
        full_path = os.path.join(WORKING_DIR, item)
        if os.path.exists(full_path):
            tar.add(full_path, arcname=item)

print(f"Created {{submission_path}}")
print(f"Size: {{os.path.getsize(submission_path)}} bytes")

# Verify contents
with tarfile.open(submission_path, "r:gz") as tar:
    print("Contents:")
    for member in tar.getmembers():
        print(f"  {{member.name}} ({{member.size}} bytes)")
"""
        with open(notebook_path, "w") as f:
            f.write(notebook_content)

        return target_path.resolve()
