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

    def make_submission(self, ids: np.ndarray, preds: np.ndarray) -> Path:
        # Locate sample submission files
        src_dir = Path("data/pokemon/pokemon-tcg-ai-battle/sample_submission")
        if not src_dir.exists():
            src_dir = Path("/home/kristian/projects/kego/data/pokemon/pokemon-tcg-ai-battle/sample_submission")

        comp_dir = Path("competitions/pokemon-tcg-ai-battle")
        if not comp_dir.exists():
            comp_dir = Path("/home/kristian/projects/kego/competitions/pokemon-tcg-ai-battle")

        # Determine which main.py and deck.csv to use
        main_py = comp_dir / "main.py"
        if not main_py.exists() or main_py.stat().st_size < 150:
            main_py = src_dir / "main.py"

        deck_csv = comp_dir / "deck.csv"
        if not deck_csv.exists():
            deck_csv = src_dir / "deck.csv"

        # Create target tar.gz inside the competition folder
        target_path = comp_dir / "submission.tar.gz"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with tarfile.open(target_path, "w:gz") as tar:
            tar.add(main_py, arcname="main.py")
            tar.add(deck_csv, arcname="deck.csv")
            cg_dir = src_dir / "cg"
            if cg_dir.exists():
                tar.add(cg_dir, arcname="cg")

        # Read contents for kernel notebook generation
        with open(main_py) as f:
            main_py_content = f.read()
        with open(deck_csv) as f:
            deck_csv_content = f.read()

        kernel_dir = comp_dir / "kernel"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        notebook_path = kernel_dir / "submission_notebook.py"

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

# Copy cg directory
cg_src = os.path.join(SAMPLE_DIR, "cg")
cg_dst = os.path.join(WORKING_DIR, "cg")
if os.path.exists(cg_src):
    if os.path.exists(cg_dst):
        shutil.rmtree(cg_dst)
    shutil.copytree(cg_src, cg_dst)

# Create submission.tar.gz
submission_path = os.path.join(WORKING_DIR, "submission.tar.gz")
with tarfile.open(submission_path, "w:gz") as tar:
    for item in ["main.py", "deck.csv", "cg"]:
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
