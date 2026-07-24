import io
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
        import sys
        from pathlib import Path

        trainable_agents = ("mcts",)
        agent = kwargs.get("agent")
        if agent not in trainable_agents:
            raise ValueError(
                f"Unknown trainable agent {agent!r}. Specify one via --agent <NAME>; "
                f"trainable agents: {sorted(trainable_agents)} "
                f"(rule agents under agents/ are heuristics and cannot be trained)."
            )

        comp_dir = Path(__file__).resolve().parent
        train_file = comp_dir / "train_agent.py"

        spec = importlib.util.spec_from_file_location("train_agent", str(train_file))
        train_module = importlib.util.module_from_spec(spec)
        # Register before exec so multiprocessing (spawn) can pickle the module's
        # worker functions by reference (pickle resolves them via sys.modules).
        sys.modules["train_agent"] = train_module
        spec.loader.exec_module(train_module)

        # Load custom config if present in kego.toml
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore

        # 1. Base defaults
        eval_games = 5
        self_play_games = 10
        num_workers = None
        eval_every = 1
        search_count = 10
        batched = False
        eval_opponents = None
        selfplay_opponents = None
        replay_buffer_size = 100000
        train_steps = 100
        init_checkpoint = None
        model_args = None
        deck_file = "decks/abomasnow.csv"

        # Load kego.toml [train] defaults
        kego_path = comp_dir / "kego.toml"
        if kego_path.exists():
            try:
                with open(kego_path, "rb") as f:
                    toml_data = tomllib.load(f)
                train_cfg = toml_data.get("train", {})
                eval_games = train_cfg.get("eval_games", eval_games)
                self_play_games = train_cfg.get("self_play_games", self_play_games)
                num_workers = train_cfg.get("num_workers", num_workers)
                eval_every = train_cfg.get("eval_every", eval_every)
                search_count = train_cfg.get("search_count", search_count)
                batched = train_cfg.get("batched", batched)
                eval_opponents = train_cfg.get("eval_opponents", eval_opponents)
                selfplay_opponents = train_cfg.get("selfplay_opponents", selfplay_opponents)
                replay_buffer_size = train_cfg.get("replay_buffer_size", replay_buffer_size)
                train_steps = train_cfg.get("train_steps", train_steps)
                init_checkpoint = train_cfg.get("init_checkpoint", init_checkpoint)
                comp_cfg = toml_data.get("competition", {})
                deck_file = comp_cfg.get("deck_file", deck_file)
            except Exception:
                pass

        # 2. Variant overrides
        variant = kwargs.get("variant")
        if not variant:
            raise ValueError("Training requires specifying a model variant via --variant <NAME>.")

        if variant.endswith(".toml"):
            variant = variant[:-5]

        variant_path = comp_dir / "configs" / "variants" / f"{variant}.toml"
        if not variant_path.exists():
            raise FileNotFoundError(f"Variant config not found at {variant_path}")

        try:
            with open(variant_path, "rb") as f:
                var_cfg = tomllib.load(f)
            deck_file = var_cfg.get("deck_file", deck_file)
            model_args = var_cfg.get("model_args", model_args)
            search_count = var_cfg.get("search_count", search_count)
            train_steps = var_cfg.get("train_steps", train_steps)
            self_play_games = var_cfg.get("self_play_games", self_play_games)
            eval_games = var_cfg.get("eval_games", eval_games)
            eval_every = var_cfg.get("eval_every", eval_every)
            eval_opponents = var_cfg.get("eval_opponents", eval_opponents)
            selfplay_opponents = var_cfg.get("selfplay_opponents", selfplay_opponents)
            replay_buffer_size = var_cfg.get("replay_buffer_size", replay_buffer_size)
            batched = var_cfg.get("batched", batched)
            features = var_cfg.get("features", {})
        except Exception as e:
            raise ValueError(f"Failed to parse variant configuration: {e}")

        # 3. Runtime overrides
        iterations = epochs if epochs is not None else 3
        out_path = output_path if output_path is not None else "outputs/mcts_model.pth"
        init_checkpoint = kwargs.get("init_checkpoint", init_checkpoint)
        num_workers = kwargs.get("num_workers", num_workers)

        if isinstance(model_args, list):
            model_args = tuple(model_args)

        from kego.training_resume import training_fingerprint

        fingerprint_config = {
            "agent": agent,
            "variant": variant,
            "deck_file": deck_file,
            "model_args": model_args,
            "search_count": search_count,
            "train_steps": train_steps,
            "self_play_games": self_play_games,
            "eval_games": eval_games,
            "eval_every": eval_every,
            "eval_opponents": eval_opponents,
            "selfplay_opponents": selfplay_opponents,
            "replay_buffer_size": replay_buffer_size,
            "batched": batched,
            "num_workers": num_workers,
            "features": features,
        }
        source_paths = [train_file, Path(__file__), variant_path, comp_dir / deck_file]
        source_paths += sorted((comp_dir / "agents" / "mcts").rglob("*.py"))
        source_paths += sorted((comp_dir / "agents").glob("*.py"))
        cg_dir = Path(train_module.cg_parent) / "cg"
        if cg_dir.exists():
            source_paths += sorted(cg_dir.rglob("*.py"))
        config_fingerprint = training_fingerprint(fingerprint_config, source_paths)

        train_module.run_training_loop(
            iterations=iterations,
            eval_games=eval_games,
            self_play_games=self_play_games,
            output_path=out_path,
            num_workers=num_workers,
            eval_every=eval_every,
            search_count=search_count,
            batched=batched,
            eval_opponents=eval_opponents,
            selfplay_opponents=selfplay_opponents,
            replay_buffer_size=replay_buffer_size,
            train_steps=train_steps,
            deck_file=deck_file,
            init_checkpoint=init_checkpoint,
            model_args=model_args,
            variant=variant,
            config_fingerprint=config_fingerprint,
            features=features,
        )

    def make_submission(self, ids: np.ndarray, preds: np.ndarray) -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        src_dir = repo_root / "data/pokemon/pokemon-tcg-ai-battle/sample_submission/sample_submission"
        if not src_dir.exists():
            src_dir = repo_root / "data/pokemon/pokemon-tcg-ai-battle/sample_submission"

        comp_dir = Path("competitions/pokemon-tcg-ai-battle")
        if not comp_dir.exists():
            comp_dir = repo_root / "competitions/pokemon-tcg-ai-battle"

        cg_candidates = [
            src_dir / "cg",
            comp_dir / "cg",
            repo_root / "cg",
            Path("cg"),
            repo_root / "data/pokemon/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg",
            repo_root / "data/pokemon/pokemon-tcg-ai-battle/sample_submission/cg",
        ]
        cg_dir = next((p for p in cg_candidates if p.exists()), None)

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
            variant_name = comp_cfg.get("variant")
            if variant_name:
                var_toml_path = comp_dir / "configs" / "variants" / f"{variant_name}.toml"
                if var_toml_path.exists():
                    with open(var_toml_path, "rb") as vf:
                        var_cfg = tomllib.load(vf)
                    if "deck_file" in var_cfg:
                        deck_file = var_cfg["deck_file"]
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

        # The agent can be a single file (rule agents) or a package directory
        # (e.g. the MCTS agent). Directory agents are packaged recursively under
        # their own basename and fronted by a thin main.py shim that re-exports
        # `agent` from the package, so Kaggle's flat /kaggle_simulations/agent/
        # layout still finds a callable main.py:agent(obs_dict) entry point.
        is_package_agent = agent_path.is_dir()
        package_name = agent_path.name if is_package_agent else None

        if is_package_agent:
            main_py_content = (
                f'"""Kaggle submission shim for the {package_name} agent package."""\n'
                "\n"
                f"from {package_name} import agent  # noqa: F401\n"
            )
        else:
            with open(agent_path) as f:
                main_py_content = f.read()

        with open(deck_path) as f:
            deck_csv_content = f.read()

        # Collect auxiliary helper files in the competition folder that might be imported by the agent
        helpers = {}
        base_path = comp_dir / "agents/base.py"
        if base_path.exists():
            try:
                with open(base_path) as f:
                    helpers["base_agent.py"] = f.read()
            except Exception:
                pass

        if is_package_agent:
            for file_path in sorted(agent_path.rglob("*")):
                if file_path.is_dir():
                    continue
                rel_parts = file_path.relative_to(agent_path).parts
                if "__pycache__" in rel_parts or file_path.suffix in (".pyc", ".pyo"):
                    continue
                arcname = "/".join((package_name, *rel_parts))
                try:
                    with open(file_path) as f:
                        helpers[arcname] = f.read()
                except Exception:
                    pass

        def submission_tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
            parts = Path(info.name).parts
            if "__pycache__" in parts or info.name.endswith((".pyc", ".pyo")):
                return None
            return info

        # Create local tar.gz
        with tarfile.open(target_path, "w:gz") as tar:
            if is_package_agent:
                main_py_bytes = main_py_content.encode()
                main_py_info = tarfile.TarInfo(name="main.py")
                main_py_info.size = len(main_py_bytes)
                tar.addfile(main_py_info, io.BytesIO(main_py_bytes))
                tar.add(agent_path, arcname=package_name, filter=submission_tar_filter)
            else:
                tar.add(agent_path, arcname="main.py")
            tar.add(deck_path, arcname="deck.csv")
            if base_path.exists():
                tar.add(base_path, arcname="base_agent.py")
            if cg_dir and cg_dir.exists():
                tar.add(cg_dir, arcname="cg", filter=submission_tar_filter)

            # Package model weights if the agent is MCTS
            if "mcts" in str(agent_path):
                for pth_name in ["mcts.pth", "mcts_model.pth"]:
                    pth_path = comp_dir / "outputs" / pth_name
                    if not pth_path.exists():
                        pth_path = comp_dir / pth_name
                    if pth_path.exists():
                        tar.add(pth_path, arcname="mcts.pth")
                        break
                if variant_name:
                    var_toml_path = comp_dir / "configs" / "variants" / f"{variant_name}.toml"
                    if var_toml_path.exists():
                        tar.add(var_toml_path, arcname="variant.toml")

        var_content = ""
        if variant_name:
            var_toml_path = comp_dir / "configs" / "variants" / f"{variant_name}.toml"
            if var_toml_path.exists():
                with open(var_toml_path) as vf:
                    var_content = vf.read()

        kernel_dir = comp_dir / "kernel"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        notebook_path = kernel_dir / "submission_notebook.py"

        # Helper codes generation for the notebook
        helpers_code = ""
        for h_name, h_content in helpers.items():
            safe_var_name = h_name.replace(".", "_").replace("-", "_").replace("/", "_")
            dest_expr = f'os.path.join(WORKING_DIR, "{h_name}")'
            makedirs_code = f"os.makedirs(os.path.dirname({dest_expr}), exist_ok=True)\n" if "/" in h_name else ""
            helpers_code += f"""
# Write helper: {h_name}
{safe_var_name}_content = {repr(h_content)}
{makedirs_code}with open({dest_expr}, "w") as f:
    f.write({safe_var_name}_content)
"""

        # Write variant.toml if present
        write_var_toml_code = ""
        if var_content:
            write_var_toml_code = f"""
# Write custom variant.toml
variant_content = {repr(var_content)}
with open(os.path.join(WORKING_DIR, "variant.toml"), "w") as f:
    f.write(variant_content)
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

{write_var_toml_code}
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
    shutil.copytree(cg_src, cg_dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
except Exception as e:
    print(f"Error copying cg: {{e}}")

# Create submission.tar.gz
# If there is a model weights file in the competition input or attached weight
# dataset, copy it into the submission archive.
for root, dirs, files in os.walk("/kaggle/input"):
    for file in files:
        if file in ("mcts.pth", "mcts_model.pth"):
            shutil.copy(os.path.join(root, file), os.path.join(WORKING_DIR, "mcts.pth"))
            break

submission_path = os.path.join(WORKING_DIR, "submission.tar.gz")
def submission_tar_filter(info):
    parts = info.name.split("/")
    if "__pycache__" in parts or info.name.endswith((".pyc", ".pyo")):
        return None
    return info

with tarfile.open(submission_path, "w:gz") as tar:
    items = ["main.py", "deck.csv", "cg"] + {list(helpers.keys())}
    if os.path.exists(os.path.join(WORKING_DIR, "mcts.pth")):
        items.append("mcts.pth")
    if os.path.exists(os.path.join(WORKING_DIR, "variant.toml")):
        items.append("variant.toml")
    for item in items:
        full_path = os.path.join(WORKING_DIR, item)
        if os.path.exists(full_path):
            tar.add(full_path, arcname=item, filter=submission_tar_filter)

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
