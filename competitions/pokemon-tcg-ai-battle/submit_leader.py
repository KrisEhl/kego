import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Resolve competition directory and add repository root to path
comp_dir = Path(__file__).parent.resolve()
repo_root = comp_dir.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(comp_dir))

from mlflow.tracking import MlflowClient
from run_league import download_checkpoint

from kego.tracking import default_tracking_uri, leaderboard


def _get_kaggle_cmd() -> list[str]:
    if shutil.which("kaggle"):
        return ["kaggle"]
    py_bin = Path(sys.executable).parent
    kaggle_bin = py_bin / "kaggle"
    if kaggle_bin.exists():
        return [str(kaggle_bin)]
    return ["kaggle"]


def main():
    # 1. Parse kego.toml for configuration
    kego_toml_path = comp_dir / "kego.toml"
    if not kego_toml_path.exists():
        print(f"Error: kego.toml not found at {kego_toml_path}")
        sys.exit(1)

    try:
        import tomli

        with open(kego_toml_path, "rb") as f:
            cfg = tomli.load(f)
        comp_cfg = cfg.get("competition", {})
        kaggle_user = comp_cfg.get("kaggle_user")
    except Exception as e:
        print(f"Error reading kego.toml: {e}")
        sys.exit(1)

    if not kaggle_user:
        print("Error: 'kaggle_user' must be configured in the [competition] section of kego.toml")
        sys.exit(1)

    # 2. Check Kaggle API credentials
    kaggle_cmd = _get_kaggle_cmd()
    try:
        subprocess.run([*kaggle_cmd, "competitions", "list"], capture_output=True, check=True)
    except Exception:
        print("\nError: Kaggle API credentials not found or invalid.")
        print("Please ensure your ~/.kaggle/kaggle.json credentials file is correctly set up.")
        sys.exit(1)

    # 3. Connect to MLflow and find the leader
    uri = default_tracking_uri()
    print(f"Connecting to MLflow registry at {uri}...")
    try:
        rows = leaderboard(uri, "pokemon-tcg-ai-battle", sort_by="elo")
    except Exception as e:
        print(f"Error querying registry leaderboard: {e}")
        sys.exit(1)

    if not rows:
        print("Error: No models registered in the MLflow model registry.")
        sys.exit(1)

    leader = rows[0]
    version = leader["version"]
    elo = leader.get("elo", "N/A")
    git_sha = leader.get("git_sha", "unknown")
    machine = leader.get("machine", "unknown")

    print("\nLeader model found:")
    print(f"  Version: {version}")
    print(f"  Elo:     {elo}")
    print(f"  Machine: {machine}")
    print(f"  Git SHA: {git_sha}")

    # Check git alignment
    try:
        curr_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        if not git_sha.startswith(curr_sha) and not curr_sha.startswith(git_sha):
            print(f"\n[WARNING] Current git commit ({curr_sha}) does not match the leader's commit ({git_sha}).")
            print("Consider checking out the matching commit if there are breaking code changes.")
    except Exception:
        pass

    # 4. Download leader checkpoint
    client = MlflowClient(tracking_uri=uri)
    try:
        versions = client.search_model_versions("name='pokemon-tcg-ai-battle'")
    except Exception as e:
        print(f"Error searching model versions: {e}")
        sys.exit(1)

    v_obj = next((v for v in versions if str(v.version) == str(version)), None)
    if v_obj is None:
        print(f"Error: Could not find registry model version {version} object.")
        sys.exit(1)

    local_dir = comp_dir / "outputs" / "latest"
    print(f"\nDownloading checkpoint for Version {version} to {local_dir}...")
    try:
        checkpoint_path = download_checkpoint(client, v_obj, str(local_dir), debug=True)
    except Exception as e:
        print(f"Failed to download checkpoint: {e}")
        sys.exit(1)
    print(f"Downloaded checkpoint to: {checkpoint_path}")

    # Copy checkpoint to outputs/mcts.pth for submission pipeline
    dest_path = comp_dir / "outputs" / "mcts.pth"
    shutil.copy(checkpoint_path, dest_path)
    print(f"Copied checkpoint to: {dest_path}")

    # 5. Create or Update Kaggle Dataset for model weights
    dataset_dir = comp_dir / "outputs" / "kaggle_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(dest_path, dataset_dir / "mcts.pth")

    dataset_slug = "kego-tcg-weights"
    dataset_id = f"{kaggle_user}/{dataset_slug}"

    metadata = {
        "title": "Kego TCG Weights",
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(dataset_dir / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Check if dataset exists on Kaggle
    print(f"\nChecking Kaggle dataset status for {dataset_id}...")
    cmd_check = [*kaggle_cmd, "datasets", "status", dataset_id]
    res_check = subprocess.run(cmd_check, capture_output=True)

    if res_check.returncode == 0:
        print("Dataset exists on Kaggle. Pushing a new version...")
        subprocess.run(
            [*kaggle_cmd, "datasets", "version", "-p", str(dataset_dir), "-m", f"Registry v{version} (Elo {elo})"],
            check=True,
        )
    else:
        print("Dataset does not exist. Creating new dataset on Kaggle...")
        subprocess.run([*kaggle_cmd, "datasets", "create", "-p", str(dataset_dir), "-r", "zip"], check=True)

    # 6. Attach dataset to kernel-metadata.json
    kernel_metadata_path = comp_dir / "kernel" / "kernel-metadata.json"
    if kernel_metadata_path.exists():
        with open(kernel_metadata_path) as f:
            k_meta = json.load(f)
        k_meta.setdefault("dataset_sources", [])
        if dataset_id not in k_meta["dataset_sources"]:
            k_meta["dataset_sources"].append(dataset_id)
            with open(kernel_metadata_path, "w") as f:
                json.dump(k_meta, f, indent=2)
            print(f"Attached dataset {dataset_id} to kernel-metadata.json")

    # 7. Update kego.toml to use MCTS agent
    with open(kego_toml_path) as f:
        content = f.read()
    content = re.sub(r'agent_file\s*=\s*".*"', 'agent_file = "agents/mcts.py"', content)
    content = re.sub(r'deck_file\s*=\s*".*"', 'deck_file = "decks/abomasnow.csv"', content)
    with open(kego_toml_path, "w") as f:
        f.write(content)
    print("Updated kego.toml to use agents/mcts.py")

    # 8. Submit to Kaggle
    print("\nRunning submission command...")
    cmd_submit = ["uv", "run", "kego", "submit", "--message", f"Registry v{version} (Elo {elo})"]
    subprocess.run(cmd_submit, check=True)


if __name__ == "__main__":
    main()
