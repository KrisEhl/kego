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


def main():
    uri = default_tracking_uri()
    print(f"Connecting to MLflow registry at {uri}...")

    # Query leaderboard to find the leader (highest Elo)
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

    # Get model version details to download the checkpoint
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

    # Download checkpoint to outputs/latest/
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

    # Update kego.toml to use the MCTS agent and deck
    kego_toml_path = comp_dir / "kego.toml"
    with open(kego_toml_path) as f:
        content = f.read()

    # Replace agent_file and deck_file lines
    content = re.sub(r'agent_file\s*=\s*".*"', 'agent_file = "agents/mcts.py"', content)
    content = re.sub(r'deck_file\s*=\s*".*"', 'deck_file = "decks/abomasnow.csv"', content)

    with open(kego_toml_path, "w") as f:
        f.write(content)
    print("Updated kego.toml to use agents/mcts.py")

    # Now run kego submit
    print("\nRunning submission command...")
    cmd = ["uv", "run", "kego", "submit", "--message", f"Registry v{version} (Elo {elo})"]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
