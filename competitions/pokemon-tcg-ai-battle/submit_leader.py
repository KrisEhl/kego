from __future__ import annotations

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


def select_elo_leader(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No models are registered in the MLflow model registry.")
    return rows[0]


def set_competition_value(content: str, key: str, value: str) -> str:
    assignment = f'{key} = "{value}"'
    section = re.search(r"(?ms)^\[competition\][^\n]*\n(?P<body>.*?)(?=^\[|\Z)", content)
    if not section:
        raise ValueError("kego.toml has no [competition] section")
    body = section.group("body")
    pattern = rf"(?m)^{re.escape(key)}\s*=\s*.*$"
    updated = re.sub(pattern, assignment, body, count=1) if re.search(pattern, body) else f"{assignment}\n{body}"
    return f"{content[: section.start('body')]}{updated}{content[section.end('body') :]}"


def validate_variant_metadata(
    version: str, deck_name: str, model_args_raw: str | None, variant_name: str, variant_path: Path
) -> tuple[int, ...]:
    import ast

    import tomllib

    parsed = ast.literal_eval(model_args_raw) if model_args_raw else None
    model_args = tuple(parsed) if isinstance(parsed, (list, tuple)) else None
    with open(variant_path, "rb") as f:
        variant_config = tomllib.load(f)
    variant_model_args = tuple(variant_config.get("model_args", ()))
    expected_deck = f"decks/{deck_name}.csv"
    if (variant_deck := variant_config.get("deck_file")) != expected_deck:
        raise ValueError(
            f"Registry version {version} deck {expected_deck!r} does not match "
            f"variant {variant_name!r} deck {variant_deck!r}."
        )
    if model_args is None or model_args != variant_model_args:
        raise ValueError(
            f"Registry version {version} model_args {model_args!r} do not match "
            f"variant {variant_name!r} model_args {variant_model_args!r}."
        )
    return model_args


def resolve_registry_model(client, uri: str, requested_version: str | None):
    if requested_version is None:
        selected = select_elo_leader(leaderboard(uri, "pokemon-tcg-ai-battle", sort_by="elo"))
        return selected, client.get_model_version("pokemon-tcg-ai-battle", selected["version"])
    version = client.get_model_version("pokemon-tcg-ai-battle", str(requested_version))
    return {"version": str(version.version), **dict(version.tags or {})}, version


def _get_kaggle_cmd() -> list[str]:
    if shutil.which("kaggle"):
        return ["kaggle"]
    py_bin = Path(sys.executable).parent
    kaggle_bin = py_bin / "kaggle"
    if kaggle_bin.exists():
        return [str(kaggle_bin)]
    return ["kaggle"]


def prepare_submission(requested_version: str | None = None) -> dict[str, object]:
    # 1. Parse kego.toml for configuration
    kego_toml_path = comp_dir / "kego.toml"
    if not kego_toml_path.exists():
        print(f"Error: kego.toml not found at {kego_toml_path}")
        sys.exit(1)

    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore

        with open(kego_toml_path, "rb") as f:
            cfg = tomllib.load(f)
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

    # 3. Connect to MLflow and resolve the requested registry version.
    uri = default_tracking_uri()
    print(f"Connecting to MLflow registry at {uri}...")
    client = MlflowClient(tracking_uri=uri)
    try:
        selected_model, v_obj = resolve_registry_model(client, uri, requested_version)
    except Exception as e:
        target = f"version {requested_version}" if requested_version is not None else "Elo leader"
        print(f"Error: Could not resolve registry {target} ({e}).")
        sys.exit(1)

    version = selected_model["version"]
    elo = selected_model.get("elo", "N/A")
    git_sha = selected_model.get("git_sha", "unknown")
    machine = selected_model.get("machine", "unknown")
    model_args_raw = selected_model.get("model_args")
    deck_name = selected_model.get("deck")
    variant_name = selected_model.get("variant")
    if not deck_name:
        print(
            f"Error: Selected registry version {version} is missing required 'deck' tag. "
            "Refusing to guess a deck for submission."
        )
        print("Set the tag explicitly, e.g.:")
        print(
            '  uv run python -c "from mlflow.tracking import MlflowClient; '
            "from kego.tracking import default_tracking_uri; "
            "MlflowClient(tracking_uri=default_tracking_uri()).set_model_version_tag("
            f"'pokemon-tcg-ai-battle', '{version}', 'deck', '<deck-name>')\""
        )
        sys.exit(1)
    if not variant_name:
        print(
            f"Error: Registry version {version} is missing required 'variant' metadata. "
            "Refusing to submit with stale local architecture settings."
        )
        sys.exit(1)
    deck_path = comp_dir / "decks" / f"{deck_name}.csv"
    variant_path = comp_dir / "configs" / "variants" / f"{variant_name}.toml"
    if not deck_path.exists():
        print(f"Error: Registry version {version} requires missing deck file {deck_path}.")
        sys.exit(1)
    if not variant_path.exists():
        print(f"Error: Registry version {version} requires missing variant file {variant_path}.")
        sys.exit(1)
    try:
        model_args = validate_variant_metadata(
            str(version), str(deck_name), model_args_raw, str(variant_name), variant_path
        )
    except Exception as e:
        print(f"Error: Could not validate registry version {version} metadata ({e}).")
        sys.exit(1)

    print("\nSelected model:")
    print(f"  Version: {version}")
    print(f"  Elo:     {elo}")
    print(f"  Machine: {machine}")
    print(f"  Git SHA: {git_sha}")
    print(f"  Deck:    {deck_name}")
    print(f"  Variant: {variant_name}")
    if model_args:
        print(f"  Model Args: {model_args}")

    # Check git alignment
    try:
        curr_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        if not git_sha.startswith(curr_sha) and not curr_sha.startswith(git_sha):
            print(
                f"\n[WARNING] Current git commit ({curr_sha}) does not match the selected model's commit ({git_sha})."
            )
            print("Consider checking out the matching commit if there are breaking code changes.")
    except Exception:
        pass

    # 4. Download the selected checkpoint.
    local_dir = comp_dir / "outputs" / "cached_registry" / f"run_{v_obj.run_id}"
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
    content = set_competition_value(content, "agent_file", "agents/mcts")
    content = set_competition_value(content, "deck_file", f"decks/{deck_name}.csv")
    content = set_competition_value(content, "variant", variant_name)
    with open(kego_toml_path, "w") as f:
        f.write(content)
    print(f"Updated kego.toml to use agents/mcts, decks/{deck_name}.csv, and variant {variant_name}")

    return {"version": str(version), "elo": elo, "message": f"Registry v{version} (Elo {elo})"}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Submit a Pokémon registry model to Kaggle.")
    parser.add_argument("version", nargs="?", help="registry version; defaults to the current Elo leader")
    args = parser.parse_args()
    cmd = ["uv", "run", "kego", "submit"]
    if args.version:
        cmd.append(args.version)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
