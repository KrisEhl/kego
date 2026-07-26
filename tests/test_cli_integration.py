import json
import subprocess


def test_kego_status_cleans_stale_runs(tmp_path):
    # Create the active runs directory
    active_runs_dir = tmp_path / ".kego" / "active_runs"
    active_runs_dir.mkdir(parents=True)

    # Write a stale run JSON file with a non-existent PID (e.g. 999999)
    run_file = active_runs_dir / "test_run_123.json"
    run_data = {
        "pid": 999999,
        "task": "pokemon-tcg-ai-battle",
        "config": "large256_actions256_abomasnow",
    }
    with open(run_file, "w") as f:
        json.dump(run_data, f)

    # Write a kego.toml in the directory
    kego_toml = tmp_path / "kego.toml"
    kego_toml.write_text('[competition]\nslug = "pokemon-tcg-ai-battle"\n')

    # Write a minimal task.py in the directory to satisfy dynamic task loading
    task_file = tmp_path / "task.py"
    task_file.write_text(
        "from kego.pipeline import register_task\n\n"
        '@register_task("pokemon-tcg-ai-battle")\n'
        "class DummySimTask:\n"
        '    name = "pokemon-tcg-ai-battle"\n'
        '    kaggle_slug = "pokemon-tcg-ai-battle"\n'
        "    is_simulation = True\n"
    )

    # Run the real kego status command in the tmp_path directory
    res = subprocess.run(
        ["uv", "run", "kego", "status"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )

    assert res.returncode == 0
    # The stale run file should be deleted (unlinked)
    assert not run_file.exists()


def test_kego_parser_help(tmp_path):
    # Verify that the CLI parser can render help output cleanly
    res = subprocess.run(
        ["uv", "run", "kego", "--help"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "usage: kego" in res.stdout
    assert "status" in res.stdout
    assert "cache" in res.stdout
