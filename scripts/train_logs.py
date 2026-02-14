#!/usr/bin/env python3
"""Parse Ray job logs and show training progress summary."""

import re
import sys


def main():
    job_id = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    lines = sys.stdin.readlines()
    text = "".join(lines)

    mode = re.search(r"Mode: (.+)", text)
    print(f"Job: {job_id}")
    if mode:
        print(f"  {mode.group(0)}")

    # Track task lifecycle
    started = set(re.findall(r"\[(\w+)\] Starting seed=(\d+)", text))
    finished = set(re.findall(r"\[(\w+)\] Finished seed=(\d+)", text))
    running = started - finished

    # Completed tasks with AUC
    completed = re.findall(
        r"\[(\d+)/(\d+)\].*?(\w[\w_]+) seed=(\d+).*?Holdout AUC: ([\d.]+)", text
    )
    if completed:
        last = completed[-1]
        print(f"  Progress: {last[0]}/{last[1]} tasks completed")
        for c in completed:
            print(f"    [{c[0]}/{c[1]}] {c[2]} seed={c[3]} — AUC: {c[4]}")

    # Currently running
    if running:
        print(f"  Running ({len(running)}):")
        for m, s in sorted(running):
            print(f"    {m} seed={s}")

    # Timeouts (each 600s = 10min between checks)
    timeouts = text.count("ray.wait timed out")
    if timeouts:
        print(f"  Waiting: ~{timeouts * 10}min ({timeouts} x 600s timeouts)")

    # Neural model fold progress (per running neural model)
    if running:
        # Extract folds_n from mode line
        folds_match = re.search(r"(\d+) folds", text)
        folds_n = int(folds_match.group(1)) if folds_match else 5
        # Count sub-model completions (Trainer.fit stopped) as fold completions
        folds_done = len(re.findall(r"max_epochs=\d+.? reached", text))
        neural_running = [
            m
            for m, s in running
            if m in ("realmlp", "realmlp_large", "resnet", "ft_transformer")
        ]
        if neural_running:
            for model_name in neural_running:
                total_folds = folds_n
                print(f"  {model_name} fold progress: ~{folds_done}/{total_folds}")

    # Ensemble results
    for line in lines:
        for pattern in [
            r"avg.*seeds",
            r"Simple Average",
            r"Ridge.*Holdout",
            r"Hill Climbing.*Holdout",
            r"Best method",
            r"Submission saved",
            r"Mean prediction",
        ]:
            if re.search(pattern, line):
                print(f"  {line.strip()}")
                break


if __name__ == "__main__":
    main()
