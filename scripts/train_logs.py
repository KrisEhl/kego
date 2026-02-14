#!/usr/bin/env python3
"""Parse Ray job logs and show training progress summary."""

import re
import sys
from datetime import datetime


def _parse_timestamps(text):
    """Extract first and last timestamps from Ray log lines."""
    # Ray logs timestamps like: 2026-02-14 19:03:03,927
    ts_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+"
    timestamps = re.findall(ts_pattern, text)
    if len(timestamps) >= 2:
        fmt = "%Y-%m-%d %H:%M:%S"
        first = datetime.strptime(timestamps[0], fmt)
        last = datetime.strptime(timestamps[-1], fmt)
        return first, last
    return None, None


def _fmt_duration(minutes):
    """Format minutes as human-readable duration."""
    if minutes < 1:
        return "<1min"
    if minutes < 60:
        return f"~{int(minutes)}min"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"~{hours}h{mins:02d}m"


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

    # Elapsed time
    t_start, t_last = _parse_timestamps(text)
    elapsed_min = (
        (t_last - t_start).total_seconds() / 60.0 if t_start and t_last else None
    )

    if completed:
        last = completed[-1]
        n_done = int(last[0])
        n_total = int(last[1])
        print(f"  Progress: {n_done}/{n_total} tasks completed")
        for c in completed:
            print(f"    [{c[0]}/{c[1]}] {c[2]} seed={c[3]} — AUC: {c[4]}")

        # Estimate remaining time from task completion rate
        if elapsed_min and n_done > 0 and n_done < n_total:
            rate = elapsed_min / n_done
            remaining = rate * (n_total - n_done)
            print(
                f"  Elapsed: {_fmt_duration(elapsed_min)}, "
                f"ETA: {_fmt_duration(remaining)} "
                f"({_fmt_duration(rate)}/task)"
            )
        elif elapsed_min and n_done == n_total:
            print(f"  Completed in {_fmt_duration(elapsed_min)}")
    elif elapsed_min:
        print(f"  Elapsed: {_fmt_duration(elapsed_min)}, no tasks completed yet")

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
        folds_match = re.search(r"(\d+) folds", text)
        folds_n = int(folds_match.group(1)) if folds_match else 5
        neural_running = [
            m
            for m, s in running
            if m in ("realmlp", "realmlp_large", "resnet", "ft_transformer")
        ]
        if neural_running:
            # Count folds per PID to get per-model progress
            fold_counts = {}
            for m in re.finditer(
                r"pid=(\d+).*?max_epochs=\d+.? reached", text, re.DOTALL
            ):
                pid = m.group(1)
                fold_counts[pid] = fold_counts.get(pid, 0) + 1

            # Map PIDs to model names via Starting messages
            pid_model = {}
            for m in re.finditer(r"pid=(\d+).*?\[(\w[\w_]*)\] Starting seed=", text):
                pid_model[m.group(1)] = m.group(2)

            shown = set()
            for pid, model_name in pid_model.items():
                if (
                    model_name in [m for m in neural_running]
                    and model_name not in shown
                ):
                    done = fold_counts.get(pid, 0)
                    print(f"  {model_name} fold progress: ~{done}/{folds_n}")
                    shown.add(model_name)

            # Show any neural models without PID mapping
            for model_name in neural_running:
                if model_name not in shown:
                    total_folds_done = sum(fold_counts.values()) if fold_counts else 0
                    print(f"  {model_name} fold progress: ~0/{folds_n}")

            # Estimate remaining from fold rate
            total_folds_done = sum(fold_counts.values()) if fold_counts else 0
            total_folds_needed = len(neural_running) * folds_n
            if elapsed_min and total_folds_done > 0:
                fold_rate = elapsed_min / total_folds_done
                remaining_folds = total_folds_needed - total_folds_done
                eta = fold_rate * remaining_folds
                print(
                    f"  Neural ETA: {_fmt_duration(eta)} "
                    f"({_fmt_duration(fold_rate)}/fold, "
                    f"{total_folds_done}/{total_folds_needed} folds done)"
                )

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
