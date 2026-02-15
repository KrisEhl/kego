#!/usr/bin/env python3
"""Parse Ray job logs and show training progress summary."""

import re
import sys
from datetime import datetime


def _parse_start_time(text):
    """Extract job start timestamp from Ray log lines.

    Skips CLI timestamps (cli.py) and finds the first actual job timestamp.
    """
    fmt = "%Y-%m-%d %H:%M:%S"
    ts_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+"
    for line in text.splitlines():
        if "cli.py" in line:
            continue
        match = re.search(ts_pattern, line)
        if match:
            return datetime.strptime(match.group(1), fmt)
    return None


def _fmt_duration(minutes):
    """Format minutes as human-readable duration."""
    if minutes < 1:
        return "<1min"
    if minutes < 60:
        return f"~{int(minutes)}min"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"~{hours}h{mins:02d}m"


GPU_MODELS = {"xgboost", "catboost", "realmlp", "resnet", "ft_transformer"}


def _is_gpu_model(model_name):
    """Check if model runs on GPU based on name prefix."""
    return any(model_name.startswith(p) for p in GPU_MODELS)


def _device_label(model_name, color=True):
    """Return GPU/CPU label, optionally colored (yellow GPU, dim CPU)."""
    if _is_gpu_model(model_name):
        return "\033[33mGPU\033[0m" if color else "GPU"
    return "\033[2mCPU\033[0m" if color else "CPU"


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

    # Per-task durations from worker Finished lines: "Finished seed=N ... (Xm YYs)"
    task_durations = {}
    for m in re.finditer(
        r"\[(\w[\w_]*)\] Finished seed=(\d+).*?\((\d+)m(\d+)s\)", text
    ):
        key = (m.group(1), m.group(2))
        mins, secs = int(m.group(3)), int(m.group(4))
        task_durations[key] = mins * 60 + secs

    # Elapsed time (from first log timestamp to now)
    t_start = _parse_start_time(text)
    elapsed_min = (datetime.now() - t_start).total_seconds() / 60.0 if t_start else None

    task_eta_min = None
    neural_eta_min = None
    neural_detail = None

    if completed:
        last = completed[-1]
        n_done = int(last[0])
        n_total = int(last[1])
        idx_width = len(last[1])  # width of "57" in "[29/57]"
        max_model_len = max(len(c[2]) for c in completed)
        print(f"  Progress: {n_done}/{n_total} tasks completed")
        top_threshold = max(float(c[4]) for c in completed) - 0.0001
        for c in completed:
            dur = task_durations.get((c[2], c[3]))
            dur_str = f"  ({dur // 60}m{dur % 60:02d}s)" if dur else ""
            idx = f"[{c[0]:>{idx_width}}/{c[1]}]"
            is_top = float(c[4]) >= top_threshold
            dev = _device_label(c[2], color=not is_top)
            hi = "\033[1;32m" if is_top else ""
            reset = "\033[0m" if is_top else ""
            print(
                f"    {hi}{idx} {c[2]:<{max_model_len}}  seed={c[3]:<4} "
                f"AUC: {c[4]}  {dev}{dur_str}{reset}"
            )

        # Compute task-rate ETA (used later in combined estimate)
        if elapsed_min and n_done > 0 and n_done < n_total:
            rate = elapsed_min / n_done
            task_eta_min = rate * (n_total - n_done)
        elif elapsed_min and n_done == n_total:
            print(f"  Completed in {_fmt_duration(elapsed_min)}")
    elif elapsed_min:
        print(f"  Elapsed: {_fmt_duration(elapsed_min)}, no tasks completed yet")

    # Currently running — show estimated duration from same-model completions
    if running:
        # Build avg duration per model type from completed tasks
        model_durations = {}
        for (model, seed), dur in task_durations.items():
            model_durations.setdefault(model, []).append(dur)
        model_avg_dur = {m: sum(ds) / len(ds) for m, ds in model_durations.items()}

        max_run_model_len = max(len(m) for m, s in running)
        print(f"  Running ({len(running)}):")
        for m, s in sorted(running):
            avg = model_avg_dur.get(m)
            if avg:
                est_str = f"  (~{int(avg) // 60}m{int(avg) % 60:02d}s est)"
            else:
                est_str = ""
            dev = _device_label(m)
            print(f"    {m:<{max_run_model_len}}  seed={s:<4} {dev}{est_str}")

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
            # Map each running neural model to its PID and log start position.
            # Ray reuses PIDs, so we take the LAST Starting message per PID.
            # Then count fold markers only after that position.
            model_pid = {}  # model_name -> (pid, start_pos)
            for m in re.finditer(r"pid=(\d+).*?\[(\w[\w_]*)\] Starting seed=", text):
                pid, mname = m.group(1), m.group(2)
                if mname in neural_running:
                    model_pid[mname] = (pid, m.start())

            fold_counts = {}
            for mname, (pid, start_pos) in model_pid.items():
                after = text[start_pos:]
                if mname.startswith("realmlp"):
                    # pytabkit: count Trainer.fit stopped, ~2 per fold
                    cnt = len(re.findall(rf"pid={pid}.*?Trainer\.fit.? stopped", after))
                    fold_counts[mname] = cnt // 2
                else:
                    # skorch: count epoch headers (1 per fold start)
                    cnt = len(re.findall(rf"pid={pid}.*?epoch\s+train_loss", after))
                    fold_counts[mname] = max(0, cnt - 1)

            for mname in neural_running:
                done = fold_counts.get(mname, 0)
                print(f"  {mname} fold progress: ~{done}/{folds_n}")

            # Compute neural fold-based ETA (used in combined estimate)
            max_done = max(fold_counts.values()) if fold_counts else 0
            if elapsed_min and max_done > 0:
                fold_rate = elapsed_min / max_done
                max_remaining = max(
                    folds_n - fold_counts.get(m, 0) for m in neural_running
                )
                neural_eta_min = fold_rate * max_remaining
                total_done = sum(fold_counts.values())
                total_needed = len(neural_running) * folds_n
                neural_detail = (
                    f"{_fmt_duration(fold_rate)}/fold, "
                    f"{total_done}/{total_needed} folds done"
                )

    # Combined ETA: max of task-rate and neural fold-rate estimates
    if elapsed_min and completed:
        last = completed[-1]
        n_done = int(last[0])
        n_total = int(last[1])
        if n_done < n_total:
            combined_eta = max(task_eta_min or 0, neural_eta_min or 0)
            parts = [f"Elapsed: {_fmt_duration(elapsed_min)}"]
            parts.append(f"ETA: {_fmt_duration(combined_eta)}")
            detail_parts = []
            if task_eta_min:
                rate = elapsed_min / n_done
                detail_parts.append(f"{_fmt_duration(rate)}/task")
            if neural_detail:
                detail_parts.append(neural_detail)
            if detail_parts:
                parts.append(f"({', '.join(detail_parts)})")
            print(f"  {' '.join(parts)}")

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
