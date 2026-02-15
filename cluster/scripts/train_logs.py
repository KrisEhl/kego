#!/usr/bin/env python3
"""Parse Ray job logs and show training progress summary."""

import re
import sys
from datetime import datetime

TS_FMT = "%Y-%m-%d %H:%M:%S"
TS_PAT = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+")


def _parse_start_time(text):
    """Extract job start timestamp from Ray log lines.

    Skips CLI timestamps (cli.py) and finds the first actual job timestamp.
    """
    for line in text.splitlines():
        if "cli.py" in line:
            continue
        match = TS_PAT.search(line)
        if match:
            return datetime.strptime(match.group(1), TS_FMT)
    return None


def _build_ts_index(text):
    """Build sorted list of (position, datetime) from all timestamps in text."""
    index = []
    for m in TS_PAT.finditer(text):
        ts = datetime.strptime(m.group(1), TS_FMT)
        index.append((m.start(), ts))
    return index


def _ts_at_pos(ts_index, pos):
    """Find the nearest timestamp at or before a position in the text."""
    best = None
    for tpos, ts in ts_index:
        if tpos <= pos:
            best = ts
        else:
            break
    return best


def _fmt_duration(minutes):
    """Format minutes as human-readable duration."""
    if minutes < 1:
        return "<1min"
    if minutes < 60:
        return f"~{int(minutes)}min"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"~{hours}h{mins:02d}m"


def _fmt_secs(secs):
    """Format seconds as Xh YYm."""
    m = int(secs) // 60
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


GPU_MODELS = {"xgboost", "catboost", "realmlp", "resnet", "ft_transformer"}
NEURAL_MODELS = {"realmlp", "realmlp_large", "resnet", "ft_transformer"}
N_GPU_WORKERS = 3

# Map worker IP → GPU label (None = head node: 2080Ti + 3090)
NODE_GPUS = {
    None: "head",
    "192.168.178.32": "head",
    "192.168.178.75": "3090",
    "192.168.178.80": "3050",
}


def _is_gpu_model(model_name):
    """Check if model runs on GPU based on name prefix."""
    return any(model_name.startswith(p) for p in GPU_MODELS)


def _is_neural(model_name):
    """Check if model is a neural model (fold-based progress tracking)."""
    return model_name in NEURAL_MODELS


def _device_label(model_name, color=True, gpu_amount=None):
    """Return GPU/CPU label, optionally colored (yellow GPU, dim CPU)."""
    if _is_gpu_model(model_name):
        amt = f"\u00d7{gpu_amount:g}" if gpu_amount else ""
        return f"\033[33mGPU{amt}\033[0m" if color else f"GPU{amt}"
    return "\033[2mCPU\033[0m" if color else "CPU"


def main():
    job_id = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    lines = sys.stdin.readlines()
    text = "".join(lines)
    now = datetime.now()

    mode = re.search(r"Mode: (.+)", text)
    print(f"Job: {job_id}")
    if mode:
        print(f"  {mode.group(0)}")

    # Track task lifecycle
    all_planned = set()
    gpu_amounts = {}  # (model, seed) -> float
    for m in re.finditer(
        r"- (\w[\w_]*) seed=(\d+) \((?:CPU|GPU(?: ([\d.]+))?)\)", text
    ):
        key = (m.group(1), m.group(2))
        all_planned.add(key)
        if m.group(3):
            gpu_amounts[key] = float(m.group(3))
    started = set(re.findall(r"\[(\w+)\] Starting seed=(\d+)", text))
    finished = set(re.findall(r"\[(\w+)\] Finished seed=(\d+)", text))
    running = started - finished
    unscheduled = all_planned - started - finished

    # Completed tasks with AUC
    completed = re.findall(
        r"\[(\d+)/(\d+)\].*?(\w[\w_]+) seed=(\d+).*?Holdout AUC: ([\d.]+)", text
    )

    # Per-task durations and IPs from worker Finished lines
    task_durations = {}
    task_ips = {}  # (model, seed) -> ip or None
    for m in re.finditer(
        r"ip=([\d.]+)\).*?\[(\w[\w_]*)\] Finished seed=(\d+).*?\((\d+)m(\d+)s\)", text
    ):
        key = (m.group(2), m.group(3))
        mins, secs = int(m.group(4)), int(m.group(5))
        task_durations[key] = mins * 60 + secs
        task_ips[key] = m.group(1)
    # Also match head node tasks (no ip= in prefix)
    for m in re.finditer(
        r"pid=(\d+)\).*?\[(\w[\w_]*)\] Finished seed=(\d+).*?\((\d+)m(\d+)s\)", text
    ):
        key = (m.group(2), m.group(3))
        if key not in task_durations:
            mins, secs = int(m.group(4)), int(m.group(5))
            task_durations[key] = mins * 60 + secs
            task_ips[key] = None

    # Average duration per model type from completed tasks
    model_durations = {}
    for (model, seed), dur in task_durations.items():
        model_durations.setdefault(model, []).append(dur)
    model_avg_dur = {m: sum(ds) / len(ds) for m, ds in model_durations.items()}

    # Build timestamp index for position-based lookups
    ts_index = _build_ts_index(text)

    # Elapsed time (from first log timestamp to now)
    t_start = _parse_start_time(text)
    elapsed_min = (now - t_start).total_seconds() / 60.0 if t_start else None

    # --- Completed tasks ---
    if completed:
        last = completed[-1]
        n_done = int(last[0])
        n_total = int(last[1])
        idx_width = len(last[1])
        max_model_len = max(len(c[2]) for c in completed)
        print(f"  Progress: {n_done}/{n_total} tasks completed")
        top_threshold = max(float(c[4]) for c in completed) - 0.0001
        for c in completed:
            dur = task_durations.get((c[2], c[3]))
            dur_str = f"  ({dur // 60}m{dur % 60:02d}s)" if dur else ""
            idx = f"[{c[0]:>{idx_width}}/{c[1]}]"
            is_top = float(c[4]) >= top_threshold
            dev = _device_label(
                c[2], color=not is_top, gpu_amount=gpu_amounts.get((c[2], c[3]))
            )
            # GPU type for completed GPU tasks
            ip = task_ips.get((c[2], c[3]))
            gpu_str = ""
            if _is_gpu_model(c[2]) and (c[2], c[3]) in task_ips:
                gpu_name = NODE_GPUS.get(ip, ip or "?")
                gpu_str = f" [{gpu_name}]"
            hi = "\033[1;32m" if is_top else ""
            reset = "\033[0m" if is_top else ""
            print(
                f"    {hi}{idx} {c[2]:<{max_model_len}}  seed={c[3]:<4} "
                f"AUC: {c[4]}  {dev}{gpu_str}{dur_str}{reset}"
            )
        if elapsed_min and n_done == n_total:
            print(f"  Completed in {_fmt_duration(elapsed_min)}")
    elif elapsed_min:
        print(f"  Elapsed: {_fmt_duration(elapsed_min)}, no tasks completed yet")

    # --- Per-task timing and fold progress ---
    folds_n = 5
    folds_match = re.search(r"(\d+) folds", text)
    if folds_match:
        folds_n = int(folds_match.group(1))

    # Parse Starting messages → (model, seed) -> (pid, start_pos, ip)
    task_pids = {}
    for m in re.finditer(
        r"pid=(\d+)(?:, ip=([\d.]+))?.*?\[(\w[\w_]*)\] Starting seed=(\d+)", text
    ):
        pid, ip, mname, seed = m.group(1), m.group(2), m.group(3), m.group(4)
        task_pids[(mname, seed)] = (pid, m.start(), ip)

    # Count folds per running neural task
    fold_counts = {}
    for mname, seed in running:
        if not _is_neural(mname):
            continue
        pid_info = task_pids.get((mname, seed))
        if not pid_info:
            continue
        pid, start_pos, _ = pid_info
        after = text[start_pos:]
        if mname.startswith("realmlp"):
            cnt = len(re.findall(rf"pid={pid}.*?LOCAL_RANK:", after))
        else:
            cnt = len(re.findall(rf"pid={pid}.*?epoch\s+train_loss", after))
        fold_counts[(mname, seed)] = min(max(0, cnt - 1), folds_n)

    # Compute how long each running task has been running
    task_elapsed = {}  # (model, seed) -> seconds running
    for mname, seed in running:
        pid_info = task_pids.get((mname, seed))
        if pid_info:
            _, start_pos, _ = pid_info
            start_ts = _ts_at_pos(ts_index, start_pos)
            if start_ts:
                task_elapsed[(mname, seed)] = (now - start_ts).total_seconds()

    # Estimate remaining time per task
    task_remaining = {}
    for mname, seed in running:
        elapsed_secs = task_elapsed.get((mname, seed))
        folds_done = fold_counts.get((mname, seed))

        # For neural tasks with fold progress: extrapolate from elapsed time
        if folds_done is not None and folds_done > 0 and elapsed_secs:
            total_est = elapsed_secs * folds_n / folds_done
            task_remaining[(mname, seed)] = max(0, total_est - elapsed_secs)
        elif folds_done is not None and folds_done == 0:
            # Neural but no folds done yet — use model avg or None
            avg = model_avg_dur.get(mname)
            task_remaining[(mname, seed)] = avg
        else:
            # Non-neural: use model average minus elapsed
            avg = model_avg_dur.get(mname)
            if avg and elapsed_secs:
                task_remaining[(mname, seed)] = max(0, avg - elapsed_secs)
            else:
                task_remaining[(mname, seed)] = None

    # Estimate duration for unscheduled tasks using completed same-model or
    # average from timestamp-based durations of completed tasks
    unsched_est = {}
    # Fallback: estimate completed task durations from timestamps if no worker timing
    if not model_avg_dur and ts_index and completed:
        for c in completed:
            mname, seed = c[2], c[3]
            # Find the [N/M] completion line timestamp
            pattern = rf"\[{c[0]}/{c[1]}\].*?{mname} seed={seed}"
            match = re.search(pattern, text)
            if match:
                finish_ts = _ts_at_pos(ts_index, match.start())
                # Find the Starting message position for this task
                pid_info = task_pids.get((mname, seed))
                if pid_info and finish_ts:
                    start_ts = _ts_at_pos(ts_index, pid_info[1])
                    if start_ts:
                        dur = (finish_ts - start_ts).total_seconds()
                        if dur > 0:
                            task_durations[(mname, seed)] = dur
                            model_durations.setdefault(mname, []).append(dur)
        # Rebuild averages
        model_avg_dur = {m: sum(ds) / len(ds) for m, ds in model_durations.items()}

    for mname, seed in unscheduled:
        avg = model_avg_dur.get(mname)
        unsched_est[(mname, seed)] = avg

    # --- Print running tasks with elapsed time and ETA ---
    if running:
        all_names = [m for m, s in running] + [m for m, s in unscheduled]
        max_name_len = max(len(n) for n in all_names) if all_names else 10
        print(f"  Running ({len(running)}):")
        for m, s in sorted(running):
            dev = _device_label(m, gpu_amount=gpu_amounts.get((m, s)))
            elapsed_secs = task_elapsed.get((m, s))
            remaining = task_remaining.get((m, s))
            folds_done = fold_counts.get((m, s))

            # GPU type from worker IP
            pid_info = task_pids.get((m, s))
            gpu_str = ""
            if _is_gpu_model(m) and pid_info:
                ip = pid_info[2]
                gpu_name = NODE_GPUS.get(ip, ip or "?")
                gpu_str = f" [{gpu_name}]"

            parts = []
            if elapsed_secs:
                parts.append(f"running {_fmt_secs(elapsed_secs)}")
            if folds_done is not None:
                parts.append(f"fold {folds_done}/{folds_n}")
            if remaining is not None:
                parts.append(f"~{_fmt_secs(remaining)} left")

            eta_str = f"  ({', '.join(parts)})" if parts else ""
            print(f"    {m:<{max_name_len}}  seed={s:<4} {dev}{gpu_str}{eta_str}")

    # --- Print unscheduled tasks with estimated durations ---
    if unscheduled:
        all_names = [m for m, s in running] + [m for m, s in unscheduled]
        max_name_len = max(len(n) for n in all_names) if all_names else 10
        print(f"  Unscheduled ({len(unscheduled)}):")
        for m, s in sorted(unscheduled):
            dev = _device_label(m, gpu_amount=gpu_amounts.get((m, s)))
            est = unsched_est.get((m, s)) or model_avg_dur.get(m)
            est_str = f"  (~{_fmt_secs(est)} est)" if est else ""
            print(f"    {m:<{max_name_len}}  seed={s:<4} {dev}{est_str}")

    # --- Combined ETA from per-task estimates ---
    if elapsed_min and completed:
        last = completed[-1]
        n_done = int(last[0])
        n_total = int(last[1])
        if n_done < n_total:
            gpu_remaining = []
            cpu_remaining = []

            for mname, seed in running:
                rem = task_remaining.get((mname, seed))
                if rem is None:
                    rem = model_avg_dur.get(mname, 300)
                if _is_gpu_model(mname):
                    gpu_remaining.append(rem)
                else:
                    cpu_remaining.append(rem)

            for mname, seed in unscheduled:
                est = unsched_est.get((mname, seed)) or model_avg_dur.get(mname, 300)
                if _is_gpu_model(mname):
                    gpu_remaining.append(est)
                else:
                    cpu_remaining.append(est)

            # GPU: total work / n_gpus; CPU: longest single task (parallel with GPU)
            gpu_eta = sum(gpu_remaining) / N_GPU_WORKERS / 60 if gpu_remaining else 0
            cpu_eta = max(cpu_remaining) / 60 if cpu_remaining else 0
            combined_eta = max(gpu_eta, cpu_eta)

            parts = [f"Elapsed: {_fmt_duration(elapsed_min)}"]
            parts.append(f"ETA: {_fmt_duration(combined_eta)}")
            detail = []
            if gpu_remaining:
                detail.append(
                    f"{len(gpu_remaining)} GPU tasks, "
                    f"~{_fmt_secs(sum(gpu_remaining))} work / {N_GPU_WORKERS} GPUs"
                )
            if cpu_remaining:
                detail.append(f"{len(cpu_remaining)} CPU tasks")
            if detail:
                parts.append(f"({'; '.join(detail)})")
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
