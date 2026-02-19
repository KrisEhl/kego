#!/usr/bin/env python3
"""Parse Ray job logs and show training progress summary.

Architecture: parse_log() -> compute_state() -> display()
"""

import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TS_FMT = "%Y-%m-%d %H:%M:%S"
TS_PAT = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+")

# Learner ID pattern: e.g. "catboost/ablation-pruned/5f" or bare "catboost"
LID = r"[\w][\w/.-]*"

GPU_MODELS = {"xgboost", "catboost", "realmlp", "resnet", "ft_transformer", "tabpfn"}
NEURAL_MODELS = {"realmlp", "realmlp_large", "resnet", "ft_transformer"}
N_GPU_WORKERS = 3

# Map worker IP -> GPU label (None = head node: 2080Ti + 3090)
NODE_GPUS = {
    None: "head",
    "192.168.178.32": "head",
    "192.168.178.75": "3090",
    "192.168.178.80": "3050",
}

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

PAT_MODE = re.compile(r"Mode: (.+)")
PAT_FOLDS = re.compile(r"(\d+) folds")
PAT_FOLD_FROM_LID = re.compile(r"/(\d+)f$")

# Planned task lines: "- catboost/raw/5f seed=42 (GPU 0.25)" or "(CPU)"
PAT_PLANNED = re.compile(rf"- ({LID}) seed=(\d+) \((?:CPU|GPU(?: ([\d.]+))?)\)")

# Driver-side completion: "[1/20] catboost/raw/5f seed=42 ... Holdout AUC: 0.9123"
PAT_DRIVER_COMPLETED = re.compile(
    rf"\[(\d+)/(\d+)\] ({LID}) seed=(\d+).*?(?:Holdout|OOF) AUC: ([\d.]+)"
)

# Driver-side failure: "[3/20] Task failed: catboost/raw/5f seed=42 ..."
PAT_DRIVER_FAILED = re.compile(rf"\[(\d+)/(\d+)\] Task failed.*")

# Worker starting: "[catboost/raw/5f] Starting seed=42"
PAT_WORKER_STARTING = re.compile(
    rf"pid=(\d+)(?:, ip=([\d.]+))?.*?\[({LID})\] Starting seed=(\d+)"
)

# Worker finished with IP: "ip=1.2.3.4) ... [model] Finished seed=42 ... (3m05s)"
PAT_WORKER_FINISHED_IP = re.compile(
    rf"ip=([\d.]+)\).*?\[({LID})\] Finished seed=(\d+).*?\((\d+)m(\d+)s\)"
)

# Worker finished on head (no ip=): "pid=123) ... [model] Finished seed=42 ... (3m05s)"
PAT_WORKER_FINISHED_HEAD = re.compile(
    rf"pid=(\d+)\).*?\[({LID})\] Finished seed=(\d+).*?\((\d+)m(\d+)s\)"
)

# Worker-side start/finish (simpler, for set-building)
PAT_STARTED = re.compile(rf"\[({LID})\] Starting seed=(\d+)")
PAT_FINISHED = re.compile(rf"\[({LID})\] Finished seed=(\d+)")

# Neural fold progress
PAT_REALMLP_FOLD = re.compile(r"Trainer\.fit.*stopped")
PAT_EPOCH_HEADER = re.compile(r"epoch\s+train_loss")
PAT_EPOCH_DUR = re.compile(r"\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)")

# Ensemble result patterns
ENSEMBLE_PATTERNS = [
    re.compile(p)
    for p in [
        r"avg.*seeds",
        r"Simple Average",
        r"Ridge.*(?:Holdout|OOF)",
        r"Hill Climbing.*(?:Holdout|OOF)",
        r"Rank Blending.*(?:Holdout|OOF)",
        r"L2 LightGBM.*(?:Holdout|OOF)",
        r"retrain-full.*method selection",
        r"Best method",
        r"Skipping calibration",
        r"Calibrat",
        r"Submission saved",
        r"Mean prediction",
    ]
]

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PlannedTask:
    learner_id: str
    seed: str
    gpu_amount: float | None  # None = CPU


@dataclass
class CompletedTask:
    index: int
    total: int
    learner_id: str
    seed: str
    auc: float
    auc_str: str  # original string from log (preserves formatting)
    duration_secs: int | None
    ip: str | None


@dataclass
class FailedTask:
    index: int
    total: int
    raw_line: str


@dataclass
class RunningTask:
    learner_id: str
    seed: str
    pid: str | None
    ip: str | None
    elapsed_secs: float | None
    folds_done: int | None
    folds_total: int | None
    remaining_secs: float | None
    gpu_amount: float | None


@dataclass
class ParsedLog:
    """Raw data extracted from log text."""

    mode: str | None
    default_folds: int
    planned: list[PlannedTask]
    gpu_amounts: dict[tuple[str, str], float]
    started: set[tuple[str, str]]
    finished: set[tuple[str, str]]
    driver_completed: list[tuple[str, str, str, str, str]]  # idx,total,lid,seed,auc
    driver_failed: list[tuple[str, str, str]]  # idx, total, raw_line
    task_pids: dict[tuple[str, str], tuple[str, int, str | None]]  # -> (pid, pos, ip)
    task_durations: dict[tuple[str, str], int]  # -> seconds
    task_ips: dict[tuple[str, str], str | None]
    ts_index: list[tuple[int, datetime]]
    start_time: datetime | None
    text: str
    lines: list[str]
    ensemble_lines: list[str]


@dataclass
class JobState:
    """Computed state ready for display."""

    job_id: str
    mode: str | None
    elapsed_min: float | None
    start_time: datetime | None

    planned: list[PlannedTask]
    completed: list[CompletedTask]
    failed: list[FailedTask]
    running: list[RunningTask]
    unscheduled: list[tuple[str, str]]  # (learner_id, seed)

    gpu_amounts: dict[tuple[str, str], float]
    model_avg_dur: dict[str, float]
    unsched_est: dict[tuple[str, str], float | None]

    # ETA data
    n_done: int
    n_total: int

    ensemble_lines: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _base_model(learner_id):
    """Extract base model name from learner ID."""
    return learner_id.split("/")[0]


def _is_gpu_model(model_name):
    """Check if model runs on GPU based on name prefix."""
    return any(model_name.startswith(p) for p in GPU_MODELS)


def _is_neural(model_name):
    """Check if model is a neural model (fold-based progress tracking)."""
    return _base_model(model_name) in NEURAL_MODELS


def _device_label(model_name, color=True, gpu_amount=None):
    """Return GPU/CPU label, optionally colored (yellow GPU, dim CPU)."""
    if _is_gpu_model(model_name):
        amt = f"\u00d7{gpu_amount:g}" if gpu_amount else ""
        return f"\033[33mGPU{amt}\033[0m" if color else f"GPU{amt}"
    return "\033[2mCPU\033[0m" if color else "CPU"


def _folds_for(learner_id, default_folds):
    """Extract fold count from learner ID (e.g. 'catboost/raw/5f' -> 5)."""
    m = PAT_FOLD_FROM_LID.search(learner_id)
    return int(m.group(1)) if m else default_folds


# ---------------------------------------------------------------------------
# parse_log: text -> ParsedLog
# ---------------------------------------------------------------------------


def parse_log(text):
    """Parse raw log text into structured data. Pure extraction, no computation."""
    lines = text.splitlines()

    # Mode
    mode_m = PAT_MODE.search(text)
    mode = mode_m.group(0) if mode_m else None

    # Default fold count
    folds_m = PAT_FOLDS.search(text)
    default_folds = int(folds_m.group(1)) if folds_m else 5

    # Planned tasks
    planned = []
    gpu_amounts = {}
    for m in PAT_PLANNED.finditer(text):
        key = (m.group(1), m.group(2))
        gpu_amt = float(m.group(3)) if m.group(3) else None
        planned.append(PlannedTask(m.group(1), m.group(2), gpu_amt))
        if gpu_amt is not None:
            gpu_amounts[key] = gpu_amt

    # Worker-side started/finished sets
    started = set(PAT_STARTED.findall(text))
    finished = set(PAT_FINISHED.findall(text))

    # Driver-side completions
    driver_completed = PAT_DRIVER_COMPLETED.findall(text)

    # Driver-side failures
    driver_failed = []
    for m in PAT_DRIVER_FAILED.finditer(text):
        driver_failed.append((m.group(1), m.group(2), m.group(0).strip()))

    # Worker Starting messages -> (model, seed) -> (pid, start_pos, ip)
    task_pids = {}
    for m in PAT_WORKER_STARTING.finditer(text):
        pid, ip, mname, seed = m.group(1), m.group(2), m.group(3), m.group(4)
        task_pids[(mname, seed)] = (pid, m.start(), ip)

    # Per-task durations and IPs from worker Finished lines
    task_durations = {}
    task_ips = {}
    for m in PAT_WORKER_FINISHED_IP.finditer(text):
        key = (m.group(2), m.group(3))
        mins, secs = int(m.group(4)), int(m.group(5))
        task_durations[key] = mins * 60 + secs
        task_ips[key] = m.group(1)
    for m in PAT_WORKER_FINISHED_HEAD.finditer(text):
        key = (m.group(2), m.group(3))
        if key not in task_durations:
            mins, secs = int(m.group(4)), int(m.group(5))
            task_durations[key] = mins * 60 + secs
            task_ips[key] = None

    # Timestamp index and start time
    ts_index = _build_ts_index(text)
    start_time = _parse_start_time(text)

    # Ensemble lines
    ensemble_lines = []
    for line in lines:
        for pat in ENSEMBLE_PATTERNS:
            if pat.search(line):
                ensemble_lines.append(line.strip())
                break

    return ParsedLog(
        mode=mode,
        default_folds=default_folds,
        planned=planned,
        gpu_amounts=gpu_amounts,
        started=started,
        finished=finished,
        driver_completed=driver_completed,
        driver_failed=driver_failed,
        task_pids=task_pids,
        task_durations=task_durations,
        task_ips=task_ips,
        ts_index=ts_index,
        start_time=start_time,
        text=text,
        lines=lines,
        ensemble_lines=ensemble_lines,
    )


# ---------------------------------------------------------------------------
# compute_state: ParsedLog -> JobState
# ---------------------------------------------------------------------------


def compute_state(parsed, now, job_id):
    """Compute display-ready state from parsed log data. Pure function."""
    # Build planned set
    all_planned = {(p.learner_id, p.seed) for p in parsed.planned}

    # Supplement worker-side sets with driver-side completions
    driver_completed_keys = {(c[2], c[3]) for c in parsed.driver_completed}
    started = parsed.started | driver_completed_keys
    finished = parsed.finished | driver_completed_keys

    # Failed tasks from driver
    failed_keys = set()
    failed = []
    for idx_s, total_s, raw in parsed.driver_failed:
        failed.append(FailedTask(int(idx_s), int(total_s), raw))
        # Try to extract learner_id/seed from the raw line to exclude from running
        m = re.search(rf"({LID}) seed=(\d+)", raw)
        if m:
            failed_keys.add((m.group(1), m.group(2)))

    # Task lifecycle
    running_keys = started - finished - failed_keys
    unscheduled_keys = all_planned - started

    # Per-model average durations
    model_durations = {}
    for (model, seed), dur in parsed.task_durations.items():
        base = _base_model(model)
        model_durations.setdefault(base, []).append(dur)
    model_avg_dur = {m: sum(ds) / len(ds) for m, ds in model_durations.items()}

    # Elapsed time
    elapsed_min = None
    if parsed.start_time:
        elapsed_min = (now - parsed.start_time).total_seconds() / 60.0

    # Fallback: estimate durations from timestamps if no worker timing
    if not model_avg_dur and parsed.ts_index and parsed.driver_completed:
        for c in parsed.driver_completed:
            mname, seed = c[2], c[3]
            pattern = rf"\[{c[0]}/{c[1]}\].*?{re.escape(mname)} seed={seed}"
            match = re.search(pattern, parsed.text)
            if match:
                finish_ts = _ts_at_pos(parsed.ts_index, match.start())
                pid_info = parsed.task_pids.get((mname, seed))
                if pid_info and finish_ts:
                    start_ts = _ts_at_pos(parsed.ts_index, pid_info[1])
                    if start_ts:
                        dur = (finish_ts - start_ts).total_seconds()
                        if dur > 0:
                            parsed.task_durations[(mname, seed)] = int(dur)
                            model_durations.setdefault(_base_model(mname), []).append(
                                dur
                            )
        model_avg_dur = {m: sum(ds) / len(ds) for m, ds in model_durations.items()}

    # --- Build completed list ---
    completed = []
    for c in parsed.driver_completed:
        key = (c[2], c[3])
        dur = parsed.task_durations.get(key)
        ip = parsed.task_ips.get(key)
        completed.append(
            CompletedTask(
                index=int(c[0]),
                total=int(c[1]),
                learner_id=c[2],
                seed=c[3],
                auc=float(c[4]),
                auc_str=c[4],
                duration_secs=dur,
                ip=ip,
            )
        )

    # --- Compute running task details ---

    # Collect all tasks per PID in log order (for elapsed reconstruction)
    pid_tasks = {}
    for (mname, seed), (pid, start_pos, ip) in parsed.task_pids.items():
        pid_tasks.setdefault(pid, []).append((start_pos, mname, seed))
    for pid in pid_tasks:
        pid_tasks[pid].sort()

    # Compute elapsed per running task via PID timeline reconstruction
    task_elapsed = {}
    for mname, seed in running_keys:
        pid_info = parsed.task_pids.get((mname, seed))
        if not pid_info or not parsed.start_time:
            continue
        pid, start_pos, ip = pid_info
        prev_duration = 0
        for pos, m, s in pid_tasks.get(pid, []):
            if pos >= start_pos:
                break
            dur = parsed.task_durations.get((m, s))
            if dur:
                prev_duration += dur
        task_start = parsed.start_time + timedelta(seconds=prev_duration)
        task_elapsed[(mname, seed)] = (now - task_start).total_seconds()

    # Count folds and epoch durations per running neural task
    fold_counts = {}
    task_epoch_secs = {}
    for mname, seed in running_keys:
        if not _is_neural(mname):
            continue
        pid_info = parsed.task_pids.get((mname, seed))
        if not pid_info:
            continue
        pid, start_pos, _ = pid_info
        after = parsed.text[start_pos:]
        if mname.startswith("realmlp"):
            cnt = len(re.findall(rf"pid={pid}.*?Trainer\.fit.*stopped", after))
        else:
            cnt = len(re.findall(rf"pid={pid}.*?epoch\s+train_loss", after))
            cnt = max(0, cnt - 1)
        folds_n = _folds_for(mname, parsed.default_folds)
        fold_counts[(mname, seed)] = min(cnt, folds_n)

        dur_values = re.findall(
            rf"pid={pid}.*?\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)", after
        )
        if dur_values:
            task_epoch_secs[(mname, seed)] = sum(float(d) for d in dur_values)

    # Estimate remaining time per running task
    task_remaining = {}
    for mname, seed in running_keys:
        elapsed_s = task_elapsed.get((mname, seed))
        folds_done = fold_counts.get((mname, seed))

        if folds_done is not None and folds_done > 0 and elapsed_s:
            task_folds = _folds_for(mname, parsed.default_folds)
            total_est = elapsed_s * task_folds / folds_done
            task_remaining[(mname, seed)] = max(0, total_est - elapsed_s)
        elif folds_done is not None and folds_done == 0:
            avg = model_avg_dur.get(_base_model(mname))
            task_remaining[(mname, seed)] = avg
        else:
            avg = model_avg_dur.get(_base_model(mname))
            if avg and elapsed_s:
                task_remaining[(mname, seed)] = max(0, avg - elapsed_s)
            else:
                task_remaining[(mname, seed)] = None

    # Build RunningTask list
    running = []
    for mname, seed in sorted(running_keys):
        pid_info = parsed.task_pids.get((mname, seed))
        folds_n = _folds_for(mname, parsed.default_folds)
        running.append(
            RunningTask(
                learner_id=mname,
                seed=seed,
                pid=pid_info[0] if pid_info else None,
                ip=pid_info[2] if pid_info else None,
                elapsed_secs=task_elapsed.get((mname, seed)),
                folds_done=fold_counts.get((mname, seed)),
                folds_total=folds_n if (mname, seed) in fold_counts else None,
                remaining_secs=task_remaining.get((mname, seed)),
                gpu_amount=parsed.gpu_amounts.get((mname, seed)),
            )
        )

    # Unscheduled estimates
    unsched_est = {}
    for mname, seed in unscheduled_keys:
        avg = model_avg_dur.get(_base_model(mname))
        unsched_est[(mname, seed)] = avg

    # n_done / n_total
    n_done = completed[-1].index if completed else 0
    n_total = completed[-1].total if completed else len(parsed.planned)

    return JobState(
        job_id=job_id,
        mode=parsed.mode,
        elapsed_min=elapsed_min,
        start_time=parsed.start_time,
        planned=parsed.planned,
        completed=completed,
        failed=failed,
        running=running,
        unscheduled=sorted(unscheduled_keys),
        gpu_amounts=parsed.gpu_amounts,
        model_avg_dur=model_avg_dur,
        unsched_est=unsched_est,
        n_done=n_done,
        n_total=n_total,
        ensemble_lines=parsed.ensemble_lines,
    )


# ---------------------------------------------------------------------------
# display: JobState -> stdout
# ---------------------------------------------------------------------------


def display(state, now):
    """Print training progress summary. Same format as previous version."""
    print(f"Job: {state.job_id}")
    if state.mode:
        print(f"  {state.mode}")

    # --- Completed tasks ---
    if state.completed:
        idx_width = len(str(state.n_total))
        max_model_len = max(len(c.learner_id) for c in state.completed)
        print(f"  Progress: {state.n_done}/{state.n_total} tasks completed")
        top_threshold = max(c.auc for c in state.completed) - 0.0001
        for c in state.completed:
            dur_str = ""
            if c.duration_secs is not None:
                dur_str = f"  ({c.duration_secs // 60}m{c.duration_secs % 60:02d}s)"
            idx = f"[{c.index:>{idx_width}}/{state.n_total}]"
            is_top = c.auc >= top_threshold
            dev = _device_label(
                c.learner_id,
                color=not is_top,
                gpu_amount=state.gpu_amounts.get((c.learner_id, c.seed)),
            )
            gpu_str = ""
            if _is_gpu_model(c.learner_id) and c.ip is not None:
                gpu_name = NODE_GPUS.get(c.ip, c.ip or "?")
                gpu_str = f" [{gpu_name}]"
            elif _is_gpu_model(c.learner_id) and c.duration_secs is not None:
                # Head node task (ip=None but has duration from head-match)
                gpu_name = NODE_GPUS.get(None, "?")
                gpu_str = f" [{gpu_name}]"
            hi = "\033[1;32m" if is_top else ""
            reset = "\033[0m" if is_top else ""
            print(
                f"    {hi}{idx} {c.learner_id:<{max_model_len}}  seed={c.seed:<4} "
                f"AUC: {c.auc_str}  {dev}{gpu_str}{dur_str}{reset}"
            )
        if state.elapsed_min and state.n_done == state.n_total:
            print(f"  Completed in {_fmt_duration(state.elapsed_min)}")
    elif state.elapsed_min:
        print(
            f"  Elapsed: {_fmt_duration(state.elapsed_min)}, " f"no tasks completed yet"
        )

    # --- Failed tasks ---
    if state.failed:
        print(f"  Failed ({len(state.failed)}):")
        for f in state.failed:
            idx = f"[{f.index}/{f.total}]"
            print(f"    \033[31m{idx} {f.raw_line}\033[0m")

    # --- Running tasks ---
    all_names = [r.learner_id for r in state.running] + [
        m for m, s in state.unscheduled
    ]
    max_name_len = max(len(n) for n in all_names) if all_names else 10

    if state.running:
        print(f"  Running ({len(state.running)}):")
        for r in state.running:
            dev = _device_label(r.learner_id, gpu_amount=r.gpu_amount)
            gpu_str = ""
            if _is_gpu_model(r.learner_id) and r.ip is not None:
                gpu_name = NODE_GPUS.get(r.ip, r.ip or "?")
                gpu_str = f" [{gpu_name}]"
            elif _is_gpu_model(r.learner_id) and r.pid is not None and r.ip is None:
                gpu_name = NODE_GPUS.get(None, "?")
                gpu_str = f" [{gpu_name}]"

            parts = []
            if r.elapsed_secs:
                parts.append(f"running {_fmt_secs(r.elapsed_secs)}")
            if r.folds_done is not None and r.folds_total is not None:
                parts.append(f"fold {r.folds_done}/{r.folds_total}")
            if r.remaining_secs is not None:
                parts.append(f"~{_fmt_secs(r.remaining_secs)} left")

            eta_str = f"  ({', '.join(parts)})" if parts else ""
            print(
                f"    {r.learner_id:<{max_name_len}}  seed={r.seed:<4} "
                f"{dev}{gpu_str}{eta_str}"
            )

    # --- Unscheduled tasks ---
    if state.unscheduled:
        print(f"  Unscheduled ({len(state.unscheduled)}):")
        for mname, seed in state.unscheduled:
            dev = _device_label(mname, gpu_amount=state.gpu_amounts.get((mname, seed)))
            est = state.unsched_est.get((mname, seed)) or state.model_avg_dur.get(
                _base_model(mname)
            )
            est_str = f"  (~{_fmt_secs(est)} est)" if est else ""
            print(f"    {mname:<{max_name_len}}  seed={seed:<4} {dev}{est_str}")

    # --- Combined ETA ---
    if state.elapsed_min and state.completed and state.n_done < state.n_total:
        gpu_remaining = []
        cpu_remaining = []

        for r in state.running:
            rem = r.remaining_secs
            if rem is None:
                rem = state.model_avg_dur.get(_base_model(r.learner_id), 300)
            if _is_gpu_model(r.learner_id):
                gpu_remaining.append(rem)
            else:
                cpu_remaining.append(rem)

        for mname, seed in state.unscheduled:
            est = state.unsched_est.get((mname, seed)) or state.model_avg_dur.get(
                _base_model(mname), 300
            )
            if _is_gpu_model(mname):
                gpu_remaining.append(est)
            else:
                cpu_remaining.append(est)

        gpu_eta = sum(gpu_remaining) / N_GPU_WORKERS / 60 if gpu_remaining else 0
        cpu_eta = max(cpu_remaining) / 60 if cpu_remaining else 0
        combined_eta = max(gpu_eta, cpu_eta)

        parts = [f"Elapsed: {_fmt_duration(state.elapsed_min)}"]
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

    # --- Ensemble results ---
    for line in state.ensemble_lines:
        print(f"  {line}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    job_id = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    text = sys.stdin.read()
    now = datetime.now()
    parsed = parse_log(text)
    state = compute_state(parsed, now, job_id)
    display(state, now)


if __name__ == "__main__":
    main()
