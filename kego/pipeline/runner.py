"""Pipeline orchestration: the verbs ``run`` / ``ensemble`` / ``tune`` / ``submit``.

Wires task + feature store + trainer (+ implicit cache) + ensembler + evaluator
+ submitter together. Each verb is a thin sequence over the stage objects; all
the reusable logic lives in those stages.
"""

from __future__ import annotations

from dataclasses import dataclass

from kego.fleet import Machine
from kego.pipeline.config import PipelineConfig
from kego.pipeline.ensemble import EnsembleResult
from kego.pipeline.evaluate import EvalReport
from kego.pipeline.executor import Executor, get_executor
from kego.pipeline.features import FeatureSets
from kego.pipeline.predictions import (
    CachingPredictionStore,
    LocalCacheStore,
    MlflowPredictionStore,
    Predictions,
    PredictionStore,
)
from kego.pipeline.submit import SubmitResult
from kego.pipeline.task import Task, get_task
from kego.pipeline.train import TrainContext, Trainer


def model_submission_stats(csv_text: str, leaderboard_csv: str | None = None) -> dict[str, dict[str, str]]:
    """Summarize Kaggle submission attempts and best public rank by registry version."""
    import csv
    import io
    import re

    base_header = "fileName,date,description,status,publicScore,privateScore"
    starts = [start for header in (f"ref,{base_header}", base_header) if (start := csv_text.find(header)) >= 0]
    if not starts:
        return {}

    stats: dict[str, dict[str, float | int | None]] = {}
    for row in csv.DictReader(io.StringIO(csv_text[min(starts) :])):
        match = re.search(r"\bRegistry v(\d+)\b", " ".join((row.get("description") or "").split()))
        if not match:
            continue
        values = stats.setdefault(match.group(1), {"attempts": 0, "best_score": None})
        values["attempts"] = int(values["attempts"] or 0) + 1
        status = (row.get("status") or "").removeprefix("SubmissionStatus.")
        if status != "COMPLETE" or not row.get("publicScore"):
            continue
        try:
            score = float(row["publicScore"])
        except (TypeError, ValueError):
            continue
        best = values["best_score"]
        values["best_score"] = score if best is None else max(float(best), score)

    public_scores = []
    if leaderboard_csv:
        for row in csv.DictReader(io.StringIO(leaderboard_csv.lstrip("\ufeff"))):
            try:
                public_scores.append(float(row["Score"]))
            except (KeyError, TypeError, ValueError):
                continue

    return {
        version: {
            "submitted": "yes" if values["attempts"] == 1 else f"yes ({values['attempts']})",
            "public_rank": (
                str(1 + sum(score > float(values["best_score"]) for score in public_scores))
                if public_scores and values["best_score"] is not None
                else "-"
            ),
        }
        for version, values in stats.items()
    }


def format_submissions(csv_text: str, description_width: int = 64, leaderboard_csv: str | None = None) -> str:
    import csv
    import io

    if description_width < 3:
        raise ValueError("description_width must be at least 3")
    base_header = "fileName,date,description,status,publicScore,privateScore"
    header_starts = [start for header in (f"ref,{base_header}", base_header) if (start := csv_text.find(header)) >= 0]
    if not header_starts:
        return "No submissions found."
    header_start = min(header_starts)
    rows = []
    for raw in csv.DictReader(io.StringIO(csv_text[header_start:])):
        full_description = " ".join((raw.get("description") or "").split())
        description = full_description
        if len(description) > description_width:
            description = f"{description[: description_width - 3]}..."
        status = (raw.get("status") or "-").removeprefix("SubmissionStatus.")
        rows.append(
            {
                "date": (raw.get("date") or "-")[:16],
                "status": status,
                "score": raw.get("publicScore") or raw.get("privateScore") or "-",
                "file": raw.get("fileName") or "-",
                "description": description or "-",
                "_date": raw.get("date") or "",
                "_description": full_description,
                "_public_score": raw.get("publicScore") or "",
            }
        )
    if not rows:
        return "No submissions found."
    rows.sort(key=lambda row: row["_date"], reverse=True)
    columns = ["date", "status", "score", "file", "description"]
    widths = {column: max(len(column), *(len(row[column]) for row in rows)) for column in columns}
    widths["description"] = min(widths["description"], description_width)

    def render(values: dict[str, str]) -> str:
        return "  ".join(
            values[column].rjust(widths[column]) if column == "score" else values[column].ljust(widths[column])
            for column in columns
        ).rstrip()

    header = render({column: column for column in columns})
    separator = render({column: "-" * widths[column] for column in columns})
    output = [header, separator, *(render(row) for row in rows)]

    import re

    current_versions = []
    for row in rows:
        if (match := re.search(r"\bRegistry v(\d+)\b", row["_description"])) and match.group(1) not in current_versions:
            current_versions.append(match.group(1))
            if len(current_versions) == 2:
                break

    models: dict[str, dict[str, float | int | None]] = {
        version: {"best_score": None, "scored": 0} for version in current_versions
    }
    for row in rows:
        match = re.search(r"\bRegistry v(\d+)\b", row["_description"])
        if not match or row["status"] != "COMPLETE" or not row["_public_score"]:
            continue
        try:
            score = float(row["_public_score"])
        except ValueError:
            continue
        version = match.group(1)
        if version not in current_versions:
            continue
        best_score = models[version]["best_score"]
        models[version]["best_score"] = score if best_score is None else max(float(best_score), score)
        models[version]["scored"] = int(models[version]["scored"] or 0) + 1

    if models:
        leaderboard_scores = []
        if leaderboard_csv:
            for leaderboard_row in csv.DictReader(io.StringIO(leaderboard_csv.lstrip("\ufeff"))):
                try:
                    leaderboard_scores.append(float(leaderboard_row["Score"]))
                except (KeyError, TypeError, ValueError):
                    continue
        ranked = sorted(
            models.items(),
            key=lambda item: (
                item[1]["best_score"] is None,
                -float(item[1]["best_score"] or 0),
                -int(item[0]),
            ),
        )
        rank_rows = [
            {
                "model": f"v{version}",
                "best_score": str(values["best_score"]) if values["best_score"] is not None else "-",
                "public_rank": (
                    str(1 + sum(score > float(values["best_score"]) for score in leaderboard_scores))
                    if leaderboard_scores and values["best_score"] is not None
                    else "-"
                ),
                "scored": str(values["scored"]),
            }
            for version, values in ranked
        ]
        rank_columns = ["model", "best_score", "public_rank", "scored"]
        rank_widths = {column: max(len(column), *(len(row[column]) for row in rank_rows)) for column in rank_columns}

        def render_rank(values: dict[str, str]) -> str:
            return "  ".join(values[column].rjust(rank_widths[column]) for column in rank_columns)

        output.extend(
            [
                "Current registry models (latest 2; best public score)",
                render_rank({column: column for column in rank_columns}),
                render_rank({column: "-" * rank_widths[column] for column in rank_columns}),
                *(render_rank(row) for row in rank_rows),
            ]
        )
    return "\n".join(output)


def _is_port_open(address: str, timeout: float = 0.2) -> bool:
    import socket
    from urllib.parse import urlparse

    try:
        parsed = urlparse(address)
        host = parsed.hostname or "localhost"
        port = parsed.port or 8265
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _resolve_dashboard_address(ray_address: str | None = None) -> str:
    """Return the HTTP Ray dashboard URL (``http://host:8265``).

    Accepts a ``ray://`` client address, an explicit ``http(s)://`` dashboard
    URL, or ``None`` (falls back to the default head node). A ``ray://`` address
    is mapped to the dashboard port 8265 on the same host.
    """
    addr = ray_address or "ray://omarchyd:10001"
    if addr.startswith(("http://", "https://")):
        return addr
    host = addr.split("://", 1)[-1].split(":")[0]
    return f"http://{host}:8265"


def _make_ray_job_client(dashboard_address: str):
    """Build a ``JobSubmissionClient`` for the given http dashboard URL.

    Ray's ``get_address_for_submission_client`` lets ``RAY_ADDRESS`` override the
    passed address; if it is a ``ray://`` client address the submission client
    routes through the (often unreachable) Ray Client port and times out. Clear
    ``RAY_ADDRESS`` during construction so the explicit http dashboard URL wins.
    """
    import os

    from ray.dashboard.modules.job.sdk import JobSubmissionClient  # ty: ignore[unresolved-import]

    saved = os.environ.pop("RAY_ADDRESS", None)
    try:
        return JobSubmissionClient(dashboard_address)
    finally:
        if saved is not None:
            os.environ["RAY_ADDRESS"] = saved


def _parse_etime(etime_str: str) -> int:
    etime_str = etime_str.strip()
    days = 0
    if "-" in etime_str:
        days_part, etime_str = etime_str.split("-", 1)
        days = int(days_part)
    parts = etime_str.split(":")
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        h = 0
        m, s = int(parts[0]), int(parts[1])
    else:
        h, m = 0, 0
        s = int(parts[0])
    return days * 86400 + h * 3600 + m * 60 + s


def _poll_machine(machine: Machine) -> dict:
    import os
    import re
    import subprocess

    remote_script = r"""
if [ "$(uname)" = "Darwin" ]; then
    CPU_UTIL=$(top -l 1 | awk '/CPU usage/ {print $3}' | tr -d '%')
    PS_CMD="ps -eo pid,etime,command"
else
    CPU_UTIL=$(top -bn1 | grep -i '%Cpu(s)' | awk '{for(i=1;i<=NF;i++) if($i ~ /id/) print 100 - $(i-1)}')
    PS_CMD="ps -eo pid,etime,cmd"
fi
echo "CPU_UTIL: $CPU_UTIL%"
echo "LOAD: $(cat /proc/loadavg 2>/dev/null || sysctl -n vm.loadavg 2>/dev/null || echo 'unknown')"
echo "CORES: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || grep -c ^processor /proc/cpuinfo 2>/dev/null || echo 'unknown')"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | while read -r gpuline; do
        echo "GPU: $gpuline"
    done
else
    echo "GPU: N/A"
fi
echo "PROCS:"
$PS_CMD 2>/dev/null | grep -E 'train-agent|train_agent|kego league|run_league' | grep -v -E 'grep|ssh' || true
echo "LOGS:"
ls -dt ~/.kego/logs/*.log 2>/dev/null | head -n 20 | while read -r logpath; do
    TAIL_LINES=$(tail -c 20000 "$logpath" 2>/dev/null | tr '\r' '\n')

    ITER_REGEX="--- Iteration ([0-9]+)/([0-9]+) ---"
    LEAGUE_REGEX="\(([0-9]+)/([0-9]+)\).*ETA: ([^]]+)"
    EVAL_REGEX="Evaluating gauntlet"
    PLAY_REGEX="Collecting self-play"
    BUFFER_REGEX="Training on buffer"
    COMPLETE_REGEX="Training complete\. avg loss: (.*)"
    DONE_REGEX="Done\. Best rule-agent avg: (.*)"
    ERROR_REGEX="Error during training: (.*)"

    CURRENT_ITER=""
    TOTAL_ITER=""
    STEP=""
    DONE_MSG=""
    KIND="train"

    while IFS= read -r line; do
        if [[ "$line" =~ $ITER_REGEX ]]; then
            CURRENT_ITER="${BASH_REMATCH[1]}"
            TOTAL_ITER="${BASH_REMATCH[2]}"
            KIND="train"
        elif [[ "$line" =~ $LEAGUE_REGEX ]]; then
            CURRENT_ITER="${BASH_REMATCH[1]}"
            TOTAL_ITER="${BASH_REMATCH[2]}"
            STEP="ETA ${BASH_REMATCH[3]}"
            KIND="league"
        elif [[ "$line" =~ $EVAL_REGEX ]]; then
            STEP="Evaluating gauntlet"
        elif [[ "$line" =~ $PLAY_REGEX ]]; then
            STEP="Collecting self-play data"
        elif [[ "$line" =~ $BUFFER_REGEX ]]; then
            STEP="Training on buffer"
        elif [[ "$line" =~ $COMPLETE_REGEX ]]; then
            LOSS_PART="${BASH_REMATCH[1]}"
            STEP="Training complete (${LOSS_PART%% |*})"
        elif [[ "$line" =~ $DONE_REGEX ]]; then
            DONE_MSG="Done (best avg: ${BASH_REMATCH[1]})"
        elif [[ "$line" =~ $ERROR_REGEX ]]; then
            DONE_MSG="Error: ${BASH_REMATCH[1]}"
        fi
    done <<< "$TAIL_LINES"

    FILENAME=$(basename "$logpath")
    RUN_ID="${FILENAME%.log}"
    echo "LOG_PARSED: run_id=$RUN_ID | curr=$CURRENT_ITER | total=$TOTAL_ITER | step=$STEP | done=$DONE_MSG | kind=$KIND"
done
"""
    try:
        from kego.fleet import machine_name

        is_local = machine.name == machine_name()
    except Exception:
        is_local = False
    connect_timeout = os.environ.get("KEGO_STATUS_CONNECT_TIMEOUT", "1")
    command_timeout = float(os.environ.get("KEGO_STATUS_TIMEOUT", "8"))
    cmd = (
        ["bash", "-lc", remote_script]
        if is_local
        else [
            "ssh",
            "-o",
            f"ConnectTimeout={connect_timeout}",
            "-o",
            "BatchMode=yes",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPath=~/.ssh/kego-%r@%h:%p",
            "-o",
            "ControlPersist=600",
            machine.ssh,
            remote_script,
        ]
    )

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=command_timeout)
        if res.returncode != 0:
            return {"name": machine.name, "status": "Offline", "error": res.stderr.strip() or "Connection failed"}

        load = "unknown"
        cores = "unknown"
        cpu_util = "unknown"
        gpus_raw = []
        procs = []
        parsed_logs = {}

        lines = res.stdout.splitlines()
        mode = None
        for line in lines:
            if line.startswith("LOAD:"):
                load_parts = line.split(":", 1)[1].strip().strip("{}").split()
                if len(load_parts) >= 3:
                    load = " ".join(load_parts[:3])
                else:
                    load = " ".join(load_parts) or "unknown"
            elif line.startswith("CORES:"):
                cores = line.split(":", 1)[1].strip()
            elif line.startswith("CPU_UTIL:"):
                cpu_util = line.split(":", 1)[1].strip()
            elif line.startswith("GPU:"):
                val = line.split(":", 1)[1].strip()
                if val and val != "N/A":
                    gpus_raw.append(val)
            elif line.startswith("PROCS:"):
                mode = "procs"
            elif line.startswith("LOGS:"):
                mode = "logs"
            elif mode == "procs":
                if line.strip():
                    procs.append(line.strip())
            elif line.startswith("LOG_PARSED:"):
                payload = line.split(":", 1)[1].strip()
                log_parts = [p.strip() for p in payload.split("|")]
                log_data = {}
                for part in log_parts:
                    if "=" in part:
                        k, v = part.split("=", 1)
                        log_data[k.strip()] = v.strip()
                if "run_id" in log_data:
                    parsed_logs[log_data["run_id"]] = log_data

        gpu = "N/A"
        if gpus_raw:
            utils = []
            total_used_mb = 0.0
            total_mem_mb = 0.0
            for val in gpus_raw:
                parts = [p.strip() for p in val.split(",")]
                if len(parts) >= 3:
                    util, used_mb, total_mb = parts[:3]
                    utils.append(f"{util}%")
                    try:
                        total_used_mb += float(used_mb)
                        total_mem_mb += float(total_mb)
                    except ValueError:
                        pass
            if utils:
                gpu = f"{'/'.join(utils)} ({total_used_mb / 1024.0:.1f}/{total_mem_mb / 1024.0:.1f} GB)"

        running_runs = []
        seen_run_ids = set()
        seen_cmds = set()
        for proc in procs:
            parts = proc.strip().split(None, 2)
            if len(parts) < 3:
                continue
            pid, etime, cmd = parts[0], parts[1], parts[2]
            if is_local and cmd.startswith("ssh ") and "kego league" in cmd:
                continue

            run_ids = re.findall(r"[a-f0-9]{32}", cmd)
            if run_ids:
                run_id = run_ids[0]
                if run_id not in seen_run_ids:
                    seen_run_ids.add(run_id)

                    log_data = parsed_logs.get(run_id, {})
                    curr = log_data.get("curr", "")
                    total = log_data.get("total", "")
                    step = log_data.get("step", "")
                    done = log_data.get("done", "")
                    kind = log_data.get("kind", "train")

                    eta_str = ""
                    if kind == "league" and step.startswith("ETA "):
                        eta_str = step.removeprefix("ETA ").strip()
                    if curr and total:
                        try:
                            curr_val = int(curr)
                            total_val = int(total)
                            if not eta_str and curr_val > 0 and total_val > curr_val:
                                elapsed_secs = _parse_etime(etime)
                                avg_iter_time = elapsed_secs / curr_val
                                rem_iters = total_val - curr_val
                                eta_secs = int(rem_iters * avg_iter_time)
                                h = eta_secs // 3600
                                m = (eta_secs % 3600) // 60
                                if h > 0:
                                    eta_str = f"{h}h {m}m"
                                else:
                                    eta_str = f"{m}m"
                        except Exception:  # noqa: S110
                            pass

                    if done:
                        progress_desc = done
                    elif curr:
                        label = "Games" if kind == "league" else "Iter"
                        progress_desc = f"{label} {curr}/{total}"
                        if step and kind != "league":
                            progress_desc += f" - {step}"
                    else:
                        progress_desc = "Running"

                    running_runs.append((pid, run_id, progress_desc, eta_str))
            else:
                if not any(x in cmd for x in ["uv run kego", "/bin/kego"]):
                    cmd_norm = cmd[:25]
                    if cmd_norm not in seen_cmds:
                        seen_cmds.add(cmd_norm)
                        running_runs.append((pid, cmd[:25], "", ""))

        load_val = f"{load} ({cores}c)" if cores != "unknown" else load
        cpu_val = f"{cpu_util} / {load_val}" if cpu_util != "unknown" else load_val
    except subprocess.TimeoutExpired:
        return {"name": machine.name, "status": "Offline", "error": "Timeout"}
    except Exception as e:
        return {"name": machine.name, "status": "Offline", "error": str(e)}
    else:
        return {
            "name": machine.name,
            "status": "Online",
            "role": machine.role,
            "load": cpu_val,
            "gpu": gpu,
            "gpu_count": len(gpus_raw),
            "runs": running_runs,
        }


@dataclass
class RunOutcome:
    predictions: list[Predictions]
    ensemble: EnsembleResult | None = None
    report: EvalReport | None = None
    submission: SubmitResult | None = None


class Pipeline:
    """High-level entry point built from a :class:`PipelineConfig`."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        store: PredictionStore | None = None,
        executor: Executor | None = None,
    ) -> None:
        self.config = config
        self.task: Task = get_task(config.task)
        self.store = store or CachingPredictionStore(LocalCacheStore(), MlflowPredictionStore())
        self.executor = executor or get_executor("serial")
        self.feature_sets = FeatureSets(config.feature_sets)
        self.trainer = Trainer(self.task, self.store, self.executor, force=config.force)

    # -- verbs ---------------------------------------------------------------

    def run(self) -> RunOutcome:
        """Full path: train grid (with cache) -> ensemble -> evaluate -> submit."""
        raise NotImplementedError

    def train_agent(self, epochs: int | None = None, output_path: str | None = None, **kwargs) -> None:
        """Run task-specific agent or model training."""
        train_fn = getattr(self.task, "train", None)
        if not callable(train_fn):
            raise NotImplementedError(f"Task '{self.task.name}' does not implement a train method.")
        init_checkpoint = kwargs.get("init_checkpoint")

        from kego.pipeline.executor import RayExecutor

        if isinstance(self.executor, RayExecutor):
            import os
            import subprocess
            import time

            try:
                import ray  # noqa: F401  # ty: ignore[unresolved-import]
            except ImportError:
                raise ImportError(
                    "Ray is not installed. Please install ray via 'pip install ray' to use the Ray executor."
                ) from None

            # 1. Determine Ray Dashboard address and connect (RAY_ADDRESS-safe).
            dashboard_address = _resolve_dashboard_address(os.environ.get("RAY_ADDRESS"))
            print(f"Connecting to Ray Dashboard at {dashboard_address}...")
            client = _make_ray_job_client(dashboard_address)

            # 2. Ship the LOCAL repo with the job so the cluster's own git checkout
            # is never used: the job's cwd is the uploaded working_dir, and both
            # `import kego` and `import cg` resolve there (cwd wins over the editable
            # .pth on sys.path). A stale cluster checkout therefore cannot matter.
            from kego.fleet import repo_root as find_repo_root

            # Repo root = the repo whose kego/ + cg/ we want to upload,
            # independent of the caller's cwd.
            repo_root = find_repo_root()

            # Output is written to an absolute cluster path so it survives the
            # discarded working_dir and can be scp'd back.
            remote_output = f"/home/kristian/projects/kego/{output_path}" if output_path else None
            cmd = f"/home/kristian/projects/kego/.venv/bin/python -m kego.pipeline.cli train-agent --task {self.task.name}"
            if kwargs.get("agent"):
                cmd += f" --agent {kwargs['agent']}"
            if kwargs.get("variant"):
                cmd += f" --variant {kwargs['variant']}"
            if epochs is not None:
                cmd += f" --epochs {epochs}"
            if remote_output:
                cmd += f" --output {remote_output}"
            if init_checkpoint:
                cmd += f" --init-checkpoint {init_checkpoint}"
            if kwargs.get("num_workers"):
                cmd += f" --num-workers {kwargs['num_workers']}"

            print(f"Submitting job to Ray cluster (working_dir={repo_root}): {cmd}")
            # Keep the upload light: drop VCS/venv/data/caches and every competition
            # except the one being trained. cg/ (the game engine) is kept.
            excludes = [
                ".git",
                ".venv",
                "**/__pycache__",
                "**/*.tar.gz",
                "data",
                "model_data",
                "outputs",
                "tmp",
                "mlruns",
            ]
            comps = repo_root / "competitions"
            if comps.is_dir():
                excludes += [
                    f"competitions/{p.name}" for p in comps.iterdir() if p.is_dir() and p.name != self.task.name
                ]

            # working_dir = repo root → packages kego/, cg/, and the active competition.
            job_id = client.submit_job(
                entrypoint=cmd,
                runtime_env={"working_dir": str(repo_root), "excludes": excludes},
            )
            print(f"Job '{job_id}' submitted successfully. Tailing logs...")

            # 3. Tail logs and wait for completion
            last_log_len = 0
            while True:
                status_info = client.get_job_status(job_id)
                try:
                    logs = client.get_job_logs(job_id)
                    if logs and len(logs) > last_log_len:
                        print(logs[last_log_len:], end="", flush=True)
                        last_log_len = len(logs)
                except Exception:  # noqa: S110
                    pass

                if status_info.is_terminal():
                    break
                time.sleep(1)

            status = client.get_job_status(job_id)
            if status != "SUCCEEDED":
                raise RuntimeError(f"Remote training job failed with status: {status}")

            print("Remote job completed successfully.")

            # 4. Download output file if specified
            if output_path:
                print(f"Downloading trained weights from omarchyd to local {output_path}...")
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                scp_cmd = ["scp", f"kristian@omarchyd:{remote_output}", output_path]
                res = subprocess.run(scp_cmd)
                if res.returncode == 0:
                    print(f"Successfully downloaded weights to {output_path}")
                else:
                    print("Error: failed to download trained weights via scp.")
            return

        import json
        import os
        from pathlib import Path

        # Outbox guard: un-synced checkpoints from a previous run of this task must reach
        # the registry before training starts — resume/lineage decisions read the registry,
        # so training against stale registry state can silently fork from the wrong parent
        # (and the new run overwrites the outputs/ files the queued entries came from).
        if os.environ.get("KEGO_IGNORE_OUTBOX") != "1":
            from kego.tracking.outbox import pending_for, sync_outbox

            if pending_for(self.task.name):
                print(f"Un-synced checkpoint registration(s) for '{self.task.name}' in the outbox; syncing...")
                for entry, model, version in sync_outbox()[0]:
                    print(f"  Synced {entry} -> {model} v{version}")
                still_pending = pending_for(self.task.name)
                if still_pending:
                    raise RuntimeError(
                        f"{len(still_pending)} checkpoint registration(s) for '{self.task.name}' are queued "
                        f"in the outbox and the registry is still unreachable. Starting a new run now could "
                        f"resume from a stale parent. Run `kego sync` once the hub is back "
                        f"(or set KEGO_IGNORE_OUTBOX=1 to train anyway)."
                    )

        active_runs_dir = Path(".kego/active_runs")
        active_runs_dir.mkdir(parents=True, exist_ok=True)

        pid = os.getpid()
        run_file = active_runs_dir / f"{pid}.json"

        variant_name = kwargs.get("variant", "default")
        run_info = {
            "task": self.task.name,
            "config": variant_name,
            "pid": pid,
            "progress": "running",
            "active_workers": [],
        }

        import contextlib

        with contextlib.suppress(Exception), open(run_file, "w") as f:
            json.dump(run_info, f)

        try:
            train_fn(self.config, epochs=epochs, output_path=output_path, **kwargs)
        finally:
            if run_file.exists():
                with contextlib.suppress(Exception):
                    run_file.unlink()

    def ensemble(
        self,
        *,
        experiments: list[str] | None = None,
        ensemble_tag: str | None = None,
    ) -> RunOutcome:
        """Re-ensemble stored predictions with no training (``--from-experiment``)."""
        raise NotImplementedError

    def tune(self, models: list[str]) -> RunOutcome:
        """Run Optuna for the named models; promote best params to a config."""
        raise NotImplementedError

    def submit(self, outcome: RunOutcome, message: str) -> SubmitResult:
        from pathlib import Path

        import numpy as np

        from kego.pipeline.submit import Submitter

        submitter = Submitter(self.task, self.config.submit)

        is_sim = "pokemon" in self.task.kaggle_slug
        filename = "submission.tar.gz" if is_sim else "submission.csv"

        submission_path = Path("outputs") / filename
        submission_path.parent.mkdir(parents=True, exist_ok=True)

        ids = np.array([])
        preds = np.array([])
        if outcome and outcome.predictions:
            # We can extract IDs and predictions if present, but for simulation it's empty
            pass

        final_path = submitter.write(ids, preds, submission_path)

        print(f"Submitting {final_path} to Kaggle ({self.task.kaggle_slug})...")
        result = submitter.submit(final_path, message)
        print(f"Submission status: {result.status}")
        if result.public_score is not None:
            print(f"Public LB Score: {result.public_score}")
        return result

    def status(self) -> None:
        import json
        from pathlib import Path

        # 1. Check local active runs
        local_found = False
        active_runs_dir = Path(".kego/active_runs")
        if active_runs_dir.exists():
            runs = list(active_runs_dir.glob("*.json"))
            active_list = []
            for run_file in runs:
                import contextlib

                with contextlib.suppress(Exception):
                    with open(run_file) as f:
                        data = json.load(f)
                    pid = data.get("pid")
                    if isinstance(pid, int):
                        import os

                        try:
                            os.kill(pid, 0)
                        except OSError:
                            # Process is dead, delete the stale active run file
                            run_file.unlink()
                            continue
                    active_list.append((run_file, data))

            if active_list:
                local_found = True
                print("Active Runs:")
                print("=" * 80)
                for run_file, data in active_list:
                    run_id = run_file.stem
                    task = data.get("task", "unknown")
                    config = data.get("config", "unknown")
                    pid = data.get("pid", "unknown")
                    progress = data.get("progress", "0/0")
                    active_workers = data.get("active_workers", [])

                    print(f"[Run {run_id}] - task: {task} | config: {config}")
                    print(f"PID: {pid} | Progress: {progress}")
                    if active_workers:
                        print("Active Workers:")
                        for w in active_workers:
                            print(f"  - {w}")
                    print("-" * 80)
                print("=" * 80)

        # 2. Check remote Ray jobs
        import os

        dashboard_address = _resolve_dashboard_address(os.environ.get("RAY_ADDRESS"))
        ray_queried = False
        active_jobs = []
        if _is_port_open(dashboard_address, timeout=0.2):
            try:
                client = _make_ray_job_client(dashboard_address)
                jobs = client.list_jobs()
                active_jobs = [j for j in jobs if not j.status.is_terminal()]
                ray_queried = True
            except Exception as e:
                # Surface that Ray is offline, and guide how to start it.
                print(f"\nRay Cluster: Offline (unreachable at {dashboard_address})")
                print(f"  -> {e}")
                print("  -> To start the cluster head, run:  make ray-head")
        else:
            # Surface that Ray is offline immediately without blocking.
            print(f"\nRay Cluster: Offline (unreachable at {dashboard_address})")
            print("  -> To start the cluster head, run:  make ray-head")

        if ray_queried and active_jobs:
            print("\nActive Remote Ray Jobs:")
            print("=" * 80)
            for job in active_jobs:
                print(f"[Job {job.job_id}] - status: {job.status} | entrypoint: {job.entrypoint}")
                if job.start_time:
                    import datetime

                    start_dt = datetime.datetime.fromtimestamp(job.start_time / 1000.0)
                    print(f"Started: {start_dt.isoformat()}")
                print("-" * 80)
            print("=" * 80)
        elif ray_queried and not local_found:
            print("No active training runs found.")

        # 3. Check Fleet Status (from fleet.toml)
        from kego.fleet import repo_root as find_repo_root

        live_run_ids: set[str] = set()
        polled_online: set[str] = set()
        fleet_path = find_repo_root() / "fleet.toml"
        if fleet_path.exists():
            from kego.fleet import load_fleet

            try:
                fleet = load_fleet(fleet_path)
                if fleet.machines:
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(fleet.machines)) as executor:
                        results = list(executor.map(_poll_machine, fleet.machines))

                    for res in results:
                        if res["status"] == "Online":
                            polled_online.add(res["name"])
                            for _pid, run_desc, _last_log, _eta in res.get("runs", []):
                                if len(run_desc) == 32:
                                    live_run_ids.add(run_desc)

                    def gpu_label(res: dict) -> str:
                        value = res.get("gpu", "-").replace(" GB", "G")
                        count = res.get("gpu_count", 0)
                        return f"{count} GPUs: {value}" if count > 1 else value

                    import contextlib

                    def cpu_label(res: dict) -> str:
                        load_str = res.get("load", "-")
                        cpu = load_str.split("/")[0].strip() if "/" in load_str else load_str
                        if cpu.endswith("%"):
                            with contextlib.suppress(ValueError):
                                cpu = f"{float(cpu.rstrip('%')):.1f}%"
                        return cpu

                    online = [res for res in results if res["status"] == "Online"]
                    name_width = max(15, *(len(res["name"]) for res in results))
                    cpu_width = max(6, *(len(cpu_label(res)) for res in online)) if online else 6
                    gpu_width = max(15, *(len(gpu_label(res)) for res in online)) if online else 15
                    eta_width = max(6, *(len(eta) for res in online for _, _, _, eta in res.get("runs", []) if eta), 0)
                    table_width = max(80, name_width + 1 + cpu_width + 2 + gpu_width + 1 + eta_width + 1 + 24)
                    print("\nFleet Status:")
                    print("=" * table_width)
                    print(
                        f"{'Machine':<{name_width}} {'CPU':>{cpu_width}}  {'GPU':<{gpu_width}} "
                        f"{'ETA':<{eta_width}} {'Active Runs / Latest Log'}"
                    )
                    print("-" * table_width)

                    for res in results:
                        name = res["name"][:name_width]
                        status_str = res["status"]
                        if status_str == "Online":
                            cpu = cpu_label(res)
                            gpu = gpu_label(res)
                            runs = res.get("runs", [])

                            if not runs:
                                print(
                                    f"{name:<{name_width}} {cpu:>{cpu_width}}  {gpu:<{gpu_width}} "
                                    f"{'-':<{eta_width}} None"
                                )
                            else:
                                first = True
                                for pid, run_desc, last_log, eta in runs:
                                    run_info = (
                                        f"PID {pid} ({run_desc[:8]})"
                                        if len(run_desc) == 32
                                        else f"PID {pid} ({run_desc})"
                                    )
                                    log_info = f" -> {last_log}" if last_log else ""
                                    if len(log_info) > 30:
                                        log_info = log_info[:27] + "..."
                                    eta_val = eta if eta else "-"
                                    lead = (
                                        f"{name:<{name_width}} {cpu:>{cpu_width}}  {gpu:<{gpu_width}}"
                                        if first
                                        else f"{'':<{name_width}} {'':>{cpu_width}}  {'':<{gpu_width}}"
                                    )
                                    print(f"{lead} {eta_val:<{eta_width}} {run_info}{log_info}")
                                    first = False
                        else:
                            err = res.get("error", "Offline")
                            if len(err) > 50:
                                err = err[:47] + "..."
                            print(f"{name:<{name_width}} Offline ({err})")
                    print("=" * table_width)
            except Exception as e:
                print(f"\nWarning: could not query fleet status from {fleet_path}: {e}")

        # 4. Recent hub runs. A run that crashed, was OOM-killed, or lost its machine
        # stays RUNNING in MLflow forever; cross-reference against the live PIDs polled
        # above so dead "RUNNING" rows are called out instead of silently looking healthy.
        from kego.tracking import default_tracking_uri

        uri = default_tracking_uri(fleet_path if fleet_path.exists() else None)
        if uri.startswith("http") and not _is_port_open(uri, timeout=0.5):
            print(f"\nRecent Hub Runs: unavailable (hub MLflow unreachable at {uri})")
            return
        try:
            from datetime import datetime

            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=uri)
            experiment = client.get_experiment_by_name(self.task.name)
            recent = (
                client.search_runs([experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=10)
                if experiment
                else []
            )
        except Exception as e:
            print(f"\nWarning: could not query recent runs from {uri}: {e}")
            return
        if not recent:
            return

        def _fmt_duration(ms: float) -> str:
            secs = int(ms / 1000)
            if secs < 60:
                return f"{secs}s"
            if secs < 3600:
                return f"{secs // 60}m"
            return f"{secs // 3600}h{(secs % 3600) // 60:02d}m"

        now_ms = datetime.now().timestamp() * 1000
        print(f"\nRecent Hub Runs ({self.task.name}):")
        print("=" * 100)
        print(f"{'RUN':<9} {'STATUS':<9} {'JOB':<18} {'MACHINE':<18} {'STARTED':<12} {'DURATION':<9} NOTE")
        print("-" * 100)
        stale_seen = False
        for r in recent:
            machine = r.data.tags.get("machine", "-")
            job = r.data.tags.get("job", "train")
            status = r.info.status
            note = ""
            if status == "RUNNING" and machine in polled_online and r.info.run_id not in live_run_ids:
                status = "RUNNING?"
                note = f"no live process on {machine}"
                stale_seen = True
            started = datetime.fromtimestamp(r.info.start_time / 1000).strftime("%m-%d %H:%M")
            duration = _fmt_duration((r.info.end_time or now_ms) - r.info.start_time)
            print(f"{r.info.run_id[:8]:<9} {status:<9} {job:<18} {machine:<18} {started:<12} {duration:<9} {note}")
        print("=" * 100)
        if stale_seen:
            print("RUNNING? = still RUNNING in MLflow but no matching process on its (online) machine — likely")
            print("crashed; check logs with:  kego logs <run-id>  or check dmesg for OOM kills.")

    def _submission_history(self) -> tuple[str, str | None]:
        import shutil
        import subprocess
        import sys
        import tempfile
        import zipfile
        from pathlib import Path

        kaggle_bin = Path(sys.executable).parent / "kaggle"
        cmd = ["kaggle"] if shutil.which("kaggle") else [str(kaggle_bin) if kaggle_bin.exists() else "kaggle"]
        competition = self.task.kaggle_slug
        result = subprocess.run(
            [*cmd, "competitions", "submissions", "-c", competition, "--csv"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "Kaggle submissions query failed")

        leaderboard_csv = None
        with tempfile.TemporaryDirectory() as tmp:
            leaderboard_result = subprocess.run(
                [*cmd, "competitions", "leaderboard", "-c", competition, "--download", "-p", tmp],
                capture_output=True,
                text=True,
            )
            archives = list(Path(tmp).glob("*.zip"))
            if leaderboard_result.returncode == 0 and archives:
                with zipfile.ZipFile(archives[0]) as archive:
                    csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
                    if csv_names:
                        leaderboard_csv = archive.read(csv_names[0]).decode("utf-8-sig")
        return result.stdout, leaderboard_csv

    def model_submission_stats(self) -> dict[str, dict[str, str]]:
        """Return submission status and best public rank keyed by registry version."""
        try:
            submissions_csv, leaderboard_csv = self._submission_history()
        except (OSError, RuntimeError):
            return {}
        return model_submission_stats(submissions_csv, leaderboard_csv)

    def submissions(self) -> None:
        try:
            submissions_csv, leaderboard_csv = self._submission_history()
        except (OSError, RuntimeError) as exc:
            print(f"Error querying Kaggle submissions: {exc}")
            return
        print(format_submissions(submissions_csv, leaderboard_csv=leaderboard_csv))

    def cache(self, action: str) -> None:
        from pathlib import Path

        from kego.pipeline.config import expand_grid

        # Expand active config grid
        specs = expand_grid(self.config)
        total_specs = len(specs)

        if action == "status":
            cached_count = 0
            for spec in specs:
                if self.store.has(spec.fingerprint):
                    cached_count += 1

            coverage = (cached_count / total_specs * 100) if total_specs > 0 else 0.0

            print("Cache Status:")
            print("-" * 40)
            print(f"Total specs in grid: {total_specs}")
            print(f"Cached specs: {cached_count}")
            print(f"Cache coverage: {coverage:.1f}%")

            # Print cache folder size if it exists
            local_root = getattr(getattr(self.store, "local", None), "root", None)
            if local_root and Path(local_root).exists():
                size_bytes = sum(f.stat().st_size for f in Path(local_root).glob("**/*") if f.is_file())
                size_mb = size_bytes / (1024 * 1024)
                print(f"Local cache size: {size_mb:.2f} MB ({local_root})")
            print("-" * 40)

        elif action == "prune":
            local_root = getattr(getattr(self.store, "local", None), "root", None)
            if not local_root or not Path(local_root).exists():
                print("No local cache directory found to prune.")
                return

            active_fingerprints = {spec.fingerprint for spec in specs}
            pruned_count = 0
            freed_bytes = 0

            # Delete all cached files in local_root that do not match active_fingerprints
            for f in Path(local_root).glob("*"):
                if f.is_file():
                    # The file stem is the fingerprint
                    if f.stem not in active_fingerprints:
                        freed_bytes += f.stat().st_size
                        f.unlink()
                        pruned_count += 1

            freed_mb = freed_bytes / (1024 * 1024)
            print("Cache Pruned:")
            print("-" * 40)
            print(f"Deleted: {pruned_count} files")
            print(f"Space freed: {freed_mb:.2f} MB")
            print("-" * 40)

    # -- helpers -------------------------------------------------------------

    def _build_context(self) -> TrainContext:
        raise NotImplementedError
