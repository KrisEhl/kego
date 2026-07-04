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

    from ray.dashboard.modules.job.sdk import JobSubmissionClient

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
    import re
    import subprocess

    remote_script = r"""
if [ "$(uname)" = "Darwin" ]; then
    CPU_UTIL=$(top -l 1 | awk '/CPU usage/ {print $3}' | tr -d '%')
else
    CPU_UTIL=$(top -bn1 | grep -i '%Cpu(s)' | awk '{for(i=1;i<=NF;i++) if($i ~ /id/) print 100 - $(i-1)}')
fi
echo "CPU_UTIL: $CPU_UTIL%"
echo "LOAD: $(cat /proc/loadavg 2>/dev/null || sysctl -n vm.loadavg 2>/dev/null || echo 'unknown')"
echo "CORES: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || grep -c ^processor /proc/cpuinfo 2>/dev/null || echo 'unknown')"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1)"
else
    echo "GPU: N/A"
fi
echo "PROCS:"
ps -eo pid,etime,cmd 2>/dev/null | grep -E 'train-agent|train_agent' | grep -v grep || true
echo "LOGS:"
ls -dt ~/.kego/logs/*.log 2>/dev/null | head -n 5 | while read -r logpath; do
    TAIL_LINES=$(tail -n 50 "$logpath" 2>/dev/null)

    ITER_REGEX="--- Iteration ([0-9]+)/([0-9]+) ---"
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

    while IFS= read -r line; do
        if [[ "$line" =~ $ITER_REGEX ]]; then
            CURRENT_ITER="${BASH_REMATCH[1]}"
            TOTAL_ITER="${BASH_REMATCH[2]}"
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
    echo "LOG_PARSED: run_id=$RUN_ID | curr=$CURRENT_ITER | total=$TOTAL_ITER | step=$STEP | done=$DONE_MSG"
done
"""
    ssh_cmd = ["ssh", "-o", "ConnectTimeout=1", "-o", "BatchMode=yes", machine.ssh, remote_script]

    try:
        res = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=2.5)
        if res.returncode != 0:
            return {"name": machine.name, "status": "Offline", "error": res.stderr.strip() or "Connection failed"}

        load = "unknown"
        cores = "unknown"
        cpu_util = "unknown"
        gpu = "N/A"
        procs = []
        parsed_logs = {}

        lines = res.stdout.splitlines()
        mode = None
        for line in lines:
            if line.startswith("LOAD:"):
                load_parts = line.split(":", 1)[1].strip().split()
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
                    parts = [p.strip() for p in val.split(",")]
                    if len(parts) >= 3:
                        util, used_mb, total_mb = parts[:3]
                        try:
                            used_gb = float(used_mb) / 1024.0
                            total_gb = float(total_mb) / 1024.0
                            gpu = f"{util}% ({used_gb:.1f}/{total_gb:.1f} GB)"
                        except ValueError:
                            gpu = val
                    else:
                        gpu = val
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

        running_runs = []
        seen_run_ids = set()
        seen_cmds = set()
        for proc in procs:
            parts = proc.strip().split(None, 2)
            if len(parts) < 3:
                continue
            pid, etime, cmd = parts[0], parts[1], parts[2]

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

                    eta_str = ""
                    if curr and total:
                        try:
                            curr_val = int(curr)
                            total_val = int(total)
                            if curr_val > 0 and total_val > curr_val:
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
                        progress_desc = f"Iter {curr}/{total}"
                        if step:
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
        if not hasattr(self.task, "train"):
            raise NotImplementedError(f"Task '{self.task.name}' does not implement a train method.")

        from kego.pipeline.executor import RayExecutor

        if isinstance(self.executor, RayExecutor):
            import os
            import subprocess
            import time

            try:
                import ray  # noqa: F401
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
            from pathlib import Path

            # Repo root = nearest ancestor of this file containing .git (the repo
            # whose kego/ + cg/ we want to upload), independent of the caller's cwd.
            repo_root = next(
                (p for p in Path(__file__).resolve().parents if (p / ".git").exists()),
                Path.cwd(),
            )

            # Output is written to an absolute cluster path so it survives the
            # discarded working_dir and can be scp'd back.
            remote_output = f"/home/kristian/projects/kego/{output_path}" if output_path else None
            cmd = f"/home/kristian/projects/kego/.venv/bin/python -m kego.pipeline.cli train-agent --task {self.task.name}"
            if epochs:
                cmd += f" --epochs {epochs}"
            if remote_output:
                cmd += f" --output {remote_output}"

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

        self.task.train(self.config, epochs=epochs, output_path=output_path, **kwargs)

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
            if runs:
                local_found = True
                print("Active Runs:")
                print("=" * 80)
                for run_file in runs:
                    try:
                        with open(run_file) as f:
                            data = json.load(f)
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
                    except Exception:  # noqa: S110
                        pass
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
            except Exception:
                # Surface that Ray is offline, and guide how to start it.
                print(f"\nRay Cluster: Offline (unreachable at {dashboard_address})")
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
        repo_root = next((p for p in Path(__file__).resolve().parents if (p / ".git").exists()), Path.cwd())
        fleet_path = repo_root / "fleet.toml"
        if fleet_path.exists():
            from kego.fleet import load_fleet

            try:
                fleet = load_fleet(fleet_path)
                if fleet.machines:
                    print("\nFleet Status:")
                    print("=" * 132)
                    print(
                        f"{'Machine':<25} {'Status':<8} {'Role':<6} {'CPU Util/Load (Cores)':<32} {'GPU Util/Mem':<18} {'ETA':<15} {'Active Runs / Latest Log'}"
                    )
                    print("-" * 132)

                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(fleet.machines)) as executor:
                        results = list(executor.map(_poll_machine, fleet.machines))

                    for res in results:
                        name = res["name"]
                        status_str = res["status"]
                        if status_str == "Online":
                            role = res.get("role", "cpu")
                            load = res.get("load", "-")
                            gpu = res.get("gpu", "-")
                            runs = res.get("runs", [])

                            if not runs:
                                print(f"{name:<25} {status_str:<8} {role:<6} {load:<32} {gpu:<18} {'-':<15} None")
                            else:
                                first = True
                                for pid, run_desc, last_log, eta in runs:
                                    run_info = (
                                        f"PID {pid} ({run_desc[:8]}...)"
                                        if len(run_desc) == 32
                                        else f"PID {pid} ({run_desc})"
                                    )
                                    log_info = f" -> {last_log}" if last_log else ""
                                    eta_val = eta if eta else "-"
                                    if first:
                                        print(
                                            f"{name:<25} {status_str:<8} {role:<6} {load:<32} {gpu:<18} {eta_val:<15} {run_info}{log_info}"
                                        )
                                        first = False
                                    else:
                                        print(
                                            f"{'':<25} {'':<8} {'':<6} {'':<32} {'':<18} {eta_val:<15} {run_info}{log_info}"
                                        )
                        else:
                            err = res.get("error", "Offline")
                            print(f"{name:<25} {status_str:<8} {'-':<6} {'-':<32} {'-':<18} {'-':<15} ({err})")
                    print("=" * 132)
            except Exception as e:
                print(f"\nWarning: could not query fleet status from {fleet_path}: {e}")

    def submissions(self) -> None:
        import shutil
        import subprocess
        import sys
        from pathlib import Path

        def get_kaggle_cmd() -> list[str]:
            if shutil.which("kaggle"):
                return ["kaggle"]
            py_bin = Path(sys.executable).parent
            kaggle_bin = py_bin / "kaggle"
            if kaggle_bin.exists():
                return [str(kaggle_bin)]
            return ["kaggle"]

        cmd = get_kaggle_cmd()
        competition = self.task.kaggle_slug

        result = subprocess.run(
            [*cmd, "competitions", "submissions", "-c", competition],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Error querying Kaggle submissions: {result.stderr}")

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
            local_root = getattr(self.store.local, "root", None)
            if local_root and Path(local_root).exists():
                size_bytes = sum(f.stat().st_size for f in Path(local_root).glob("**/*") if f.is_file())
                size_mb = size_bytes / (1024 * 1024)
                print(f"Local cache size: {size_mb:.2f} MB ({local_root})")
            print("-" * 40)

        elif action == "prune":
            local_root = getattr(self.store.local, "root", None)
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
