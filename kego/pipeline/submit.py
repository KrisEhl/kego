"""Kaggle submission: write CSV, submit, poll for the LB score, log it back.

Generic over the competition via the :class:`Task` (which supplies the slug and
submission schema). Wraps the ``kaggle`` CLI; polling logic mirrors the proven
flow in the s6e2 script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kego.pipeline.config import SubmitConfig
from kego.pipeline.task import Task


@dataclass
class SubmitResult:
    path: Path
    public_score: float | None = None
    status: str = "unknown"


class Submitter:
    def __init__(self, task: Task, config: SubmitConfig) -> None:
        self.task = task
        self.config = config

    def write(self, ids: np.ndarray, preds: np.ndarray, path: Path) -> Path:
        """Build the submission frame via the task and write it to ``path``."""
        out = self.task.make_submission(ids, preds)
        if isinstance(out, (str, Path)):
            import shutil

            shutil.copy(out, path)
            return path

        import pandas as pd

        if isinstance(out, pd.DataFrame):
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(path, index=False)
            return path

        raise TypeError(f"Unexpected submission output type: {type(out)}")

    def submit(self, path: Path, message: str) -> SubmitResult:
        """Upload via the kaggle CLI and poll until scored or timed out."""
        import csv
        import json
        import shutil
        import subprocess
        import sys
        import time

        competition = self.task.kaggle_slug

        # Get the correct kaggle command executable path
        def _get_kaggle_cmd() -> list[str]:
            if shutil.which("kaggle"):
                return ["kaggle"]
            py_bin = Path(sys.executable).parent
            kaggle_bin = py_bin / "kaggle"
            if kaggle_bin.exists():
                return [str(kaggle_bin)]
            return ["kaggle"]

        cmd = _get_kaggle_cmd()

        # Check if this is a simulation competition
        is_simulation = getattr(self.task, "is_simulation", False) or "pokemon" in competition

        if is_simulation:
            # Locate kernel directory
            kernel_dir = Path("competitions") / self.task.name / "kernel"
            if not kernel_dir.exists():
                kernel_dir = Path("competitions") / self.task.name.replace("-", "_") / "kernel"

            metadata_path = kernel_dir / "kernel-metadata.json"
            if not metadata_path.exists():
                return SubmitResult(path=path, status=f"failed: kernel-metadata.json not found at {metadata_path}")

            with open(metadata_path) as f:
                metadata = json.load(f)
            kernel_slug = metadata.get("id")
            if not kernel_slug:
                return SubmitResult(path=path, status="failed: 'id' not found in kernel-metadata.json")

            # 1. Push kernel
            print(f"Pushing Kaggle kernel from {kernel_dir}...")
            version = None
            try:
                push_res = subprocess.run(
                    [*cmd, "kernels", "push", "-p", str(kernel_dir)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                push_out = push_res.stdout.strip()
                print(push_out)

                import re

                match = re.search(r"version\s+(\d+)\s+successfully", push_out, re.IGNORECASE)
                if match:
                    version = match.group(1)
            except subprocess.CalledProcessError as e:
                return SubmitResult(path=path, status=f"failed kernel push: {e.stderr.strip()}")

            # 2. Poll kernel until complete
            print(f"Polling kernel {kernel_slug} status...")
            start_time = time.time()
            timeout = self.config.poll_timeout_s
            poll_interval = 10
            status = "queued"

            while time.time() - start_time < timeout:
                try:
                    result = subprocess.run(
                        [*cmd, "kernels", "status", kernel_slug],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    out = result.stdout.strip()
                    if 'has status "' in out:
                        parts = out.split('has status "')
                        if len(parts) > 1:
                            status = parts[1].rstrip('"').strip()
                    else:
                        status = out
                except subprocess.CalledProcessError:
                    pass

                print(f"Kernel status: {status}")
                if "complete" in status.lower() or "failed" in status.lower():
                    break
                time.sleep(poll_interval)

            if "complete" not in status.lower():
                return SubmitResult(path=path, status=f"failed: kernel run status is {status}")

            # 3. Submit the kernel output
            print(f"Submitting kernel {kernel_slug} (version {version}) output {path.name} to {competition}...")
            try:
                submit_args = [
                    *cmd,
                    "competitions",
                    "submit",
                    "-c",
                    competition,
                    "-k",
                    kernel_slug,
                    "-f",
                    path.name,
                    "-m",
                    message,
                ]
                if version:
                    submit_args += ["-v", version]

                subprocess.run(
                    submit_args,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                return SubmitResult(path=path, status=f"failed kernel submit: {e.stderr.strip()}")

        else:
            # Standard CSV submission
            try:
                subprocess.run(
                    [
                        *cmd,
                        "competitions",
                        "submit",
                        "-c",
                        competition,
                        "-f",
                        str(path),
                        "-m",
                        message,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                return SubmitResult(path=path, status=f"failed: {e.stderr.strip()}")

        # 4. Poll for submission scoring
        public_score = None
        timeout = self.config.poll_timeout_s
        poll_interval = 10
        start_time = time.time()
        status = "pending"

        while time.time() - start_time < timeout:
            time.sleep(poll_interval)
            result = subprocess.run(
                [
                    *cmd,
                    "competitions",
                    "submissions",
                    "-c",
                    competition,
                    "--csv",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                continue

            csv_lines = [l for l in result.stdout.splitlines() if not l.startswith("Warning:")]
            if not csv_lines:
                continue

            reader = csv.DictReader(csv_lines)
            for row in reader:
                status = row.get("status", "").lower()
                if "complete" in status or "successful" in status:
                    score_str = row.get("publicScore", "")
                    if score_str and score_str != "None":
                        try:
                            public_score = float(score_str)
                            status = "complete"
                        except ValueError:
                            pass
                elif "error" in status:
                    status = "error"
                break  # only check the latest submission

            if status in ("complete", "error") or public_score is not None:
                break

        return SubmitResult(path=path, public_score=public_score, status=status)
