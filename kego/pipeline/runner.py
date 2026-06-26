"""Pipeline orchestration: the verbs ``run`` / ``ensemble`` / ``tune`` / ``submit``.

Wires task + feature store + trainer (+ implicit cache) + ensembler + evaluator
+ submitter together. Each verb is a thin sequence over the stage objects; all
the reusable logic lives in those stages.
"""

from __future__ import annotations

from dataclasses import dataclass

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

        active_runs_dir = Path(".kego/active_runs")
        if not active_runs_dir.exists():
            print("No active training runs found.")
            return

        runs = list(active_runs_dir.glob("*.json"))
        if not runs:
            print("No active training runs found.")
            return

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
