"""Crash-safe MLflow wrapper for fleet training runs.

Telemetry must never break a training run: if MLflow is missing, misconfigured, or the
store is unreachable, the tracker degrades to a silent no-op instead of raising.
"""

from __future__ import annotations


class Tracker:
    def __init__(self, active: bool) -> None:
        self._active = active

    @classmethod
    def noop(cls) -> Tracker:
        return cls(active=False)

    @classmethod
    def open(
        cls,
        uri: str,
        experiment: str,
        run_id: str | None = None,
        run_name: str | None = None,
        tags: dict | None = None,
    ) -> Tracker:
        """Open (or attach to) an MLflow run. Returns a no-op tracker if anything fails."""
        try:
            import mlflow

            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment)
            mlflow.start_run(run_id=run_id, run_name=run_name, tags=tags or {})
            return cls(active=True)
        except Exception:  # missing/unreachable MLflow -> degrade to no-op
            return cls(active=False)

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        if not self._active:
            return
        try:
            import mlflow

            mlflow.log_metric(name, value, step=step)
        except Exception:  # never let telemetry break the run
            return

    def set_tags(self, tags: dict) -> None:
        if not self._active:
            return
        try:
            import mlflow

            mlflow.set_tags(tags)
        except Exception:
            return

    def close(self) -> None:
        if not self._active:
            return
        try:
            import mlflow

            mlflow.end_run()
        except Exception:
            return

    def __enter__(self) -> Tracker:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
