"""Crash-safe MLflow wrapper for fleet training runs.

Telemetry must never break a training run: if MLflow is missing, misconfigured, or the
store is unreachable, the tracker degrades to a silent no-op instead of raising.
"""

from __future__ import annotations


def create_run(uri: str, experiment: str, run_name: str | None = None, tags: dict | None = None) -> str | None:
    """Create an MLflow run (left RUNNING) and return its id, so a dispatched remote worker
    can attach to it via ``KEGO_MLFLOW_RUN_ID``. Returns ``None`` if MLflow is unreachable."""
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=uri)
        exp = client.get_experiment_by_name(experiment)
        exp_id = exp.experiment_id if exp else client.create_experiment(experiment)
        run_tags = dict(tags or {})
        if run_name:
            run_tags["mlflow.runName"] = run_name
        return client.create_run(exp_id, tags=run_tags).info.run_id
    except Exception:  # missing/unreachable MLflow -> caller falls back
        return None


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

    def log_params(self, params: dict[str, object]) -> None:
        if not self._active:
            return
        try:
            import mlflow

            mlflow.log_params({k: str(v) for k, v in params.items()})
        except Exception:
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
