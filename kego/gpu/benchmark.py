"""GPU benchmarking utilities: epoch timing and MLflow logging.

Requires skorch (optional dependency). Import lazily to avoid hard dependency.
"""

import time


class EpochTimer:
    """Skorch callback that records per-epoch wall time.

    Attach as a callback to a skorch NeuralNet to track training speed.

    Usage::

        from kego.gpu.benchmark import EpochTimer

        timer = EpochTimer()
        net = NeuralNetClassifier(..., callbacks=[("timer", timer)])
        net.fit(X, y)
        print(f"Avg epoch: {sum(timer.epoch_times) / len(timer.epoch_times):.2f}s")
    """

    def __init__(self) -> None:
        self.epoch_times: list[float] = []
        self._epoch_start: float | None = None

    def on_epoch_begin(self, net, **kwargs) -> None:  # type: ignore[override]
        self._epoch_start = time.time()

    def on_epoch_end(self, net, **kwargs) -> None:  # type: ignore[override]
        if self._epoch_start is not None:
            self.epoch_times.append(time.time() - self._epoch_start)

    # Skorch will call these methods as a callback without requiring inheritance.


def log_benchmark_to_mlflow(
    model_name: str,
    params: dict,
    epoch_times: list[float],
    total_time: float,
    n_samples: int,
    gpu_monitor=None,
    experiment: str = "benchmark",
    tracking_uri: str = "",
) -> None:
    """Log timing, throughput, and GPU utilization metrics to MLflow.

    Args:
        model_name: Name of the benchmarked model.
        params: Dict of hyperparameters to log (e.g. batch_size, num_workers).
        epoch_times: Per-epoch wall times in seconds (empty for non-epoch models).
        total_time: Total training wall time in seconds.
        n_samples: Number of training samples.
        gpu_monitor: Optional GPUMonitor instance with collected samples.
        experiment: MLflow experiment name.
        tracking_uri: MLflow tracking URI. Uses MLFLOW_TRACKING_URI env var if empty.
    """
    import os

    try:
        import mlflow
    except ImportError:
        return

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri:
        return

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)

    try:
        import torch

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        gpu_name = "unknown"

    run_name = f"{model_name}_" + "_".join(f"{k}{v}" for k, v in params.items())

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"model": model_name, "gpu": gpu_name, **params})

        metrics: dict[str, float] = {"total_time": total_time}

        if epoch_times:
            avg = sum(epoch_times) / len(epoch_times)
            metrics["avg_epoch_time"] = avg
            metrics["throughput_samples_per_sec"] = n_samples / avg if avg > 0 else 0.0
            for i, t in enumerate(epoch_times, 1):
                metrics[f"epoch_{i}_time"] = t
        else:
            metrics["throughput_samples_per_sec"] = (
                n_samples / total_time if total_time > 0 else 0.0
            )

        if gpu_monitor is not None and gpu_monitor.samples:
            metrics["avg_gpu_util"] = gpu_monitor.avg_gpu_util
            metrics["max_gpu_util"] = gpu_monitor.max_gpu_util
            metrics["avg_mem_util"] = gpu_monitor.avg_mem_util
            metrics["max_mem_util"] = gpu_monitor.max_mem_util
            metrics["gpu_samples"] = float(len(gpu_monitor.samples))

        mlflow.log_metrics(metrics)
