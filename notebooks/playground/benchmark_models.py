"""Benchmark neural models on S6E2 dataset.

Standalone script (no Ray) that trains a single model for N epochs,
reports timing, and logs everything to MLflow under the "benchmark" experiment.

Usage:
    python benchmark_models.py --model resnet --epochs 3
    python benchmark_models.py --model ft_transformer --batch-size 8192
    python benchmark_models.py --model realmlp --num-workers 2
    python benchmark_models.py --model resnet --profile
"""

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from train_s6e2_baseline import (  # noqa: E402
    CAT_FEATURES,
    DATA_DIR,
    TARGET,
    TE_FEATURES,
    ScaledRealMLP,
    SkorchFTTransformer,
    SkorchResNet,
    _engineer_features,
    _impute_cholesterol,
    get_models,
    make_te_preprocess,
)

from kego.datasets.split import split_dataset  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CHOICES = ["realmlp", "resnet", "ft_transformer"]


def _prepare_data(sample=None):
    """Load, preprocess, and split S6E2 data.

    Args:
        sample: If set, subsample to this many rows after splitting.
    """
    train_full = pd.read_csv(DATA_DIR / "train.csv")
    original = pd.read_csv(DATA_DIR / "Heart_Disease_Prediction.csv")

    train_full[TARGET] = train_full[TARGET].map({"Presence": 1, "Absence": 0})
    original[TARGET] = original[TARGET].map({"Presence": 1, "Absence": 0})
    original["id"] = -1
    train_full = pd.concat([train_full, original], ignore_index=True)

    train, holdout, _ = split_dataset(
        train_full, train_size=0.8, validate_size=0.2, stratify_column=TARGET
    )
    train = train.reset_index(drop=True)
    holdout = holdout.reset_index(drop=True)

    if sample is not None and sample < len(train):
        train = train.sample(n=sample, random_state=42).reset_index(drop=True)
        holdout = holdout.sample(
            n=min(sample // 4, len(holdout)), random_state=42
        ).reset_index(drop=True)
        logger.info(f"Subsampled to {len(train)} train / {len(holdout)} holdout rows")

    train = _impute_cholesterol(train)
    holdout = _impute_cholesterol(holdout)
    train = _engineer_features(train)
    holdout = _engineer_features(holdout)

    features = [c for c in train.columns if c not in ["id", TARGET]]

    # Apply target encoding
    te_preprocess = make_te_preprocess(TE_FEATURES)
    train, holdout, _, _ = te_preprocess(
        train, train[TARGET], holdout, holdout.copy(), holdout.copy()
    )
    features = [c for c in train.columns if c not in ["id", TARGET]]

    return train, holdout, features


def _build_model(model_name, models_config, batch_size, num_workers, epochs, compile):
    """Instantiate a model with overridden batch_size/num_workers/epochs."""
    config = models_config[model_name]
    kwargs = config["kwargs"].copy()

    if model_name == "realmlp":
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        if epochs is not None:
            kwargs["n_epochs"] = epochs
        return ScaledRealMLP(**kwargs)

    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if num_workers is not None:
        kwargs["num_workers"] = num_workers
    if epochs is not None:
        kwargs["max_epochs"] = epochs
    # Disable early stopping for benchmarks — train full epochs
    kwargs["patience"] = epochs or kwargs.get("max_epochs", 200)

    if model_name == "resnet":
        model = SkorchResNet(**kwargs)
        model._compile = compile
        return model
    elif model_name == "ft_transformer":
        model = SkorchFTTransformer(**kwargs)
        model._compile = compile
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")


class EpochTimer:
    """Skorch-compatible callback that records per-epoch wall time."""

    def __init__(self):
        self.epoch_times = []
        self._epoch_start = None

    def on_epoch_begin(self, net, **kwargs):
        self._epoch_start = time.time()

    def on_epoch_end(self, net, **kwargs):
        if self._epoch_start is not None:
            self.epoch_times.append(time.time() - self._epoch_start)


def _resolve_smi_index(cuda_index=0):
    """Map PyTorch CUDA device index to nvidia-smi device index via UUID."""
    if not torch.cuda.is_available():
        return cuda_index
    try:
        # Get UUID from PyTorch (format: bytes, no "GPU-" prefix)
        props = torch.cuda.get_device_properties(cuda_index)
        cuda_uuid = str(props.uuid) if hasattr(props, "uuid") else None
        if cuda_uuid is None:
            return cuda_index

        # Query nvidia-smi for index,uuid mapping
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return cuda_index

        for line in result.stdout.strip().split("\n"):
            parts = line.split(",")
            smi_idx = int(parts[0].strip())
            smi_uuid = parts[1].strip()
            # PyTorch uuid may or may not have "GPU-" prefix
            if cuda_uuid in smi_uuid or smi_uuid in cuda_uuid:
                return smi_idx
    except Exception:
        pass
    return cuda_index


class GPUMonitor:
    """Sample GPU utilization via nvidia-smi in a background thread."""

    def __init__(self, device_index=0, interval=1.0):
        self.device_index = _resolve_smi_index(device_index)
        self.interval = interval
        self.samples = []
        self.mem_samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _poll(self):
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.device_index}",
                        "--query-gpu=utilization.gpu,utilization.memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(",")
                    self.samples.append(float(parts[0].strip()))
                    self.mem_samples.append(float(parts[1].strip()))
            except Exception:
                pass
            self._stop.wait(self.interval)

    @property
    def avg_gpu_util(self):
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def max_gpu_util(self):
        return max(self.samples) if self.samples else 0.0

    @property
    def avg_mem_util(self):
        return (
            sum(self.mem_samples) / len(self.mem_samples) if self.mem_samples else 0.0
        )

    @property
    def max_mem_util(self):
        return max(self.mem_samples) if self.mem_samples else 0.0


def _patch_compile(model, compile_flag):
    """Monkey-patch model.fit to control torch.compile behavior."""
    original_fit = model.fit

    def patched_fit(X, y, **kwargs):
        # Temporarily override the compile step inside fit
        model._compile = compile_flag
        return original_fit(X, y, **kwargs)

    model.fit = patched_fit


def _benchmark_skorch(model, model_name, train, holdout, features, profile):
    """Benchmark a skorch-based model (resnet, ft_transformer)."""
    timer = EpochTimer()
    compile_flag = getattr(model, "_compile", True)

    # Prepare data the same way the model's fit() does
    if model_name == "resnet":
        te_preprocess = make_te_preprocess(TE_FEATURES, drop_original=True)
        train_prep, holdout_prep, _, _ = te_preprocess(
            train.copy(),
            train[TARGET],
            holdout.copy(),
            holdout.copy(),
            holdout.copy(),
        )
        feat = [c for c in train_prep.columns if c not in ["id", TARGET]]
        X_train = train_prep[feat].values.astype(np.float32)
    else:
        feat = features
        X_train = train[feat]

    y_train = train[TARGET].values.astype(np.float32)

    # Build the skorch net manually so we can inject the timer callback
    torch.manual_seed(model.random_state)

    if model_name == "resnet":
        import torch.nn as nn
        from sklearn.preprocessing import QuantileTransformer
        from skorch.callbacks import EarlyStopping
        from train_s6e2_baseline import (
            AMPNeuralNetBinaryClassifier,
            ResNetModule,
        )

        scaler = QuantileTransformer(
            output_distribution="normal", random_state=model.random_state
        )
        X_np = scaler.fit_transform(X_train).astype(np.float32)
        d_in = X_np.shape[1]

        net = AMPNeuralNetBinaryClassifier(
            ResNetModule,
            module__d_in=d_in,
            module__d_out=1,
            module__n_blocks=model.n_blocks,
            module__d_block=model.d_block,
            module__d_hidden_multiplier=model.d_hidden_multiplier,
            module__dropout1=model.dropout1,
            module__dropout2=model.dropout2,
            criterion=nn.BCEWithLogitsLoss,
            optimizer=torch.optim.AdamW,
            lr=model.lr,
            max_epochs=model.max_epochs,
            batch_size=model.batch_size,
            device="cuda",
            iterator_train__pin_memory=True,
            iterator_valid__pin_memory=True,
            iterator_train__num_workers=model.num_workers,
            iterator_valid__num_workers=model.num_workers,
            callbacks=[
                EarlyStopping(patience=model.patience, monitor="valid_loss"),
                ("epoch_timer", timer),
            ],
            verbose=1,
        )
        net.initialize()
        if compile_flag:
            net.module_ = torch.compile(net.module_)

        X_fit, y_fit = X_np, y_train

    elif model_name == "ft_transformer":
        import torch.nn as nn
        from skorch.callbacks import EarlyStopping
        from train_s6e2_baseline import (
            AMPNeuralNetBinaryClassifier,
            FTTransformerModule,
        )

        X_prep = model._prepare(train[feat], fit=True)
        n_cont = len(model.cont_cols) if model.cont_cols else X_prep.shape[1]

        net = AMPNeuralNetBinaryClassifier(
            FTTransformerModule,
            module__n_cont_features=n_cont,
            module__cat_cardinalities=model.cat_cardinalities,
            module__d_out=1,
            module__n_blocks=model.n_blocks,
            module__d_block=model.d_block,
            module__attention_n_heads=model.attention_n_heads,
            module__attention_dropout=model.attention_dropout,
            module__ffn_d_hidden_multiplier=model.ffn_d_hidden_multiplier,
            module__ffn_dropout=model.ffn_dropout,
            module__residual_dropout=model.residual_dropout,
            criterion=nn.BCEWithLogitsLoss,
            optimizer=torch.optim.AdamW,
            lr=model.lr,
            max_epochs=model.max_epochs,
            batch_size=model.batch_size,
            device="cuda",
            iterator_train__pin_memory=True,
            iterator_valid__pin_memory=True,
            iterator_train__num_workers=model.num_workers,
            iterator_valid__num_workers=model.num_workers,
            callbacks=[
                EarlyStopping(patience=model.patience, monitor="valid_loss"),
                ("epoch_timer", timer),
            ],
            verbose=1,
        )
        net.initialize()
        if compile_flag:
            net.module_ = torch.compile(net.module_)

        X_fit, y_fit = X_prep, y_train

    n_samples = len(X_fit)
    logger.info(f"Training {model_name} on {n_samples} samples...")

    gpu_monitor = GPUMonitor()
    gpu_monitor.start()

    if profile:
        trace_path = Path(__file__).parent / f"trace_{model_name}.json"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            t0 = time.time()
            net.fit(X_fit, y_fit)
            total_time = time.time() - t0

        prof.export_chrome_trace(str(trace_path))
        logger.info(f"Chrome trace saved to {trace_path}")
        logger.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    else:
        t0 = time.time()
        net.fit(X_fit, y_fit)
        total_time = time.time() - t0

    gpu_monitor.stop()
    return timer.epoch_times, total_time, n_samples, gpu_monitor


def _benchmark_realmlp(model, train, features, profile):
    """Benchmark RealMLP (no skorch, just time the full fit call)."""
    X_train = train[features]
    y_train = train[TARGET]
    n_samples = len(X_train)
    logger.info(f"Training realmlp on {n_samples} samples...")

    gpu_monitor = GPUMonitor()
    gpu_monitor.start()

    if profile:
        trace_path = Path(__file__).parent / "trace_realmlp.json"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            t0 = time.time()
            model.fit(X_train, y_train)
            total_time = time.time() - t0

        prof.export_chrome_trace(str(trace_path))
        logger.info(f"Chrome trace saved to {trace_path}")
        logger.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    else:
        t0 = time.time()
        model.fit(X_train, y_train)
        total_time = time.time() - t0

    gpu_monitor.stop()
    return [], total_time, n_samples, gpu_monitor


def _log_to_mlflow(
    model_name,
    batch_size,
    num_workers,
    compile_flag,
    epochs,
    epoch_times,
    total_time,
    n_samples,
    gpu_monitor,
):
    """Log benchmark results to MLflow."""
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed, skipping logging")
        return

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        logger.warning("MLFLOW_TRACKING_URI not set, skipping MLflow logging")
        return

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("benchmark")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    run_name = f"{model_name}_{batch_size}bs_{num_workers}w"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model": model_name,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "compile": compile_flag,
                "epochs": epochs,
                "gpu": gpu_name,
            }
        )

        metrics = {"total_time": total_time}

        if epoch_times:
            avg_epoch = sum(epoch_times) / len(epoch_times)
            throughput = n_samples / avg_epoch if avg_epoch > 0 else 0
            metrics["avg_epoch_time"] = avg_epoch
            metrics["throughput_samples_per_sec"] = throughput
            for i, t in enumerate(epoch_times, 1):
                metrics[f"epoch_{i}_time"] = t
        else:
            # RealMLP: no per-epoch times
            throughput = n_samples / total_time if total_time > 0 else 0
            metrics["throughput_samples_per_sec"] = throughput

        if gpu_monitor.samples:
            metrics["avg_gpu_util"] = gpu_monitor.avg_gpu_util
            metrics["max_gpu_util"] = gpu_monitor.max_gpu_util
            metrics["avg_mem_util"] = gpu_monitor.avg_mem_util
            metrics["max_mem_util"] = gpu_monitor.max_mem_util
            metrics["gpu_samples"] = len(gpu_monitor.samples)

        mlflow.log_metrics(metrics)

    logger.info(f"MLflow: logged run '{run_name}' to experiment 'benchmark'")


def main():
    parser = argparse.ArgumentParser(description="Benchmark neural models on S6E2")
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_CHOICES,
        help="Model to benchmark",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override default batch size"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None, help="DataLoader num_workers"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile()",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable PyTorch profiler and export Chrome trace",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Subsample training data to N rows (for quick debug runs)",
    )
    args = parser.parse_args()

    compile_flag = not args.no_compile
    model_name = args.model

    logger.info(
        f"Benchmark: model={model_name}, batch_size={args.batch_size}, "
        f"num_workers={args.num_workers}, epochs={args.epochs}, "
        f"compile={compile_flag}, profile={args.profile}"
    )

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No CUDA GPU available, running on CPU")

    # Load and preprocess data
    train, holdout, features = _prepare_data(sample=args.sample)
    n_features = len(features)

    # Get default model configs
    models_config = get_models(n_features, neural=True)

    # Build model with overrides
    model = _build_model(
        model_name,
        models_config,
        args.batch_size,
        args.num_workers,
        args.epochs,
        compile_flag,
    )

    # Resolve effective batch_size/num_workers for logging
    if model_name == "realmlp":
        effective_bs = models_config[model_name]["kwargs"].get("batch_size", "default")
        effective_nw = 0
    else:
        effective_bs = model.batch_size
        effective_nw = model.num_workers

    if args.batch_size is not None:
        effective_bs = args.batch_size
    if args.num_workers is not None:
        effective_nw = args.num_workers

    # Run benchmark
    if model_name == "realmlp":
        epoch_times, total_time, n_samples, gpu_monitor = _benchmark_realmlp(
            model, train, features, args.profile
        )
    else:
        epoch_times, total_time, n_samples, gpu_monitor = _benchmark_skorch(
            model, model_name, train, holdout, features, args.profile
        )

    # Report results
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    logger.info(f"\n{'='*50}")
    logger.info(f"Model: {model_name}")
    logger.info(f"GPU: {gpu_name}")
    logger.info(f"Total time: {total_time:.2f}s")
    if epoch_times:
        avg = sum(epoch_times) / len(epoch_times)
        logger.info(f"Epochs trained: {len(epoch_times)}")
        logger.info(f"Avg epoch time: {avg:.3f}s")
        logger.info(f"Throughput: {n_samples / avg:.0f} samples/sec")
        for i, t in enumerate(epoch_times, 1):
            logger.info(f"  Epoch {i}: {t:.3f}s")
    else:
        logger.info(f"Throughput: {n_samples / total_time:.0f} samples/sec")
    if gpu_monitor.samples:
        logger.info(
            f"GPU utilization: avg={gpu_monitor.avg_gpu_util:.1f}%, "
            f"max={gpu_monitor.max_gpu_util:.1f}% "
            f"({len(gpu_monitor.samples)} samples)"
        )
        logger.info(
            f"GPU memory util: avg={gpu_monitor.avg_mem_util:.1f}%, "
            f"max={gpu_monitor.max_mem_util:.1f}%"
        )
    logger.info(f"{'='*50}")

    # Log to MLflow
    _log_to_mlflow(
        model_name,
        effective_bs,
        effective_nw,
        compile_flag,
        args.epochs,
        epoch_times,
        total_time,
        n_samples,
        gpu_monitor,
    )


if __name__ == "__main__":
    main()
