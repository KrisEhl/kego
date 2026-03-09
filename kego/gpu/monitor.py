"""GPU utilization monitoring via nvidia-smi."""

import subprocess
import threading


def resolve_smi_index(cuda_index: int = 0) -> int:
    """Map a PyTorch CUDA device index to its nvidia-smi device index via UUID.

    PyTorch and nvidia-smi may use different device orderings when
    CUDA_VISIBLE_DEVICES is set. This function resolves the correct index
    by matching UUIDs.

    Args:
        cuda_index: PyTorch CUDA device index (e.g. 0 for torch.device("cuda:0")).

    Returns:
        Corresponding nvidia-smi device index. Falls back to cuda_index on error.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return cuda_index
        props = torch.cuda.get_device_properties(cuda_index)
        cuda_uuid = str(props.uuid) if hasattr(props, "uuid") else None
        if cuda_uuid is None:
            return cuda_index

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
            if cuda_uuid in smi_uuid or smi_uuid in cuda_uuid:
                return smi_idx
    except Exception:
        pass
    return cuda_index


class GPUMonitor:
    """Sample GPU utilization via nvidia-smi in a background thread.

    Usage::

        monitor = GPUMonitor(device_index=0, interval=1.0)
        monitor.start()
        # ... run workload ...
        monitor.stop()
        print(f"Avg GPU util: {monitor.avg_gpu_util:.1f}%")

    Args:
        device_index: PyTorch CUDA device index (resolved to nvidia-smi index internally).
        interval: Polling interval in seconds.
    """

    def __init__(self, device_index: int = 0, interval: float = 1.0):
        self.device_index = resolve_smi_index(device_index)
        self.interval = interval
        self.samples: list[float] = []
        self.mem_samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start background polling thread."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background polling thread and wait for it to finish."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _poll(self) -> None:
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
    def avg_gpu_util(self) -> float:
        """Average GPU compute utilization (%) across all samples."""
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def max_gpu_util(self) -> float:
        """Peak GPU compute utilization (%) across all samples."""
        return max(self.samples) if self.samples else 0.0

    @property
    def avg_mem_util(self) -> float:
        """Average GPU memory bandwidth utilization (%) across all samples."""
        return (
            sum(self.mem_samples) / len(self.mem_samples) if self.mem_samples else 0.0
        )

    @property
    def max_mem_util(self) -> float:
        """Peak GPU memory bandwidth utilization (%) across all samples."""
        return max(self.mem_samples) if self.mem_samples else 0.0
