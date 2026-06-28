"""Pluggable execution backend for the learner grid.

The trainer maps a training function over many :class:`LearnerSpec`s. Whether
that runs serially (debug, tests), across a Ray cluster, or in a local process
pool is hidden behind :class:`Executor`, so the core stays testable without Ray.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class Executor(Protocol):
    def map(self, fn: Callable[[T], R], items: list[T]) -> list[R]: ...


class SerialExecutor:
    """Run tasks one at a time. Default for ``--debug`` and unit tests."""

    def map(self, fn: Callable[[T], R], items: list[T]) -> list[R]:
        return [fn(item) for item in items]


class RayExecutor:
    """Fan out across a Ray cluster. ``ray`` is imported lazily."""

    def __init__(self, num_cpus: float | None = None, num_gpus: float | None = None) -> None:
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus

    def map(self, fn: Callable[[T], R], items: list[T]) -> list[R]:
        import os

        try:
            import ray
        except ImportError:
            raise ImportError(
                "Ray is not installed. Please install ray via 'pip install ray' to use the Ray executor."
            ) from None

        if not ray.is_initialized():
            # If RAY_ADDRESS is set in the environment or if auto-detection is preferred
            address = os.environ.get("RAY_ADDRESS")
            ray.init(address=address)

        options = {}
        if self.num_cpus is not None:
            options["num_cpus"] = self.num_cpus
        if self.num_gpus is not None:
            options["num_gpus"] = self.num_gpus

        if options:
            remote_fn = ray.remote(fn).options(**options)
        else:
            remote_fn = ray.remote(fn)

        futures = [remote_fn.remote(item) for item in items]
        return ray.get(futures)


def get_executor(kind: str = "serial", **kwargs) -> Executor:
    if kind == "serial":
        return SerialExecutor()
    if kind == "ray":
        return RayExecutor(**kwargs)
    raise ValueError(f"Unknown executor {kind!r}")
