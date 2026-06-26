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
        raise NotImplementedError


def get_executor(kind: str = "serial", **kwargs) -> Executor:
    if kind == "serial":
        return SerialExecutor()
    if kind == "ray":
        return RayExecutor(**kwargs)
    raise ValueError(f"Unknown executor {kind!r}")
