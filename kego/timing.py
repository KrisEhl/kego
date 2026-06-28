"""Lightweight wall-clock timing aggregation for keeping run phases under surveillance.

Accumulate elapsed time per named label (across calls and across worker processes),
then print a sorted report. Output goes through ``print(flush=True)`` so it shows up
in Ray cluster logs.

Usage::

    from kego.timing import Timings, timer, timed, report

    t = Timings()
    with t.timer("self_play"):
        ...
    t.report("iter 1")

    # module-level default registry (handy inside worker processes):
    with timer("nn_eval"):
        model(...)

    @timed()              # label defaults to the function name
    def collect(...): ...

Worker processes get their own default registry; return ``as_dict()`` and
``merge()`` it back in the parent to aggregate.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps


class Timings:
    """Accumulates total elapsed seconds and a call count per label."""

    def __init__(self) -> None:
        self.total: dict[str, float] = {}
        self.count: dict[str, int] = {}

    def add(self, label: str, seconds: float) -> None:
        self.total[label] = self.total.get(label, 0.0) + seconds
        self.count[label] = self.count.get(label, 0) + 1

    def merge(self, other: Timings | dict[str, tuple[float, int]]) -> None:
        """Fold another registry (or its ``as_dict()``) into this one."""
        data = other.as_dict() if isinstance(other, Timings) else other
        for label, (seconds, n) in data.items():
            self.total[label] = self.total.get(label, 0.0) + seconds
            self.count[label] = self.count.get(label, 0) + n

    def as_dict(self) -> dict[str, tuple[float, int]]:
        return {label: (self.total[label], self.count[label]) for label in self.total}

    def reset(self) -> None:
        self.total.clear()
        self.count.clear()

    @contextmanager
    def timer(self, label: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(label, time.perf_counter() - start)

    def timed(self, label: str | None = None) -> Callable:
        def decorate(fn: Callable) -> Callable:
            name = label or fn.__name__

            @wraps(fn)
            def wrapper(*args, **kwargs):
                with self.timer(name):
                    return fn(*args, **kwargs)

            return wrapper

        return decorate

    def report(self, prefix: str = "timings") -> None:
        """Print one line per label, sorted by total time descending."""
        if not self.total:
            return
        grand = sum(self.total.values())
        print(f"[{prefix}] total tracked {grand:.2f}s", flush=True)
        for label, seconds in sorted(self.total.items(), key=lambda kv: kv[1], reverse=True):
            n = self.count[label]
            avg_ms = seconds / n * 1000.0 if n else 0.0
            print(f"    {label:<14} {seconds:9.2f}s  x{n:<7} avg {avg_ms:9.2f}ms", flush=True)


# Module-level default registry, convenient inside worker processes that instrument
# hot functions without threading a Timings instance through every call.
DEFAULT = Timings()


@contextmanager
def timer(label: str, into: Timings | None = None):
    with (into or DEFAULT).timer(label):
        yield


def timed(label: str | None = None, into: Timings | None = None) -> Callable:
    return (into or DEFAULT).timed(label)


def report(prefix: str = "timings", of: Timings | None = None) -> None:
    (of or DEFAULT).report(prefix)


def reset(of: Timings | None = None) -> None:
    (of or DEFAULT).reset()
