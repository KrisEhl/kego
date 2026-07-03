"""Fleet machine registry — parses ``fleet.toml`` (see the fleet-training-fabric spec)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tomllib


@dataclass(frozen=True)
class Hub:
    name: str
    mlflow: str


@dataclass(frozen=True)
class Machine:
    name: str
    ssh: str
    role: str  # hub | gpu | cpu
    repo: str
    data: str | None = None
    gpus: tuple[str, ...] = ()


@dataclass(frozen=True)
class Fleet:
    hub: Hub
    machines: tuple[Machine, ...]

    def machine(self, name: str) -> Machine:
        for m in self.machines:
            if m.name == name:
                return m
        raise KeyError(f"no machine named {name!r} (have {[m.name for m in self.machines]})")


def load_fleet(path: str | Path) -> Fleet:
    data = tomllib.loads(Path(path).read_text())
    hub = Hub(**data["hub"])
    machines = tuple(
        Machine(
            name=m["name"],
            ssh=m["ssh"],
            role=m["role"],
            repo=m["repo"],
            data=m.get("data"),
            gpus=tuple(m.get("gpus", ())),
        )
        for m in data.get("machine", [])
    )
    return Fleet(hub=hub, machines=machines)
