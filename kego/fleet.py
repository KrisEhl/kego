"""Fleet machine registry — parses ``fleet.toml`` (see the fleet-training-fabric spec)."""

from __future__ import annotations

import getpass
import json
import os
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib  # ty: ignore[unresolved-import]  # py3.11+ stdlib
except ImportError:
    import tomli as tomllib  # ty: ignore[unresolved-import]


def _tailscale_short(dns: str) -> str:
    """A Tailscale DNSName's short MagicDNS label: ``host.tailnet.ts.net.`` -> ``host``."""
    return dns.strip(".").split(".")[0]


def _tailscale_name() -> str | None:
    """This machine's Tailscale MagicDNS name (e.g. ``kristians-macbook-pro``), or ``None``
    if Tailscale is not installed/running. The whole fleet shares a tailnet, so this is the
    name other machines actually reach it by."""
    try:
        out = subprocess.run(
            ["tailscale", "status", "--json"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        dns = json.loads(out.stdout).get("Self", {}).get("DNSName", "")
    except Exception:
        return None
    return _tailscale_short(dns) or None


def machine_name() -> str:
    """This machine's fleet name: the ``KEGO_MACHINE`` env (set by the dispatcher), else the
    Tailscale MagicDNS name, else the OS hostname."""
    return os.environ.get("KEGO_MACHINE") or _tailscale_name() or socket.gethostname()


def git_sha(path: str | Path) -> str:
    """Short git SHA of the repo at ``path``, or ``"unknown"`` if it is not a git repo."""
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


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


def _detect_gpus() -> list[str]:
    """GPU slugs from ``nvidia-smi`` (e.g. ``rtx3090``); empty on CPU-only machines."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []
    slugs = []
    for line in out.stdout.splitlines():
        name = line.strip()
        if name:
            slugs.append(name.lower().replace("nvidia", "").replace("geforce", "").replace(" ", ""))
    return slugs


def detect_machine(repo: str | Path | None = None, gpus: list[str] | None = None) -> Machine:
    """This machine's fleet entry, auto-detected: fleet name, ``user@host`` ssh target,
    repo path (``repo`` or the cwd), and role/gpus (``gpu`` if any NVIDIA GPU is present)."""
    if gpus is None:
        gpus = _detect_gpus()
    host = _tailscale_name() or socket.gethostname()
    return Machine(
        name=machine_name(),
        ssh=f"{getpass.getuser()}@{host}",
        role="gpu" if gpus else "cpu",
        repo=str(repo) if repo is not None else str(Path.cwd()),
        gpus=tuple(gpus),
    )


def _machine_toml(m: Machine) -> str:
    """Serialize a Machine as a ``[[machine]]`` TOML block (stdlib tomllib is read-only)."""
    lines = ["", "[[machine]]", f'name = "{m.name}"', f'ssh = "{m.ssh}"', f'role = "{m.role}"', f'repo = "{m.repo}"']
    if m.data:
        lines.append(f'data = "{m.data}"')
    if m.gpus:
        lines.append("gpus = [" + ", ".join(f'"{g}"' for g in m.gpus) + "]")
    return "\n".join(lines) + "\n"


def registration_summary(machine: Machine, added: bool) -> str:
    """A human reminder of what got registered, and how to fix a wrong hostname."""
    role = machine.role + ("/" + ",".join(machine.gpus) if machine.gpus else "")
    verb = "Registered" if added else "Already registered as"
    return (
        f"{verb} '{machine.name}' -> ssh {machine.ssh} ({role}) in fleet.toml\n"
        f"Reminder: '{machine.name}' was auto-detected (Tailscale MagicDNS name if available, "
        f"else the OS hostname). If that isn't how the fleet reaches this machine, re-run with "
        f"KEGO_MACHINE set or edit the name/ssh in fleet.toml."
    )


def register_self(fleet_path: str | Path, machine: Machine | None = None) -> bool:
    """Append this machine to ``fleet.toml`` unless an entry with its name already exists.

    Returns ``True`` if a new entry was appended, ``False`` if already registered. Machines
    self-register (``make fleet-register``) instead of being hand-listed in the file.
    """
    machine = machine or detect_machine()
    if any(m.name == machine.name for m in load_fleet(fleet_path).machines):
        return False
    path = Path(fleet_path)
    path.write_text(path.read_text() + _machine_toml(machine))
    return True
