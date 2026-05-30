"""kego status — show Ray job queue with run names and statuses."""

from __future__ import annotations

import argparse
import datetime
import json
import urllib.request


def _fetch_jobs(ray_address: str) -> list[dict]:
    url = ray_address.rstrip("/") + "/api/jobs/"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            return json.loads(resp.read())
    except Exception as e:
        raise RuntimeError(f"Cannot reach Ray at {ray_address}: {e}") from e


def _ago(ms: int | None) -> str:
    if not ms:
        return "?"
    delta = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.datetime.fromtimestamp(
        ms / 1000, tz=datetime.timezone.utc
    )
    hours = int(delta.total_seconds() // 3600)
    if hours >= 24:
        return (
            datetime.datetime.fromtimestamp(ms / 1000, tz=datetime.timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
        )
    if hours > 0:
        return f"{hours}h"
    return f"{int(delta.total_seconds() // 60)}m"


def _short_error(job: dict) -> str:
    msg = job.get("message", "")
    if not msg:
        return ""
    # grab the most useful line: first non-empty line after "last available logs"
    for line in reversed(msg.splitlines()):
        line = line.strip()
        if line and not line.startswith("20") and "INFO" not in line and "Running entrypoint" not in line:
            return line[:60]
    return ""


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "status",
        help="Show Ray job queue with run names and statuses",
        description="List Ray jobs on the cluster, newest first, with kego run names and fold numbers.",
        epilog=("Examples:\n  uv run kego status\n  uv run kego status --all\n  uv run kego status --status running\n"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--status",
        choices=["pending", "running", "succeeded", "failed", "stopped"],
        metavar="STATUS",
        help="Filter by Ray job status: pending, running, succeeded, failed, stopped",
    )
    p.add_argument(
        "--all",
        action="store_true",
        dest="show_all",
        help="Show all jobs (default: last 20)",
    )
    p.set_defaults(func=_status)


def _status(args: argparse.Namespace, extra_args: list[str]) -> int:
    from kego.cli import config as cfg_module

    config = cfg_module.load_config()

    try:
        jobs = _fetch_jobs(config.cluster.ray_address)
    except RuntimeError as e:
        print(str(e))
        return 1

    # Sort newest first
    jobs.sort(key=lambda j: j.get("start_time") or 0, reverse=True)

    if args.status:
        jobs = [j for j in jobs if j.get("status", "").lower() == args.status]

    if not args.show_all:
        jobs = jobs[:20]

    if not jobs:
        print("No jobs found.")
        return 0

    header = f"{'ID':<8} {'NAME':<26} {'FOLD':<6} {'STATUS':<12} {'AGO':<18} {'NOTE'}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for job in jobs:
        env = (job.get("runtime_env") or {}).get("env_vars", {})
        kego_id = env.get("KEGO_EXPERIMENT_ID", "?")[:6]
        name = env.get("KEGO_RUN_NAME", "?")[:26]
        cli_params = json.loads(env.get("KEGO_CLI_PARAMS", "{}"))
        fold = str(cli_params.get("fold", "—"))
        status = job.get("status", "?")
        ago = _ago(job.get("start_time"))
        note = ""
        if status == "FAILED":
            note = _short_error(job)
        elif status == "RUNNING":
            note = job.get("submission_id", "")[-8:]
        print(f"{kego_id:<8} {name:<26} {fold:<6} {status:<12} {ago:<18} {note}")

    print(sep)
    total = len(jobs)
    counts = {}
    for j in jobs:
        s = j.get("status", "?")
        counts[s] = counts.get(s, 0) + 1
    summary = "  ".join(f"{s}: {n}" for s, n in sorted(counts.items()))
    print(f"  {total} job(s)  {summary}")

    return 0
