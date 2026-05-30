"""Shared Ray dashboard helpers (HTTP API, no ray binary needed)."""

from __future__ import annotations

import json
import urllib.request


def job_statuses(ray_address: str) -> dict[str, str]:
    """Map Ray submission_id → status. Empty dict if Ray is unreachable."""
    url = ray_address.rstrip("/") + "/api/jobs/"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            jobs = json.loads(resp.read())
    except Exception:
        return {}
    return {j["submission_id"]: j.get("status", "") for j in jobs if j.get("submission_id")}
