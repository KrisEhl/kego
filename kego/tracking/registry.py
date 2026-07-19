"""MLflow Model Registry helpers — the cross-fleet leaderboard (see the spec, §5.3)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingResume:
    version: str
    completed_iterations: int
    path: Path


def register_checkpoint(
    uri: str,
    name: str,
    checkpoint_path: str,
    tags: dict,
    *,
    training_state_path: str | None = None,
) -> str:
    """Log ``checkpoint_path`` as an artifact and register a Model Registry version of
    ``name`` carrying ``tags`` (values coerced to strings). Uses the active MLflow run if
    one is open, else creates an ephemeral one. Returns the new version string."""
    import mlflow
    from mlflow.tracking import MlflowClient

    from kego.tracking.tracker import fail_fast_http

    fail_fast_http()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    try:
        client.get_registered_model(name)
    except Exception:  # not found -> create it
        client.create_registered_model(name)

    str_tags = {**{k: str(v) for k, v in tags.items()}, "checkpoint_filename": Path(checkpoint_path).name}
    if training_state_path:
        str_tags["training_state_filename"] = Path(training_state_path).name
    run = mlflow.active_run()
    owns_run = run is None
    if owns_run:
        mlflow.set_experiment("Default")
        run = mlflow.start_run()
    try:
        mlflow.log_artifact(checkpoint_path, artifact_path="checkpoint")
        if training_state_path:
            mlflow.log_artifact(training_state_path, artifact_path="checkpoint")
        source = mlflow.get_artifact_uri("checkpoint")
        version = client.create_model_version(name, source=source, run_id=run.info.run_id, tags=str_tags)
        return str(version.version)
    finally:
        if owns_run:
            mlflow.end_run()


def resolve_training_resume(
    uri: str,
    name: str,
    fingerprint: str,
    target_iterations: int,
    cache_dir: str | Path,
) -> TrainingResume | None:
    """Download the most advanced compatible training state at or below the target."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    try:
        versions = client.search_model_versions(f"name='{name}'")
    except Exception:
        return None

    candidates = []
    for version in versions:
        tags = dict(version.tags or {})
        if tags.get("training_fingerprint") != fingerprint or not tags.get("training_state_filename"):
            continue
        if getattr(version, "current_stage", None) == "Archived":
            continue
        if tags.get("dropped") == "true" or tags.get("status") == "archived":
            continue
        if not version.run_id:
            continue
        try:
            completed = int(tags["completed_iterations"])
        except (KeyError, TypeError, ValueError):
            continue
        if completed <= target_iterations:
            candidates.append((completed, version, tags["training_state_filename"], version.run_id))
    if not candidates:
        return None

    completed, version, filename, run_id = max(candidates, key=lambda candidate: candidate[0])
    downloaded = Path(client.download_artifacts(run_id, "checkpoint", dst_path=str(cache_dir)))
    paths = [downloaded] if downloaded.is_file() else list(downloaded.rglob(filename))
    if not paths:
        raise FileNotFoundError(f"Training state {filename!r} is missing from registry:{version.version}")
    return TrainingResume(version=str(version.version), completed_iterations=completed, path=paths[0])


def leaderboard(uri: str, name: str, sort_by: str = "elo", desc: bool = True) -> list[dict]:
    """Return every registered version of ``name`` as ``{"version", **tags}`` dicts, sorted
    by the numeric ``sort_by`` tag (versions missing/unparsable rank last)."""
    from datetime import datetime, timezone

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    versions = client.search_model_versions(f"name='{name}'")
    rows = []
    for v in versions:
        if getattr(v, "current_stage", None) == "Archived":
            continue
        if v.tags and (v.tags.get("dropped") == "true" or v.tags.get("status") == "archived"):
            continue
        dt = datetime.fromtimestamp(v.creation_timestamp / 1000.0, tz=timezone.utc)
        created_str = dt.astimezone().strftime("%Y-%m-%d %H:%M")
        rows.append({"version": str(v.version), "run_id": v.run_id, "created": created_str, **dict(v.tags)})

    def key(row: dict) -> float:
        try:
            return float(row[sort_by])
        except (KeyError, TypeError, ValueError):
            return -math.inf

    return sorted(rows, key=key, reverse=desc)


def read_ratings(uri: str, name: str) -> dict[str, dict]:
    """version -> {"elo", "elo_rd", "games"} for versions already carrying an ``elo`` tag."""
    out: dict[str, dict] = {}
    for row in leaderboard(uri, name, sort_by="version"):
        if "elo" not in row:
            continue
        out[row["version"]] = {
            "elo": float(row["elo"]),
            "elo_rd": float(row.get("elo_rd", 350.0)),
            "games": int(float(row.get("games", 0))),
        }
    return out


def write_ratings(uri: str, name: str, ratings: dict[str, dict]) -> None:
    """Write ``elo``/``elo_rd``/``games``/``rating_status`` tags onto each registry version."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    for version, r in ratings.items():
        client.set_model_version_tag(name, str(version), "elo", str(round(r["elo"], 1)))
        client.set_model_version_tag(name, str(version), "elo_rd", str(round(r["elo_rd"], 1)))
        client.set_model_version_tag(name, str(version), "games", str(int(r["games"])))
        client.set_model_version_tag(name, str(version), "rating_status", "rated")


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _ansi_elo(value: str, elo_rd: Any) -> str:
    try:
        rd = float(elo_rd)
    except (TypeError, ValueError):
        return value
    color = "32" if rd <= 1 else "33" if rd <= 5 else "31"
    return f"\x1b[{color}m{value}\x1b[0m"


def _visible_len(value: str) -> int:
    return len(ANSI_RE.sub("", value))


def format_leaderboard(
    rows: list[dict],
    columns: list[str],
    max_widths: dict[str, int] | None = None,
    color_elo: bool = False,
) -> str:
    """Render leaderboard ``rows`` (already sorted) as an aligned table with a rank column;
    missing cells show ``-``."""
    if not rows:
        return "(no models registered)"
    headers = ["rank", *columns]
    max_widths = max_widths or {}

    def cell(row: dict, column: str, value: object) -> str:
        text = str(value)
        width = max_widths.get(column)
        text = text if width is None or len(text) <= width else text[:width]
        return _ansi_elo(text, row.get("elo_rd")) if color_elo and column == "elo" else text

    table = [[str(i), *(cell(row, c, row.get(c, "-")) for c in columns)] for i, row in enumerate(rows, 1)]
    widths = [max(len(headers[j]), *(_visible_len(r[j]) for r in table)) for j in range(len(headers))]

    def line(cells: list[str]) -> str:
        return "  ".join(c + " " * (widths[j] - _visible_len(c)) for j, c in enumerate(cells))

    return "\n".join([line(headers), *(line(r) for r in table)])
