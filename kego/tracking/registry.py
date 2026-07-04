"""MLflow Model Registry helpers — the cross-fleet leaderboard (see the spec, §5.3)."""

from __future__ import annotations

import math


def register_checkpoint(uri: str, name: str, checkpoint_path: str, tags: dict) -> str:
    """Log ``checkpoint_path`` as an artifact and register a Model Registry version of
    ``name`` carrying ``tags`` (values coerced to strings). Uses the active MLflow run if
    one is open, else creates an ephemeral one. Returns the new version string."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    try:
        client.get_registered_model(name)
    except Exception:  # not found -> create it
        client.create_registered_model(name)

    str_tags = {k: str(v) for k, v in tags.items()}
    run = mlflow.active_run()
    owns_run = run is None
    if owns_run:
        run = mlflow.start_run()
    try:
        mlflow.log_artifact(checkpoint_path, artifact_path="checkpoint")
        source = mlflow.get_artifact_uri("checkpoint")
        version = client.create_model_version(name, source=source, run_id=run.info.run_id, tags=str_tags)
        return str(version.version)
    finally:
        if owns_run:
            mlflow.end_run()


def leaderboard(uri: str, name: str, sort_by: str = "elo", desc: bool = True) -> list[dict]:
    """Return every registered version of ``name`` as ``{"version", **tags}`` dicts, sorted
    by the numeric ``sort_by`` tag (versions missing/unparsable rank last)."""
    from datetime import datetime, timezone

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    versions = client.search_model_versions(f"name='{name}'")
    rows = []
    for v in versions:
        dt = datetime.fromtimestamp(v.creation_timestamp / 1000.0, tz=timezone.utc)
        created_str = dt.astimezone().strftime("%Y-%m-%d %H:%M")
        rows.append({"version": str(v.version), "created": created_str, **dict(v.tags)})

    def key(row: dict) -> float:
        try:
            return float(row[sort_by])
        except (KeyError, TypeError, ValueError):
            return -math.inf

    return sorted(rows, key=key, reverse=desc)


def format_leaderboard(rows: list[dict], columns: list[str]) -> str:
    """Render leaderboard ``rows`` (already sorted) as an aligned table with a rank column;
    missing cells show ``-``."""
    if not rows:
        return "(no models registered)"
    headers = ["rank", *columns]
    table = [[str(i), *(str(row.get(c, "-")) for c in columns)] for i, row in enumerate(rows, 1)]
    widths = [max(len(headers[j]), *(len(r[j]) for r in table)) for j in range(len(headers))]

    def line(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[j]) for j, c in enumerate(cells))

    return "\n".join([line(headers), *(line(r) for r in table)])
