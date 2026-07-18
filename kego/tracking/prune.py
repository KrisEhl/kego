"""Retire / restore model-registry versions (manual list or statistical drop-worse)."""

from __future__ import annotations

from typing import Any


def filter_worse_versions(versions: list[Any], min_games: int = 20, k: float = 1.96) -> tuple[list[Any], list[Any]]:
    """Group versions by (deck, model_args) and keep:
    1. Unrated or insufficiently rated versions (played < min_games).
    2. Rated versions that are NOT statistically significantly worse than the best rated version.

    A version A is statistically significantly better than B if:
        Elo_A - Elo_B > k * (rd_A + rd_B)
    """
    if not versions:
        return [], []

    groups: dict[Any, list[Any]] = {}
    for v in versions:
        tags = v.tags if getattr(v, "tags", None) else None
        variant = tags.get("variant") if tags else None
        groups.setdefault(variant, []).append(v)

    filtered: list[Any] = []
    dropped: list[Any] = []

    for group in groups.values():

        def parse_version_stats(v_obj: Any) -> tuple[float, float, int]:
            tags = v_obj.tags if getattr(v_obj, "tags", None) else {}
            try:
                elo = float(tags.get("elo", "-inf"))
            except (TypeError, ValueError):
                elo = -float("inf")
            try:
                rd = float(tags.get("elo_rd", 350.0))
            except (TypeError, ValueError):
                rd = 350.0
            try:
                games = int(float(tags.get("games", 0)))
            except (TypeError, ValueError):
                games = 0
            return elo, rd, games

        rated_group = []
        unrated_group = []
        for v in group:
            elo, rd, games = parse_version_stats(v)
            if games >= min_games and elo != -float("inf"):
                rated_group.append((v, elo, rd))
            else:
                unrated_group.append(v)

        filtered.extend(unrated_group)

        if not rated_group:
            continue

        best_rated_v, best_elo, best_rd = max(rated_group, key=lambda item: item[1])
        filtered.append(best_rated_v)

        for v, elo, rd in rated_group:
            if v.version == best_rated_v.version:
                continue
            if best_elo - elo > k * (best_rd + rd):
                dropped.append(v)
            else:
                filtered.append(v)

    seen_versions: set[str] = set()
    unique_filtered: list[Any] = []
    for v in filtered:
        if v.version not in seen_versions:
            seen_versions.add(v.version)
            unique_filtered.append(v)

    seen_dropped: set[str] = set()
    unique_dropped: list[Any] = []
    for v in dropped:
        if v.version not in seen_versions and v.version not in seen_dropped:
            seen_dropped.add(v.version)
            unique_dropped.append(v)

    def get_version_num(v_obj: Any) -> int:
        try:
            return int(v_obj.version)
        except (TypeError, ValueError):
            return 0

    return sorted(unique_filtered, key=get_version_num), sorted(unique_dropped, key=get_version_num)


def archive_version(client: Any, name: str, version: str, *, dropped: bool = False) -> None:
    """Archive a model version and mark it inactive in tags."""
    client.transition_model_version_stage(name=name, version=str(version), stage="Archived")
    client.set_model_version_tag(name=name, version=str(version), key="status", value="archived")
    if dropped:
        client.set_model_version_tag(name=name, version=str(version), key="dropped", value="true")


def unarchive_version(client: Any, name: str, version: str) -> None:
    """Restore an archived model version to the active pool."""
    client.transition_model_version_stage(name=name, version=str(version), stage="None")
    client.set_model_version_tag(name=name, version=str(version), key="status", value="active")
    client.set_model_version_tag(name=name, version=str(version), key="dropped", value="false")


def active_model_versions(client: Any, name: str) -> list[Any]:
    """Return non-archived / non-dropped registry versions for ``name``."""
    versions = []
    for v in client.search_model_versions(f"name='{name}'"):
        if getattr(v, "current_stage", None) == "Archived":
            continue
        tags = v.tags or {}
        if tags.get("dropped") == "true" or tags.get("status") == "archived":
            continue
        versions.append(v)
    return versions
