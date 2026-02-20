import copy
import hashlib
import json


def make_learner_id(model_name: str, feature_set: str, folds_n: int) -> str:
    """Build a unique learner identifier string."""
    return f"{model_name}/{feature_set}/{folds_n}f"


def get_seeds_for_learner(
    learner_index: int, seed_pool: list[int], n_seeds: int | None
) -> list[int]:
    """Rotate through the seed pool so each learner gets a different subset."""
    if n_seeds is None or n_seeds >= len(seed_pool):
        return seed_pool  # all seeds, no rotation (backward compat)
    offset = learner_index % len(seed_pool)
    rotated = seed_pool[offset:] + seed_pool[:offset]
    return rotated[:n_seeds]


def filter_model_config(config: dict, active_features: set[str]) -> dict:
    """Deep-copy config and filter cat_features to match a feature set."""
    config = copy.deepcopy(config)
    if "cat_features" in config.get("kwargs", {}):
        config["kwargs"]["cat_features"] = [
            c for c in config["kwargs"]["cat_features"] if c in active_features
        ]
    if "categorical_feature" in config.get("kwargs_fit", {}):
        config["kwargs_fit"]["categorical_feature"] = [
            c
            for c in config["kwargs_fit"]["categorical_feature"]
            if c in active_features
        ]
    return config


def task_fingerprint(model_name, seed, folds_n, feature_set, features, model_config):
    """Deterministic hash of all parameters that define a training task."""
    blob = json.dumps(
        {
            "model": model_name,
            "seed": seed,
            "folds_n": folds_n,
            "feature_set": feature_set,
            "features": sorted(features),
            "kwargs": {k: repr(v) for k, v in sorted(model_config["kwargs"].items())},
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
