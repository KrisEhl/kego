"""Configuration objects and the learner fingerprint.

The whole pipeline is configured from these dataclasses, which are typically
hydrated from a YAML file (see :func:`load_config`) and optionally overridden
from the CLI. The single most important object here is :class:`LearnerSpec`:
its :pyattr:`~LearnerSpec.fingerprint` is the cache key that makes training
results reusable across runs without the user asking for it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

# Bump (or set per-task in YAML) to invalidate every cached prediction at once,
# e.g. after changing feature engineering or the train/holdout split logic.
DATA_VERSION = "v1"

MetricDirection = Literal["maximize", "minimize"]
FoldKind = Literal["kfold", "stratified", "group", "time"]


@dataclass(frozen=True)
class ModelConfig:
    """A single model and its hyper-parameters."""

    name: str
    hyper_params: dict[str, Any] = field(default_factory=dict)
    # How categorical columns are handled by this model.
    categorical: Literal["native", "encode", "ignore"] = "native"
    early_stopping_rounds: int | None = None
    # Optional HP search space, only consulted by the tuner. Maps param -> spec.
    hp_space: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class FoldScheme:
    """Cross-validation scheme. Part of the fingerprint."""

    kind: FoldKind = "stratified"
    n: int = 5
    shuffle: bool = True
    group_col: str | None = None


@dataclass(frozen=True)
class GridConfig:
    """The orthogonal axes whose cartesian product defines the learner grid."""

    feature_sets: list[str] = field(default_factory=lambda: ["baseline"])
    folds: list[FoldScheme] = field(default_factory=lambda: [FoldScheme()])
    seeds: list[int] = field(default_factory=lambda: [42])
    # If set, each learner only uses this many seeds (rotating through the pool).
    seeds_per_learner: int | None = None


@dataclass(frozen=True)
class EnsembleConfig:
    name: str = "auto"
    # Strategies to try; the best by `select_by` metric wins.
    methods: list[str] = field(default_factory=lambda: ["mean", "rank_mean", "ridge_stack"])
    select_by: Literal["holdout", "oof"] = "holdout"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluatorConfig:
    name: str = "default"
    importance: bool = True
    fold_stability: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubmitConfig:
    enabled: bool = False
    message: str = ""
    poll_timeout_s: int = 300


@dataclass(frozen=True)
class BattleConfig:
    agent1: str | None = None
    agent2: str | None = None
    games: int = 10
    deck1: str | None = None
    deck2: str | None = None


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level config: everything needed for a full run."""

    task: str
    data_version: str = DATA_VERSION
    feature_sets: dict[str, list[str]] = field(default_factory=dict)
    models: list[ModelConfig] = field(default_factory=list)
    grid: GridConfig = field(default_factory=GridConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    submit: SubmitConfig = field(default_factory=SubmitConfig)
    battle: BattleConfig = field(default_factory=BattleConfig)
    # Behavioural toggles.
    tune: bool = False
    do_ensemble: bool = True
    # When True, ignore the cache and retrain everything.
    force: bool = False


@dataclass(frozen=True)
class LearnerSpec:
    """One cell of the experiment grid: model x feature_set x fold x seed.

    Its :pyattr:`fingerprint` is the deterministic cache key used by the
    :class:`~kego.pipeline.predictions.PredictionStore` to skip re-training.
    """

    model: ModelConfig
    feature_set: str
    fold_scheme: FoldScheme
    seed: int
    data_version: str = DATA_VERSION

    @property
    def fingerprint(self) -> str:
        payload = {
            "model": self.model.name,
            "params": _canonical(self.model.hyper_params),
            "categorical": self.model.categorical,
            "feature_set": self.feature_set,
            "fold": asdict(self.fold_scheme),
            "seed": self.seed,
            "data_version": self.data_version,
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        # Not security-sensitive: this is a cache key, so sha1 is fine and fast.
        return hashlib.sha1(blob.encode(), usedforsecurity=False).hexdigest()[:16]

    @property
    def label(self) -> str:
        return f"{self.model.name}.{self.feature_set}.f{self.fold_scheme.n}.s{self.seed}"


def _canonical(d: dict[str, Any]) -> dict[str, Any]:
    """Stable, hashable view of a params dict (sorted, JSON-safe)."""
    return json.loads(json.dumps(d, sort_keys=True, default=str))


def expand_grid(config: PipelineConfig) -> list[LearnerSpec]:
    """Cartesian product of models x feature_sets x folds x seeds."""
    specs = []
    learner_idx = 0
    for model in config.models:
        for feature_set in config.grid.feature_sets:
            for fold_scheme in config.grid.folds:
                seeds_pool = config.grid.seeds
                if config.grid.seeds_per_learner is not None:
                    k = config.grid.seeds_per_learner
                    start = (learner_idx * k) % len(seeds_pool)
                    seeds_to_use = [seeds_pool[(start + j) % len(seeds_pool)] for j in range(k)]
                else:
                    seeds_to_use = seeds_pool

                for seed in seeds_to_use:
                    specs.append(
                        LearnerSpec(
                            model=model,
                            feature_set=feature_set,
                            fold_scheme=fold_scheme,
                            seed=seed,
                            data_version=config.data_version,
                        )
                    )
                learner_idx += 1
    return specs


def _hydrate_config(d: dict[str, Any]) -> PipelineConfig:
    # Hydrate models
    models = []
    for m in d.get("models", []):
        models.append(
            ModelConfig(
                name=m.get("name", ""),
                hyper_params=dict(m.get("hyper_params", m.get("params", {}))),
                categorical=m.get("categorical", "native"),
                early_stopping_rounds=m.get("early_stopping_rounds"),
                hp_space=dict(m.get("hp_space", {})),
            )
        )

    # Hydrate grid
    grid_raw = d.get("grid", {})
    folds = []
    for f in grid_raw.get("folds", []):
        folds.append(
            FoldScheme(
                kind=f.get("kind", "stratified"),
                n=f.get("n", 5),
                shuffle=f.get("shuffle", True),
                group_col=f.get("group_col"),
            )
        )
    grid = GridConfig(
        feature_sets=list(grid_raw.get("feature_sets", ["baseline"])),
        folds=folds if folds else [FoldScheme()],
        seeds=list(grid_raw.get("seeds", [42])),
        seeds_per_learner=grid_raw.get("seeds_per_learner"),
    )

    # Hydrate ensemble
    e = d.get("ensemble", {})
    ensemble = EnsembleConfig(
        name=e.get("name", "auto"),
        methods=list(e.get("methods", ["mean", "rank_mean", "ridge_stack"])),
        select_by=e.get("select_by", "holdout"),
        params=dict(e.get("params", {})),
    )

    # Hydrate evaluator
    ev = d.get("evaluator", {})
    evaluator = EvaluatorConfig(
        name=ev.get("name", "default"),
        importance=ev.get("importance", True),
        fold_stability=ev.get("fold_stability", True),
        params=dict(ev.get("params", {})),
    )

    # Hydrate submit
    s = d.get("submit", {})
    submit = SubmitConfig(
        enabled=s.get("enabled", False),
        message=s.get("message", ""),
        poll_timeout_s=s.get("poll_timeout_s", 300),
    )

    # Hydrate battle
    b = d.get("battle", {})
    battle = BattleConfig(
        agent1=b.get("agent1"),
        agent2=b.get("agent2"),
        games=b.get("games", 10),
        deck1=b.get("deck1"),
        deck2=b.get("deck2"),
    )

    return PipelineConfig(
        task=d.get("task", ""),
        data_version=d.get("data_version", DATA_VERSION),
        feature_sets=dict(d.get("feature_sets", {})),
        models=models,
        grid=grid,
        ensemble=ensemble,
        evaluator=evaluator,
        submit=submit,
        battle=battle,
        tune=d.get("tune", False),
        do_ensemble=d.get("do_ensemble", True),
        force=d.get("force", False),
    )


def load_config(path: str, overrides: list[str] | None = None, task_name: str | None = None) -> PipelineConfig:
    """Load YAML into a :class:`PipelineConfig`, applying CLI dotlist overrides.

    Resolution order: built-in defaults < YAML file < ``overrides``.
    Implemented with OmegaConf (imported lazily).
    """
    from omegaconf import OmegaConf

    p = Path(path)
    if not p.exists():
        paths_to_try = []
        if task_name:
            task_clean = task_name.replace("-", "_")
            paths_to_try.extend(
                [
                    Path("competitions") / task_name / "configs" / f"{path}.yaml",
                    Path("competitions") / task_name / "configs" / f"{path}.yml",
                    Path("competitions") / task_name / f"{path}.yaml",
                    Path("competitions") / task_name / f"{path}.yml",
                    Path("competitions") / task_clean / "configs" / f"{path}.yaml",
                    Path("competitions") / task_clean / "configs" / f"{path}.yml",
                    Path("competitions") / task_clean / f"{path}.yaml",
                    Path("competitions") / task_clean / f"{path}.yml",
                ]
            )
        paths_to_try.extend(
            [
                Path("configs") / f"{path}.yaml",
                Path("configs") / f"{path}.yml",
            ]
        )

        found = False
        for candidate in paths_to_try:
            if candidate.exists():
                p = candidate
                found = True
                break

        if not found:
            raise FileNotFoundError(f"Config file not found: {path}")

    # Load YAML file config without strict schema
    yaml_conf = OmegaConf.load(p)

    # Parse and apply overrides
    if overrides:
        for o in overrides:
            if ":" in o and "=" not in o:
                k, v = o.split(":", 1)
            elif "=" in o:
                k, v = o.split("=", 1)
            else:
                k, v = o, "true"

            if k == "featureset":
                k = "grid.feature_sets"
                if not (v.startswith("[") and v.endswith("]")):
                    v = f"[{v}]"

            # Parse value to python primitive or object
            if v.startswith("[") and v.endswith("]"):
                inner = v[1:-1].strip()
                val_parsed = []
                if inner:
                    for x in inner.split(","):
                        item = x.strip().strip('"').strip("'")
                        try:
                            if item.lower() == "true":
                                val_parsed.append(True)
                            elif item.lower() == "false":
                                val_parsed.append(False)
                            elif "." in item:
                                val_parsed.append(float(item))
                            else:
                                val_parsed.append(int(item))
                        except ValueError:
                            val_parsed.append(item)
            elif v.startswith("{") and v.endswith("}"):
                try:
                    import json

                    val_parsed = json.loads(v.replace("'", '"'))
                except Exception:
                    val_parsed = v
            else:
                try:
                    if v.lower() == "true":
                        val_parsed = True
                    elif v.lower() == "false":
                        val_parsed = False
                    elif "." in v:
                        val_parsed = float(v)
                    else:
                        val_parsed = int(v)
                except ValueError:
                    val_parsed = v

            OmegaConf.update(yaml_conf, k, val_parsed)

    config_dict = OmegaConf.to_container(yaml_conf, resolve=True)
    if not isinstance(config_dict, dict):
        config_dict = {}
    return _hydrate_config(cast(dict, config_dict))
