"""Hyper-parameter tuning via Optuna.

Search spaces are *data* (``HPSpace``), declared in config or parsed from the
CLI shorthand (``depth::4:10:int``), rather than per-model ``suggest`` functions.

:func:`run_study` is the engine: it turns a list of ``HPSpace`` into Optuna
suggestions and maximises (or minimises) any ``objective(params) -> score``, so
scripts can tune with their own validation scheme. :class:`Tuner` runs the same
engine over a :class:`ModelConfig` with a scoring function; once the trainer's
cross-validation exists it becomes the default score.

Optuna is an optional dependency (``kego[tune]``) and is imported only when a
study runs.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from kego.pipeline.config import ModelConfig
from kego.pipeline.train import TrainContext, Trainer

HPType = Literal["int", "float", "categorical"]


@dataclass(frozen=True)
class HPSpace:
    name: str
    type: HPType
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: list[Any] | None = None

    @classmethod
    def parse(cls, spec: str) -> HPSpace:
        """Parse the CLI shorthand, e.g. ``"depth::4:10:int"`` or ``name::a:b:c``.

        Format: ``<name>::<low>:<high>:<type>[:log]`` (numeric) or
        ``<name>::<v1>,<v2>,...:categorical``.
        """
        if "::" not in spec:
            raise ValueError(f"Invalid HPSpace spec: {spec}")

        name, rest = spec.split("::", 1)
        if rest.endswith(":categorical"):
            choices_str = rest[: -len(":categorical")]
            choices = choices_str.split(",")
            typed_choices = []
            for c in choices:
                try:
                    typed_choices.append(int(c))
                except ValueError:
                    try:
                        typed_choices.append(float(c))
                    except ValueError:
                        typed_choices.append(c)
            return cls(name=name, type="categorical", choices=typed_choices)

        parts = rest.split(":")
        if len(parts) < 2:
            raise ValueError(f"Numeric HPSpace spec requires at least low:high. Got: {rest}")

        low = float(parts[0])
        high = float(parts[1])
        hp_type = "float"
        log = False

        if len(parts) > 2:
            if parts[-1].lower() == "log":
                log = True
                type_part = (
                    parts[-2] if len(parts) > 3 else ("int" if (parts[0].isdigit() and parts[1].isdigit()) else "float")
                )
            else:
                type_part = parts[2]

            if type_part == "int":
                hp_type = "int"
                low = int(low)
                high = int(high)
            elif type_part == "float":
                hp_type = "float"
            elif type_part == "log":
                log = True
                if parts[0].isdigit() and parts[1].isdigit():
                    hp_type = "int"
                    low = int(low)
                    high = int(high)
                else:
                    hp_type = "float"

        return cls(name=name, type=hp_type, low=low, high=high, log=log)

    @classmethod
    def from_config(cls, name: str, spec: dict[str, Any]) -> HPSpace:
        """Build from a ``ModelConfig.hp_space`` entry, e.g. ``{"type": "int", "low": 2, "high": 10}``."""
        return cls(
            name=name,
            type=spec["type"],
            low=spec.get("low"),
            high=spec.get("high"),
            log=bool(spec.get("log", False)),
            choices=spec.get("choices"),
        )

    def suggest(self, trial: Any) -> Any:
        """Draw this parameter from an Optuna trial."""
        if self.type == "categorical":
            if not self.choices:
                raise ValueError(f"Categorical HPSpace {self.name!r} has no choices")
            return trial.suggest_categorical(self.name, self.choices)
        if self.low is None or self.high is None:
            raise ValueError(f"Numeric HPSpace {self.name!r} needs low and high")
        if self.type == "int":
            return trial.suggest_int(self.name, int(self.low), int(self.high), log=self.log)
        return trial.suggest_float(self.name, float(self.low), float(self.high), log=self.log)


def spaces_from_model(model: ModelConfig) -> list[HPSpace]:
    """The search space declared on a model config."""
    return [HPSpace.from_config(name, spec) for name, spec in model.hp_space.items()]


@dataclass
class TuneResult:
    model: str
    best_params: dict[str, Any]
    best_score: float
    n_trials: int
    elapsed_seconds: float = 0.0


def run_study(
    objective: Callable[[dict[str, Any]], float],
    spaces: list[HPSpace],
    *,
    name: str = "model",
    n_trials: int = 50,
    timeout_seconds: float | None = None,
    direction: Literal["maximize", "minimize"] = "maximize",
    seed: int = 42,
    fixed_params: dict[str, Any] | None = None,
) -> TuneResult:
    """Optimise ``objective`` over ``spaces`` with Optuna's TPE sampler.

    Each trial calls ``objective({**fixed_params, **suggested})``. The study stops
    after ``n_trials`` trials or ``timeout_seconds``, whichever comes first.
    ``best_params`` holds only the tuned parameters, not ``fixed_params``.
    """
    # Optional dependency (kego[tune]): imported here so kego works without it.
    import optuna

    if not spaces:
        raise ValueError("run_study needs at least one HPSpace")
    fixed_params = fixed_params or {}

    def optuna_objective(trial: Any) -> float:
        suggested = {space.name: space.suggest(trial) for space in spaces}
        return objective({**fixed_params, **suggested})

    started = time.time()
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(optuna_objective, n_trials=n_trials, timeout=timeout_seconds)
    return TuneResult(
        model=name,
        best_params=study.best_params,
        best_score=study.best_value,
        n_trials=len(study.trials),
        elapsed_seconds=time.time() - started,
    )


ModelScore = Callable[[ModelConfig, TrainContext], float]


class Tuner:
    def __init__(
        self,
        trainer: Trainer,
        n_trials: int = 50,
        timeout_seconds: float | None = None,
        direction: Literal["maximize", "minimize"] = "maximize",
        seed: int = 42,
    ) -> None:
        self.trainer = trainer
        self.n_trials = n_trials
        self.timeout_seconds = timeout_seconds
        self.direction = direction
        self.seed = seed

    def tune(
        self,
        base: ModelConfig,
        spaces: list[HPSpace],
        ctx: TrainContext,
        score: ModelScore | None = None,
    ) -> TuneResult:
        """Tune ``base`` over ``spaces`` (or its own ``hp_space``) and return the best params.

        ``score`` evaluates one candidate config, e.g. on a validation split. Without it the
        trainer's cross-validation would be used, which is not implemented yet.
        """
        if score is None:
            raise NotImplementedError(
                "Tuner.tune needs a score function until Trainer._train_one (cross-validation) is implemented"
            )
        spaces = spaces or spaces_from_model(base)

        def objective(params: dict[str, Any]) -> float:
            candidate = dataclasses.replace(base, hyper_params={**base.hyper_params, **params})
            return score(candidate, ctx)

        return run_study(
            objective,
            spaces,
            name=base.name,
            n_trials=self.n_trials,
            timeout_seconds=self.timeout_seconds,
            direction=self.direction,
            seed=self.seed,
        )
