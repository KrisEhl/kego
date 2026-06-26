"""Hyper-parameter tuning via Optuna.

Search spaces are *data* (``HPSpace``), declared in config or parsed from the
CLI shorthand (``depth::4:10:int``), rather than per-model ``suggest`` functions.
The objective reuses the trainer's CV on the configured fold scheme. The best
params can be written back out as a promoted config.
"""

from __future__ import annotations

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


@dataclass
class TuneResult:
    model: str
    best_params: dict[str, Any]
    best_score: float
    n_trials: int


class Tuner:
    def __init__(self, trainer: Trainer, n_trials: int = 50) -> None:
        self.trainer = trainer
        self.n_trials = n_trials

    def tune(
        self,
        base: ModelConfig,
        spaces: list[HPSpace],
        ctx: TrainContext,
    ) -> TuneResult:
        """Run an Optuna study (imported lazily) and return the best params."""
        raise NotImplementedError
