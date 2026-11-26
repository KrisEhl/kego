"""Tests for the Optuna tuning engine in kego.pipeline.tune."""

import pytest

from kego.pipeline.config import ModelConfig
from kego.pipeline.tune import HPSpace, Tuner, run_study, spaces_from_model

pytest.importorskip("optuna")


def test_from_config_round_trips_parse() -> None:
    parsed = HPSpace.parse("learning_rate::0.01:0.3:float:log")
    from_config = HPSpace.from_config(
        "learning_rate", {"type": "float", "low": 0.01, "high": 0.3, "log": True, "choices": None}
    )
    assert parsed == from_config


def test_run_study_finds_maximum_of_quadratic() -> None:
    spaces = [HPSpace(name="x", type="float", low=-5.0, high=5.0)]
    result = run_study(lambda params: -((params["x"] - 2.0) ** 2), spaces, n_trials=60, seed=0)
    assert result.n_trials == 60
    assert abs(result.best_params["x"] - 2.0) < 0.3


def test_run_study_passes_fixed_params_and_handles_int_and_categorical() -> None:
    seen = []

    def objective(params: dict) -> float:
        seen.append(params)
        return params["depth"] + (1.0 if params["booster"] == "gbtree" else 0.0)

    spaces = [
        HPSpace(name="depth", type="int", low=2, high=6),
        HPSpace(name="booster", type="categorical", choices=["gbtree", "dart"]),
    ]
    result = run_study(objective, spaces, n_trials=30, seed=0, fixed_params={"n_jobs": 1})
    assert all(params["n_jobs"] == 1 and isinstance(params["depth"], int) for params in seen)
    assert "n_jobs" not in result.best_params
    assert result.best_params == {"depth": 6, "booster": "gbtree"}


def test_tuner_uses_model_hp_space_and_score() -> None:
    base = ModelConfig(
        name="toy",
        hyper_params={"offset": 1.0},
        hp_space={"x": {"type": "float", "low": 0.0, "high": 1.0}},
    )
    assert [space.name for space in spaces_from_model(base)] == ["x"]

    def score(candidate: ModelConfig, ctx: object) -> float:
        return candidate.hyper_params["offset"] + candidate.hyper_params["x"]

    result = Tuner(trainer=None, n_trials=20, seed=0).tune(base, [], ctx=None, score=score)  # type: ignore[arg-type]
    assert result.model == "toy"
    assert result.best_score > 1.9


def test_tuner_without_score_is_not_implemented() -> None:
    base = ModelConfig(name="toy", hp_space={"x": {"type": "float", "low": 0.0, "high": 1.0}})
    with pytest.raises(NotImplementedError):
        Tuner(trainer=None).tune(base, [], ctx=None)  # type: ignore[arg-type]
