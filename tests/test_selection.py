import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from kego.features.selection import (
    SelectionResult,
    drop_one_ablation,
    eval_multiseed,
    forward_selection,
    greedy_add_one_screening,
)


@pytest.fixture
def classification_data():
    """Binary classification data with one useful feature and one random feature.

    'useful' is strongly correlated with the target (target = useful > 0).
    'random' is pure noise drawn from the same distribution.
    """
    rng = np.random.RandomState(42)
    n_train, n_holdout = 200, 100

    useful_train = rng.randn(n_train)
    useful_holdout = rng.randn(n_holdout)
    random_train = rng.randn(n_train)
    random_holdout = rng.randn(n_holdout)

    y_train = (useful_train > 0).astype(int)
    y_holdout = (useful_holdout > 0).astype(int)

    X_train = pd.DataFrame({"useful": useful_train, "random": random_train})
    X_holdout = pd.DataFrame({"useful": useful_holdout, "random": random_holdout})

    return X_train, y_train, X_holdout, y_holdout


@pytest.fixture
def model():
    return DecisionTreeClassifier(max_depth=3)


@pytest.fixture
def shared_kwargs(model):
    return dict(seeds=[42, 123], metric="roc_auc", model=model)


class TestEvalMultiseed:
    def test_returns_float(self, classification_data, shared_kwargs):
        X_train, y_train, X_holdout, y_holdout = classification_data
        score, iterations = eval_multiseed(
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            features=["useful", "random"],
            **shared_kwargs,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(iterations, int)
        assert iterations >= 0

    def test_useful_feature_scores_higher_than_random(
        self, classification_data, shared_kwargs
    ):
        X_train, y_train, X_holdout, y_holdout = classification_data
        score_useful, _ = eval_multiseed(
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            features=["useful"],
            **shared_kwargs,
        )
        score_random, _ = eval_multiseed(
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            features=["random"],
            **shared_kwargs,
        )
        assert score_useful > score_random

    def test_accepts_series_labels(self, classification_data, shared_kwargs):
        X_train, y_train, X_holdout, y_holdout = classification_data
        score, _ = eval_multiseed(
            X_train,
            pd.Series(y_train),
            X_holdout,
            pd.Series(y_holdout),
            features=["useful"],
            **shared_kwargs,
        )
        assert isinstance(score, float)


class TestDropOneAblation:
    def test_returns_selection_result(self, classification_data, shared_kwargs):
        X_train, y_train, X_holdout, y_holdout = classification_data
        result = drop_one_ablation(
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            features=["useful", "random"],
            verbose=False,
            **shared_kwargs,
        )
        assert isinstance(result, SelectionResult)
        assert isinstance(result.baseline_score, float)
        assert len(result.feature_results) == 2

    def test_feature_results_are_dicts(self, classification_data, shared_kwargs):
        X_train, y_train, X_holdout, y_holdout = classification_data
        result = drop_one_ablation(
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            features=["useful", "random"],
            verbose=False,
            **shared_kwargs,
        )
        for entry in result.feature_results:
            assert "feature" in entry
            assert "score" in entry
            assert "delta" in entry
            assert "iterations" in entry
            assert entry["iterations"] >= 0

    def test_removing_useful_hurts(self, classification_data, shared_kwargs):
        X_train, y_train, X_holdout, y_holdout = classification_data
        result = drop_one_ablation(
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            features=["useful", "random"],
            verbose=False,
            **shared_kwargs,
        )
        useful_entry = next(
            entry for entry in result.feature_results if entry["feature"] == "useful"
        )
        # Removing useful should hurt -> negative delta
        assert useful_entry["delta"] < 0

    def test_selected_features_exclude_harmful(
        self, classification_data, shared_kwargs
    ):
        X_train, y_train, X_holdout, y_holdout = classification_data
        result = drop_one_ablation(
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            features=["useful", "random"],
            verbose=False,
            **shared_kwargs,
        )
        # 'useful' must be in selected (removing it hurts)
        assert "useful" in result.selected_features

    def test_selected_score_is_valid(self, classification_data, shared_kwargs):
        X_train, y_train, X_holdout, y_holdout = classification_data
        result = drop_one_ablation(
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            features=["useful", "random"],
            verbose=False,
            **shared_kwargs,
        )
        assert 0.0 <= result.selected_score <= 1.0


class TestForwardSelection:
    def test_returns_selection_result(self, classification_data, shared_kwargs):
        X_train, y_train, X_holdout, y_holdout = classification_data
        result = forward_selection(
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            features_ordered=["useful", "random"],
            verbose=False,
            **shared_kwargs,
        )
        assert isinstance(result, SelectionResult)
        assert len(result.selected_features) >= 1
        assert len(result.feature_results) == 2

    def test_useful_first_selects_at_least_useful(
        self, classification_data, shared_kwargs
    ):
        X_train, y_train, X_holdout, y_holdout = classification_data
        result = forward_selection(
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            features_ordered=["useful", "random"],
            verbose=False,
            **shared_kwargs,
        )
        assert "useful" in result.selected_features

    def test_score_improves_with_useful_feature(
        self, classification_data, shared_kwargs
    ):
        X_train, y_train, X_holdout, y_holdout = classification_data
        result = forward_selection(
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            features_ordered=["useful", "random"],
            verbose=False,
            **shared_kwargs,
        )
        # First step (useful only) should score well
        assert result.feature_results[0]["score"] > 0.5

    def test_selected_score_matches_best_prefix(
        self, classification_data, shared_kwargs
    ):
        X_train, y_train, X_holdout, y_holdout = classification_data
        result = forward_selection(
            X_train,
            y_train,
            X_holdout,
            y_holdout,
            features_ordered=["useful", "random"],
            verbose=False,
            **shared_kwargs,
        )
        best_step_score = max(entry["score"] for entry in result.feature_results)
        assert result.selected_score == best_step_score


class TestGreedyAddOneScreening:
    def test_returns_selection_result(self, classification_data, shared_kwargs):
        X_train, y_train, X_holdout, y_holdout = classification_data
        result = greedy_add_one_screening(
            baseline_features=["useful"],
            candidate_features=["random"],
            X_train=X_train,
            y_train=y_train,
            X_holdout=X_holdout,
            y_holdout=y_holdout,
            verbose=False,
            **shared_kwargs,
        )
        assert isinstance(result, SelectionResult)
        assert len(result.feature_results) == 1

    def test_random_candidate_has_small_delta(self, classification_data, shared_kwargs):
        X_train, y_train, X_holdout, y_holdout = classification_data
        result = greedy_add_one_screening(
            baseline_features=["useful"],
            candidate_features=["random"],
            X_train=X_train,
            y_train=y_train,
            X_holdout=X_holdout,
            y_holdout=y_holdout,
            verbose=False,
            **shared_kwargs,
        )
        random_entry = result.feature_results[0]
        assert random_entry["feature"] == "random"
        # Random feature should not help much (delta near zero or negative)
        assert random_entry["delta"] < 0.05

    def test_useful_candidate_has_positive_delta(
        self, classification_data, shared_kwargs
    ):
        X_train, y_train, X_holdout, y_holdout = classification_data
        result = greedy_add_one_screening(
            baseline_features=["random"],
            candidate_features=["useful"],
            X_train=X_train,
            y_train=y_train,
            X_holdout=X_holdout,
            y_holdout=y_holdout,
            verbose=False,
            **shared_kwargs,
        )
        useful_entry = result.feature_results[0]
        assert useful_entry["feature"] == "useful"
        assert useful_entry["delta"] > 0

    def test_selected_includes_helpful_candidates(
        self, classification_data, shared_kwargs
    ):
        X_train, y_train, X_holdout, y_holdout = classification_data
        result = greedy_add_one_screening(
            baseline_features=["random"],
            candidate_features=["useful"],
            X_train=X_train,
            y_train=y_train,
            X_holdout=X_holdout,
            y_holdout=y_holdout,
            verbose=False,
            **shared_kwargs,
        )
        # 'useful' should be added to selected since it helps
        assert "useful" in result.selected_features
        assert "random" in result.selected_features  # baseline is always included

    def test_accepts_precomputed_baseline_score(
        self, classification_data, shared_kwargs
    ):
        X_train, y_train, X_holdout, y_holdout = classification_data
        result = greedy_add_one_screening(
            baseline_features=["useful"],
            candidate_features=["random"],
            baseline_score=0.9,
            X_train=X_train,
            y_train=y_train,
            X_holdout=X_holdout,
            y_holdout=y_holdout,
            verbose=False,
            **shared_kwargs,
        )
        assert result.baseline_score == 0.9
