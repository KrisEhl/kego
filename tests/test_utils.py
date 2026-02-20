from kego.utils import (
    filter_model_config,
    get_seeds_for_learner,
    make_learner_id,
    task_fingerprint,
)


class TestMakeLearnerId:
    def test_basic(self):
        assert make_learner_id("catboost", "raw", 5) == "catboost/raw/5f"

    def test_with_variant_name(self):
        assert (
            make_learner_id("xgboost_deep", "ablation-pruned", 10)
            == "xgboost_deep/ablation-pruned/10f"
        )


class TestGetSeedsForLearner:
    def test_full_pool_when_n_seeds_none(self):
        pool = [42, 123, 777]
        result = get_seeds_for_learner(0, pool, n_seeds=None)
        assert result == [42, 123, 777]

    def test_full_pool_when_n_seeds_exceeds(self):
        pool = [42, 123, 777]
        result = get_seeds_for_learner(0, pool, n_seeds=5)
        assert result == [42, 123, 777]

    def test_rotation_index_0(self):
        pool = [42, 123, 777]
        result = get_seeds_for_learner(0, pool, n_seeds=2)
        assert result == [42, 123]

    def test_rotation_index_1(self):
        pool = [42, 123, 777]
        result = get_seeds_for_learner(1, pool, n_seeds=2)
        assert result == [123, 777]

    def test_rotation_index_2(self):
        pool = [42, 123, 777]
        result = get_seeds_for_learner(2, pool, n_seeds=2)
        assert result == [777, 42]

    def test_rotation_wraps_around(self):
        pool = [42, 123, 777]
        # index 3 wraps to offset 0
        result = get_seeds_for_learner(3, pool, n_seeds=2)
        assert result == [42, 123]


class TestFilterModelConfig:
    def test_filters_cat_features(self):
        config = {
            "kwargs": {"cat_features": ["Sex", "Age", "Thallium"]},
        }
        active = {"Sex", "Max HR"}
        result = filter_model_config(config, active)
        assert result["kwargs"]["cat_features"] == ["Sex"]

    def test_does_not_mutate_original(self):
        config = {
            "kwargs": {"cat_features": ["Sex", "Age"]},
        }
        original_cats = config["kwargs"]["cat_features"].copy()
        filter_model_config(config, {"Sex"})
        assert config["kwargs"]["cat_features"] == original_cats

    def test_filters_kwargs_fit_categorical_feature(self):
        config = {
            "kwargs": {},
            "kwargs_fit": {"categorical_feature": ["Sex", "Thallium"]},
        }
        result = filter_model_config(config, {"Thallium"})
        assert result["kwargs_fit"]["categorical_feature"] == ["Thallium"]

    def test_no_cat_features_key(self):
        config = {"kwargs": {"learning_rate": 0.1}}
        result = filter_model_config(config, {"Sex"})
        assert result["kwargs"] == {"learning_rate": 0.1}


class TestTaskFingerprint:
    def test_deterministic(self):
        config = {"kwargs": {"lr": 0.1, "depth": 6}}
        fp1 = task_fingerprint("catboost", 42, 10, "raw", ["Age", "Sex"], config)
        fp2 = task_fingerprint("catboost", 42, 10, "raw", ["Age", "Sex"], config)
        assert fp1 == fp2

    def test_feature_order_independent(self):
        config = {"kwargs": {"lr": 0.1}}
        fp1 = task_fingerprint("catboost", 42, 10, "raw", ["Age", "Sex"], config)
        fp2 = task_fingerprint("catboost", 42, 10, "raw", ["Sex", "Age"], config)
        assert fp1 == fp2

    def test_changes_with_seed(self):
        config = {"kwargs": {"lr": 0.1}}
        fp1 = task_fingerprint("catboost", 42, 10, "raw", ["Age"], config)
        fp2 = task_fingerprint("catboost", 99, 10, "raw", ["Age"], config)
        assert fp1 != fp2

    def test_changes_with_model(self):
        config = {"kwargs": {"lr": 0.1}}
        fp1 = task_fingerprint("catboost", 42, 10, "raw", ["Age"], config)
        fp2 = task_fingerprint("lightgbm", 42, 10, "raw", ["Age"], config)
        assert fp1 != fp2

    def test_returns_12_char_hex(self):
        config = {"kwargs": {"lr": 0.1}}
        fp = task_fingerprint("catboost", 42, 10, "raw", ["Age"], config)
        assert len(fp) == 12
        int(fp, 16)  # should not raise
