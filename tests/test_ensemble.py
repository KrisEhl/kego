import numpy as np

from kego.ensemble import compute_ensemble, hill_climbing, l2_stacking


class TestHillClimbing:
    def test_returns_valid_weights(self, oof_matrix, binary_labels):
        weights = hill_climbing(
            oof_matrix, binary_labels, model_names=["a", "b", "c", "d", "e"]
        )
        assert weights.shape == (5,)
        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= 0).all()

    def test_at_least_as_good_as_uniform(self, oof_matrix, binary_labels):
        from sklearn.metrics import roc_auc_score

        weights = hill_climbing(
            oof_matrix, binary_labels, model_names=["a", "b", "c", "d", "e"]
        )
        hc_auc = roc_auc_score(binary_labels, oof_matrix @ weights)
        uniform = np.ones(5) / 5
        uniform_auc = roc_auc_score(binary_labels, oof_matrix @ uniform)
        assert hc_auc >= uniform_auc - 1e-10


class TestL2Stacking:
    def test_returns_correct_shapes(self, oof_matrix, binary_labels):
        holdout_matrix = oof_matrix[:50]
        test_matrix = oof_matrix[:30]
        l2_oof, l2_holdout, l2_test = l2_stacking(
            oof_matrix, holdout_matrix, test_matrix, binary_labels
        )
        assert l2_oof.shape == (200,)
        assert l2_holdout.shape == (50,)
        assert l2_test.shape == (30,)

    def test_with_extra_features(self, oof_matrix, binary_labels):
        rng = np.random.RandomState(42)
        holdout_matrix = oof_matrix[:50]
        test_matrix = oof_matrix[:30]
        train_feat = rng.randn(200, 3)
        holdout_feat = rng.randn(50, 3)
        test_feat = rng.randn(30, 3)
        l2_oof, l2_holdout, l2_test = l2_stacking(
            oof_matrix,
            holdout_matrix,
            test_matrix,
            binary_labels,
            train_features=train_feat,
            holdout_features=holdout_feat,
            test_features=test_feat,
        )
        assert l2_oof.shape == (200,)
        assert l2_holdout.shape == (50,)
        assert l2_test.shape == (30,)


class TestComputeEnsemble:
    def _make_preds(self):
        rng = np.random.RandomState(42)
        n_train, n_holdout, n_test = 200, 50, 30
        labels_train = rng.randint(0, 2, n_train)
        labels_holdout = rng.randint(0, 2, n_holdout)
        names = ["m1", "m2", "m3"]
        oof, holdout, test = {}, {}, {}
        for name in names:
            noise = rng.normal(0, 0.3, n_train)
            oof[name] = np.clip(labels_train + noise, 0, 1)
            holdout[name] = np.clip(
                labels_holdout + rng.normal(0, 0.3, n_holdout), 0, 1
            )
            test[name] = rng.uniform(0, 1, n_test)
        return names, oof, holdout, test, labels_train, labels_holdout

    def test_returns_structured_result(self):
        names, oof, holdout, test, train_labels, holdout_labels = self._make_preds()
        result = compute_ensemble(
            names, oof, holdout, test, train_labels, holdout_labels
        )
        assert result.best_method in result.all_aucs
        assert result.best_test_preds.shape == (30,)
        assert 0 < result.best_auc <= 1
        assert len(result.methods) >= 5  # average, ridge, hc, rank_blend, l2_preds_only
        assert isinstance(result.calibrated, bool)

    def test_all_aucs_populated(self):
        names, oof, holdout, test, train_labels, holdout_labels = self._make_preds()
        result = compute_ensemble(
            names, oof, holdout, test, train_labels, holdout_labels
        )
        assert "average" in result.all_aucs
        assert "ridge" in result.all_aucs
        assert "hill_climbing" in result.all_aucs
        assert "rank_blending" in result.all_aucs
        assert "l2_preds_only" in result.all_aucs

    def test_without_holdout_labels(self):
        names, oof, holdout, test, train_labels, _ = self._make_preds()
        result = compute_ensemble(
            names, oof, holdout, test, train_labels, holdout_labels=None
        )
        assert result.best_method in result.all_aucs
        assert result.best_test_preds.shape == (30,)

    def test_with_l2_feature_configs(self):
        names, oof, holdout, test, train_labels, holdout_labels = self._make_preds()
        rng = np.random.RandomState(42)
        l2_configs = [
            ("raw", rng.randn(200, 3), rng.randn(50, 3), rng.randn(30, 3)),
        ]
        result = compute_ensemble(
            names,
            oof,
            holdout,
            test,
            train_labels,
            holdout_labels,
            l2_feature_configs=l2_configs,
        )
        assert "l2_preds_only" in result.all_aucs
        assert "l2_raw" in result.all_aucs
