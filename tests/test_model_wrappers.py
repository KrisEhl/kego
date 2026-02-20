import numpy as np

from kego.models.wrappers import ScaledLogisticRegression


class TestScaledLogisticRegression:
    def test_fit_predict_shapes(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = rng.randint(0, 2, 100)
        model = ScaledLogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (100, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

        preds = model.predict(X)
        assert preds.shape == (100,)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_between_0_and_1(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        y = rng.randint(0, 2, 50)
        model = ScaledLogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert (proba >= 0).all()
        assert (proba <= 1).all()
