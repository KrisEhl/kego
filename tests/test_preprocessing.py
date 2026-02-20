import numpy as np
import pandas as pd

from kego.preprocessing import make_te_preprocess


def _make_data():
    """Build small train/valid/test/holdout DataFrames for TE testing."""
    rng = np.random.RandomState(42)
    n = 50

    def _df():
        return pd.DataFrame(
            {
                "Cat1": rng.choice(["A", "B", "C"], n),
                "Cat2": rng.choice(["X", "Y"], n),
                "Num1": rng.randn(n),
            }
        )

    x_train = _df()
    y_train = pd.Series(rng.randint(0, 2, n))
    x_valid = _df()
    x_test = _df()
    x_holdout = _df()
    return x_train, y_train, x_valid, x_test, x_holdout


class TestMakeTePreprocess:
    def test_adds_te_columns(self):
        x_train, y_train, x_valid, x_test, x_holdout = _make_data()
        preprocess = make_te_preprocess(["Cat1", "Cat2"])
        x_train, x_valid, x_test, x_holdout = preprocess(
            x_train, y_train, x_valid, x_test, x_holdout
        )
        assert "Cat1_te" in x_train.columns
        assert "Cat2_te" in x_train.columns
        assert "Cat1_te" in x_valid.columns
        assert "Cat1_te" in x_test.columns
        assert "Cat1_te" in x_holdout.columns

    def test_keeps_originals_by_default(self):
        x_train, y_train, x_valid, x_test, x_holdout = _make_data()
        preprocess = make_te_preprocess(["Cat1"])
        x_train, x_valid, x_test, x_holdout = preprocess(
            x_train, y_train, x_valid, x_test, x_holdout
        )
        assert "Cat1" in x_train.columns
        assert "Cat1_te" in x_train.columns

    def test_drop_original(self):
        x_train, y_train, x_valid, x_test, x_holdout = _make_data()
        preprocess = make_te_preprocess(["Cat1"], drop_original=True)
        x_train, x_valid, x_test, x_holdout = preprocess(
            x_train, y_train, x_valid, x_test, x_holdout
        )
        assert "Cat1" not in x_train.columns
        assert "Cat1" not in x_valid.columns
        assert "Cat1_te" in x_train.columns

    def test_skips_missing_columns(self):
        x_train, y_train, x_valid, x_test, x_holdout = _make_data()
        preprocess = make_te_preprocess(["NonExistent"])
        x_train, x_valid, x_test, x_holdout = preprocess(
            x_train, y_train, x_valid, x_test, x_holdout
        )
        assert "NonExistent_te" not in x_train.columns

    def test_te_values_are_numeric(self):
        x_train, y_train, x_valid, x_test, x_holdout = _make_data()
        preprocess = make_te_preprocess(["Cat1"])
        x_train, x_valid, x_test, x_holdout = preprocess(
            x_train, y_train, x_valid, x_test, x_holdout
        )
        assert x_train["Cat1_te"].dtype in [np.float64, np.float32]
        assert not x_train["Cat1_te"].isna().any()
