"""Model wrappers for common classifiers.

All classes in this module use the sklearn fit/predict/predict_proba API.
Optional dependencies (xgboost, tabpfn, pytabkit) are imported lazily.

This module is never auto-imported from kego.models — use explicit imports:
    from kego.models.wrappers import ScaledLogisticRegression
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class ScaledLogisticRegression:
    """LogisticRegression with StandardScaler preprocessing."""

    def __init__(self, **kwargs):
        self.pipe = make_pipeline(StandardScaler(), LogisticRegression(**kwargs))

    def fit(self, X, y, **kwargs):
        self.pipe.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)

    def predict(self, X):
        return self.pipe.predict(X)


class SubsampledTabPFN:
    """TabPFN with stratified subsampling for large datasets."""

    def __init__(
        self, cat_features=None, max_train_rows=10000, random_state=42, **kwargs
    ):
        self.cat_features = cat_features or []
        self.max_train_rows = max_train_rows
        self.random_state = random_state
        self.kwargs = kwargs

    def _prepare(self, X):
        if isinstance(X, pd.DataFrame) and self.cat_features:
            X = X.copy()
            for c in self.cat_features:
                if c in X.columns:
                    X[c] = X[c].astype("category")
        return X

    def fit(self, X, y, **kwargs):
        from tabpfn import TabPFNClassifier

        X = self._prepare(X)
        if len(X) > self.max_train_rows:
            from sklearn.model_selection import train_test_split

            X, _, y, _ = train_test_split(
                X,
                y,
                train_size=self.max_train_rows,
                stratify=y,
                random_state=self.random_state,
            )
        self.model = TabPFNClassifier(random_state=self.random_state, **self.kwargs)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._prepare(X))

    def predict(self, X):
        return self.model.predict(self._prepare(X))


def _get_gpu_xgb_base():
    """Lazily resolve the XGBClassifier base class."""
    from xgboost import XGBClassifier

    return XGBClassifier


class _GPUXGBMeta(type):
    """Metaclass that sets XGBClassifier as base when the class is first used."""

    _resolved = False

    def __instancecheck__(cls, instance):
        cls._ensure_base()
        return super().__instancecheck__(instance)

    def _ensure_base(cls):
        if not cls._resolved:
            from xgboost import XGBClassifier

            cls.__bases__ = (XGBClassifier,)
            cls._resolved = True

    def __call__(cls, *args, **kwargs):
        cls._ensure_base()
        return super().__call__(*args, **kwargs)


class GPUXGBClassifier(metaclass=_GPUXGBMeta):
    """XGBClassifier that uses DMatrix for GPU-native prediction.

    Inherits from XGBClassifier at first instantiation (lazy import).
    """

    def predict_proba(self, X, **kwargs):
        import xgboost as xgb

        dmat = xgb.DMatrix(
            X, enable_categorical=getattr(self, "enable_categorical", False)
        )
        preds = self.get_booster().predict(dmat)
        return np.column_stack([1 - preds, preds])

    def predict(self, X, **kwargs):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class ScaledRealMLP:
    """RealMLP_TD_Classifier with native categorical feature support."""

    def __init__(self, cat_features=None, random_state=42, **kwargs):
        self.cat_features = cat_features or []
        self.random_state = random_state
        self.kwargs = kwargs

    def _prepare(self, X):
        if isinstance(X, pd.DataFrame) and self.cat_features:
            X = X.copy()
            for c in self.cat_features:
                if c in X.columns:
                    X[c] = X[c].astype("category")
            return X
        return X.values if isinstance(X, pd.DataFrame) else X

    def fit(self, X, y, **kwargs):
        from pytabkit import RealMLP_TD_Classifier

        X_prep = self._prepare(X)
        y_np = y.values if hasattr(y, "values") else y
        self.model = RealMLP_TD_Classifier(
            random_state=self.random_state, **self.kwargs
        )
        self.model.fit(X_prep, y_np)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._prepare(X))

    def predict(self, X):
        return self.model.predict(self._prepare(X))
