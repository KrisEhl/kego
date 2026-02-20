import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rtdl_num_embeddings import (
    PeriodicEmbeddings,
    PiecewiseLinearEmbeddings,
    compute_bins,
)
from rtdl_revisiting_models import FTTransformer
from sklearn.preprocessing import QuantileTransformer
from skorch.callbacks import EarlyStopping

from .amp import AMPNeuralNetBinaryClassifier
from .noise import GaussianNoise


class FTTransformerModule(nn.Module):
    """Wraps rtdl FTTransformer for skorch (continuous + categorical features).

    Replaces the default LinearEmbeddings for continuous features with
    PeriodicEmbeddings for richer numerical representations.
    """

    def __init__(
        self,
        n_cont_features=1,
        cat_cardinalities=None,
        d_out=1,
        n_blocks=3,
        d_block=96,
        attention_n_heads=8,
        attention_dropout=0.2,
        ffn_d_hidden_multiplier=4 / 3,
        ffn_dropout=0.1,
        residual_dropout=0.0,
        n_frequencies=48,
        frequency_init_scale=0.01,
        noise_std=0.01,
        bins=None,
    ):
        super().__init__()
        self.n_cont = n_cont_features
        self.noise = GaussianNoise(std=noise_std)
        self.net = FTTransformer(
            n_cont_features=n_cont_features,
            cat_cardinalities=cat_cardinalities or [],
            d_out=d_out,
            n_blocks=n_blocks,
            d_block=d_block,
            attention_n_heads=attention_n_heads,
            attention_dropout=attention_dropout,
            ffn_d_hidden_multiplier=ffn_d_hidden_multiplier,
            ffn_dropout=ffn_dropout,
            residual_dropout=residual_dropout,
        )
        # Replace the default LinearEmbeddings with PLE or PeriodicEmbeddings
        if n_cont_features > 0:
            if bins is not None:
                self.net.cont_embeddings = PiecewiseLinearEmbeddings(
                    bins=bins,
                    d_embedding=d_block,
                    activation=True,
                    version="B",
                )
            else:
                self.net.cont_embeddings = PeriodicEmbeddings(
                    n_features=n_cont_features,
                    d_embedding=d_block,
                    n_frequencies=n_frequencies,
                    frequency_init_scale=frequency_init_scale,
                    activation=True,
                    lite=False,
                )

    def forward(self, X):
        x_cont = X[:, : self.n_cont]
        x_cont = self.noise(x_cont)  # Add noise to continuous features during training
        x_cat = X[:, self.n_cont :].long() if X.shape[1] > self.n_cont else None
        return self.net(x_cont, x_cat=x_cat).squeeze(-1)


class SkorchFTTransformer:
    """FTTransformer with categorical embeddings, wrapped via skorch."""

    def __init__(
        self,
        cat_features=None,
        n_blocks=3,
        d_block=96,
        attention_n_heads=8,
        attention_dropout=0.2,
        ffn_d_hidden_multiplier=4 / 3,
        ffn_dropout=0.1,
        residual_dropout=0.0,
        lr=1e-4,
        max_epochs=200,
        patience=20,
        batch_size=256,
        num_workers=0,
        random_state=42,
        embedding_type="periodic",
        n_bins=48,
    ):
        self.cat_features = cat_features or []
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.ffn_d_hidden_multiplier = ffn_d_hidden_multiplier
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state
        self.embedding_type = embedding_type
        self.n_bins = n_bins

    def _prepare(self, X, fit=False):
        if isinstance(X, pd.DataFrame):
            cont_cols = [c for c in X.columns if c not in self.cat_features]
            cat_cols = [c for c in X.columns if c in self.cat_features]
        else:
            if fit:
                self.cont_cols = []
                self.cat_cols = []
                self.cat_cardinalities = []
                self.scaler = QuantileTransformer(
                    output_distribution="normal", random_state=self.random_state
                )
                return self.scaler.fit_transform(X).astype(np.float32)
            return self.scaler.transform(X).astype(np.float32)

        if fit:
            self.cont_cols = cont_cols
            self.cat_cols = cat_cols
            self.scaler = QuantileTransformer(
                output_distribution="normal", random_state=self.random_state
            )
            self.cat_encoders = {}
            for c in cat_cols:
                vals = sorted(X[c].unique())
                self.cat_encoders[c] = {v: i for i, v in enumerate(vals)}
            self.cat_cardinalities = [len(self.cat_encoders[c]) for c in cat_cols]
            X_cont = self.scaler.fit_transform(X[cont_cols].values).astype(np.float32)
        else:
            X_cont = self.scaler.transform(X[self.cont_cols].values).astype(np.float32)

        if self.cat_cols:
            X_cat = np.column_stack(
                [X[c].map(self.cat_encoders[c]).values for c in self.cat_cols]
            ).astype(np.float32)
            return np.hstack([X_cont, X_cat])
        return X_cont

    def fit(self, X, y, **kwargs):
        torch.manual_seed(self.random_state)
        X_prep = self._prepare(X, fit=True)
        y_np = (y.values if hasattr(y, "values") else y).astype(np.float32)
        n_cont = len(self.cont_cols) if self.cont_cols else X_prep.shape[1]

        bins = None
        if self.embedding_type == "ple" and n_cont > 0:
            X_cont = X_prep[:, :n_cont]
            X_tensor = torch.from_numpy(X_cont)
            y_tensor = torch.from_numpy(y_np)
            bins = compute_bins(
                X_tensor,
                n_bins=self.n_bins,
                tree_kwargs={"min_samples_leaf": 64},
                y=y_tensor,
                regression=False,
            )

        self.net = AMPNeuralNetBinaryClassifier(
            FTTransformerModule,
            module__n_cont_features=n_cont,
            module__cat_cardinalities=self.cat_cardinalities,
            module__d_out=1,
            module__n_blocks=self.n_blocks,
            module__d_block=self.d_block,
            module__attention_n_heads=self.attention_n_heads,
            module__attention_dropout=self.attention_dropout,
            module__ffn_d_hidden_multiplier=self.ffn_d_hidden_multiplier,
            module__ffn_dropout=self.ffn_dropout,
            module__residual_dropout=self.residual_dropout,
            module__bins=bins,
            criterion=nn.BCEWithLogitsLoss,
            optimizer=torch.optim.AdamW,
            lr=self.lr,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            device="cuda",
            iterator_train__pin_memory=True,
            iterator_valid__pin_memory=True,
            iterator_train__num_workers=self.num_workers,
            iterator_valid__num_workers=self.num_workers,
            callbacks=[
                EarlyStopping(patience=self.patience, monitor="valid_loss"),
            ],
            verbose=1,
        )
        self.net.initialize()
        self.net.module_ = torch.compile(self.net.module_)
        self.net.fit(X_prep, y_np)
        return self

    def predict_proba(self, X):
        X_prep = self._prepare(X)
        return self.net.predict_proba(X_prep)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
