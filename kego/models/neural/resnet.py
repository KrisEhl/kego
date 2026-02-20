import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rtdl_num_embeddings import (
    PeriodicEmbeddings,
    PiecewiseLinearEmbeddings,
    compute_bins,
)
from rtdl_revisiting_models import ResNet
from sklearn.preprocessing import QuantileTransformer
from skorch.callbacks import EarlyStopping

from .amp import AMPNeuralNetBinaryClassifier
from .noise import GaussianNoise


class ResNetModule(nn.Module):
    """Wraps rtdl ResNet for skorch compatibility, with periodic numerical embeddings."""

    def __init__(
        self,
        d_in=1,
        d_out=1,
        n_blocks=3,
        d_block=192,
        d_hidden_multiplier=2.0,
        dropout1=0.15,
        dropout2=0.0,
        n_frequencies=48,
        frequency_init_scale=0.01,
        d_embedding=24,
        noise_std=0.01,
        bins=None,
    ):
        super().__init__()
        self.noise = GaussianNoise(std=noise_std)
        if bins is not None:
            self.num_embeddings = PiecewiseLinearEmbeddings(
                bins=bins,
                d_embedding=d_embedding,
                activation=True,
                version="B",
            )
        else:
            self.num_embeddings = PeriodicEmbeddings(
                n_features=d_in,
                d_embedding=d_embedding,
                n_frequencies=n_frequencies,
                frequency_init_scale=frequency_init_scale,
                activation=True,
                lite=False,
            )
        self.net = ResNet(
            d_in=d_in * d_embedding,
            d_out=d_out,
            n_blocks=n_blocks,
            d_block=d_block,
            d_hidden_multiplier=d_hidden_multiplier,
            dropout1=dropout1,
            dropout2=dropout2,
        )

    def forward(self, X):
        X = self.noise(X)  # Add noise during training only
        X = self.num_embeddings(X)  # (B, n_feat) -> (B, n_feat, d_emb)
        X = X.flatten(1)  # (B, n_feat * d_emb)
        return self.net(X).squeeze(-1)


class SkorchResNet:
    """ResNet with QuantileTransformer + periodic/PLE embeddings + Gaussian noise."""

    def __init__(
        self,
        d_block=192,
        n_blocks=3,
        d_hidden_multiplier=2.0,
        dropout1=0.15,
        dropout2=0.0,
        lr=1e-3,
        max_epochs=200,
        patience=20,
        batch_size=256,
        num_workers=0,
        random_state=42,
        embedding_type="periodic",
        n_bins=48,
    ):
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.d_hidden_multiplier = d_hidden_multiplier
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state
        self.embedding_type = embedding_type
        self.n_bins = n_bins

    def fit(self, X, y, **kwargs):
        torch.manual_seed(self.random_state)
        self.scaler = QuantileTransformer(
            output_distribution="normal", random_state=self.random_state
        )
        X_np = self.scaler.fit_transform(
            X.values if isinstance(X, pd.DataFrame) else X
        ).astype(np.float32)
        y_np = (y.values if hasattr(y, "values") else y).astype(np.float32)
        d_in = X_np.shape[1]

        bins = None
        if self.embedding_type == "ple":
            X_tensor = torch.from_numpy(X_np)
            y_tensor = torch.from_numpy(y_np)
            bins = compute_bins(
                X_tensor,
                n_bins=self.n_bins,
                tree_kwargs={"min_samples_leaf": 64},
                y=y_tensor,
                regression=False,
            )

        self.net = AMPNeuralNetBinaryClassifier(
            ResNetModule,
            module__d_in=d_in,
            module__d_out=1,
            module__n_blocks=self.n_blocks,
            module__d_block=self.d_block,
            module__d_hidden_multiplier=self.d_hidden_multiplier,
            module__dropout1=self.dropout1,
            module__dropout2=self.dropout2,
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
        self.net.fit(X_np, y_np)
        return self

    def predict_proba(self, X):
        X_np = self.scaler.transform(
            X.values if isinstance(X, pd.DataFrame) else X
        ).astype(np.float32)
        return self.net.predict_proba(X_np)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
