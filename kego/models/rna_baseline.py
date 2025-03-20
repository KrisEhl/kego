from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNABaselineModel(nn.Module):
    """Baseline model for RNA 3D structure prediction.

    This model uses:
    1. Embedding layer to convert nucleotides to vectors
    2. LSTM to process the sequence
    3. MLPs to predict x, y, z coordinates for each position
    """

    def __init__(
        self,
        num_nucleotides: int = 4,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize the model.

        Args:
            num_nucleotides: Number of different nucleotides (A, C, G, U)
            embedding_dim: Dimension of nucleotide embeddings
            hidden_dim: Dimension of LSTM hidden states
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()

        # Embedding layer for nucleotides
        self.embedding = nn.Embedding(num_nucleotides, embedding_dim)

        # Bidirectional LSTM to process the sequence
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True,
            batch_first=True,
        )

        # MLP for coordinate prediction
        lstm_output_dim = hidden_dim * 2  # *2 because bidirectional
        self.coordinate_mlp = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, 3),  # 3 for x, y, z coordinates
        )

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            sequences: Tensor of shape (batch_size, seq_length) containing nucleotide indices

        Returns:
            Tensor of shape (batch_size, seq_length, 3) containing predicted coordinates
        """
        # Embed nucleotides
        embedded = self.embedding(sequences)  # (batch_size, seq_length, embedding_dim)

        # Process with LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_length, hidden_dim*2)

        # Predict coordinates for each position
        coordinates = self.coordinate_mlp(lstm_out)  # (batch_size, seq_length, 3)

        return coordinates

    def compute_loss(
        self, pred_coords: torch.Tensor, true_coords: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE loss between predicted and true coordinates.

        Args:
            pred_coords: Predicted coordinates of shape (batch_size, seq_length, 3)
            true_coords: True coordinates of shape (batch_size, seq_length, 3)

        Returns:
            MSE loss value
        """
        return F.mse_loss(pred_coords, true_coords)
