from typing import List, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


class RNADataset(Dataset):
    """Dataset for RNA 3D structure prediction."""

    # Mapping of nucleotides to integers
    NUCLEOTIDE_MAP = {"A": 0, "C": 1, "G": 2, "U": 3}

    def __init__(self, sequences_df: pl.DataFrame, labels_df: pl.DataFrame):
        """Initialize the dataset.

        Args:
            sequences_df: DataFrame containing RNA sequences
            labels_df: DataFrame containing 3D coordinates
        """
        self.sequences_df = sequences_df
        self.labels_df = labels_df
        self.process_data()

    def process_data(self):
        """Process the raw data into tensors."""
        self.sequences = []
        self.coordinates = []

        # Process each RNA sequence
        for row in self.sequences_df.rows(named=True):
            seq_id = row["target_id"]
            sequence = row["sequence"]

            # Get coordinates for this sequence
            coords = self.labels_df.filter(pl.col("ID").str.contains(seq_id))
            if len(coords) == len(sequence):  # Ensure matching lengths
                # Convert sequence to numerical representation
                seq_tensor = torch.tensor(
                    [
                        self.NUCLEOTIDE_MAP[nt]
                        for nt in sequence
                        if nt in self.NUCLEOTIDE_MAP
                    ]
                )

                # Get coordinates as tensor
                coord_tensor = torch.tensor(
                    coords.select(["x_1", "y_1", "z_1"]).to_numpy()
                )

                self.sequences.append(seq_tensor)
                self.coordinates.append(coord_tensor)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sequence and its corresponding 3D coordinates."""
        return self.sequences[idx], self.coordinates[idx]

    @staticmethod
    def collate_fn(
        batch: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Custom collate function to handle variable length sequences."""
        sequences, coordinates = zip(*batch)

        # Pad sequences to same length
        seq_lengths = [len(seq) for seq in sequences]
        max_len = max(seq_lengths)

        padded_seqs = torch.zeros(len(sequences), max_len, dtype=torch.long)
        padded_coords = torch.zeros(len(sequences), max_len, 3)

        for i, (seq, coord) in enumerate(zip(sequences, coordinates)):
            padded_seqs[i, : len(seq)] = seq
            padded_coords[i, : len(coord)] = coord

        return padded_seqs, padded_coords
