import sys
from pathlib import Path
from typing import Optional

import polars as pl
import torch
from torch.utils.data import DataLoader

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from kego.models.rna_baseline import RNABaselineModel
from kego.models.rna_dataset import RNADataset


def train_rna_model(
    train_sequences_path: Path,
    train_labels_path: Path,
    model_save_dir: Path,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Train the RNA structure prediction model.

    Args:
        train_sequences_path: Path to training sequences CSV
        train_labels_path: Path to training labels CSV
        model_save_dir: Directory to save model checkpoints
        batch_size: Batch size for training
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        device: Device to train on ("cuda" or "cpu")
    """
    # Load data
    train_sequences = pl.read_csv(train_sequences_path)
    train_labels = pl.read_csv(train_labels_path)

    # Create dataset and dataloader
    dataset = RNADataset(train_sequences, train_labels)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn
    )

    # Initialize model
    model = RNABaselineModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for sequences, coordinates in dataloader:
            # Move data to device
            sequences = sequences.to(device)
            coordinates = coordinates.to(device)

            # Forward pass
            pred_coordinates = model(sequences)
            loss = model.compute_loss(pred_coordinates, coordinates)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            num_batches += 1

        # Compute average epoch loss
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = model_save_dir / f"rna_model_epoch_{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_epoch_loss,
                },
                checkpoint_path,
            )

    # Save final model
    final_model_path = model_save_dir / "rna_model_final.pt"
    torch.save(model.state_dict(), final_model_path)


if __name__ == "__main__":
    # Set paths relative to the project root
    data_dir = project_root / "data" / "stanford" / "stanford-rna-3d-folding"
    models_dir = project_root / "models" / "stanford"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Example usage
    train_rna_model(
        train_sequences_path=data_dir / "train_sequences.csv",
        train_labels_path=data_dir / "train_labels.csv",
        model_save_dir=models_dir,
    )
