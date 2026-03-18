"""BirdCLEF+ 2026 — Training script.

EfficientNet/ConvNeXt CNN on log-mel spectrograms, 5-second clips, multilabel BCE.

Usage:
    python train.py --fold 0 --backbone efficientnet_b0 --gpu 0
    python train.py --fold 0 --backbone convnext_small --gpu 1 --epochs 50
    # Full 5-fold (run 2 at a time on 2 GPUs):
    python train.py --fold 0 --gpu 0 &
    python train.py --fold 1 --gpu 1 &
"""

import argparse
import ast
import os
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA = Path(os.getenv("KEGO_PATH_DATA", "data")) / "birdclef" / "birdclef-2026"
OUT = Path("outputs")
# Species list comes from taxonomy (234 species), not just train (206 species).
# 28 species have zero training examples — model outputs 0 for them by default.

SR = 32000
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 1024
FMIN = 20
FMAX = 16000
CLIP_DURATION = 5  # seconds
CLIP_SAMPLES = SR * CLIP_DURATION

# ImageNet mean/std (applied per-channel on stacked 3-channel spectrogram)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------


def load_audio(path: Path, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def crop_or_pad(y: np.ndarray, length: int = CLIP_SAMPLES) -> np.ndarray:
    if len(y) >= length:
        start = np.random.randint(0, len(y) - length + 1)
        return y[start : start + length]
    return np.pad(y, (0, length - len(y)))


def make_melspec(y: np.ndarray, sr: int = SR) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
        fmin=FMIN,
        fmax=FMAX,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def spec_to_tensor(spec: np.ndarray) -> torch.Tensor:
    """Normalize and stack to 3-channel tensor for ImageNet-pretrained backbone."""
    # spec: (n_mels, time) in dB, range roughly [-80, 0]
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    img = np.stack([spec, spec, spec], axis=0)  # (3, n_mels, time)
    t = torch.from_numpy(img)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """Mixup augmentation on a batch."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]


def specaugment(
    spec: np.ndarray, freq_mask: int = 20, time_mask: int = 30
) -> np.ndarray:
    """SpecAugment: random frequency and time masking."""
    spec = spec.copy()
    n_mels, t = spec.shape
    # Frequency masking
    f = np.random.randint(0, freq_mask)
    f0 = np.random.randint(0, n_mels - f + 1)
    spec[f0 : f0 + f, :] = spec.mean()
    # Time masking
    t_mask = np.random.randint(0, time_mask)
    t0 = np.random.randint(0, t - t_mask + 1)
    spec[:, t0 : t0 + t_mask] = spec.mean()
    return spec


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BirdDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        species_to_idx: dict[str, int],
        n_species: int,
        audio_dir: Path,
        augment: bool = False,
        secondary_weight: float = 0.5,
    ):
        self.df = df.reset_index(drop=True)
        self.species_to_idx = species_to_idx
        self.n_species = n_species
        self.audio_dir = audio_dir
        self.augment = augment
        self.secondary_weight = secondary_weight

    def __len__(self) -> int:
        return len(self.df)

    def _make_label(self, row: pd.Series) -> np.ndarray:
        label = np.zeros(self.n_species, dtype=np.float32)
        primary = str(row.get("primary_label", ""))
        if primary in self.species_to_idx:
            label[self.species_to_idx[primary]] = 1.0
        # Secondary labels as soft positives
        secondary_raw = row.get("secondary_labels", "[]")
        if isinstance(secondary_raw, str) and secondary_raw not in ("[]", ""):
            try:
                for s in ast.literal_eval(secondary_raw):
                    if s in self.species_to_idx:
                        label[self.species_to_idx[s]] = self.secondary_weight
            except (ValueError, SyntaxError):
                pass
        return label

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # filename already includes subdirectory e.g. "64898/XC12345.ogg"
        path = self.audio_dir / row["filename"]

        y = load_audio(path)
        y = crop_or_pad(y)

        spec = make_melspec(y)
        if self.augment:
            spec = specaugment(spec)

        x = spec_to_tensor(spec)
        label = self._make_label(row)
        return x, torch.from_numpy(label)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class BirdModel(nn.Module):
    def __init__(self, backbone: str, n_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=n_classes,
            in_chans=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_mixup: bool = True,
    label_smoothing: float = 0.05,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if use_mixup:
            x, y = mixup(x, y)
        # Label smoothing
        y = y * (1 - label_smoothing) + label_smoothing / 2
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def val_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        total_loss += loss.item()
    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke", action="store_true", help="Smoke test: 2 epochs, 64 samples"
    )
    args = parser.parse_args()
    if args.smoke:
        args.epochs = 2
        args.batch_size = 8
        args.workers = 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Backbone: {args.backbone} | Fold: {args.fold}")

    # Load metadata — file is train.csv (not train_metadata.csv)
    meta = pd.read_csv(DATA / "train.csv")
    audio_dir = DATA / "train_audio"

    # Build species index from taxonomy (234 species = all submission targets)
    # 28 of these have no training data — we still include them in the output head
    taxonomy = pd.read_csv(DATA / "taxonomy.csv")
    species = sorted(taxonomy["primary_label"].astype(str).tolist())
    species_to_idx = {s: i for i, s in enumerate(species)}
    n_species = len(species)
    meta["primary_label"] = meta["primary_label"].astype(str)
    print(f"Species: {n_species} (taxonomy) | Train recordings: {len(meta)}")

    # Stratified K-fold by primary_label (only fold over species with training data)
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    splits = list(skf.split(meta, meta["primary_label"]))
    train_idx, val_idx = splits[args.fold]

    train_df = meta.iloc[train_idx]
    val_df = meta.iloc[val_idx]
    print(f"Fold {args.fold}: train={len(train_df)}, val={len(val_df)}")

    if args.smoke:
        train_df = train_df.head(64)
        val_df = val_df.head(32)

    train_ds = BirdDataset(train_df, species_to_idx, n_species, audio_dir, augment=True)
    val_ds = BirdDataset(val_df, species_to_idx, n_species, audio_dir, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = BirdModel(args.backbone, n_species).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    OUT.mkdir(exist_ok=True)
    best_val_loss = float("inf")
    best_path = OUT / f"{args.backbone}_fold{args.fold}.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = val_epoch(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        marker = " *" if val_loss < best_val_loss else ""
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_loss:.4f} val={val_loss:.4f} | "
            f"{elapsed:.0f}s{marker}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "val_loss": val_loss},
                best_path,
            )

    print(f"\nBest val loss: {best_val_loss:.4f} → {best_path}")


if __name__ == "__main__":
    main()
