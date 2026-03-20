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
import soundfile as sf
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
OUT = Path(__file__).parent / "outputs"
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
CLIP_FRAMES = CLIP_SAMPLES // HOP_LENGTH  # ~312 mel time frames per 5s clip
CACHE_DIR = DATA / "specs_cache"

# ImageNet mean/std (applied per-channel on stacked 3-channel spectrogram)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Baseline config — matches public SED baseline (LB 0.862)
N_MELS_BASELINE = 224
N_FFT_BASELINE = 2048
CACHE_DIR_BASELINE = DATA / "specs_cache_224"


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------


def load_audio(path: Path, sr: int = SR) -> np.ndarray:
    """Load audio, reading only a random 5s chunk to avoid loading long files fully."""
    try:
        info = sf.info(path)
        total_frames = info.frames
        native_sr = info.samplerate
        clip_frames_native = int(CLIP_DURATION * native_sr)
        if total_frames > clip_frames_native:
            max_start = total_frames - clip_frames_native
            start = np.random.randint(0, max_start + 1)
            offset = start / native_sr
        else:
            offset = 0.0
        y, _ = librosa.load(
            path, sr=sr, mono=True, offset=offset, duration=CLIP_DURATION
        )
    except Exception:
        y, _ = librosa.load(path, sr=sr, mono=True, duration=CLIP_DURATION)
    return y


def crop_or_pad(y: np.ndarray, length: int = CLIP_SAMPLES) -> np.ndarray:
    if len(y) >= length:
        return y[:length]
    return np.pad(y, (0, length - len(y)))


def make_melspec(
    y: np.ndarray, sr: int = SR, n_mels: int = N_MELS, n_fft: int = N_FFT
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        hop_length=HOP_LENGTH,
        n_fft=n_fft,
        fmin=FMIN,
        fmax=FMAX,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def spec_to_tensor(spec: np.ndarray) -> torch.Tensor:
    """Z-score + ImageNet normalisation (default pipeline)."""
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    img = np.stack([spec, spec, spec], axis=0)  # (3, n_mels, time)
    t = torch.from_numpy(img)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


def spec_to_tensor_minmax(spec: np.ndarray) -> torch.Tensor:
    """Per-sample min-max normalisation (baseline pipeline, no ImageNet stats)."""
    s_min, s_max = spec.min(), spec.max()
    spec = (spec - s_min) / (s_max - s_min + 1e-7)
    img = np.stack([spec, spec, spec], axis=0)
    return torch.from_numpy(img).float()


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """Mixup augmentation on a batch."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]


def specaugment(
    spec: np.ndarray,
    freq_mask: int = 20,
    time_mask: int = 30,
    n_freq_masks: int = 1,
    n_time_masks: int = 1,
) -> np.ndarray:
    """SpecAugment: random frequency and time masking."""
    spec = spec.copy()
    n_mels, t = spec.shape
    fill = spec.mean()
    for _ in range(n_freq_masks):
        f = np.random.randint(0, freq_mask + 1)
        f0 = np.random.randint(0, max(n_mels - f, 1))
        spec[f0 : f0 + f, :] = fill
    for _ in range(n_time_masks):
        t_mask = np.random.randint(0, time_mask + 1)
        t0 = np.random.randint(0, max(t - t_mask, 1))
        spec[:, t0 : t0 + t_mask] = fill
    return spec


# ---------------------------------------------------------------------------
# Spec cache helpers
# ---------------------------------------------------------------------------


def load_spec_crop(
    filename: str,
    augment: bool = False,
    cache_dir: Path = CACHE_DIR,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    freq_mask: int = 20,
    time_mask: int = 30,
    n_freq_masks: int = 1,
    n_time_masks: int = 1,
) -> np.ndarray:
    """Load a 5s mel spec crop from cache (fast) or compute on-the-fly (slow fallback)."""
    stem = Path(filename).stem
    subdir = Path(filename).parent
    cache_path = cache_dir / subdir / f"{stem}.npy"
    clip_frames = CLIP_SAMPLES // HOP_LENGTH

    if cache_path.exists():
        spec = np.load(cache_path).astype(np.float32)  # (n_mels, T)
        t = spec.shape[1]
        if t > clip_frames:
            start = np.random.randint(0, t - clip_frames + 1)
            spec = spec[:, start : start + clip_frames]
        elif t < clip_frames:
            spec = np.pad(spec, ((0, 0), (0, clip_frames - t)))
    else:
        path = DATA / "train_audio" / filename
        y = load_audio(path)
        y = crop_or_pad(y)
        spec = make_melspec(y, n_mels=n_mels, n_fft=n_fft)

    if augment:
        spec = specaugment(
            spec,
            freq_mask=freq_mask,
            time_mask=time_mask,
            n_freq_masks=n_freq_masks,
            n_time_masks=n_time_masks,
        )
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
        cache_dir: Path = CACHE_DIR,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        minmax_norm: bool = False,
        freq_mask: int = 20,
        time_mask: int = 30,
        n_freq_masks: int = 1,
        n_time_masks: int = 1,
    ):
        self.df = df.reset_index(drop=True)
        self.species_to_idx = species_to_idx
        self.n_species = n_species
        self.audio_dir = audio_dir
        self.augment = augment
        self.secondary_weight = secondary_weight
        self.cache_dir = cache_dir
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.minmax_norm = minmax_norm
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

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
        spec = load_spec_crop(
            row["filename"],
            augment=self.augment,
            cache_dir=self.cache_dir,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            freq_mask=self.freq_mask,
            time_mask=self.time_mask,
            n_freq_masks=self.n_freq_masks,
            n_time_masks=self.n_time_masks,
        )
        x = spec_to_tensor_minmax(spec) if self.minmax_norm else spec_to_tensor(spec)
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


class BirdModelSED(nn.Module):
    """SED head: frame-level predictions + attention pooling over time.

    Keeps spatial feature maps from the backbone (no global pool), pools over
    the frequency axis, then applies attention-weighted pooling over time frames.
    Returns clip-level probabilities (not logits) — use F.binary_cross_entropy.
    """

    def __init__(self, backbone: str, n_classes: int, pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="", in_chans=3
        )
        feat_dim = self.encoder.num_features
        self.fc = nn.Linear(feat_dim, n_classes)
        self.att = nn.Linear(feat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)  # (B, C, H', T') — spatial feature maps
        feat = feat.mean(dim=2)  # pool over freq → (B, C, T')
        feat = feat.permute(0, 2, 1)  # (B, T', C)
        logit = self.fc(feat)  # (B, T', n_classes) — frame logits
        att = self.att(feat)  # (B, T', n_classes) — attention logits
        # Attention-weighted clip-level prediction (PANNs-style)
        clip = (torch.sigmoid(logit) * torch.sigmoid(att)).sum(1) / (
            torch.sigmoid(att).sum(1) + 1e-7
        )
        return clip  # (B, n_classes) probabilities


class GEMFreqPool(nn.Module):
    """Generalised-mean pooling over the frequency axis (learnable exponent)."""

    def __init__(self, p_init: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p_init))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W)
        p = self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)  # (B, C, W)


class AttentionSEDHead(nn.Module):
    """1D-conv attention head with tanh+softmax — matches public baseline.

    Returns (clip_probs, frame_logits) during training for dual loss.
    Returns clip_probs only during inference.
    """

    def __init__(self, feat_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.att_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)
        self.cls_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):  # x: (B, C, T)
        x = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, C, T)
        att = F.softmax(torch.tanh(self.att_conv(x)), dim=-1)  # (B, n_classes, T)
        cls = self.cls_conv(x)  # (B, n_classes, T) — logits
        clip_probs = torch.sigmoid((att * cls).sum(dim=-1))  # (B, n_classes)
        if self.training:
            return clip_probs, cls  # also return frame logits for dual loss
        return clip_probs


class BirdModelBaseline(nn.Module):
    """Replicates public SED baseline: GEMFreqPool + AttentionSEDHead.

    Default backbone: tf_efficientnet_b0.ns_jft_in1k (NoisyStudent JFT pretrained).
    Training: returns (clip_probs, frame_logits) for dual clip+frame loss.
    Inference: returns clip_probs only.
    """

    def __init__(
        self,
        backbone: str,
        n_classes: int,
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="", in_chans=3
        )
        feat_dim = self.encoder.num_features
        self.gem_pool = GEMFreqPool(p_init=3.0)
        self.head = AttentionSEDHead(feat_dim, n_classes, dropout)

    def forward(self, x: torch.Tensor):
        feat = self.encoder(x)  # (B, C, H', T')
        feat = self.gem_pool(feat)  # (B, C, T')
        return self.head(feat)  # clip_probs or (clip_probs, frame_logits)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _bce(preds: torch.Tensor, targets: torch.Tensor, sed: bool) -> torch.Tensor:
    """BCE loss — with_logits for plain model, standard BCE for SED (probs)."""
    if sed:
        return F.binary_cross_entropy(preds, targets)
    return F.binary_cross_entropy_with_logits(preds, targets)


def _dual_loss(
    out, targets: torch.Tensor, label_smoothing: float = 0.05
) -> torch.Tensor:
    """Dual clip+frame loss for BirdModelBaseline.

    out: (clip_probs, frame_logits) where frame_logits is (B, n_classes, T).
    Clip loss: BCE on attention-pooled probs.
    Frame loss: BCE-with-logits averaged over time frames (each frame supervised
    with the clip-level label, as in the public baseline).
    """
    clip_probs, frame_logits = out
    targets_smooth = targets * (1 - label_smoothing) + label_smoothing / 2
    clip_loss = F.binary_cross_entropy(clip_probs, targets_smooth)
    # frame_logits: (B, n_classes, T) — expand targets to (B, n_classes, T)
    frame_targets = targets_smooth.unsqueeze(-1).expand_as(frame_logits)
    frame_loss = F.binary_cross_entropy_with_logits(frame_logits, frame_targets)
    return 0.5 * clip_loss + 0.5 * frame_loss


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_mixup: bool = True,
    label_smoothing: float = 0.05,
    sed: bool = False,
    baseline: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if use_mixup:
            x, y = mixup(x, y)
        out = model(x)
        if baseline:
            loss = _dual_loss(out, y, label_smoothing)
        else:
            y_smooth = y * (1 - label_smoothing) + label_smoothing / 2
            loss = _bce(out, y_smooth, sed)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    sed: bool = False,
    baseline: bool = False,
) -> float:
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if baseline:
            # val: model is in eval mode → returns clip_probs only
            loss = F.binary_cross_entropy(out, y)
        else:
            loss = _bce(out, y, sed)
        total_loss += loss.item()
    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (0 = disabled)",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--freq_mask",
        type=int,
        default=0,
        help="SpecAugment max freq mask width (0=auto)",
    )
    parser.add_argument(
        "--time_mask",
        type=int,
        default=0,
        help="SpecAugment max time mask width (0=auto)",
    )
    parser.add_argument("--n_freq_masks", type=int, default=2)
    parser.add_argument("--n_time_masks", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke", action="store_true", help="Smoke test: 2 epochs, 64 samples"
    )
    parser.add_argument(
        "--sed", action="store_true", help="Use SED head (attention pooling over time)"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Replicate public baseline: n_mels=224, n_fft=2048, GEMFreqPool+AttentionSEDHead, NoisyStudent backbone, minmax norm",
    )
    parser.add_argument(
        "--hard-labels",
        action="store_true",
        help="Treat secondary labels as hard positives (weight=1.0) instead of soft (0.5)",
    )
    args = parser.parse_args()
    if args.smoke:
        args.epochs = 2
        args.batch_size = 8
        args.workers = 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Mel / cache config
    if args.baseline:
        cache_dir = CACHE_DIR_BASELINE
        n_mels_cfg = N_MELS_BASELINE
        n_fft_cfg = N_FFT_BASELINE
        minmax_norm = True
        default_backbone = "tf_efficientnet_b0.ns_jft_in1k"
        # SpecAugment defaults matching public baseline
        freq_mask = args.freq_mask if args.freq_mask > 0 else 30
        time_mask = args.time_mask if args.time_mask > 0 else 30
    else:
        cache_dir = CACHE_DIR
        n_mels_cfg = N_MELS
        n_fft_cfg = N_FFT
        minmax_norm = False
        default_backbone = "efficientnet_b0"
        freq_mask = args.freq_mask if args.freq_mask > 0 else 20
        time_mask = args.time_mask if args.time_mask > 0 else 30

    if args.backbone == "efficientnet_b0" and args.baseline:
        args.backbone = default_backbone

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if args.baseline:
        head = "baseline"
    elif args.sed:
        head = "SED"
    else:
        head = "plain"
    print(
        f"Device: {device} | Backbone: {args.backbone} | Head: {head} | Fold: {args.fold}"
    )
    print(f"Mel: n_mels={n_mels_cfg}, n_fft={n_fft_cfg} | Cache: {cache_dir.name}")
    print(
        f"SpecAugment: freq_mask={freq_mask}×{args.n_freq_masks}, time_mask={time_mask}×{args.n_time_masks}"
    )
    print(f"Secondary labels: {'hard (1.0)' if args.hard_labels else 'soft (0.5)'}")

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

    secondary_weight = 1.0 if args.hard_labels else 0.5
    ds_kwargs = dict(
        cache_dir=cache_dir,
        n_mels=n_mels_cfg,
        n_fft=n_fft_cfg,
        minmax_norm=minmax_norm,
        freq_mask=freq_mask,
        time_mask=time_mask,
        n_freq_masks=args.n_freq_masks,
        n_time_masks=args.n_time_masks,
        secondary_weight=secondary_weight,
    )
    train_ds = BirdDataset(
        train_df, species_to_idx, n_species, audio_dir, augment=True, **ds_kwargs
    )
    val_ds = BirdDataset(
        val_df, species_to_idx, n_species, audio_dir, augment=False, **ds_kwargs
    )

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

    if args.baseline:
        model = BirdModelBaseline(args.backbone, n_species).to(device)
    elif args.sed:
        model = BirdModelSED(args.backbone, n_species).to(device)
    else:
        model = BirdModel(args.backbone, n_species).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    OUT.mkdir(exist_ok=True)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    if args.baseline:
        suffix = "_baseline"
    elif args.sed:
        suffix = "_sed"
    else:
        suffix = ""
    best_path = OUT / f"{args.backbone}{suffix}_fold{args.fold}.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        is_prob_model = args.sed or args.baseline
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            sed=is_prob_model,
            baseline=args.baseline,
        )
        val_loss = val_epoch(
            model, val_loader, device, sed=is_prob_model, baseline=args.baseline
        )
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
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_loss": val_loss,
                    "sed": args.sed,
                    "baseline": args.baseline,
                    "backbone": args.backbone,
                    "n_mels": n_mels_cfg,
                    "n_fft": n_fft_cfg,
                    "minmax_norm": minmax_norm,
                    "hard_labels": args.hard_labels,
                },
                best_path,
            )
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(
                    f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)"
                )
                break

    print(f"\nBest val loss: {best_val_loss:.4f} → {best_path}")


if __name__ == "__main__":
    main()
