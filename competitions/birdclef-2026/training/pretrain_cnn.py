"""Pre-train EfficientNet-B0 on BirdCLEF 2021-2024 combined data.

Trains on the union of all historical BirdCLEF primary labels (~900 species)
with multi-class CrossEntropy. The backbone learns cross-species bird-audio
representations — the same knowledge that makes Perch powerful.

After training, only the backbone encoder weights are saved. These slot directly
into train_cnn.py via --pretrained-backbone <path>, replacing ImageNet init.

Two-stage approach (same as v7 but with better backbone init):
  Stage 0 (this script): pre-train on 2021-2024, ~900 species, CrossEntropy
  Stage 1+2 (train_cnn.py): fine-tune on 2026, 234 species, BCE + soundscape labels

Usage:
    # On cluster — run both GPUs in parallel (split years or just use GPU 0)
    KEGO_PATH_DATA=/home/kristian/projects/kego/data \\
    uv run python competitions/birdclef-2026/training/pretrain_cnn.py \\
        --years 2021 2022 2023 2024 --epochs 15 --gpu 0 --tag pretrain-v1

    # Check progress
    tail -f /tmp/pretrain-v1.log
"""

import argparse
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
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Config (must match v7 / train_cnn.py baseline HTK settings)
# ---------------------------------------------------------------------------

SR = 32000
N_MELS = 224
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 0  # HTK: fmin=0
FMAX = 16000
CLIP_DURATION = 5
CLIP_SAMPLES = SR * CLIP_DURATION
CLIP_FRAMES = CLIP_SAMPLES // HOP_LENGTH

BACKBONE = "tf_efficientnet_b0.ns_jft_in1k"

DATA_ROOT = Path(os.getenv("KEGO_PATH_DATA", "data")) / "birdclef"


# ---------------------------------------------------------------------------
# Audio / spec helpers (identical to train_cnn.py HTK baseline pipeline)
# ---------------------------------------------------------------------------


def load_audio(path: Path) -> np.ndarray:
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
        y, _ = librosa.load(path, sr=SR, mono=True, offset=offset, duration=CLIP_DURATION)
    except Exception:
        y, _ = librosa.load(path, sr=SR, mono=True, duration=CLIP_DURATION)
    return y


def crop_or_pad(y: np.ndarray) -> np.ndarray:
    if len(y) >= CLIP_SAMPLES:
        return y[:CLIP_SAMPLES]
    return np.pad(y, (0, CLIP_SAMPLES - len(y)))


def make_melspec(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
        fmin=FMIN,
        fmax=FMAX,
        htk=True,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def spec_to_tensor(spec: np.ndarray) -> torch.Tensor:
    """Per-sample min-max → (3, H, W) — identical to v7 baseline."""
    s_min, s_max = spec.min(), spec.max()
    spec = (spec - s_min) / (s_max - s_min + 1e-7)
    img = np.stack([spec, spec, spec], axis=0)
    return torch.from_numpy(img).float()


def specaugment(spec: np.ndarray, freq_mask: int = 20, time_mask: int = 30) -> np.ndarray:
    spec = spec.copy()
    n_mels, t = spec.shape
    fill = spec.mean()
    f = np.random.randint(0, freq_mask + 1)
    f0 = np.random.randint(0, max(n_mels - f, 1))
    spec[f0 : f0 + f, :] = fill
    t_mask = np.random.randint(0, time_mask + 1)
    t0 = np.random.randint(0, max(t - t_mask, 1))
    spec[:, t0 : t0 + t_mask] = fill
    return spec


def gain_augment(spec: np.ndarray, max_db: float = 12.0) -> np.ndarray:
    return spec + np.random.uniform(-max_db, max_db)


def load_spec_crop(audio_path: Path, cache_path: Path | None, augment: bool) -> np.ndarray:
    """Load from cache (.npy) if available, else decode audio on-the-fly."""
    if cache_path is not None and cache_path.exists():
        spec = np.load(cache_path).astype(np.float32)
        t = spec.shape[1]
        if t > CLIP_FRAMES:
            start = np.random.randint(0, t - CLIP_FRAMES + 1)
            spec = spec[:, start : start + CLIP_FRAMES]
        elif t < CLIP_FRAMES:
            spec = np.pad(spec, ((0, 0), (0, CLIP_FRAMES - t)))
    else:
        try:
            y = load_audio(audio_path)
            y = crop_or_pad(y)
            spec = make_melspec(y)
        except Exception:
            spec = np.zeros((N_MELS, CLIP_FRAMES), dtype=np.float32)

    if augment:
        spec = gain_augment(spec)
        spec = specaugment(spec)
    return spec


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _load_csv(year_dir: Path) -> pd.DataFrame | None:
    """Load train CSV from a BirdCLEF year directory, handling format differences."""
    for name in ("train_metadata.csv", "train.csv"):
        p = year_dir / name
        if p.exists():
            df = pd.read_csv(p)
            # Normalise column names
            if "primary_label" not in df.columns:
                # 2021 uses 'ebird_code' as the species column
                for alt in ("ebird_code", "species"):
                    if alt in df.columns:
                        df = df.rename(columns={alt: "primary_label"})
                        break
            if "filename" not in df.columns:
                for alt in ("file_path", "path"):
                    if alt in df.columns:
                        df = df.rename(columns={alt: "filename"})
                        break
            if "primary_label" in df.columns and "filename" in df.columns:
                return df[["filename", "primary_label"]].dropna()
    return None


class MultiYearDataset(Dataset):
    """Combined dataset over BirdCLEF 2021-2024 training clips.

    Each item is (spec_tensor, class_index) for CrossEntropy pre-training.
    Tries spec cache first (fast), falls back to on-the-fly audio decode (slow).
    Cache expected at: data/birdclef/birdclef-YEAR/specs_cache_pretrain/
    """

    def __init__(
        self,
        years: list[int],
        species_to_idx: dict[str, int],
        augment: bool = False,
    ):
        self.species_to_idx = species_to_idx
        self.augment = augment
        self.rows: list[tuple[Path, Path | None, int]] = []  # (audio_path, cache_path, label)

        for year in years:
            year_dir = DATA_ROOT / f"birdclef-{year}"
            # 2021 uses train_short_audio/, all other years use train_audio/
            audio_dir = year_dir / ("train_short_audio" if year == 2021 else "train_audio")
            cache_dir = year_dir / "specs_cache_pretrain"

            df = _load_csv(year_dir)
            if df is None:
                print(f"  WARNING: no CSV found for {year}, skipping", flush=True)
                continue

            n_before = len(self.rows)
            n_unknown = 0
            for _, row in df.iterrows():
                label = str(row["primary_label"]).strip()
                if label not in species_to_idx:
                    n_unknown += 1
                    continue
                fname = str(row["filename"])
                # Some years store filename with subdirectory already, some without
                audio_path = audio_dir / fname
                if not audio_path.exists():
                    # Try without extension or with .ogg
                    audio_path = audio_dir / (fname if fname.endswith(".ogg") else fname + ".ogg")

                stem = Path(fname).stem
                subdir = Path(fname).parent
                cache_path = cache_dir / subdir / f"{stem}.npy" if cache_dir.exists() else None

                self.rows.append((audio_path, cache_path, species_to_idx[label]))

            n_added = len(self.rows) - n_before
            print(
                f"  {year}: {n_added} clips added, {n_unknown} unknown labels skipped",
                flush=True,
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        audio_path, cache_path, label = self.rows[idx]
        spec = load_spec_crop(audio_path, cache_path, self.augment)
        return spec_to_tensor(spec), label


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PretrainModel(nn.Module):
    """EfficientNet-B0 with GAP + linear head for multi-class pre-training.

    Uses global_pool='avg' so the encoder can be extracted and used as
    BirdModelBaseline.encoder (which uses global_pool='' + GEMFreqPool).
    The backbone weights are fully compatible — only the head differs.
    """

    def __init__(self, n_classes: int, backbone: str = BACKBONE):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg", in_chans=3)
        feat_dim = self.encoder.num_features
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    def backbone_state_dict(self) -> dict:
        """Return encoder-only weights — compatible with BirdModelBaseline.encoder."""
        return {k.removeprefix("encoder."): v for k, v in self.state_dict().items() if k.startswith("encoder.")}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_epoch(model, loader, optimiser, device, scaler) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimiser.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        total_loss += loss.item() * len(x)
        n += len(x)
    return total_loss / n


@torch.no_grad()
def eval_epoch(model, loader, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * len(x)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(x)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tag", type=str, default="pretrain-v1")
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.05,
        help="Fraction of data held out for validation (quick accuracy check)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save backbone checkpoint every N epochs",
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    print(f"Device: {device}", flush=True)
    print(f"Years: {args.years}", flush=True)
    print(f"Data root: {DATA_ROOT}", flush=True)

    # Build combined species vocabulary
    all_species: set[str] = set()
    for year in args.years:
        year_dir = DATA_ROOT / f"birdclef-{year}"
        df = _load_csv(year_dir)
        if df is not None:
            all_species.update(df["primary_label"].dropna().astype(str).str.strip().unique())
    species_list = sorted(all_species)
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    n_classes = len(species_list)
    print(f"Combined species vocabulary: {n_classes} species", flush=True)

    # Save vocab for reference
    vocab_path = out_dir / f"{args.tag}_vocab.txt"
    vocab_path.write_text("\n".join(species_list))
    print(f"Vocab saved: {vocab_path}", flush=True)

    # Build dataset
    print("Loading dataset...", flush=True)
    full_ds = MultiYearDataset(args.years, species_to_idx, augment=False)
    print(f"Total clips: {len(full_ds)}", flush=True)

    # Train/val split (random, not stratified — pre-training only needs rough accuracy)
    n_val = max(1, int(len(full_ds) * args.val_frac))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    # Enable augmentation on train split via wrapper
    class AugWrapper(Dataset):
        def __init__(self, subset):
            self.subset = subset

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            audio_path, cache_path, label = self.subset.dataset.rows[self.subset.indices[idx]]
            spec = load_spec_crop(audio_path, cache_path, augment=True)
            return spec_to_tensor(spec), label

    train_loader = DataLoader(
        AugWrapper(train_ds),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    print(f"Train: {n_train} clips, Val: {n_val} clips", flush=True)
    print(f"Batches/epoch: {len(train_loader)}", flush=True)

    # Model, optimiser, scheduler
    model = PretrainModel(n_classes=n_classes).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    print(f"\nStarting pre-training: {args.epochs} epochs, lr={args.lr}", flush=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimiser, device, scaler)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
            f"t={elapsed:.0f}s",
            flush=True,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = out_dir / f"{args.tag}_best_backbone.pt"
            torch.save(model.backbone_state_dict(), ckpt_path)
            print(f"  → Best backbone saved: {ckpt_path}", flush=True)

        if epoch % args.save_every == 0:
            ckpt_path = out_dir / f"{args.tag}_ep{epoch:02d}_backbone.pt"
            torch.save(model.backbone_state_dict(), ckpt_path)
            print(f"  → Checkpoint: {ckpt_path}", flush=True)

    # Always save final
    final_path = out_dir / f"{args.tag}_final_backbone.pt"
    torch.save(model.backbone_state_dict(), final_path)
    print(f"\nDone. Final backbone: {final_path}", flush=True)
    print(f"Best val accuracy: {best_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
