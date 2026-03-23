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
from torch.utils.data import ConcatDataset, DataLoader, Dataset

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

# HTK mel config — fmin=0, htk=True (public baseline exact match)
FMIN_HTK = 0
CACHE_DIR_BASELINE_HTK = DATA / "specs_cache_224_htk"
SOUNDSCAPE_CACHE_DIR_BASELINE_HTK = DATA / "specs_cache_soundscape_224_htk"

# BirdSet config — EfficientNet-B1 pretrained on 9,736 Xeno-Canto species
N_MELS_BIRDSET = 256
N_FFT_BIRDSET = 2048
HOP_LENGTH_BIRDSET = 256
CACHE_DIR_BIRDSET = DATA / "specs_cache_256"
SOUNDSCAPE_CACHE_DIR_BASELINE = DATA / "specs_cache_soundscape_224"
CLIP_FRAMES_BIRDSET = CLIP_SAMPLES // HOP_LENGTH_BIRDSET  # 625 frames per 5s


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
    y: np.ndarray,
    sr: int = SR,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    fmin: int = FMIN,
    htk: bool = False,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        fmin=fmin,
        fmax=FMAX,
        htk=htk,
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


def spec_to_tensor_birdset(spec: np.ndarray) -> torch.Tensor:
    """1-channel per-sample min-max normalisation for BirdSet EfficientNet-B1."""
    s_min, s_max = spec.min(), spec.max()
    spec = (spec - s_min) / (s_max - s_min + 1e-7)
    return torch.from_numpy(spec[np.newaxis]).float()  # (1, n_mels, time)


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


def gain_augment(spec: np.ndarray, max_db: float = 12.0) -> np.ndarray:
    """Apply random gain shift ±max_db dB to the spectrogram (log-mel space)."""
    gain = np.random.uniform(-max_db, max_db)
    return spec + gain


# ---------------------------------------------------------------------------
# Spec cache helpers
# ---------------------------------------------------------------------------


def load_spec_crop(
    filename: str,
    augment: bool = False,
    cache_dir: Path = CACHE_DIR,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    freq_mask: int = 20,
    time_mask: int = 30,
    n_freq_masks: int = 1,
    n_time_masks: int = 1,
    fmin: int = FMIN,
    htk: bool = False,
    gain_aug: bool = False,
    gain_db: float = 12.0,
) -> np.ndarray:
    """Load a 5s mel spec crop from cache (fast) or compute on-the-fly (slow fallback)."""
    stem = Path(filename).stem
    subdir = Path(filename).parent
    cache_path = cache_dir / subdir / f"{stem}.npy"
    clip_frames = CLIP_SAMPLES // hop_length

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
        spec = make_melspec(y, n_mels=n_mels, n_fft=n_fft, fmin=fmin, htk=htk)

    if augment:
        if gain_aug:
            spec = gain_augment(spec, max_db=gain_db)
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
        hop_length: int = HOP_LENGTH,
        minmax_norm: bool = False,
        birdset_norm: bool = False,
        freq_mask: int = 20,
        time_mask: int = 30,
        n_freq_masks: int = 1,
        n_time_masks: int = 1,
        bg_pool: list | None = None,
        bg_noise_p: float = 0.0,
        bg_cache_dir: Path | None = None,
        fmin: int = FMIN,
        htk: bool = False,
        gain_aug: bool = False,
        gain_db: float = 12.0,
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
        self.hop_length = hop_length
        self.minmax_norm = minmax_norm
        self.birdset_norm = birdset_norm
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.bg_pool = bg_pool or []
        self.bg_noise_p = bg_noise_p
        self.bg_cache_dir = bg_cache_dir
        self.fmin = fmin
        self.htk = htk
        self.gain_aug = gain_aug
        self.gain_db = gain_db

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
            hop_length=self.hop_length,
            freq_mask=self.freq_mask,
            time_mask=self.time_mask,
            n_freq_masks=self.n_freq_masks,
            n_time_masks=self.n_time_masks,
            fmin=self.fmin,
            htk=self.htk,
            gain_aug=self.gain_aug,
            gain_db=self.gain_db,
        )
        if self.augment and self.bg_pool and np.random.random() < self.bg_noise_p:
            spec = add_background_noise(
                spec,
                self.bg_pool,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                fmin=self.fmin,
                htk=self.htk,
                bg_cache_dir=self.bg_cache_dir,
            )
        if self.birdset_norm:
            x = spec_to_tensor_birdset(spec)
        elif self.minmax_norm:
            x = spec_to_tensor_minmax(spec)
        else:
            x = spec_to_tensor(spec)
        label = self._make_label(row)
        return x, torch.from_numpy(label)


# ---------------------------------------------------------------------------
# Soundscape pseudo-label dataset
# ---------------------------------------------------------------------------


class SoundscapeDataset(Dataset):
    """Dataset over pseudo-labeled soundscape 5s chunks.

    Reads rows from soundscape_pseudo_labels.csv (produced by pseudo_label_self.py).
    Each row is a (soundscape_file, start_sec) pair with a semicolon-separated
    primary_label of predicted species.  Audio is loaded on-the-fly (no cache).

    Optionally mixes in a random background noise segment from the soundscape
    pool (background_noise_p > 0) to regularise XC-clip training.
    """

    def __init__(
        self,
        pseudo_csv: Path,
        soundscape_dir: Path,
        species_to_idx: dict[str, int],
        n_species: int,
        augment: bool = False,
        label_weight: float = 0.5,  # treat pseudo-labels as soft (not 1.0)
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        minmax_norm: bool = False,
        freq_mask: int = 20,
        time_mask: int = 30,
        n_freq_masks: int = 1,
        n_time_masks: int = 1,
    ):
        import csv

        self.soundscape_dir = soundscape_dir
        self.species_to_idx = species_to_idx
        self.n_species = n_species
        self.augment = augment
        self.label_weight = label_weight
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.minmax_norm = minmax_norm
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

        with open(pseudo_csv) as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.rows)

    def _make_label(self, primary_label: str) -> np.ndarray:
        label = np.zeros(self.n_species, dtype=np.float32)
        for sp in primary_label.split(";"):
            sp = sp.strip()
            if sp in self.species_to_idx:
                label[self.species_to_idx[sp]] = self.label_weight
        return label

    def _load_chunk(self, filename: str, start_sec: int) -> np.ndarray:
        path = self.soundscape_dir / filename
        try:
            y, _ = librosa.load(
                path, sr=SR, mono=True, offset=float(start_sec), duration=5.0
            )
        except Exception:
            return np.zeros(CLIP_SAMPLES, dtype=np.float32)
        return crop_or_pad(y)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        y = self._load_chunk(row["soundscape_filename"], int(row["start_sec"]))
        spec = make_melspec(
            y, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length
        )
        if self.augment:
            spec = specaugment(
                spec,
                freq_mask=self.freq_mask,
                time_mask=self.time_mask,
                n_freq_masks=self.n_freq_masks,
                n_time_masks=self.n_time_masks,
            )
        if self.minmax_norm:
            x = spec_to_tensor_minmax(spec)
        else:
            x = spec_to_tensor(spec)
        label = self._make_label(row["primary_label"])
        return x, torch.from_numpy(label)


class SoundscapeLabelsDataset(Dataset):
    """Labeled soundscape segments from train_soundscapes_labels.csv.

    These are real, human-annotated labels on in-domain passive recordings —
    same distribution as the test set. Adding them to training is the single
    most impactful change to close the gap to the public baseline (LB 0.862).

    Fast path: slices precomputed 60s specs from specs_cache_soundscape_224/.
    Slow path: loads audio on-the-fly with librosa.

    CSV columns: filename, start (HH:MM:SS), end (HH:MM:SS), primary_label (;-sep)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        soundscape_dir: Path,
        species_to_idx: dict[str, int],
        n_species: int,
        augment: bool = False,
        n_mels: int = N_MELS_BASELINE,
        n_fft: int = N_FFT_BASELINE,
        hop_length: int = HOP_LENGTH,
        minmax_norm: bool = True,
        freq_mask: int = 30,
        time_mask: int = 30,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        cache_dir: Path | None = None,
        fmin: int = FMIN,
        htk: bool = False,
        gain_aug: bool = False,
        gain_db: float = 12.0,
    ):
        self.df = df.reset_index(drop=True)
        self.soundscape_dir = soundscape_dir
        self.species_to_idx = species_to_idx
        self.n_species = n_species
        self.augment = augment
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.minmax_norm = minmax_norm
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.cache_dir = cache_dir
        self.clip_frames = CLIP_SAMPLES // hop_length
        self.fmin = fmin
        self.htk = htk
        self.gain_aug = gain_aug
        self.gain_db = gain_db

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _parse_seconds(t: str) -> int:
        """Parse HH:MM:SS → total seconds."""
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)

    def _make_label(self, primary_label: str) -> np.ndarray:
        label = np.zeros(self.n_species, dtype=np.float32)
        for sp in str(primary_label).split(";"):
            sp = sp.strip()
            if sp in self.species_to_idx:
                label[self.species_to_idx[sp]] = 1.0
        return label

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        start_sec = self._parse_seconds(str(row["start"]))

        if self.cache_dir is not None:
            stem = Path(str(row["filename"])).stem
            cache_path = self.cache_dir / f"{stem}.npy"
            if cache_path.exists():
                full_spec = np.load(cache_path).astype(np.float32)
                start_frame = int(start_sec * SR / self.hop_length)
                spec = full_spec[:, start_frame : start_frame + self.clip_frames]
                if spec.shape[1] < self.clip_frames:
                    spec = np.pad(spec, ((0, 0), (0, self.clip_frames - spec.shape[1])))
            else:
                spec = self._load_from_audio(row["filename"], start_sec)
        else:
            spec = self._load_from_audio(row["filename"], start_sec)

        if self.augment:
            if self.gain_aug:
                spec = gain_augment(spec, max_db=self.gain_db)
            spec = specaugment(
                spec,
                freq_mask=self.freq_mask,
                time_mask=self.time_mask,
                n_freq_masks=self.n_freq_masks,
                n_time_masks=self.n_time_masks,
            )
        x = spec_to_tensor_minmax(spec) if self.minmax_norm else spec_to_tensor(spec)
        return x, torch.from_numpy(self._make_label(row["primary_label"]))

    def _load_from_audio(self, filename: str, start_sec: int) -> np.ndarray:
        path = self.soundscape_dir / filename
        try:
            y, _ = librosa.load(
                path, sr=SR, mono=True, offset=float(start_sec), duration=5.0
            )
        except Exception:
            return np.zeros((self.n_mels, self.clip_frames), dtype=np.float32)
        y = crop_or_pad(y)
        return make_melspec(
            y,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            htk=self.htk,
        )


class PerchDataset(Dataset):
    """Soundscape windows with Perch soft labels for stage-1 pretraining.

    Reads perch_pseudo_labels_soft.npz (output of pseudo_label_perch.py).
    Labels are raw Perch sigmoid probabilities — used as soft BCE targets.
    Species are remapped from NPZ order to the sorted train.py species order.

    Fast path: if cache_dir is set, loads precomputed 60s mel specs from
    specs_cache_soundscape_224/ (run precompute_specs.py --soundscapes first).
    Slow path: loads audio on-the-fly via librosa (CPU bottleneck, not recommended).
    """

    def __init__(
        self,
        npz_path: Path,
        soundscape_dir: Path,
        species_to_idx: dict[str, int],
        n_species: int,
        augment: bool = False,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        minmax_norm: bool = False,
        freq_mask: int = 20,
        time_mask: int = 30,
        n_freq_masks: int = 1,
        n_time_masks: int = 1,
        cache_dir: Path | None = None,
        min_max_prob: float = 0.0,
    ):
        data = np.load(npz_path, allow_pickle=True)
        all_labels = data["labels"].astype(np.float32)  # (N, 234) Perch probs
        npz_species = list(data["species"])
        # Map NPZ species order → train.py sorted species order (-1 = absent)
        self.remap = np.array(
            [species_to_idx.get(sp, -1) for sp in npz_species], dtype=np.int32
        )
        all_entries = []
        for key in data["filenames"]:
            fname, start = str(key).rsplit(":", 1)
            all_entries.append((fname, int(start)))

        # Filter to windows with meaningful Perch signal
        if min_max_prob > 0.0:
            keep = all_labels.max(axis=1) >= min_max_prob
            self.labels = all_labels[keep]
            self.entries = [e for e, k in zip(all_entries, keep) if k]
        else:
            self.labels = all_labels
            self.entries = all_entries

        self.soundscape_dir = soundscape_dir
        self.cache_dir = cache_dir
        self.n_species = n_species
        self.augment = augment
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.minmax_norm = minmax_norm
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.clip_frames = CLIP_SAMPLES // hop_length

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        fname, start_sec = self.entries[idx]

        if self.cache_dir is not None:
            # Fast path: slice precomputed full-soundscape mel spec
            cache_path = self.cache_dir / f"{Path(fname).stem}.npy"
            if cache_path.exists():
                full_spec = np.load(cache_path).astype(np.float32)  # (n_mels, T_full)
                start_frame = int(start_sec * SR / self.hop_length)
                spec = full_spec[:, start_frame : start_frame + self.clip_frames]
                if spec.shape[1] < self.clip_frames:
                    spec = np.pad(spec, ((0, 0), (0, self.clip_frames - spec.shape[1])))
            else:
                spec = np.zeros((self.n_mels, self.clip_frames), dtype=np.float32)
        else:
            # Slow path: on-the-fly audio loading
            try:
                y, _ = librosa.load(
                    self.soundscape_dir / fname,
                    sr=SR,
                    mono=True,
                    offset=float(start_sec),
                    duration=5.0,
                )
            except Exception:
                y = np.zeros(CLIP_SAMPLES, dtype=np.float32)
            y = crop_or_pad(y)
            spec = make_melspec(
                y, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length
            )

        if self.augment:
            spec = specaugment(
                spec,
                freq_mask=self.freq_mask,
                time_mask=self.time_mask,
                n_freq_masks=self.n_freq_masks,
                n_time_masks=self.n_time_masks,
            )
        x = spec_to_tensor_minmax(spec) if self.minmax_norm else spec_to_tensor(spec)
        # Remap Perch probs from NPZ species order → train sorted species order
        raw_probs = self.labels[idx]
        label = np.zeros(self.n_species, dtype=np.float32)
        valid = self.remap >= 0
        label[self.remap[valid]] = raw_probs[valid]
        return x, torch.from_numpy(label)


def load_background_pool(soundscape_dir: Path, max_files: int = 500) -> list[Path]:
    """Return a list of soundscape paths to use as background noise sources."""
    files = sorted(soundscape_dir.glob("*.ogg"))
    if len(files) > max_files:
        rng = np.random.default_rng(42)
        files = list(rng.choice(files, max_files, replace=False))  # type: ignore[arg-type]
    return files


def add_background_noise(
    spec: np.ndarray,
    bg_pool: list[Path],
    alpha: float = 0.3,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    fmin: int = FMIN,
    htk: bool = False,
    bg_cache_dir: Path | None = None,
) -> np.ndarray:
    """Mix a random background noise segment into the spectrogram (in dB space).

    If bg_cache_dir is provided, loads precomputed full-soundscape specs (.npy)
    and slices a random 5s window — avoids decoding raw .ogg on every sample.
    Falls back to librosa.load if cache file is missing.
    """
    if not bg_pool:
        return spec
    path = bg_pool[np.random.randint(len(bg_pool))]
    try:
        if bg_cache_dir is not None:
            cache_path = bg_cache_dir / (path.stem + ".npy")
            if cache_path.exists():
                full_spec = np.load(cache_path).astype(np.float32)  # (n_mels, T_full)
                clip_frames = spec.shape[1]
                max_start = max(0, full_spec.shape[1] - clip_frames)
                start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
                spec_bg = full_spec[:, start : start + clip_frames]
                # Pad if edge slice is short
                if spec_bg.shape[1] < clip_frames:
                    spec_bg = np.pad(
                        spec_bg, ((0, 0), (0, clip_frames - spec_bg.shape[1]))
                    )
                return spec + alpha * spec_bg
        # Fallback: decode raw audio
        info = sf.info(path)
        max_start = max(0, info.frames - int(5.0 * info.samplerate))
        offset = (
            np.random.randint(0, max_start + 1) / info.samplerate
            if max_start > 0
            else 0.0
        )
        y_bg, _ = librosa.load(path, sr=SR, mono=True, offset=offset, duration=5.0)
        y_bg = crop_or_pad(y_bg)
        spec_bg = make_melspec(
            y_bg, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmin=fmin, htk=htk
        )
        return spec + alpha * spec_bg
    except Exception:
        return spec


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


class BirdModelBirdSet(nn.Module):
    """EfficientNet-B1 pretrained on BirdSet XCL (9,736 Xeno-Canto species).

    HuggingFace backbone + GEMFreqPool + AttentionSEDHead.
    Input: 1-channel mel spec (256 mels, hop=256).
    Training: returns (clip_probs, frame_logits) for dual loss.
    Inference: returns clip_probs only.
    """

    FEAT_DIM = 1280  # EfficientNet-B1 final feature channels

    def __init__(self, n_classes: int, dropout: float = 0.1):
        super().__init__()
        from transformers import EfficientNetModel

        self.encoder = EfficientNetModel.from_pretrained(
            "DBD-research-group/EfficientNet-B1-BirdSet-XCL"
        )
        self.gem_pool = GEMFreqPool(p_init=3.0)
        self.head = AttentionSEDHead(self.FEAT_DIM, n_classes, dropout)

    def forward(self, x: torch.Tensor):
        # x: (B, 1, H, W) — 1-channel mel spec
        out = self.encoder(pixel_values=x)
        feat = out.last_hidden_state  # (B, C, H', W')
        feat = self.gem_pool(feat)  # (B, C, T') — pool over freq axis
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
    out, targets: torch.Tensor, label_smoothing: float = 0.05, ce_loss: bool = False
) -> torch.Tensor:
    """Dual clip+frame loss for BirdModelBaseline.

    out: (clip_probs, frame_logits) where frame_logits is (B, n_classes, T).
    When ce_loss=True: cross-entropy on frame logits (soft targets, dim=1).
    When ce_loss=False: BCE on attention-pooled probs + BCE-with-logits on frames.
    """
    clip_probs, frame_logits = out
    n_classes = targets.shape[-1]
    if ce_loss:
        targets_smooth = targets * (1 - label_smoothing) + label_smoothing / n_classes
        # Clip loss: CE on time-averaged frame logits
        clip_loss = F.cross_entropy(frame_logits.mean(dim=-1), targets_smooth)
        # Frame loss: CE on per-frame logits, targets broadcast over time
        frame_targets = targets_smooth.unsqueeze(-1).expand_as(frame_logits)
        frame_loss = F.cross_entropy(frame_logits, frame_targets)
    else:
        targets_smooth = targets * (1 - label_smoothing) + label_smoothing / 2
        clip_loss = F.binary_cross_entropy(clip_probs, targets_smooth)
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
    ce_loss: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if use_mixup:
            x, y = mixup(x, y)
        out = model(x)
        if baseline:
            loss = _dual_loss(out, y, label_smoothing, ce_loss=ce_loss)
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
    ce_loss: bool = False,
) -> tuple[float, float, float]:
    """Returns (val_loss, mean_preds_above_thresh, mean_top2_score).

    mean_preds_above_thresh: avg number of classes > 0.3 per sample.
      BCE-trained models: ~2–5. CE-trained models: ~0 (multilabel collapse).
    mean_top2_score: avg sigmoid score of the 2nd-highest class per sample.
      BCE: meaningful (>0.1). CE collapse: near-zero.
    """
    model.eval()
    total_loss = 0.0
    all_preds: list[torch.Tensor] = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if baseline:
            loss = F.binary_cross_entropy(out, y)
            all_preds.append(out.cpu())
        else:
            loss = _bce(out, y, sed)
            preds = torch.sigmoid(out) if not sed else out
            all_preds.append(preds.cpu())
        total_loss += loss.item()
    preds_cat = torch.cat(all_preds, dim=0)  # (N, C)
    mean_above = (preds_cat > 0.3).float().sum(dim=1).mean().item()
    top2 = preds_cat.topk(2, dim=1).values[:, 1].mean().item()
    return total_loss / len(loader), mean_above, top2


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
    parser.add_argument(
        "--birdset",
        action="store_true",
        help="Use EfficientNet-B1 pretrained on BirdSet XCL (9,736 Xeno-Canto species)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Experiment tag included in checkpoint filename to prevent collisions (e.g. 'birdset-v1')",
    )
    parser.add_argument(
        "--pseudo-csv",
        type=str,
        default="",
        help="Path to soundscape_pseudo_labels.csv (from pseudo_label_self.py). "
        "Pseudo-labeled soundscape segments are concatenated with XC training data.",
    )
    parser.add_argument(
        "--pseudo-weight",
        type=float,
        default=0.5,
        help="Label weight for pseudo-labeled soundscape segments (default 0.5 = soft label)",
    )
    parser.add_argument(
        "--bg-noise-p",
        type=float,
        default=0.0,
        help="Probability of mixing a random soundscape segment as background noise (0=off)",
    )
    parser.add_argument(
        "--perch-npz",
        type=str,
        default="",
        help="Path to perch_pseudo_labels_soft.npz. If set, run stage-1 pretraining on "
        "Perch soft labels before fine-tuning on XC hard labels.",
    )
    parser.add_argument(
        "--perch-epochs",
        type=int,
        default=10,
        help="Epochs for stage-1 Perch pretraining (default 10)",
    )
    parser.add_argument(
        "--perch-min-prob",
        type=float,
        default=0.0,
        help="Only use Perch windows where max species prob >= this threshold (default 0 = all). "
        "Recommended: 0.1 (keeps ~1%% of windows with real signal, drops empty noise windows).",
    )
    parser.add_argument(
        "--soundscape-labels",
        action="store_true",
        help="Include train_soundscapes_labels.csv in training. A fraction of soundscapes "
        "is held out for validation (see --soundscape-val-frac). When set, early stopping "
        "uses soundscape val loss instead of XC val loss — much closer to LB metric.",
    )
    parser.add_argument(
        "--htk",
        action="store_true",
        help="Use HTK mel scale (fmin=0, htk=True) — matches public baseline config. "
        "Requires separate spec cache (specs_cache_224_htk/). Run precompute_specs.py --baseline --htk first.",
    )
    parser.add_argument(
        "--warm-restarts",
        action="store_true",
        help="Use CosineAnnealingWarmRestarts(T_0=5) instead of CosineAnnealingLR. "
        "Matches public baseline scheduler.",
    )
    parser.add_argument(
        "--warm-restarts-t0",
        type=int,
        default=5,
        help="T_0 for CosineAnnealingWarmRestarts (default 5)",
    )
    parser.add_argument(
        "--gain-aug",
        action="store_true",
        help="Apply random gain augmentation ±12dB to training specs.",
    )
    parser.add_argument(
        "--gain-db",
        type=float,
        default=12.0,
        help="Max gain in dB for gain augmentation (default 12.0)",
    )
    parser.add_argument(
        "--ce-loss",
        action="store_true",
        help="Use cross-entropy loss instead of BCE. Applies softmax over classes "
        "on frame logits (matches public baseline loss_type=CE).",
    )
    args = parser.parse_args()
    if args.smoke:
        args.epochs = 2
        args.batch_size = 8
        args.workers = 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Mel / cache config
    hop_length_cfg = HOP_LENGTH
    birdset_norm = False
    htk_cfg = args.htk
    fmin_cfg = FMIN_HTK if args.htk else FMIN
    if args.birdset:
        cache_dir = CACHE_DIR_BIRDSET
        n_mels_cfg = N_MELS_BIRDSET
        n_fft_cfg = N_FFT_BIRDSET
        hop_length_cfg = HOP_LENGTH_BIRDSET
        minmax_norm = False
        birdset_norm = True
        freq_mask = args.freq_mask if args.freq_mask > 0 else 30
        time_mask = args.time_mask if args.time_mask > 0 else 30
    elif args.baseline:
        if args.htk:
            cache_dir = CACHE_DIR_BASELINE_HTK
        else:
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
    if args.birdset:
        head = "birdset"
    elif args.baseline:
        head = "baseline"
    elif args.sed:
        head = "SED"
    else:
        head = "plain"
    backbone_name = "EfficientNet-B1-BirdSet-XCL" if args.birdset else args.backbone
    print(
        f"Device: {device} | Backbone: {backbone_name} | Head: {head} | Fold: {args.fold}"
    )
    print(f"Mel: n_mels={n_mels_cfg}, n_fft={n_fft_cfg} | Cache: {cache_dir.name}")
    print(
        f"SpecAugment: freq_mask={freq_mask}×{args.n_freq_masks}, time_mask={time_mask}×{args.n_time_masks}"
    )
    print(f"Secondary labels: {'hard (1.0)' if args.hard_labels else 'soft (0.5)'}")
    print(
        f"HTK mel: {args.htk} (fmin={fmin_cfg}) | Gain aug: {args.gain_aug} (±{args.gain_db}dB) | "
        f"Scheduler: {'WarmRestarts T_0=' + str(args.warm_restarts_t0) if args.warm_restarts else 'CosineAnnealingLR'}"
    )

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
        hop_length=hop_length_cfg,
        minmax_norm=minmax_norm,
        birdset_norm=birdset_norm,
        freq_mask=freq_mask,
        time_mask=time_mask,
        n_freq_masks=args.n_freq_masks,
        n_time_masks=args.n_time_masks,
        secondary_weight=secondary_weight,
        fmin=fmin_cfg,
        htk=htk_cfg,
    )
    # Background noise pool (shared by BirdDataset and optionally SoundscapeDataset)
    bg_pool: list = []
    bg_cache_dir: Path | None = None
    if args.bg_noise_p > 0:
        soundscape_dir = DATA / "train_soundscapes"
        bg_pool = load_background_pool(soundscape_dir)
        # Use precomputed spec cache for bg noise to avoid slow .ogg decoding per sample
        sc_cache_candidate = (
            SOUNDSCAPE_CACHE_DIR_BASELINE_HTK
            if args.htk
            else SOUNDSCAPE_CACHE_DIR_BASELINE
        )
        if sc_cache_candidate.exists():
            bg_cache_dir = sc_cache_candidate
            print(
                f"Background noise pool: {len(bg_pool)} soundscapes (p={args.bg_noise_p})"
                f" — using spec cache: {bg_cache_dir.name}"
            )
        else:
            print(
                f"Background noise pool: {len(bg_pool)} soundscapes (p={args.bg_noise_p})"
                f" — WARNING: spec cache not found, falling back to slow .ogg decoding"
            )

    train_ds = BirdDataset(
        train_df,
        species_to_idx,
        n_species,
        audio_dir,
        augment=True,
        bg_pool=bg_pool,
        bg_noise_p=args.bg_noise_p,
        bg_cache_dir=bg_cache_dir,
        gain_aug=args.gain_aug,
        gain_db=args.gain_db,
        **ds_kwargs,
    )
    val_ds = BirdDataset(
        val_df, species_to_idx, n_species, audio_dir, augment=False, **ds_kwargs
    )

    # Labeled soundscape segments (train_soundscapes_labels.csv)
    if args.soundscape_labels:
        sc_labels_path = DATA / "train_soundscapes_labels.csv"
        sc_labels = pd.read_csv(sc_labels_path)
        soundscape_dir = DATA / "train_soundscapes"
        # Select HTK or standard soundscape cache
        if args.htk:
            sc_cache_candidate = SOUNDSCAPE_CACHE_DIR_BASELINE_HTK
        else:
            sc_cache_candidate = SOUNDSCAPE_CACHE_DIR_BASELINE
        sc_cache = sc_cache_candidate if sc_cache_candidate.exists() else None

        sc_ds = SoundscapeLabelsDataset(
            sc_labels,
            soundscape_dir=soundscape_dir,
            species_to_idx=species_to_idx,
            n_species=n_species,
            augment=True,
            n_mels=n_mels_cfg,
            n_fft=n_fft_cfg,
            hop_length=hop_length_cfg,
            minmax_norm=minmax_norm,
            freq_mask=freq_mask,
            time_mask=time_mask,
            n_freq_masks=args.n_freq_masks,
            n_time_masks=args.n_time_masks,
            cache_dir=sc_cache,
            fmin=fmin_cfg,
            htk=htk_cfg,
            gain_aug=args.gain_aug,
            gain_db=args.gain_db,
        )
        train_ds = ConcatDataset([train_ds, sc_ds])
        # Val stays as XC OOF fold — all 66 soundscapes used for training
        n_sc_files = sc_labels["filename"].nunique()
        print(
            f"Soundscape labels: {len(sc_ds)} segments ({n_sc_files} soundscapes) added to train"
        )
        print(f"Combined train set: {len(train_ds)} samples | Val: XC OOF fold")
        if sc_cache:
            print(f"  Using spec cache: {sc_cache.name}")
        else:
            print(
                "  WARNING: soundscape spec cache not found — slow on-the-fly loading"
            )
            print("  Run: python precompute_specs.py --soundscapes")

    # Pseudo-labeled soundscape segments
    if args.pseudo_csv:
        soundscape_dir = DATA / "train_soundscapes"
        pseudo_ds = SoundscapeDataset(
            pseudo_csv=Path(args.pseudo_csv),
            soundscape_dir=soundscape_dir,
            species_to_idx=species_to_idx,
            n_species=n_species,
            augment=True,
            label_weight=args.pseudo_weight,
            n_mels=n_mels_cfg,
            n_fft=n_fft_cfg,
            hop_length=hop_length_cfg,
            minmax_norm=minmax_norm,
            freq_mask=freq_mask,
            time_mask=time_mask,
            n_freq_masks=args.n_freq_masks,
            n_time_masks=args.n_time_masks,
        )
        train_ds = ConcatDataset([train_ds, pseudo_ds])
        print(
            f"Pseudo-labels: {len(pseudo_ds)} segments added (weight={args.pseudo_weight})"
        )
        print(f"Combined train set: {len(train_ds)} samples")

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

    if args.birdset:
        model = BirdModelBirdSet(n_species).to(device)
    elif args.baseline:
        model = BirdModelBaseline(args.backbone, n_species).to(device)
    elif args.sed:
        model = BirdModelSED(args.backbone, n_species).to(device)
    else:
        model = BirdModel(args.backbone, n_species).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.warm_restarts:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.warm_restarts_t0, eta_min=1e-5
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.patience * 3, eta_min=1e-5
        )

    OUT.mkdir(exist_ok=True)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    if args.tag:
        best_path = OUT / f"{args.tag}_fold{args.fold}.pt"
    else:
        if args.birdset:
            suffix = "_birdset"
        elif args.baseline:
            suffix = "_baseline"
        elif args.sed:
            suffix = "_sed"
        else:
            suffix = ""
        ckpt_name = "efficientnet_b1" if args.birdset else args.backbone
        best_path = OUT / f"{ckpt_name}{suffix}_fold{args.fold}.pt"

    # ── Stage 1: Perch soft-label pretraining ──────────────────────────────────
    if args.perch_npz:
        soundscape_dir = DATA / "train_soundscapes"
        perch_cache = (
            SOUNDSCAPE_CACHE_DIR_BASELINE
            if SOUNDSCAPE_CACHE_DIR_BASELINE.exists()
            else None
        )
        if perch_cache:
            print(f"Perch spec cache: {perch_cache}")
        else:
            print(
                "WARNING: Soundscape spec cache not found — using slow on-the-fly loading."
            )
            print("  Run: python precompute_specs.py --soundscapes")
        perch_ds = PerchDataset(
            npz_path=Path(args.perch_npz),
            soundscape_dir=soundscape_dir,
            species_to_idx=species_to_idx,
            n_species=n_species,
            augment=True,
            n_mels=n_mels_cfg,
            n_fft=n_fft_cfg,
            hop_length=hop_length_cfg,
            minmax_norm=minmax_norm,
            freq_mask=freq_mask,
            time_mask=time_mask,
            n_freq_masks=args.n_freq_masks,
            n_time_masks=args.n_time_masks,
            cache_dir=perch_cache,
            min_max_prob=args.perch_min_prob,
        )
        perch_loader = DataLoader(
            perch_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )
        print(
            f"\n── Stage 1: Perch pretraining — {len(perch_ds)} windows, "
            f"{args.perch_epochs} epochs ──"
        )
        is_prob_model = args.sed or args.baseline or args.birdset
        use_dual_loss = args.baseline or args.birdset
        for epoch in range(1, args.perch_epochs + 1):
            t0 = time.time()
            train_loss = train_epoch(
                model,
                perch_loader,
                optimizer,
                device,
                sed=is_prob_model,
                baseline=use_dual_loss,
            )
            scheduler.step()
            print(
                f"  [Perch] Epoch {epoch:02d}/{args.perch_epochs} | "
                f"train={train_loss:.4f} | {time.time() - t0:.0f}s",
                flush=True,
            )
        print("── Stage 1 done. Starting stage 2 (XC fine-tuning) ──\n")

    # ── Stage 2: XC hard-label fine-tuning (with early stopping) ───────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        is_prob_model = args.sed or args.baseline or args.birdset
        use_dual_loss = args.baseline or args.birdset
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            sed=is_prob_model,
            baseline=use_dual_loss,
            ce_loss=args.ce_loss,
        )
        val_loss, ml_above, ml_top2 = val_epoch(
            model,
            val_loader,
            device,
            sed=is_prob_model,
            baseline=use_dual_loss,
            ce_loss=args.ce_loss,
        )
        scheduler.step()

        elapsed = time.time() - t0
        marker = " *" if val_loss < best_val_loss else ""
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_loss:.4f} val={val_loss:.4f} | "
            f"ml_above={ml_above:.1f} top2={ml_top2:.3f} | "
            f"{elapsed:.0f}s{marker}",
            flush=True,
        )
        if ml_above < 0.5 or ml_top2 < 0.02:
            print(
                f"  ⚠ MULTILABEL COLLAPSE: ml_above={ml_above:.2f} top2={ml_top2:.4f} — "
                "model predicts near-zero for co-occurring species. "
                "Check loss function (CE loss incompatible with multilabel).",
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
                    "birdset": args.birdset,
                    "backbone": args.backbone,
                    "n_mels": n_mels_cfg,
                    "n_fft": n_fft_cfg,
                    "hop_length": hop_length_cfg,
                    "minmax_norm": minmax_norm,
                    "hard_labels": args.hard_labels,
                    "tag": args.tag,
                    "dual_loss": use_dual_loss,
                    "freq_mask": freq_mask,
                    "time_mask": time_mask,
                    "n_freq_masks": args.n_freq_masks,
                    "n_time_masks": args.n_time_masks,
                    "patience": args.patience,
                    "lr": args.lr,
                    "seed": args.seed,
                    "fold": args.fold,
                    "n_folds": args.n_folds,
                    "pseudo_csv": args.pseudo_csv,
                    "bg_noise_p": args.bg_noise_p,
                    "perch_npz": args.perch_npz,
                    "perch_epochs": args.perch_epochs,
                    "perch_min_prob": args.perch_min_prob,
                    "soundscape_labels": args.soundscape_labels,
                    "htk": args.htk,
                    "fmin": fmin_cfg,
                    "warm_restarts": args.warm_restarts,
                    "gain_aug": args.gain_aug,
                    "gain_db": args.gain_db,
                    "ce_loss": args.ce_loss,
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
