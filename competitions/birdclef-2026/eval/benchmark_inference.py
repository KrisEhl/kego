"""Local inference benchmark — times each pipeline component per soundscape.

Runs against the 66 labeled soundscapes (same ogg files used by the scoring
environment), so extrapolation to the test set is realistic.


Usage:
    uv run python competitions/birdclef-2026/benchmark_inference.py \
        --tag soundscape-v7 \
        --n-folds 4 \
        [--soundscapes <path>]  # default: data/birdclef/train_soundscapes/

Reports:
    - Per-soundscape: load / mel+batch / inference / total (seconds)
    - Summary stats: mean ± std, min/max per component
    - Extrapolation: estimated time for N test soundscapes at same rate
    - Bottleneck identification
"""

import argparse

# ── Repo path setup ─────────────────────────────────────────────────────────
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

import os

os.environ.setdefault("KEGO_PATH_DATA", str(_repo / "data"))

# ── Imports from train.py ────────────────────────────────────────────────────

# We reuse the model classes from kaggle_inference.ipynb by importing train.py
# for the mel utility, and defining the model classes inline (same as notebook).
import timm
import torch.nn as nn
import torch.nn.functional as F

SR = 32000
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 1024
FMIN = 20
FMAX = 16000
CLIP_DURATION = 5
CLIP_SAMPLES = SR * CLIP_DURATION

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GEMFreqPool(nn.Module):
    def __init__(self, p_init: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p_init))
        self.eps = eps

    def forward(self, x):
        p = self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)


class AttentionSEDHead(nn.Module):
    def __init__(self, feat_dim, num_classes, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)
        )
        self.att_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)
        self.cls_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)
        att = F.softmax(torch.tanh(self.att_conv(x)), dim=-1)
        cls = self.cls_conv(x)
        return torch.sigmoid((att * cls).sum(dim=-1))


class BirdModel(nn.Module):
    def __init__(self, backbone, n_classes, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=n_classes, in_chans=3
        )

    def forward(self, x):
        return self.backbone(x)


class BirdModelBaseline(nn.Module):
    def __init__(self, backbone, n_classes, pretrained=False, dropout=0.1):
        super().__init__()
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="", in_chans=3
        )
        # Probe actual output dim — num_features is unreliable for some backbones (e.g. hgnetv2)
        with torch.no_grad():
            _dummy = torch.zeros(1, 3, 64, 128)
            feat_dim = self.encoder(_dummy).shape[1]
        self.gem_pool = GEMFreqPool(p_init=3.0)
        self.head = AttentionSEDHead(feat_dim, n_classes, dropout)

    def forward(self, x):
        return self.head(self.gem_pool(self.encoder(x)))


def make_melspec(y, n_mels=128, n_fft=1024, hop_length=512, fmin=FMIN, htk=False):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        fmin=fmin,
        fmax=FMAX,
        htk=htk,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def spec_to_tensor_minmax(spec):
    mn, mx = spec.min(), spec.max()
    spec = (spec - mn) / (mx - mn + 1e-6)
    img = np.stack([spec, spec, spec], axis=0)
    return torch.from_numpy(img)


def load_checkpoints(tag: str, n_folds: int, device: torch.device, data_dir: Path):
    """Load model checkpoints for a given experiment tag."""
    outputs_dir = _repo / "competitions" / "birdclef-2026" / "outputs"
    models_cfg = []
    for fold in range(n_folds):
        ckpt_path = outputs_dir / f"{tag}_fold{fold}.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: {ckpt_path} not found, skipping")
            continue
        ckpt = torch.load(ckpt_path, map_location=device)
        backbone = ckpt.get("backbone", "tf_efficientnet_b0.ns_jft_in1k")
        n_mels = ckpt.get("n_mels", 128)
        n_fft = ckpt.get("n_fft", 1024)
        hop_length = ckpt.get("hop_length", 512)
        fmin = ckpt.get("fmin", FMIN)
        htk = ckpt.get("htk", False)

        # Load taxonomy for n_classes
        tax_path = data_dir / "taxonomy.csv"
        import pandas as pd

        taxonomy = pd.read_csv(tax_path)
        n_classes = len(taxonomy)

        # Detect model type from checkpoint key schema (same as kaggle_inference.ipynb)
        if "head.cls_conv.bias" in ckpt["model"]:
            model = BirdModelBaseline(backbone=backbone, n_classes=n_classes).to(device)
        elif "backbone.head.fc.bias" in ckpt["model"]:
            model = BirdModel(backbone=backbone, n_classes=n_classes).to(device)
        else:
            print(f"  WARNING: unknown checkpoint schema in {ckpt_path.name}, skipping")
            continue
        model.load_state_dict(ckpt["model"])
        model.eval()
        models_cfg.append((model, n_mels, n_fft, hop_length, "minmax", fmin, htk))
        val_loss = ckpt.get("val_loss", float("nan"))
        print(f"  Loaded {ckpt_path.name} — {backbone}, val_loss={val_loss:.4f}")
    return models_cfg


def benchmark_file(path: Path, models_cfg: list, device: torch.device) -> dict:
    """Time each phase for a single soundscape file."""
    OVERLAP_STRIDE_S = 2.5
    stride_samples = int(OVERLAP_STRIDE_S * SR)

    # Phase 1: load
    t0 = time.perf_counter()
    y, _ = librosa.load(path, sr=SR, mono=True)
    t_load = time.perf_counter() - t0

    n_slots = len(y) // CLIP_SAMPLES
    positions = list(range(0, len(y) - CLIP_SAMPLES + 1, stride_samples))
    n_windows = len(positions)

    if n_windows == 0:
        return {
            "name": path.name,
            "load": t_load,
            "mel": 0,
            "infer": 0,
            "total": t_load,
            "n_slots": 0,
            "n_windows": 0,
        }

    # Phase 2: mel spectrogram (CPU, separate from inference)
    t1 = time.perf_counter()
    all_chunks = []
    for model, n_mels, n_fft, hop_length, norm_type, fmin, htk in models_cfg:
        chunks = []
        for pos in positions:
            chunk = y[pos : pos + CLIP_SAMPLES]
            spec = make_melspec(
                chunk,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                fmin=fmin,
                htk=htk,
            )
            tensor = spec_to_tensor_minmax(spec)
            chunks.append(tensor)
        all_chunks.append(torch.stack(chunks))
    t_mel = time.perf_counter() - t1

    # Phase 3: model inference (GPU/CPU forward pass)
    t2 = time.perf_counter()
    with torch.no_grad():
        for i, (model, *_) in enumerate(models_cfg):
            batch = all_chunks[i].to(device)
            _ = model(batch).cpu().numpy()
    t_infer = time.perf_counter() - t2

    t_total = time.perf_counter() - t0
    return {
        "name": path.name,
        "load": t_load,
        "mel": t_mel,
        "infer": t_infer,
        "total": t_total,
        "n_slots": n_slots,
        "n_windows": n_windows,
    }


def main():
    parser = argparse.ArgumentParser(description="BirdCLEF inference benchmark")
    parser.add_argument("--tag", default="soundscape-v7", help="Checkpoint tag")
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument(
        "--soundscapes", type=str, default=None, help="Path to soundscape directory"
    )
    parser.add_argument(
        "--n-test-soundscapes",
        type=int,
        default=780,
        help="Number of test soundscapes to extrapolate to",
    )
    parser.add_argument(
        "--max-files", type=int, default=None, help="Limit files for quick testing"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(os.environ["KEGO_PATH_DATA"]) / "birdclef" / "birdclef-2026"
    if args.soundscapes:
        sc_dir = Path(args.soundscapes)
    else:
        sc_dir = data_dir / "train_soundscapes"

    print(f"Data dir:   {data_dir}")
    print(f"Soundscapes: {sc_dir}")

    print(f"\nLoading checkpoints (tag={args.tag}, folds={args.n_folds})...")
    models_cfg = load_checkpoints(args.tag, args.n_folds, device, data_dir)
    if not models_cfg:
        print("ERROR: No checkpoints found. Check --tag and --n-folds.")
        return

    soundscapes = sorted(sc_dir.glob("*.ogg"))
    if not soundscapes:
        for ext in ("*.wav", "*.flac"):
            soundscapes.extend(sorted(sc_dir.glob(ext)))
    if args.max_files:
        soundscapes = soundscapes[: args.max_files]

    print(f"\nBenchmarking {len(soundscapes)} soundscapes...")
    print("-" * 80)

    results = []
    t_run_start = time.perf_counter()
    for i, path in enumerate(soundscapes):
        r = benchmark_file(path, models_cfg, device)
        results.append(r)
        elapsed = time.perf_counter() - t_run_start
        avg = elapsed / (i + 1)
        eta = avg * (len(soundscapes) - i - 1)
        print(
            f"  [{i + 1:3d}/{len(soundscapes)}] {path.name:<40s} "
            f"load={r['load']:.2f}s  mel={r['mel']:.2f}s  infer={r['infer']:.2f}s  "
            f"total={r['total']:.2f}s  ETA={eta:.0f}s"
        )

    # ── Summary stats ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for key in ("load", "mel", "infer", "total"):
        vals = np.array([r[key] for r in results])
        pct = 100 * vals.mean() / np.array([r["total"] for r in results]).mean()
        print(
            f"  {key:6s}: mean={vals.mean():.2f}s  std={vals.std():.2f}s  "
            f"min={vals.min():.2f}s  max={vals.max():.2f}s  "
            f"({pct:.0f}% of total)"
        )

    total_vals = np.array([r["total"] for r in results])
    mean_per_file = total_vals.mean()
    n_test = args.n_test_soundscapes
    est_total_s = mean_per_file * n_test
    budget_s = 90 * 60

    print(f"\n  Files benchmarked:       {len(results)}")
    print(f"  Mean time per file:      {mean_per_file:.2f}s")
    print(f"  Models loaded:           {len(models_cfg)}")
    print(
        f"  Windows per file (mean): {np.mean([r['n_windows'] for r in results]):.1f}"
    )
    print()
    print(f"  Extrapolation to {n_test} test soundscapes:")
    print(f"    Estimated total:       {est_total_s:.0f}s = {est_total_s / 60:.1f} min")
    print(f"    Kaggle budget:         {budget_s}s = {budget_s / 60:.0f} min")
    margin_s = budget_s - est_total_s
    margin_pct = 100 * margin_s / budget_s
    if margin_s > 0:
        print(f"    Margin:                +{margin_s:.0f}s ({margin_pct:.0f}%) — OK")
    else:
        print(
            f"    OVER BUDGET:           {-margin_s:.0f}s ({-margin_pct:.0f}%) over limit!"
        )

    # Bottleneck
    bottleneck = max(
        ("load", "mel", "infer"), key=lambda k: np.mean([r[k] for r in results])
    )
    print(f"\n  Bottleneck: {bottleneck}")


if __name__ == "__main__":
    main()
