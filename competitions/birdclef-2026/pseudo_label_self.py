"""
Self-training pseudo-labeler for BirdCLEF 2026 train soundscapes.

Runs the current best 5-fold ensemble on all 10,658 train soundscapes
(each 60s → 12 × 5s windows = 127,896 segments) to produce pseudo-labels
for all 234 competition species.

Output:
    data/birdclef/birdclef-2026/soundscape_pseudo_labels.csv
        Columns: soundscape_filename, start_sec, end_sec, primary_label, max_prob
        One row per (soundscape, 5s window) where max prediction > THRESHOLD.
        primary_label: semicolon-separated predicted species.

Usage (on GPU server):
    KEGO_PATH_DATA=/home/kristian/projects/kego/data \\
        CUDA_VISIBLE_DEVICES=0 uv run python \\
        competitions/birdclef-2026/pseudo_label_self.py \\
        --ckpt-pattern "tf_efficientnet_b0.ns_jft_in1k_baseline_fold*.pt"

    # Use both GPUs (split soundscapes):
    ... --start-idx 0    --end-idx 5329 --gpu 0 &
    ... --start-idx 5329 --end-idx 10658 --gpu 1 &
    # Then merge output CSVs.
"""

import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn

# ── paths ──────────────────────────────────────────────────────────────────────
DATA = Path(os.getenv("KEGO_PATH_DATA", "data")) / "birdclef" / "birdclef-2026"
SOUNDSCAPE_DIR = DATA / "train_soundscapes"
TAXONOMY_CSV = DATA / "taxonomy.csv"
OUT_DIR = DATA

SR = 32_000
CLIP_SAMPLES = SR * 5


# ── model definitions (copied from train.py) ───────────────────────────────────
def _load_checkpoint_meta(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    return {
        "n_mels": ckpt.get("n_mels", 224),
        "n_fft": ckpt.get("n_fft", 2048),
        "hop_length": ckpt.get("hop_length", 512),
        "minmax_norm": ckpt.get("minmax_norm", True),
        "baseline": ckpt.get("baseline", True),
        "birdset": ckpt.get("birdset", False),
        "backbone": ckpt.get("backbone", "tf_efficientnet_b0.ns_jft_in1k"),
        "state_dict": ckpt["model"],
    }


def _make_melspec(
    y: np.ndarray, n_mels: int, n_fft: int, hop_length: int
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        fmin=20,
        fmax=16000,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def _spec_to_tensor(spec: np.ndarray, minmax: bool) -> torch.Tensor:
    if minmax:
        s_min, s_max = spec.min(), spec.max()
        spec = (spec - s_min) / (s_max - s_min + 1e-7)
    else:
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    img = np.stack([spec, spec, spec], axis=0)
    return torch.from_numpy(img).float()


def _build_model(meta: dict, n_species: int, device: torch.device) -> nn.Module:
    import timm

    backbone = meta["backbone"]
    in_ch = 1 if meta["birdset"] else 3
    n_mels = meta["n_mels"]

    class GEMFreqPool(nn.Module):
        def __init__(self, p_init: float = 3.0):
            super().__init__()
            self.p = nn.Parameter(torch.ones(1) * p_init)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            p = torch.clamp(self.p, 1.0, 10.0)
            return (x.clamp(min=1e-6).pow(p).mean(dim=2)).pow(1.0 / p)

    class AttentionSEDHead(nn.Module):
        def __init__(self, feat_dim: int, n_cls: int, dropout: float = 0.1):
            super().__init__()
            self.att = nn.Linear(feat_dim, n_cls)
            self.cls = nn.Linear(feat_dim, n_cls)
            self.drop = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor):
            att = torch.softmax(self.att(self.drop(x)), dim=1)
            cls = self.cls(self.drop(x))
            return (att * torch.sigmoid(cls)).sum(dim=1)

    if meta["birdset"]:
        from transformers import EfficientNetConfig, EfficientNetModel

        cfg_path = next(DATA.parent.parent.rglob("*birdset*config*.json"), None)
        assert cfg_path, "BirdSet config JSON not found"
        cfg = EfficientNetConfig.from_pretrained(str(cfg_path))
        enc = EfficientNetModel(cfg)
        feat_dim = 1280

        class _BirdSet(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = enc
                self.gem = GEMFreqPool()
                self.head = AttentionSEDHead(feat_dim, n_species)

            def forward(self, x):
                out = self.encoder(pixel_values=x)
                f = out.last_hidden_state
                f = self.gem(f)
                return self.head(f)

        model = _BirdSet()
    else:
        enc = timm.create_model(
            backbone, pretrained=False, in_chans=in_ch, num_classes=0
        )
        feat_dim = enc.num_features

        class _Baseline(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = enc
                self.gem = GEMFreqPool()
                self.head = AttentionSEDHead(feat_dim, n_species)

            def forward(self, x):
                f = self.encoder.forward_features(x)
                f = self.gem(f)
                return self.head(f)

        model = _Baseline()

    model.load_state_dict(meta["state_dict"])
    return model.eval().to(device)


# ── inference ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_soundscape(
    y: np.ndarray,
    models: list[tuple],  # [(model, meta), ...]
    device: torch.device,
    species_list: list[str],
) -> np.ndarray:
    """Returns (n_windows, n_species) mean sigmoid predictions across all folds."""
    n_windows = len(y) // CLIP_SAMPLES
    if n_windows == 0:
        return np.zeros((0, len(species_list)), dtype=np.float32)

    all_fold_preds = []
    for model, meta in models:
        fold_preds = []
        for w in range(n_windows):
            chunk = y[w * CLIP_SAMPLES : (w + 1) * CLIP_SAMPLES]
            spec = _make_melspec(
                chunk, meta["n_mels"], meta["n_fft"], meta["hop_length"]
            )
            x = _spec_to_tensor(spec, meta["minmax_norm"]).unsqueeze(0).to(device)
            out = model(x)  # (1, n_species)
            fold_preds.append(torch.sigmoid(out).cpu().numpy()[0])
        all_fold_preds.append(np.stack(fold_preds))  # (n_windows, n_species)

    return np.mean(all_fold_preds, axis=0)  # (n_windows, n_species)


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    import csv
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-pattern",
        default="tf_efficientnet_b0.ns_jft_in1k_baseline_fold*.pt",
        help="Glob pattern for checkpoint files in outputs/",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Min sigmoid prob to include a species in pseudo-label",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load taxonomy
    with open(TAXONOMY_CSV) as f:
        species_list = [row["primary_label"] for row in csv.DictReader(f)]
    n_species = len(species_list)

    # load checkpoints
    ckpt_dir = Path(__file__).parent / "outputs"
    ckpt_paths = sorted(ckpt_dir.glob(args.ckpt_pattern))
    assert ckpt_paths, (
        f"No checkpoints found matching {args.ckpt_pattern} in {ckpt_dir}"
    )
    print(f"Loading {len(ckpt_paths)} fold checkpoints...")
    models = []
    for p in ckpt_paths:
        meta = _load_checkpoint_meta(p)
        model = _build_model(meta, n_species, device)
        models.append((model, meta))
        print(
            f"  {p.name}  n_mels={meta['n_mels']}  epoch={torch.load(p, map_location='cpu').get('epoch')}"
        )

    # soundscape files
    sc_files = sorted(SOUNDSCAPE_DIR.glob("*.ogg"))
    end_idx = args.end_idx or len(sc_files)
    sc_files = sc_files[args.start_idx : end_idx]
    print(f"\nSoundscapes to process: {len(sc_files)}")

    suffix = f"_{args.start_idx}_{end_idx}" if args.start_idx or args.end_idx else ""
    out_csv = OUT_DIR / f"soundscape_pseudo_labels{suffix}.csv"

    t_start = time.time()
    n_rows_written = 0

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "soundscape_filename",
                "start_sec",
                "end_sec",
                "primary_label",
                "max_prob",
                "n_species",
            ],
        )
        writer.writeheader()

        for i, sc_path in enumerate(sc_files):
            try:
                y, _ = librosa.load(sc_path, sr=SR, mono=True)
            except Exception as e:
                print(f"  WARN: {sc_path.name}: {e}", flush=True)
                continue

            preds = predict_soundscape(
                y, models, device, species_list
            )  # (n_windows, 234)

            for w in range(len(preds)):
                row_preds = preds[w]
                predicted = np.where(row_preds >= args.threshold)[0]
                if len(predicted) == 0:
                    continue
                predicted_species = [species_list[j] for j in predicted]
                writer.writerow(
                    {
                        "soundscape_filename": sc_path.name,
                        "start_sec": w * 5,
                        "end_sec": (w + 1) * 5,
                        "primary_label": ";".join(predicted_species),
                        "max_prob": float(row_preds.max()),
                        "n_species": len(predicted_species),
                    }
                )
                n_rows_written += 1

            if (i + 1) % 100 == 0 or i == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (len(sc_files) - i - 1) / rate
                print(
                    f"  [{i + 1}/{len(sc_files)}]  {rate:.1f} files/s  "
                    f"pseudo-rows so far: {n_rows_written}  ETA {eta / 60:.1f} min",
                    flush=True,
                )

    total_windows = len(sc_files) * 12
    print(
        f"\nPseudo-labeled rows: {n_rows_written} / {total_windows} windows "
        f"({100 * n_rows_written / total_windows:.1f}%)"
    )
    print(f"Saved → {out_csv}")
    print(f"Total time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
