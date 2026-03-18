"""BirdCLEF+ 2026 — Inference on test soundscapes.

Sliding window over each soundscape (5s chunks, 2.5s stride).
Max-pools predictions per soundscape, writes submission.csv.

Usage:
    python predict.py --models outputs/efficientnet_b0_fold*.pt
    python predict.py --models outputs/efficientnet_b3_fold*.pt outputs/convnext_small_fold*.pt
"""

import argparse
import glob
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from train import (
    CLIP_SAMPLES,
    DATA,
    SR,
    make_melspec,
    spec_to_tensor,
)


class BirdModel(nn.Module):
    def __init__(self, backbone: str, n_classes: int):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=False, num_classes=n_classes, in_chans=3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


@torch.no_grad()
def predict_soundscape(
    path: Path,
    models: list,
    device: torch.device,
    stride: float = 2.5,
) -> np.ndarray:
    """Sliding window inference. Returns (n_chunks, n_species) sigmoid preds."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    stride_samples = int(stride * SR)

    chunks = []
    pos = 0
    while pos + CLIP_SAMPLES <= len(y):
        chunk = y[pos : pos + CLIP_SAMPLES]
        spec = make_melspec(chunk)
        chunks.append(spec_to_tensor(spec))
        pos += stride_samples

    if not chunks:
        return np.zeros((1, models[0].backbone.num_classes))

    batch = torch.stack(chunks).to(device)  # (n_chunks, 3, n_mels, time)

    all_preds = []
    for model in models:
        logits = model(batch)
        all_preds.append(torch.sigmoid(logits).cpu().numpy())

    # Average across models, max-pool across chunks
    preds = np.mean(all_preds, axis=0)  # (n_chunks, n_species)
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", required=True, help="Checkpoint paths or glob"
    )
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Resolve globs in model paths
    model_paths = []
    for pattern in args.models:
        model_paths.extend(sorted(glob.glob(pattern)) or [pattern])

    # Species list from taxonomy (234 species = all submission targets)
    taxonomy = pd.read_csv(DATA / "taxonomy.csv")
    species = sorted(taxonomy["primary_label"].astype(str).tolist())
    n_species = len(species)

    # Load models
    models = []
    for path in model_paths:
        ckpt = torch.load(path, map_location=device)
        model = BirdModel(args.backbone, n_species).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models.append(model)
        print(f"Loaded {path} (val_loss={ckpt.get('val_loss', '?'):.4f})")

    # Find test soundscapes
    test_dir = DATA / "test_soundscapes"
    soundscapes = sorted(test_dir.rglob("*.ogg")) + sorted(test_dir.rglob("*.wav"))
    print(f"\nRunning inference on {len(soundscapes)} soundscapes...")

    # Load sample submission to get expected format
    sample_sub = pd.read_csv(DATA / "sample_submission.csv")
    print(f"Sample submission shape: {sample_sub.shape}")
    print(sample_sub.head(3))

    rows = []
    for sc_path in soundscapes:
        preds = predict_soundscape(sc_path, models, device, stride=5.0)
        # Row IDs: {soundscape_stem}_{end_second} at 5, 10, 15, ...
        for i, chunk_preds in enumerate(preds):
            end_sec = (i + 1) * 5
            row_id = f"{sc_path.stem}_{end_sec}"
            row = {"row_id": row_id}
            for j, sp in enumerate(species):
                row[sp] = float(chunk_preds[j])
            rows.append(row)

    sub = pd.DataFrame(rows)
    # Keep only columns that exist in sample submission
    sub_cols = [c for c in sample_sub.columns if c in sub.columns]
    sub = sub[["row_id"] + [c for c in sub_cols if c != "row_id"]]

    out_path = Path("submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"\nSubmission saved: {out_path} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
