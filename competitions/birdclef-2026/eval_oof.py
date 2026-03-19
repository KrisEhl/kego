"""BirdCLEF+ 2026 — Out-of-fold evaluation.

Runs inference on each fold's validation set using the corresponding
checkpoint and computes:
  - cmAP  (class-mean average precision, the competition metric)
  - macro ROC-AUC
  - per-class AP (saved to CSV for inspection)

Also evaluates on the 66 labeled train soundscapes (sliding-window, ensemble
of all fold checkpoints) — this is the closest local proxy to LB.

Usage:
    python eval_oof.py
    python eval_oof.py --backbone efficientnet_b3
    python eval_oof.py --skip-soundscapes   # clip OOF only
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import (
    CLIP_SAMPLES,
    DATA,
    SR,
    BirdDataset,
    make_melspec,
    spec_to_tensor,
)

OUT = Path("outputs")


class BirdModel(nn.Module):
    def __init__(self, backbone: str, n_classes: int):
        import timm

        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=False, num_classes=n_classes, in_chans=3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


@torch.no_grad()
def run_inference(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (preds, labels) both shape (N, n_species)."""
    model.eval()
    all_preds, all_labels = [], []
    for x, y in tqdm(loader, desc="Inference", leave=False):
        x = x.to(device)
        logits = model(x)
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_cmap(preds: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    """Class-mean average precision. Skips classes with no positive labels.
    Labels are binarized (>0) to handle soft secondary-label targets.
    """
    bin_labels = (labels > 0).astype(np.int32)
    n_classes = labels.shape[1]
    aps = []
    for c in range(n_classes):
        if bin_labels[:, c].sum() == 0:
            aps.append(np.nan)
            continue
        ap = average_precision_score(bin_labels[:, c], preds[:, c])
        aps.append(ap)
    cmap = float(np.nanmean(aps))
    return cmap, np.array(aps)


@torch.no_grad()
def eval_soundscapes(
    models: list[nn.Module],
    species: list[str],
    species_to_idx: dict[str, int],
    device: torch.device,
) -> float:
    """Sliding-window inference on the 66 labeled train soundscapes.

    Uses the full model ensemble (all fold checkpoints averaged).
    Returns soundscape-level cmAP — much closer to LB than clip OOF.
    """
    sc_labels = pd.read_csv(DATA / "train_soundscapes_labels.csv")
    sc_dir = DATA / "train_soundscapes"
    n_species = len(species)

    all_preds, all_labels = [], []

    for filename in tqdm(sc_labels["filename"].unique(), desc="Soundscapes"):
        sc_path = sc_dir / filename
        if not sc_path.exists():
            continue

        # Load full soundscape
        y, _ = librosa.load(sc_path, sr=SR, mono=True)

        # Slice into 5s non-overlapping chunks
        chunks = []
        pos = 0
        while pos + CLIP_SAMPLES <= len(y):
            chunk = y[pos : pos + CLIP_SAMPLES]
            chunks.append(spec_to_tensor(make_melspec(chunk)))
            pos += CLIP_SAMPLES
        if not chunks:
            continue

        batch = torch.stack(chunks).to(device)  # (n_chunks, 3, n_mels, t)

        # Ensemble: average sigmoid predictions across all models
        fold_preds = []
        for model in models:
            model.eval()
            logits = model(batch)
            fold_preds.append(torch.sigmoid(logits).cpu().numpy())
        pred_chunks = np.mean(fold_preds, axis=0)  # (n_chunks, n_species)

        # Match labels: sc_labels has one row per 5s chunk
        sc_rows = sc_labels[sc_labels["filename"] == filename].copy()

        # Convert "HH:MM:SS" end time to chunk index
        def end_to_idx(end_str: str) -> int:
            h, m, s = map(int, end_str.split(":"))
            return (h * 3600 + m * 60 + s) // 5 - 1

        for _, row in sc_rows.iterrows():
            chunk_idx = end_to_idx(row["end"])
            if chunk_idx >= len(pred_chunks):
                continue
            label = np.zeros(n_species, dtype=np.float32)
            for sp in str(row["primary_label"]).split(";"):
                sp = sp.strip()
                if sp in species_to_idx:
                    label[species_to_idx[sp]] = 1.0
            all_preds.append(pred_chunks[chunk_idx])
            all_labels.append(label)

    if not all_preds:
        print("No soundscape predictions collected.")
        return float("nan")

    preds = np.stack(all_preds)  # (N_chunks, n_species)
    labels = np.stack(all_labels)  # (N_chunks, n_species)

    cmap, aps = compute_cmap(preds, labels)

    # Per-class breakdown (only species that appear in soundscape labels)
    present = np.where(labels.sum(axis=0) > 0)[0]
    print("\n--- Soundscape eval ---")
    print(f"Chunks: {len(preds)} | Species with labels: {len(present)}/234")
    print(f"Soundscape cmAP: {cmap:.4f}")
    return cmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-soundscapes",
        action="store_true",
        help="Skip soundscape-level evaluation",
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    folds_to_eval = args.folds or list(range(args.n_folds))

    meta = pd.read_csv(DATA / "train.csv")
    taxonomy = pd.read_csv(DATA / "taxonomy.csv")
    species = sorted(taxonomy["primary_label"].astype(str).tolist())
    species_to_idx = {s: i for i, s in enumerate(species)}
    n_species = len(species)
    meta["primary_label"] = meta["primary_label"].astype(str)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    splits = list(skf.split(meta, meta["primary_label"]))

    all_preds = np.zeros((len(meta), n_species), dtype=np.float32)
    all_labels = np.zeros((len(meta), n_species), dtype=np.float32)

    fold_results = []
    for fold in folds_to_eval:
        ckpt_path = OUT / f"{args.backbone}_fold{fold}.pt"
        if not ckpt_path.exists():
            print(f"[fold {fold}] checkpoint not found: {ckpt_path} — skipping")
            continue

        _, val_idx = splits[fold]
        val_df = meta.iloc[val_idx]

        val_ds = BirdDataset(
            val_df, species_to_idx, n_species, DATA / "train_audio", augment=False
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

        ckpt = torch.load(ckpt_path, map_location=device)
        model = BirdModel(args.backbone, n_species).to(device)
        model.load_state_dict(ckpt["model"])

        preds, labels = run_inference(model, val_loader, device)
        all_preds[val_idx] = preds
        all_labels[val_idx] = labels

        fold_cmap, fold_aps = compute_cmap(preds, labels)
        try:
            fold_auc = roc_auc_score((labels > 0).astype(int), preds, average="macro")
        except ValueError:
            fold_auc = float("nan")

        print(
            f"Fold {fold} | cmAP={fold_cmap:.4f} | ROC-AUC={fold_auc:.4f} "
            f"| val_loss(BCE)={ckpt.get('val_loss', float('nan')):.4f}"
        )
        fold_results.append({"fold": fold, "cmap": fold_cmap, "roc_auc": fold_auc})

        del model

    # OOF aggregate
    print("\n--- OOF aggregate ---")
    oof_cmap, oof_aps = compute_cmap(all_preds, all_labels)
    try:
        oof_auc = roc_auc_score(
            (all_labels > 0).astype(int), all_preds, average="macro"
        )
    except ValueError:
        oof_auc = float("nan")
    print(f"OOF cmAP   : {oof_cmap:.4f}")
    print(f"OOF ROC-AUC: {oof_auc:.4f}")

    # Per-class AP breakdown
    ap_df = pd.DataFrame(
        {
            "primary_label": species,
            "ap": oof_aps,
            "n_positive": (all_labels > 0).sum(axis=0).astype(int),
        }
    ).merge(
        taxonomy[["primary_label", "common_name", "class_name"]],
        on="primary_label",
        how="left",
    )

    ap_df_sorted = ap_df.sort_values("ap")
    print("\n--- Bottom 15 species by AP (hardest) ---")
    print(
        ap_df_sorted.head(15)[
            ["primary_label", "common_name", "class_name", "ap", "n_positive"]
        ].to_string(index=False)
    )
    print("\n--- Top 15 species by AP ---")
    print(
        ap_df_sorted.tail(15)[
            ["primary_label", "common_name", "class_name", "ap", "n_positive"]
        ].to_string(index=False)
    )

    out_csv = OUT / "oof_per_class_ap.csv"
    ap_df.sort_values("ap").to_csv(out_csv, index=False)
    print(f"\nPer-class AP saved → {out_csv}")

    # Soundscape-level evaluation (closest proxy to LB)
    if not args.skip_soundscapes and (DATA / "train_soundscapes_labels.csv").exists():
        print("\n" + "=" * 50)
        print("SOUNDSCAPE-LEVEL EVALUATION (LB proxy)")
        print("=" * 50)
        # Load all available fold checkpoints as an ensemble
        sc_models = []
        for fold in range(args.n_folds):
            ckpt_path = OUT / f"{args.backbone}_fold{fold}.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device)
                m = BirdModel(args.backbone, n_species).to(device)
                m.load_state_dict(ckpt["model"])
                sc_models.append(m)
        print(f"Ensemble: {len(sc_models)} fold checkpoints")
        sc_cmap = eval_soundscapes(sc_models, species, species_to_idx, device)
        print("\nSummary:")
        print(f"  Clip OOF cmAP        : {oof_cmap:.4f}")
        print(f"  Soundscape cmAP      : {sc_cmap:.4f}  ← LB proxy")
        print("  LB (reference)       : check submissions")


if __name__ == "__main__":
    main()
