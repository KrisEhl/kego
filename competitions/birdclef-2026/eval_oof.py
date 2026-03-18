"""BirdCLEF+ 2026 — Out-of-fold evaluation.

Runs inference on each fold's validation set using the corresponding
checkpoint and computes:
  - cmAP  (class-mean average precision, the competition metric)
  - macro ROC-AUC
  - per-class AP (saved to CSV for inspection)

Usage:
    python eval_oof.py
    python eval_oof.py --backbone convnext_small  # if you trained a different arch
    python eval_oof.py --folds 0 1 2              # subset of folds
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import (
    DATA,
    BirdDataset,
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
    """Class-mean average precision. Skips classes with no positive labels."""
    n_classes = labels.shape[1]
    aps = []
    valid_mask = []
    for c in range(n_classes):
        if labels[:, c].sum() == 0:
            valid_mask.append(False)
            aps.append(np.nan)
            continue
        ap = average_precision_score(labels[:, c], preds[:, c])
        aps.append(ap)
        valid_mask.append(True)
    cmap = np.nanmean(aps)
    return cmap, np.array(aps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
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
            fold_auc = roc_auc_score(labels, preds, average="macro")
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
        oof_auc = roc_auc_score(all_labels, all_preds, average="macro")
    except ValueError:
        oof_auc = float("nan")
    print(f"OOF cmAP   : {oof_cmap:.4f}")
    print(f"OOF ROC-AUC: {oof_auc:.4f}")

    # Per-class AP breakdown
    ap_df = pd.DataFrame(
        {
            "primary_label": species,
            "ap": oof_aps,
            "n_positive": all_labels.sum(axis=0).astype(int),
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


if __name__ == "__main__":
    main()
