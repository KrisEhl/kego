"""Calibrate Perch + CNN blend weight α on 59 labeled training soundscapes.

Uses:
- Perch raw logits from jaejohn/perch-meta cache (708 windows × 234 species)
- CNN (soundscape-v7) predictions on the same 59 soundscape files
- Ground truth from train_soundscapes_labels.csv

Outputs the cmAP curve over α ∈ [0, 1] so we can pick the best blend weight.

Usage:
    python eval/calibrate_blend_alpha.py
    python eval/calibrate_blend_alpha.py --ckpt-pattern "soundscape-v7_fold*.pt"
    python eval/calibrate_blend_alpha.py --perch-meta /path/to/perch-meta/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "training"))

from train_cnn import (  # noqa: E402
    SOUNDSCAPE_CACHE_DIR_BASELINE_HTK,
    BirdModel,
    BirdModelBaseline,
    SoundscapeLabelsDataset,
)

DATA = Path("/home/kristian/projects/kego/data") / "birdclef" / "birdclef-2026"
LABELS_CSV = DATA / "train_soundscapes_labels.csv"
SOUNDSCAPE_DIR = DATA / "train_soundscapes"
CKPT_DIR = ROOT / "outputs"
DEFAULT_PERCH_META = Path("/home/kristian/projects/kego/data/perch-meta")


def load_model(path: Path, n_species: int, device: torch.device) -> nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    backbone = ckpt.get("backbone", "tf_efficientnet_b0.ns_jft_in1k")
    if "head.cls_conv.bias" in ckpt["model"]:
        model = BirdModelBaseline(backbone=backbone, n_classes=n_species)
    else:
        model = BirdModel(backbone=backbone, n_classes=n_species)
    model.load_state_dict(ckpt["model"])
    return model.eval().to(device)


def load_meta(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "backbone": ckpt.get("backbone", "tf_efficientnet_b0.ns_jft_in1k"),
        "n_mels": ckpt.get("n_mels", 224),
        "n_fft": ckpt.get("n_fft", 2048),
        "hop_length": ckpt.get("hop_length", 512),
        "minmax_norm": ckpt.get("minmax_norm", True),
        "htk": ckpt.get("htk", True),
        "fmin": ckpt.get("fmin", 0),
        "is_baseline": "head.cls_conv.bias" in ckpt["model"],
    }


def compute_cmap(preds: np.ndarray, labels: np.ndarray) -> float:
    bin_labels = (labels > 0).astype(np.int32)
    aps = []
    for c in range(labels.shape[1]):
        if bin_labels[:, c].sum() > 0:
            aps.append(average_precision_score(bin_labels[:, c], preds[:, c]))
    return float(np.mean(aps)) if aps else float("nan")


@torch.no_grad()
def get_cnn_preds(
    df: pd.DataFrame,
    ckpt_paths: list[Path],
    meta: dict,
    species_to_idx: dict,
    n_species: int,
    device: torch.device,
) -> np.ndarray:
    """Return averaged CNN predictions (n_windows, n_species) for given segments."""
    cache_dir = SOUNDSCAPE_CACHE_DIR_BASELINE_HTK if meta["htk"] else None
    ds = SoundscapeLabelsDataset(
        df,
        soundscape_dir=SOUNDSCAPE_DIR,
        species_to_idx=species_to_idx,
        n_species=n_species,
        augment=False,
        n_mels=meta["n_mels"],
        n_fft=meta["n_fft"],
        hop_length=meta["hop_length"],
        minmax_norm=meta["minmax_norm"],
        cache_dir=cache_dir,
        fmin=meta["fmin"],
        htk=meta["htk"],
    )
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    fold_preds = []
    for p in ckpt_paths:
        model = load_model(p, n_species, device)
        preds_list, labels_list = [], []
        for x, y in loader:
            out = model(x.to(device))
            if not meta["is_baseline"]:
                out = torch.sigmoid(out)
            preds_list.append(out.cpu().numpy())
            labels_list.append(y.numpy())
        fold_preds.append(np.concatenate(preds_list))

    labels_arr = np.concatenate(labels_list)
    cnn_preds = np.mean(fold_preds, axis=0)
    return cnn_preds, labels_arr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-pattern", default="soundscape-v7_fold*.pt")
    parser.add_argument("--perch-meta", default=str(DEFAULT_PERCH_META))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Species from taxonomy
    taxonomy = pd.read_csv(DATA / "taxonomy.csv")
    species = sorted(taxonomy["primary_label"].astype(str).tolist())
    species_to_idx = {s: i for i, s in enumerate(species)}
    n_species = len(species)
    print(f"Species: {n_species}")

    # Load perch-meta cache
    perch_meta_dir = Path(args.perch_meta)
    npz_path = perch_meta_dir / "full_perch_arrays.npz"
    parquet_path = perch_meta_dir / "full_perch_meta.parquet"
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found. Download jaejohn/perch-meta dataset first.")
        sys.exit(1)

    npz = np.load(npz_path)
    scores_raw = npz["scores_full_raw"]  # (708, 234) raw Perch logits
    print(f"Perch cache: {scores_raw.shape} windows × species")

    perch_df = pd.read_parquet(parquet_path)  # row_id, filename, site, hour_utc
    print(f"Perch meta: {len(perch_df)} rows, {perch_df['filename'].nunique()} soundscapes")

    # Perch raw logits → probabilities via sigmoid
    perch_probs = 1.0 / (1.0 + np.exp(-scores_raw.astype(np.float64))).astype(np.float32)

    # Load CNN checkpoints
    ckpt_paths = sorted(CKPT_DIR.glob(args.ckpt_pattern))
    if not ckpt_paths:
        print(f"ERROR: No checkpoints matching '{args.ckpt_pattern}' in {CKPT_DIR}")
        sys.exit(1)
    print(f"\nLoading {len(ckpt_paths)} CNN checkpoints ({args.ckpt_pattern})")
    meta = load_meta(ckpt_paths[0])
    print(f"  n_mels={meta['n_mels']} htk={meta['htk']} fmin={meta['fmin']}")

    # Build labels DF for the same 59 soundscapes × 12 windows
    # Join perch row_ids against train_soundscapes_labels.csv
    sc_labels = pd.read_csv(LABELS_CSV)

    # For each perch window (row_id = filename_stem + "_" + end_sec),
    # find matching labeled segment in sc_labels
    # sc_labels has: filename, start (HH:MM:SS), end (HH:MM:SS), primary_label
    # perch row_id: BC2026_Train_NNNN_SXX_..._5 → filename BC2026_Train_NNNN_SXX_....ogg, end_sec=5

    # Build a segment DataFrame that matches the perch window format
    # Each perch window is a 5s slot at end_sec = 5,10,...,60
    # Map to sc_labels where start_sec = end_sec - 5

    def parse_seconds(t: str) -> int:
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)

    sc_labels["start_sec"] = sc_labels["start"].apply(parse_seconds)
    sc_labels["end_sec"] = sc_labels["end"].apply(parse_seconds)

    # Build a lookup: (filename, end_sec) → list of labels
    label_lookup: dict[tuple, list[str]] = {}
    for _, row in sc_labels.iterrows():
        key = (row["filename"], int(row["end_sec"]))
        labels = [s.strip() for s in str(row["primary_label"]).split(";") if s.strip()]
        label_lookup.setdefault(key, []).extend(labels)

    # Build label matrix for perch windows
    labels_arr = np.zeros((len(perch_df), n_species), dtype=np.float32)
    has_label = np.zeros(len(perch_df), dtype=bool)
    for i, row in perch_df.iterrows():
        fname = row["filename"]
        end_sec = int(row["row_id"].split("_")[-1])
        key = (fname, end_sec)
        if key in label_lookup:
            for sp in label_lookup[key]:
                if sp in species_to_idx:
                    labels_arr[i, species_to_idx[sp]] = 1.0
            has_label[i] = True

    n_labeled = has_label.sum()
    print(f"\nPerch windows with ground-truth labels: {n_labeled}/{len(perch_df)}")
    if n_labeled == 0:
        print("ERROR: No labeled windows found — check filename/end_sec matching")
        sys.exit(1)

    # Get CNN predictions for the same soundscape files
    # Build a DataFrame of (filename, start, end, primary_label) for the perch soundscapes
    perch_files = set(perch_df["filename"].tolist())
    sc_for_cnn = sc_labels[sc_labels["filename"].isin(perch_files)].reset_index(drop=True)
    print(f"CNN inference: {len(sc_for_cnn)} labeled segments from {sc_for_cnn['filename'].nunique()} soundscapes")

    cnn_preds_labeled, cnn_labels_labeled = get_cnn_preds(
        sc_for_cnn, ckpt_paths, meta, species_to_idx, n_species, device
    )
    print(f"CNN predictions: {cnn_preds_labeled.shape}")

    # Match CNN predictions to perch windows by (filename, end_sec)
    # sc_for_cnn rows correspond to cnn_preds_labeled rows in order
    # For each perch window, find the matching sc_for_cnn row
    cnn_for_perch = np.zeros((len(perch_df), n_species), dtype=np.float32)
    cnn_matched = np.zeros(len(perch_df), dtype=bool)
    sc_for_cnn_key = {(row["filename"], int(row["end_sec"])): idx for idx, (_, row) in enumerate(sc_for_cnn.iterrows())}
    for i, row in perch_df.iterrows():
        fname = row["filename"]
        end_sec = int(row["row_id"].split("_")[-1])
        key = (fname, end_sec)
        if key in sc_for_cnn_key:
            cnn_for_perch[i] = cnn_preds_labeled[sc_for_cnn_key[key]]
            cnn_matched[i] = True

    n_matched = cnn_matched.sum()
    print(f"CNN-Perch window matches: {n_matched}/{len(perch_df)}")

    # Use windows that have BOTH ground-truth labels AND CNN predictions
    eval_mask = has_label  # all labeled windows (even if CNN is 0 for unlabeled ones)
    eval_perch = perch_probs[eval_mask]
    eval_cnn = cnn_for_perch[eval_mask]
    eval_labels = labels_arr[eval_mask]

    print(f"\nEval windows: {eval_mask.sum()} (with ground-truth labels)")
    n_pos_classes = (eval_labels.sum(0) > 0).sum()
    print(f"Classes with positives: {n_pos_classes}/{n_species}")

    # cmAP for Perch alone and CNN alone
    cmap_perch = compute_cmap(eval_perch, eval_labels)
    cmap_cnn = compute_cmap(eval_cnn, eval_labels)
    print(f"\nPerch alone  cmAP: {cmap_perch:.4f}")
    print(f"CNN alone    cmAP: {cmap_cnn:.4f}")

    # Grid-search α: final = α * cnn + (1-α) * perch
    print("\n--- α sweep (α = CNN weight, 1-α = Perch weight) ---")
    best_alpha = 0.0
    best_cmap = cmap_perch
    alphas = np.arange(0.0, 1.01, 0.05)
    for alpha in alphas:
        blended = alpha * eval_cnn + (1.0 - alpha) * eval_perch
        cmap = compute_cmap(blended, eval_labels)
        marker = " ← best" if cmap > best_cmap else ""
        print(f"  α={alpha:.2f}  cmAP={cmap:.4f}{marker}")
        if cmap > best_cmap:
            best_cmap = cmap
            best_alpha = alpha

    print(f"\nBest α (CNN weight): {best_alpha:.2f}  →  cmAP={best_cmap:.4f}")
    print(f"  Blend: {best_alpha:.2f} × CNN  +  {1 - best_alpha:.2f} × Perch")
    print(f"  vs Perch alone: {cmap_perch:.4f}  (+{best_cmap - cmap_perch:.4f})")
    print(f"  vs CNN alone:   {cmap_cnn:.4f}  (+{best_cmap - cmap_cnn:.4f})")


if __name__ == "__main__":
    main()
