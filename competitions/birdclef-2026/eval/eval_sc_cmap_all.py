"""Evaluate sc_cmap for all trained model families on labeled soundscape segments.

Two evaluation modes per group:
 - ALL  : all 1478 labeled segments (leaky for sc-label models)
 - HELD : 14-file held-out val (val_frac=0.15, seed=42) — leakage-free

Usage:
    python eval/eval_sc_cmap_all.py
"""

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
    SOUNDSCAPE_CACHE_DIR_BASELINE,
    SOUNDSCAPE_CACHE_DIR_BASELINE_HTK,
    SOUNDSCAPE_CACHE_DIR_HGNETV2,
    BirdModel,
    BirdModelBaseline,
    SoundscapeLabelsDataset,
)

DATA = Path("/home/kristian/projects/kego/data") / "birdclef" / "birdclef-2026"
LABELS_CSV = DATA / "train_soundscapes_labels.csv"
SOUNDSCAPE_DIR = DATA / "train_soundscapes"

# --- groups: (label, ckpt_dir_relative, glob_pattern, LB or None, trained_with_sc_labels) ---
GROUPS = [
    # (label,                   subdir,           pattern,                          LB,    sc_labels)
    ("b0-5fold", "outputs", "efficientnet_b0_fold*.pt", 0.758, False),
    ("b3-5fold", "outputs", "efficientnet_b3_fold*.pt", 0.776, False),
    ("b3-sed", "outputs", "efficientnet_b3_sed_fold*.pt", 0.750, False),
    ("birdset-b1", "outputs", "efficientnet_b1_birdset_fold*.pt", 0.782, False),
    ("soundscape-v1", "outputs", "soundscape-v1_fold*.pt", 0.827, True),
    ("soundscape-v2", "outputs", "soundscape-v2_fold*.pt", 0.854, True),
    ("soundscape-v3", "outputs", "soundscape-v3_fold*.pt", 0.858, True),
    ("soundscape-v4-CE-dead", "outputs", "soundscape-v4_fold*.pt", 0.723, True),
    ("soundscape-v5", "outputs", "soundscape-v5_fold*.pt", None, True),
    ("soundscape-v6-b1", "outputs", "soundscape-v6-b1_fold*.pt", None, True),
    ("soundscape-v7", "outputs", "soundscape-v7_fold*.pt", 0.882, True),
    ("soundscape-v8-htk", "outputs", "soundscape-v8_fold*.pt", None, True),
    ("soundscape-v8-hgnetv2", "outputs", "soundscape-v8-hgnetv2_fold*.pt", None, True),
    (
        "soundscape-v8-hgnetv2-b",
        "outputs",
        "soundscape-v8-hgnetv2-b_fold*.pt",
        None,
        True,
    ),
    ("perch-v1", "outputs", "perch-v1_fold*.pt", None, True),
    ("perch-v2", "outputs", "perch-v2_fold*.pt", None, True),
    ("soundscape-v9-4fold", "training/outputs", "soundscape-v9_fold*.pt", None, True),
    ("soundscape-v9b-1ep", "training/outputs", "soundscape-v9b_fold*.pt", None, True),
    ("soundscape-v9c-2ep", "training/outputs", "soundscape-v9c_fold*.pt", None, True),
    (
        "soundscape-v9d-1ep-p03",
        "training/outputs",
        "soundscape-v9d_fold*.pt",
        None,
        True,
    ),
]

VAL_FRAC = 0.15
SEED = 42


def get_held_out_files(sc_labels: pd.DataFrame) -> set:
    """Reproduce the soundscape-val-frac split from train_cnn.py."""
    unique_files = sc_labels["filename"].unique()
    stations = np.array([str(f).split("_")[3] for f in unique_files])
    rng = np.random.RandomState(SEED)
    val_files: set[str] = set()
    for st in np.unique(stations):
        st_files = unique_files[stations == st]
        n_val = max(1, round(len(st_files) * VAL_FRAC))
        chosen = rng.choice(st_files, size=n_val, replace=False)
        val_files.update(chosen.tolist())
    return val_files


def load_meta(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m = ckpt["model"]
    is_baseline = "head.cls_conv.bias" in m
    # HGNetV2 has a different head
    is_hgnetv2 = "head.fc.bias" in m and not is_baseline
    return {
        "backbone": ckpt.get("backbone", "tf_efficientnet_b0.ns_jft_in1k"),
        "n_mels": ckpt.get("n_mels", 128),
        "n_fft": ckpt.get("n_fft", 1024),
        "hop_length": ckpt.get("hop_length", 512),
        "minmax_norm": ckpt.get("minmax_norm", False),
        "htk": ckpt.get("htk", False),
        "fmin": ckpt.get("fmin", 20),
        "is_baseline": is_baseline,
        "is_hgnetv2": is_hgnetv2,
    }


def load_model(path: Path, n_species: int, meta: dict, device: torch.device) -> nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if meta["is_baseline"]:
        model = BirdModelBaseline(backbone=meta["backbone"], n_classes=n_species)
    else:
        model = BirdModel(backbone=meta["backbone"], n_classes=n_species)
    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as e:
        print(f"    state_dict mismatch: {e}")
        return None
    return model.eval().to(device)


def make_loader(df: pd.DataFrame, meta: dict, species_to_idx: dict, n_species: int) -> DataLoader:
    htk = meta["htk"]
    # Pick cache that matches mel config
    if meta["n_mels"] > 128 and htk:
        cache_dir = SOUNDSCAPE_CACHE_DIR_BASELINE_HTK
    elif meta["n_mels"] > 128 and not htk:
        cache_dir = SOUNDSCAPE_CACHE_DIR_BASELINE
    else:
        # 128-mel or other non-standard → no cache
        cache_dir = None
    # HGNetV2 non-HTK uses its own cache (256-mel, hop=625)
    if meta["is_hgnetv2"] and not htk:
        cache_dir = SOUNDSCAPE_CACHE_DIR_HGNETV2

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
        htk=htk,
    )
    return DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)


@torch.no_grad()
def compute_cmap(fold_preds: list, labels_arr: np.ndarray) -> float:
    preds_arr = np.mean(fold_preds, axis=0)
    bin_labels = (labels_arr > 0).astype(np.int32)
    aps = []
    for c in range(labels_arr.shape[1]):
        if bin_labels[:, c].sum() > 0:
            aps.append(average_precision_score(bin_labels[:, c], preds_arr[:, c]))
    return float(np.mean(aps)) if aps else float("nan")


def eval_group(
    label: str,
    ckpt_dir: Path,
    pattern: str,
    lb: float | None,
    sc_labels_used: bool,
    sc_labels_all: pd.DataFrame,
    sc_labels_held: pd.DataFrame,
    species_to_idx: dict,
    n_species: int,
    device: torch.device,
):
    ckpt_paths = sorted(ckpt_dir.glob(pattern))
    if not ckpt_paths:
        print(f"  [{label}] NO CHECKPOINTS")
        return None, None, lb, 0

    print(f"\n[{label}] {len(ckpt_paths)} fold(s)")
    try:
        meta = load_meta(ckpt_paths[0])
    except Exception as e:
        print(f"  meta load failed: {e}")
        return None, None, lb, 0

    print(f"  backbone={meta['backbone']} n_mels={meta['n_mels']} htk={meta['htk']} fmin={meta['fmin']}")

    loader_all = make_loader(sc_labels_all, meta, species_to_idx, n_species)
    loader_held = make_loader(sc_labels_held, meta, species_to_idx, n_species)

    fold_preds_all, fold_preds_held = [], []
    labels_all_arr = labels_held_arr = None

    for p in ckpt_paths:
        model = load_model(p, n_species, meta, device)
        if model is None:
            continue

        preds_all, preds_held = [], []
        la, lh = [], []
        with torch.no_grad():
            for x, y in loader_all:
                out = model(x.to(device))
                if not meta["is_baseline"]:
                    out = torch.sigmoid(out)
                preds_all.append(out.cpu().numpy())
                la.append(y.numpy())
            for x, y in loader_held:
                out = model(x.to(device))
                if not meta["is_baseline"]:
                    out = torch.sigmoid(out)
                preds_held.append(out.cpu().numpy())
                lh.append(y.numpy())

        fold_preds_all.append(np.concatenate(preds_all))
        fold_preds_held.append(np.concatenate(preds_held))
        labels_all_arr = np.concatenate(la)
        labels_held_arr = np.concatenate(lh)
        print(f"  {p.name}: OK")

    if not fold_preds_all:
        return None, None, lb, 0

    cmap_all = compute_cmap(fold_preds_all, labels_all_arr)
    cmap_held = compute_cmap(fold_preds_held, labels_held_arr)
    print(f"  sc_cmap ALL={cmap_all:.4f}  HELD={cmap_held:.4f}  (sc_labels={sc_labels_used})")
    return cmap_all, cmap_held, lb, len(ckpt_paths)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Species from taxonomy
    taxonomy = pd.read_csv(DATA / "taxonomy.csv")
    species = sorted(taxonomy["primary_label"].astype(str).tolist())
    species_to_idx = {s: i for i, s in enumerate(species)}
    n_species = len(species)
    print(f"Species: {n_species}")

    sc_labels = pd.read_csv(LABELS_CSV)
    print(f"Soundscape labels: {len(sc_labels)} segments, {sc_labels['filename'].nunique()} files")

    held_out_files = get_held_out_files(sc_labels)
    sc_held = sc_labels[sc_labels["filename"].isin(held_out_files)].reset_index(drop=True)
    sc_all = sc_labels.reset_index(drop=True)
    print(f"Held-out val: {sc_held['filename'].nunique()} files, {len(sc_held)} segments")

    results = []
    for label, subdir, pattern, lb, sc_labels_used in GROUPS:
        ckpt_dir = ROOT / subdir
        cmap_all, cmap_held, lb, nf = eval_group(
            label=label,
            ckpt_dir=ckpt_dir,
            pattern=pattern,
            lb=lb,
            sc_labels_used=sc_labels_used,
            sc_labels_all=sc_all,
            sc_labels_held=sc_held,
            species_to_idx=species_to_idx,
            n_species=n_species,
            device=device,
        )
        results.append((label, cmap_all, cmap_held, lb, sc_labels_used, nf))

    # Summary table
    print("\n\n" + "=" * 88)
    print(f"{'Model':<28} {'sc_cmap(all)':>12}  {'sc_cmap(held)':>13}  {'LB':>7}  {'scL':>4}  {'F':>2}")
    print("-" * 88)
    for label, ca, ch, lb, scl, nf in results:
        ca_s = f"{ca:.4f}" if ca is not None else "   N/A"
        ch_s = f"{ch:.4f}" if ch is not None else "   N/A"
        lb_s = f"{lb:.3f}" if lb is not None else "  ---"
        scl_s = "Y" if scl else "N"
        print(f"{label:<28} {ca_s:>12}  {ch_s:>13}  {lb_s:>7}  {scl_s:>4}  {nf:>2}")
    print("=" * 88)
    print("scL=trained with soundscape labels  F=number of folds")
    print("sc_cmap(all) = leaky for scL=Y models | sc_cmap(held) = 14-file holdout (val_frac=0.15)")


if __name__ == "__main__":
    main()
