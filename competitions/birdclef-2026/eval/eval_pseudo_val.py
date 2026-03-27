"""Evaluate CNN ensemble on a held-out pseudo-val set from unlabeled soundscapes.

Selects 500 soundscapes from S01/S02/S13 stations (deterministic, seed=42),
runs the N-fold ensemble, and computes cmAP against CNN soft labels.

Key caveat: this is CNN-vs-CNN evaluation (circular). Use only to detect large
regressions (>0.015 drop); not reliable for fine-grained ranking.

Usage:
    python eval/eval_pseudo_val.py
    python eval/eval_pseudo_val.py --ckpt-pattern "soundscape-v9*_fold*.pt"
    python eval/eval_pseudo_val.py --n-val 200  # faster, fewer files
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from train_cnn import (  # noqa: E402
    SOUNDSCAPE_CACHE_DIR_BASELINE_HTK,
    BirdModel,
    BirdModelBaseline,
)

DATA = Path(os.getenv("KEGO_PATH_DATA", "data")) / "birdclef" / "birdclef-2026"
SOUNDSCAPE_DIR = DATA / "train_soundscapes"
CKPT_DIR = Path(__file__).parent.parent / "training" / "outputs"
DEFAULT_NPZ = DATA / "cnn_pseudo_labels_soft.npz"

# Stations with NO labeled soundscapes → zero leakage
# (S13 has 2 labeled files so excluded; S01+S02 give 4846 total files)
PSEUDO_VAL_STATIONS = {"S01", "S02"}
N_VAL_DEFAULT = 500
SEED = 42


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
        "n_mels": ckpt.get("n_mels", 224),
        "n_fft": ckpt.get("n_fft", 2048),
        "hop_length": ckpt.get("hop_length", 512),
        "minmax_norm": ckpt.get("minmax_norm", True),
        "htk": ckpt.get("htk", True),
        "fmin": ckpt.get("fmin", 0),
    }


def select_pseudo_val_files(
    npz_data: dict, n_val: int, rng: np.random.RandomState
) -> list[str]:
    """Return list of soundscape filenames from PSEUDO_VAL_STATIONS, deduplicated."""
    all_filenames = [str(k).split(":")[0] for k in npz_data["filenames"]]
    unique = sorted(set(all_filenames))
    # Filename format: BC2026_Train_NNNN_SXX_YYYYMMDD_HHMMSS.ogg → station at index 3
    candidates = [f for f in unique if f.split("_")[3] in PSEUDO_VAL_STATIONS]
    if len(candidates) == 0:
        # Fall back to all unlabeled files if station filter finds nothing
        print(
            f"WARNING: No files matching stations {PSEUDO_VAL_STATIONS} — using all files"
        )
        candidates = unique
    if len(candidates) > n_val:
        chosen = rng.choice(candidates, size=n_val, replace=False)
        return sorted(chosen.tolist())
    return candidates


@torch.no_grad()
def run_ensemble(
    filenames: list[str],
    models_meta: list[tuple[nn.Module, dict]],
    npz_labels: np.ndarray,
    npz_filenames: list[str],
    npz_species: list[str],
    species_to_idx: dict[str, int],
    n_species: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ensemble on pseudo-val files. Returns (preds, labels) both (N_windows, n_species)."""
    import librosa
    from train_cnn import (
        CLIP_SAMPLES,
        SR,
        make_melspec,
        spec_to_tensor,
        spec_to_tensor_minmax,
    )

    meta = models_meta[0][1]
    hop = meta["hop_length"]
    clip_frames = CLIP_SAMPLES // hop

    # Build lookup: filename:start_sec → row index in npz
    fname_start_to_row: dict[str, int] = {}
    for i, key in enumerate(npz_filenames):
        fname_start_to_row[key] = i

    # Remap npz species order → training species order
    remap = np.array([species_to_idx.get(sp, -1) for sp in npz_species], dtype=np.int32)

    cache_dir = SOUNDSCAPE_CACHE_DIR_BASELINE_HTK if meta["htk"] else None

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    n_windows_per_file = 12  # 12 × 5s = 60s
    t0 = time.time()

    for i, fname in enumerate(filenames):
        stem = Path(fname).stem

        # Load spec from cache (fast) or audio (slow)
        if cache_dir is not None:
            cache_path = cache_dir / f"{stem}.npy"
            if cache_path.exists():
                full_spec = np.load(cache_path).astype(np.float32)
            else:
                full_spec = None
        else:
            full_spec = None

        if full_spec is None:
            try:
                y, _ = librosa.load(SOUNDSCAPE_DIR / fname, sr=SR, mono=True)
            except Exception as e:
                print(f"  WARN: {fname}: {e}")
                continue
        else:
            y = None

        # Build windows
        fold_preds_windows: list[np.ndarray] = []
        for model, m in models_meta:
            window_tensors = []
            for w in range(n_windows_per_file):
                if full_spec is not None:
                    start_f = w * clip_frames
                    spec = full_spec[:, start_f : start_f + clip_frames]
                    if spec.shape[1] < clip_frames:
                        spec = np.pad(spec, ((0, 0), (0, clip_frames - spec.shape[1])))
                else:
                    start_s = w * (SR * 5)
                    chunk = y[start_s : start_s + CLIP_SAMPLES]
                    if len(chunk) < CLIP_SAMPLES:
                        chunk = np.pad(chunk, (0, CLIP_SAMPLES - len(chunk)))
                    spec = make_melspec(
                        chunk,
                        n_mels=m["n_mels"],
                        n_fft=m["n_fft"],
                        hop_length=hop,
                        fmin=m["fmin"],
                        htk=m["htk"],
                    )
                t = (
                    spec_to_tensor_minmax(spec)
                    if m["minmax_norm"]
                    else spec_to_tensor(spec)
                )
                window_tensors.append(t)
            batch = torch.stack(window_tensors).to(device)
            out = model(
                batch
            )  # (n_windows, n_species) — sigmoid already applied for baseline
            fold_preds_windows.append(out.cpu().numpy())

        if not fold_preds_windows:
            continue
        file_preds = np.mean(fold_preds_windows, axis=0)  # (n_windows, n_species)

        # Get pseudo-labels for this file from npz
        for w in range(n_windows_per_file):
            key = f"{fname}:{w * 5}"
            if key not in fname_start_to_row:
                continue
            row_idx = fname_start_to_row[key]
            raw_probs = npz_labels[row_idx]
            # Remap npz species → training species
            label = np.zeros(n_species, dtype=np.float32)
            valid = remap >= 0
            label[remap[valid]] = raw_probs[valid]

            all_preds.append(file_preds[w])
            all_labels.append(label)

        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  [{i + 1}/{len(filenames)}]  {rate:.1f} files/s", flush=True)

    return np.stack(all_preds), np.stack(all_labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt-pattern",
        default="soundscape-v7_fold*.pt",
        help="Glob pattern for checkpoint files.",
    )
    parser.add_argument(
        "--npz",
        default=str(DEFAULT_NPZ),
        help="Path to cnn_pseudo_labels_soft.npz",
    )
    parser.add_argument(
        "--n-val",
        type=int,
        default=N_VAL_DEFAULT,
        help="Number of pseudo-val soundscape files to evaluate (default 500)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Threshold on soft labels to compute binary cmAP (default 0.2)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pseudo-label NPZ
    npz_path = Path(args.npz)
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found. Run data/pseudo_label_cnn_soft.py first.")
        sys.exit(1)
    data = np.load(npz_path, allow_pickle=True)
    npz_labels = data["labels"].astype(np.float32)
    npz_species = list(data["species"])
    npz_filenames_raw = [str(k) for k in data["filenames"]]
    print(f"NPZ: {npz_labels.shape[0]:,} windows × {npz_labels.shape[1]} species")

    # Species ordering (sorted)
    n_species = len(npz_species)
    species_to_idx = {sp: i for i, sp in enumerate(sorted(npz_species))}

    # Select pseudo-val files
    rng = np.random.RandomState(SEED)
    val_files = select_pseudo_val_files(
        {"filenames": data["filenames"]}, args.n_val, rng
    )
    print(f"Pseudo-val files: {len(val_files)} from stations {PSEUDO_VAL_STATIONS}")
    station_counts: dict[str, int] = {}
    for f in val_files:
        st = f.split("_")[3]
        station_counts[st] = station_counts.get(st, 0) + 1
    for st, cnt in sorted(station_counts.items()):
        print(f"  {st}: {cnt} files")

    # Load checkpoints
    ckpt_paths = sorted(CKPT_DIR.glob(args.ckpt_pattern))
    if not ckpt_paths:
        print(
            f"ERROR: No checkpoints found matching '{args.ckpt_pattern}' in {CKPT_DIR}"
        )
        sys.exit(1)
    print(f"\nLoading {len(ckpt_paths)} checkpoints...")
    models_meta = []
    for p in ckpt_paths:
        meta = load_meta(p)
        model = load_model(p, n_species, device)
        models_meta.append((model, meta))
        print(f"  {p.name}  n_mels={meta['n_mels']}  htk={meta['htk']}")

    # Run inference
    print(f"\nRunning ensemble on {len(val_files)} files...")
    preds, labels = run_ensemble(
        val_files,
        models_meta,
        npz_labels,
        npz_filenames_raw,
        npz_species,
        species_to_idx,
        n_species,
        device,
    )
    print(f"Collected {len(preds)} windows")

    if len(preds) == 0:
        print("ERROR: No predictions collected.")
        sys.exit(1)

    # cmAP vs soft labels (binarized at threshold)
    bin_labels = (labels >= args.threshold).astype(np.int32)
    n_pos_classes = (bin_labels.sum(0) > 0).sum()
    print(f"\n--- Pseudo-val cmAP (threshold={args.threshold}) ---")
    print(f"Classes with positives: {n_pos_classes}/{n_species}")

    aps = []
    for c in range(n_species):
        if bin_labels[:, c].sum() == 0:
            continue
        aps.append(average_precision_score(bin_labels[:, c], preds[:, c]))

    cmap = float(np.mean(aps))
    print(f"Pseudo-val cmAP: {cmap:.4f}")
    print(f"  (lower bound: random = {bin_labels.mean():.4f}, upper = 1.0)")
    print(f"  Mean max-prob (preds): {preds.max(axis=1).mean():.4f}")
    print(
        f"  Windows with pred≥0.3: {(preds.max(axis=1) >= 0.3).sum():,}/{len(preds):,}"
    )

    # Top/bottom classes
    ap_arr = np.full(n_species, np.nan)
    for c in range(n_species):
        if bin_labels[:, c].sum() > 0:
            ap_arr[c] = average_precision_score(bin_labels[:, c], preds[:, c])
    present = np.where(~np.isnan(ap_arr))[0]
    top5 = present[np.argsort(-ap_arr[present])[:5]]
    bot5 = present[np.argsort(ap_arr[present])[:5]]
    sorted_species = sorted(npz_species)
    print("\nTop-5 classes:")
    for c in top5:
        print(f"  {sorted_species[c]:<30} AP={ap_arr[c]:.4f}")
    print("Bottom-5 classes:")
    for c in bot5:
        print(f"  {sorted_species[c]:<30} AP={ap_arr[c]:.4f}")


if __name__ == "__main__":
    main()
