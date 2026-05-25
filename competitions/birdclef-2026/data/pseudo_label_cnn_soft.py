"""Generate CNN soft pseudo-labels for all (unlabeled) soundscapes.

Runs the N-fold soundscape-v7 ensemble on soundscape files and saves full
soft probability arrays — no threshold filter. Output is in PerchDataset
.npz format so it can be used directly with --perch-npz in train_cnn.py.

Output npz keys:
    filenames : (N_windows,) object array of "filename.ogg:start_sec" strings
    labels    : (N_windows, 234) float32 mean ensemble predictions
    species   : (234,) object array of sorted species names

Usage (on GPU server, split across both GPUs):
    KEGO_PATH_DATA=/home/kristian/projects/kego/data \\
    CUDA_VISIBLE_DEVICES=0 uv run python \\
        competitions/birdclef-2026/data/pseudo_label_cnn_soft.py \\
        --ckpt-pattern "soundscape-v7_fold*.pt" \\
        --start-idx 0 --end-idx 5296

    KEGO_PATH_DATA=/home/kristian/projects/kego/data \\
    CUDA_VISIBLE_DEVICES=1 uv run python \\
        competitions/birdclef-2026/data/pseudo_label_cnn_soft.py \\
        --ckpt-pattern "soundscape-v7_fold*.pt" \\
        --start-idx 5296

Then merge shards:
    uv run python competitions/birdclef-2026/data/pseudo_label_cnn_soft.py \\
        --merge --output cnn_pseudo_labels_soft.npz

To exclude labeled soundscapes from the output (they have leakage):
    --exclude-labeled   (default: include all)
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from train_cnn import BirdModel, BirdModelBaseline  # noqa: E402

# ── paths ──────────────────────────────────────────────────────────────────────
DATA = Path(os.getenv("KEGO_PATH_DATA", "data")) / "birdclef" / "birdclef-2026"
SOUNDSCAPE_DIR = DATA / "train_soundscapes"
TAXONOMY_CSV = DATA / "taxonomy.csv"
TRAIN_SOUNDSCAPE_LABELS = DATA / "train_soundscapes_labels.csv"
CKPT_DIR = Path(__file__).parent.parent / "outputs"

# Spec cache dirs (fast path) — HTK 224-mel cache covers all 10,658 files
SPEC_CACHE_HTK = DATA.parent.parent / "specs_cache_soundscape_224_htk"
SPEC_CACHE_STD = DATA.parent.parent / "specs_cache_soundscape_224"

SR = 32_000
CLIP_SAMPLES = SR * 5  # 5-second windows
N_WINDOWS = 12  # 12 × 5s = 60s per soundscape file


# ── helpers ───────────────────────────────────────────────────────────────────


def _load_checkpoint(path: Path, n_species: int, device: torch.device) -> nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    backbone = ckpt.get("backbone", "tf_efficientnet_b0.ns_jft_in1k")
    if "head.cls_conv.bias" in ckpt["model"]:
        model = BirdModelBaseline(backbone=backbone, n_classes=n_species)
    else:
        model = BirdModel(backbone=backbone, n_classes=n_species)
    model.load_state_dict(ckpt["model"])
    return model.eval().to(device)


def _load_checkpoint_meta(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "n_mels": ckpt.get("n_mels", 224),
        "n_fft": ckpt.get("n_fft", 2048),
        "hop_length": ckpt.get("hop_length", 512),
        "minmax_norm": ckpt.get("minmax_norm", True),
        "htk": ckpt.get("htk", True),
        "fmin": ckpt.get("fmin", 0),
    }


def _spec_to_tensor(spec: np.ndarray, minmax: bool) -> torch.Tensor:
    if minmax:
        s_min, s_max = spec.min(), spec.max()
        spec = (spec - s_min) / (s_max - s_min + 1e-7)
    else:
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    return torch.from_numpy(np.stack([spec, spec, spec], axis=0)).float()


def _load_windows_from_cache(
    stem: str,
    meta: dict,
    n_windows: int = N_WINDOWS,
) -> torch.Tensor | None:
    """Load pre-computed mel spec from cache, slice into N_WINDOWS tensors.

    Returns tensor of shape (n_windows, 3, n_mels, clip_frames) or None if
    cache miss.
    """
    hop = meta["hop_length"]
    n_mels = meta["n_mels"]
    clip_frames = CLIP_SAMPLES // hop

    # Pick correct cache dir based on htk flag
    cache_dir = SPEC_CACHE_HTK if meta["htk"] else SPEC_CACHE_STD
    cache_path = cache_dir / f"{stem}.npy"
    if not cache_path.exists():
        return None

    full_spec = np.load(cache_path).astype(np.float32)  # (n_mels, T_full)
    if full_spec.shape[0] != n_mels:
        return None

    tensors = []
    for w in range(n_windows):
        start = w * clip_frames
        spec = full_spec[:, start : start + clip_frames]
        if spec.shape[1] < clip_frames:
            spec = np.pad(spec, ((0, 0), (0, clip_frames - spec.shape[1])))
        tensors.append(_spec_to_tensor(spec, meta["minmax_norm"]))

    return torch.stack(tensors)  # (n_windows, 3, n_mels, clip_frames)


def _load_windows_from_audio(path: Path, meta: dict) -> torch.Tensor:
    """Slow path: load audio, compute mel spec on-the-fly."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    hop = meta["hop_length"]
    clip_frames = CLIP_SAMPLES // hop
    tensors = []
    for w in range(N_WINDOWS):
        chunk = y[w * CLIP_SAMPLES : (w + 1) * CLIP_SAMPLES]
        if len(chunk) < CLIP_SAMPLES:
            chunk = np.pad(chunk, (0, CLIP_SAMPLES - len(chunk)))
        mel = librosa.feature.melspectrogram(
            y=chunk,
            sr=SR,
            n_mels=meta["n_mels"],
            hop_length=hop,
            n_fft=meta["n_fft"],
            fmin=meta["fmin"],
            htk=meta["htk"],
        )
        spec = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        if spec.shape[1] > clip_frames:
            spec = spec[:, :clip_frames]
        elif spec.shape[1] < clip_frames:
            spec = np.pad(spec, ((0, 0), (0, clip_frames - spec.shape[1])))
        tensors.append(_spec_to_tensor(spec, meta["minmax_norm"]))
    return torch.stack(tensors)


@torch.no_grad()
def predict_file(
    path: Path,
    models: list[tuple[nn.Module, dict]],
    device: torch.device,
) -> np.ndarray:
    """Returns (N_WINDOWS, n_species) mean ensemble predictions."""
    meta = models[0][1]
    batch = _load_windows_from_cache(path.stem, meta)
    if batch is None:
        batch = _load_windows_from_audio(path, meta)
    batch = batch.to(device)  # (N_WINDOWS, 3, n_mels, clip_frames)

    fold_preds = []
    for model, _ in models:
        out = model(batch)  # (N_WINDOWS, n_species) — sigmoid already applied
        fold_preds.append(out.cpu().numpy())

    return np.mean(fold_preds, axis=0).astype(np.float32)  # (N_WINDOWS, n_species)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt-pattern",
        default="soundscape-v7_fold*.pt",
        help="Glob pattern for checkpoint files in outputs/ dir.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index into sorted soundscape file list (for parallelism).",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End index (exclusive). Defaults to end of file list.",
    )
    parser.add_argument(
        "--exclude-labeled",
        action="store_true",
        help="Skip the 66 labeled soundscape files (those in train_soundscapes_labels.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .npz filename (default: auto-named shard or merged file).",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all shard .npz files in DATA dir into one cnn_pseudo_labels_soft.npz.",
    )
    args = parser.parse_args()

    # ── merge mode ────────────────────────────────────────────────────────────
    if args.merge:
        out_name = args.output or "cnn_pseudo_labels_soft.npz"
        shards = sorted(DATA.glob("cnn_pseudo_labels_soft_shard_*.npz"))
        if not shards:
            print("No shard files found. Run generation first.")
            sys.exit(1)
        print(f"Merging {len(shards)} shards...")
        all_filenames, all_labels, species = [], [], None
        for shard in shards:
            d = np.load(shard, allow_pickle=True)
            all_filenames.extend(d["filenames"].tolist())
            all_labels.append(d["labels"])
            if species is None:
                species = d["species"]
        all_labels_arr = np.concatenate(all_labels, axis=0)
        out_path = DATA / out_name
        np.savez_compressed(
            out_path,
            filenames=np.array(all_filenames, dtype=object),
            labels=all_labels_arr,
            species=species,
        )
        print(f"Merged {len(all_filenames)} windows → {out_path}")
        print(f"Shape: {all_labels_arr.shape}  ({all_labels_arr.nbytes / 1e6:.1f} MB)")
        return

    # ── generation mode ───────────────────────────────────────────────────────
    with open(TAXONOMY_CSV) as f:
        species_list = [row["primary_label"] for row in csv.DictReader(f)]
    n_species = len(species_list)

    # Optionally exclude labeled soundscape files
    labeled_stems: set[str] = set()
    if args.exclude_labeled and TRAIN_SOUNDSCAPE_LABELS.exists():
        with open(TRAIN_SOUNDSCAPE_LABELS) as f:
            for row in csv.DictReader(f):
                labeled_stems.add(Path(row["filename"]).stem)
        print(f"Excluding {len(labeled_stems)} labeled soundscape files.")

    # Load checkpoints
    ckpt_paths = sorted(CKPT_DIR.glob(args.ckpt_pattern))
    if not ckpt_paths:
        print(f"ERROR: No checkpoints found matching '{args.ckpt_pattern}' in {CKPT_DIR}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {len(ckpt_paths)} checkpoints...")

    models = []
    for p in ckpt_paths:
        meta = _load_checkpoint_meta(p)
        model = _load_checkpoint(p, n_species, device)
        models.append((model, meta))
        ckpt_info = torch.load(p, map_location="cpu", weights_only=False)
        print(
            f"  {p.name}  backbone={meta.get('backbone', '?')}"
            f"  n_mels={meta['n_mels']}  htk={meta['htk']}"
            f"  epoch={ckpt_info.get('epoch', '?')}",
            flush=True,
        )

    # Check spec cache availability
    cache_dir = SPEC_CACHE_HTK if models[0][1]["htk"] else SPEC_CACHE_STD
    if cache_dir.exists():
        n_cached = len(list(cache_dir.glob("*.npy")))
        print(f"Spec cache: {cache_dir.name}/ — {n_cached} files (fast path)")
    else:
        print(f"WARNING: Spec cache not found at {cache_dir} — using slow librosa path")

    # Collect soundscape files
    all_files = sorted(SOUNDSCAPE_DIR.glob("*.ogg"))
    if labeled_stems:
        all_files = [f for f in all_files if f.stem not in labeled_stems]
    end_idx = args.end_idx or len(all_files)
    files = all_files[args.start_idx : end_idx]
    print(f"\nFiles to process: {len(files)} (indices {args.start_idx}–{end_idx})")
    print(f"Total windows: {len(files) * N_WINDOWS:,}")

    # Output path
    if args.output:
        out_path = DATA / args.output
    elif args.start_idx > 0 or args.end_idx:
        out_path = DATA / f"cnn_pseudo_labels_soft_shard_{args.start_idx}_{end_idx}.npz"
    else:
        out_path = DATA / "cnn_pseudo_labels_soft.npz"

    # Run inference
    all_filenames = []
    all_labels = []
    n_cache_hits = 0
    t_start = time.time()

    for i, path in enumerate(files):
        if args.exclude_labeled and path.stem in labeled_stems:
            continue
        try:
            preds = predict_file(path, models, device)  # (N_WINDOWS, n_species)
        except Exception as e:
            print(f"  WARN: {path.name}: {e}", flush=True)
            continue

        # Check if cache was used (heuristic: if cache file exists)
        if (cache_dir / f"{path.stem}.npy").exists():
            n_cache_hits += 1

        for w in range(N_WINDOWS):
            all_filenames.append(f"{path.name}:{w * 5}")
        all_labels.append(preds)

        if (i + 1) % 200 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta_min = (len(files) - i - 1) / rate / 60
            mean_max = np.mean([p.max() for p in all_labels[-min(200, len(all_labels)) :]])
            print(
                f"  [{i + 1}/{len(files)}]  {rate:.1f} files/s  ETA {eta_min:.1f} min  mean_max_prob={mean_max:.3f}",
                flush=True,
            )

    labels_arr = np.concatenate(all_labels, axis=0).astype(np.float32)
    filenames_arr = np.array(all_filenames, dtype=object)
    species_arr = np.array(species_list, dtype=object)

    np.savez_compressed(
        out_path,
        filenames=filenames_arr,
        labels=labels_arr,
        species=species_arr,
    )

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed / 60:.1f} min")
    print(f"Windows: {len(all_filenames):,}  ({labels_arr.nbytes / 1e6:.1f} MB uncompressed)")
    print(f"Cache hits: {n_cache_hits}/{len(files)} ({100 * n_cache_hits / max(len(files), 1):.0f}%)")
    print(f"Label density (mean_max_prob): {labels_arr.max(axis=1).mean():.4f}")
    print(f"Windows with max_prob ≥ 0.1: {(labels_arr.max(axis=1) >= 0.1).sum():,}")
    print(f"Windows with max_prob ≥ 0.3: {(labels_arr.max(axis=1) >= 0.3).sum():,}")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
