"""
Extend the Perch embedding cache with the 7 missing labeled soundscapes.

Reads existing full_perch_arrays.npz + full_perch_meta.parquet,
runs Perch on the 7 missing files, and saves an extended version.

Run on cluster:
  KEGO_PATH_DATA=/home/kristian/projects/kego/data \\
  PERCH_MODEL_DIR=/home/kristian/perch_v2_cpu \\
  uv run python competitions/birdclef-2026/training/extend_perch_cache.py
"""

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_ROOT = Path(os.environ.get("KEGO_PATH_DATA", "data"))
SC_DIR = DATA_ROOT / "birdclef" / "birdclef-2026" / "train_soundscapes"
TAXONOMY_CSV = DATA_ROOT / "birdclef" / "birdclef-2026" / "taxonomy.csv"
META_DIR = DATA_ROOT / "perch-meta"
MODEL_DIR = Path(os.environ.get("PERCH_MODEL_DIR", str(Path.home() / "perch_v2_cpu")))

SR = 32_000
WINDOW_SAMPLES = SR * 5
FILE_SAMPLES = SR * 60
N_WINDOWS = 12

MISSING_FILES = [
    "BC2026_Train_0006_S09_20250828_000000.ogg",
    "BC2026_Train_0007_S09_20250829_000000.ogg",
    "BC2026_Train_0008_S09_20250831_000000.ogg",
    "BC2026_Train_0009_S09_20250828_000000.ogg",
    "BC2026_Train_0010_S09_20250828_000000.ogg",
    "BC2026_Train_0015_S18_20211016_011500.ogg",
    "BC2026_Train_0026_S22_20211128_004500.ogg",
]

_SC_RE = re.compile(r"BC2026_Train_(\d+)_(S\w+)_(\d{8})_(\d{6})\.ogg", re.IGNORECASE)


def parse_filename(name: str) -> dict:
    m = _SC_RE.match(name)
    if not m:
        return {"site": "UNKNOWN", "hour_utc": -1}
    time_str = m.group(4)
    return {"site": m.group(2), "hour_utc": int(time_str[:2])}


def read_soundscape(path: Path) -> np.ndarray:
    y, _ = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    return y[:FILE_SAMPLES]


# ---------------------------------------------------------------------------
# Load existing cache
# ---------------------------------------------------------------------------
print("Loading existing cache...", flush=True)
npz = np.load(META_DIR / "full_perch_arrays.npz")
emb_existing = npz["emb_full"].astype(np.float32)
scores_existing = npz["scores_full_raw"].astype(np.float32)
meta_existing = pd.read_parquet(META_DIR / "full_perch_meta.parquet")
N_CLASSES = scores_existing.shape[1]
print(f"  Existing: {len(emb_existing)} windows, {N_CLASSES} classes", flush=True)

# ---------------------------------------------------------------------------
# Load taxonomy + Perch label mapping
# ---------------------------------------------------------------------------
tax = pd.read_csv(TAXONOMY_CSV)
TARGET_SPECIES = sorted(tax["primary_label"].tolist())
assert len(TARGET_SPECIES) == N_CLASSES

# ---------------------------------------------------------------------------
# Load Perch model
# ---------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf  # noqa: E402

print(f"Loading Perch from {MODEL_DIR}...", flush=True)
birdclassifier = tf.saved_model.load(str(MODEL_DIR))
infer_fn = birdclassifier.signatures["serving_default"]

# Build species mapping
perch_labels_path = next(MODEL_DIR.rglob("labels.csv"), None) or next(
    MODEL_DIR.rglob("label.csv"), None
)
if perch_labels_path is None:
    print("ERROR: label(s).csv not found", flush=True)
    sys.exit(1)
perch_labels = pd.read_csv(perch_labels_path, header=None)[0].tolist()
perch_label_set = {lbl: i for i, lbl in enumerate(perch_labels)}
mapped_our_idx, mapped_perch_idx = [], []
for i, sp in enumerate(TARGET_SPECIES):
    if sp in perch_label_set:
        mapped_our_idx.append(i)
        mapped_perch_idx.append(perch_label_set[sp])
MAPPED_OUR = np.array(mapped_our_idx, dtype=np.int32)
MAPPED_PERCH = np.array(mapped_perch_idx, dtype=np.int32)
print(f"  Mapped {len(MAPPED_OUR)}/{N_CLASSES} species", flush=True)

# ---------------------------------------------------------------------------
# Run Perch on missing files
# ---------------------------------------------------------------------------
n_new = len(MISSING_FILES) * N_WINDOWS
emb_new = np.zeros((n_new, 1536), dtype=np.float32)
scores_new = np.zeros((n_new, N_CLASSES), dtype=np.float32)
row_ids_new, filenames_new, sites_new = [], [], []
hours_new = np.zeros(n_new, dtype=np.int16)

print(f"\nRunning Perch on {len(MISSING_FILES)} missing files...", flush=True)
t0 = time.perf_counter()
write_row = 0
for fn in MISSING_FILES:
    path = SC_DIR / fn
    if not path.exists():
        print(f"  ERROR: {fn} not found!", flush=True)
        sys.exit(1)

    audio = read_soundscape(path)
    windows = [
        audio[i * WINDOW_SAMPLES : (i + 1) * WINDOW_SAMPLES] for i in range(N_WINDOWS)
    ]
    x = np.stack(windows).astype(np.float32)

    out = infer_fn(inputs=tf.convert_to_tensor(x))
    logits = out["label"].numpy()
    emb = out["embedding"].numpy()

    emb_new[write_row : write_row + N_WINDOWS] = emb
    scores_new[write_row : write_row + N_WINDOWS, MAPPED_OUR] = logits[
        :N_WINDOWS, MAPPED_PERCH
    ]

    meta = parse_filename(fn)
    stem = Path(fn).stem
    for i in range(N_WINDOWS):
        row_ids_new.append(f"{stem}_{(i + 1) * 5}")
        filenames_new.append(fn)
        sites_new.append(meta["site"])
        hours_new[write_row + i] = meta["hour_utc"]

    write_row += N_WINDOWS
    elapsed = time.perf_counter() - t0
    print(f"  {fn}: done ({elapsed:.0f}s)", flush=True)

# ---------------------------------------------------------------------------
# Merge and save
# ---------------------------------------------------------------------------
print("\nMerging...", flush=True)
emb_full = np.concatenate([emb_existing, emb_new], axis=0)
scores_full = np.concatenate([scores_existing, scores_new], axis=0)
print(
    f"  Extended: {len(emb_full)} windows ({len(emb_existing)} + {n_new} new)",
    flush=True,
)

meta_new_df = pd.DataFrame(
    {
        "row_id": row_ids_new,
        "filename": filenames_new,
        "site": sites_new,
        "hour_utc": hours_new,
    }
)
# Ensure same columns as existing meta
for col in meta_existing.columns:
    if col not in meta_new_df.columns:
        meta_new_df[col] = None
meta_new_df = meta_new_df[meta_existing.columns]
meta_full = pd.concat([meta_existing, meta_new_df], ignore_index=True)

# Save to new files (keep originals as backup)
out_npz = META_DIR / "full_perch_arrays_66.npz"
out_meta = META_DIR / "full_perch_meta_66.parquet"
np.savez_compressed(out_npz, emb_full=emb_full, scores_full_raw=scores_full)
meta_full.to_parquet(out_meta, index=False)

print(f"Saved:\n  {out_npz}\n  {out_meta}", flush=True)
print(
    "Next step: run precompute_probe_scores.py with --npz full_perch_arrays_66.npz",
    flush=True,
)
