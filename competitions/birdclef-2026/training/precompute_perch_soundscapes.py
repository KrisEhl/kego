"""
Precompute Perch v2 predictions + embeddings on all training soundscapes.

Outputs (saved to $KEGO_PATH_DATA/birdclef/birdclef-2026/perch_soundscape_cache/):
  - perch_sc_scores.npy     : float32, shape (N_windows, 234)  — mapped to 234 target classes
  - perch_sc_embeddings.npy : float32, shape (N_windows, 1536)
  - perch_sc_meta.parquet   : row_id, filename, site, hour_utc, window_idx

Run on cluster:
  KEGO_PATH_DATA=/home/kristian/projects/kego/data \\
  PERCH_MODEL_DIR=/home/kristian/perch_v2_cpu \\
  uv run python competitions/birdclef-2026/training/precompute_perch_soundscapes.py

Download Perch model first (run once):
  kaggle models instances versions download \\
    google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1 \\
    -p ~/perch_v2_cpu --untar
"""

import gc
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
LABELS_CSV = DATA_ROOT / "birdclef" / "birdclef-2026" / "train_soundscapes_labels.csv"

MODEL_DIR = Path(
    os.environ.get(
        "PERCH_MODEL_DIR",
        str(Path.home() / "perch_v2_cpu"),
    )
)

OUT_DIR = DATA_ROOT / "birdclef" / "birdclef-2026" / "perch_soundscape_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SR = 32_000
WINDOW_SAMPLES = SR * 5  # 160,000
FILE_SAMPLES = SR * 60  # 1,920,000
N_WINDOWS = 12
BATCH_FILES = 16

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SC_RE = re.compile(r"BC2026_Train_(\d+)_(S\w+)_(\d{8})_(\d{6})\.ogg", re.IGNORECASE)


def parse_filename(name: str) -> dict:
    m = _SC_RE.match(name)
    if not m:
        return {"site": "UNKNOWN", "hour_utc": -1}
    date_str, time_str = m.group(3), m.group(4)
    hour = int(time_str[:2])
    return {"site": m.group(2), "hour_utc": hour}


def read_soundscape(path: Path) -> np.ndarray:
    y, _ = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    return y[:FILE_SAMPLES]


# ---------------------------------------------------------------------------
# Load taxonomy + build species index
# ---------------------------------------------------------------------------
print("Loading taxonomy...", flush=True)
tax = pd.read_csv(TAXONOMY_CSV)
TARGET_SPECIES = sorted(tax["primary_label"].tolist())
N_CLASSES = len(TARGET_SPECIES)
print(f"  Target species: {N_CLASSES}", flush=True)

# ---------------------------------------------------------------------------
# Load Perch model
# ---------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf  # noqa: E402

print(f"TensorFlow: {tf.__version__}", flush=True)
print(f"Loading Perch from {MODEL_DIR} ...", flush=True)
t0 = time.perf_counter()
birdclassifier = tf.saved_model.load(str(MODEL_DIR))
infer_fn = birdclassifier.signatures["serving_default"]
print(f"  Loaded in {time.perf_counter() - t0:.1f}s", flush=True)

# Warm up XLA
print("XLA warmup...", flush=True)
_dummy = tf.zeros([BATCH_FILES * N_WINDOWS, WINDOW_SAMPLES], dtype=tf.float32)
_ = infer_fn(inputs=_dummy)
del _dummy
print("  Done.", flush=True)

# Build mapping: Perch 14795-class → our 234 target classes
out_keys = list(infer_fn.structured_outputs.keys())
print(f"  Output keys: {out_keys}", flush=True)

# Load Perch label vocab from the model directory
vocab_path = MODEL_DIR / "assets" / "label.csv"
if not vocab_path.exists():
    # Try alternate location
    for p in MODEL_DIR.rglob("label.csv"):
        vocab_path = p
        break

if vocab_path.exists():
    perch_labels = pd.read_csv(vocab_path, header=None)[0].tolist()
    print(f"  Perch vocab size: {len(perch_labels)}", flush=True)

    # Map our 234 target species → Perch indices
    perch_label_set = {lbl: i for i, lbl in enumerate(perch_labels)}
    mapped_our_idx = []
    mapped_perch_idx = []
    for our_i, sp in enumerate(TARGET_SPECIES):
        # Try direct match, then scientific name lookup
        if sp in perch_label_set:
            mapped_our_idx.append(our_i)
            mapped_perch_idx.append(perch_label_set[sp])

    MAPPED_OUR = np.array(mapped_our_idx, dtype=np.int32)
    MAPPED_PERCH = np.array(mapped_perch_idx, dtype=np.int32)
    print(f"  Mapped {len(MAPPED_OUR)}/{N_CLASSES} species directly", flush=True)
else:
    print("  WARNING: label.csv not found — saving raw logits (14795 dim)", flush=True)
    MAPPED_OUR = None
    MAPPED_PERCH = None

# ---------------------------------------------------------------------------
# Find all soundscape files
# ---------------------------------------------------------------------------
sc_paths = sorted(SC_DIR.glob("*.ogg"))
print(f"\nSoundscape files found: {len(sc_paths)}", flush=True)
if len(sc_paths) == 0:
    print(f"ERROR: no .ogg files in {SC_DIR}", flush=True)
    sys.exit(1)

N_FILES = len(sc_paths)
N_TOTAL_WINDOWS = N_FILES * N_WINDOWS

# ---------------------------------------------------------------------------
# Allocate output arrays
# ---------------------------------------------------------------------------
scores_out = np.zeros((N_TOTAL_WINDOWS, N_CLASSES), dtype=np.float32)
emb_out = np.zeros((N_TOTAL_WINDOWS, 1536), dtype=np.float32)
row_ids = np.empty(N_TOTAL_WINDOWS, dtype=object)
filenames = np.empty(N_TOTAL_WINDOWS, dtype=object)
sites = np.empty(N_TOTAL_WINDOWS, dtype=object)
hours = np.zeros(N_TOTAL_WINDOWS, dtype=np.int16)

# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------
batches = [sc_paths[s : s + BATCH_FILES] for s in range(0, N_FILES, BATCH_FILES)]
n_batches = len(batches)
write_row = 0
t_total_start = time.perf_counter()

print(f"\nRunning Perch on {N_FILES} files ({n_batches} batches)...", flush=True)

for bi, batch in enumerate(batches):
    bn = len(batch)

    # Load audio
    windows_list = []
    for path in batch:
        audio = read_soundscape(path)
        meta = parse_filename(path.name)
        bstart = write_row + len(windows_list) // N_WINDOWS * N_WINDOWS  # noqa
        for i in range(N_WINDOWS):
            windows_list.append(audio[i * WINDOW_SAMPLES : (i + 1) * WINDOW_SAMPLES])

    x = np.stack(windows_list).astype(np.float32)

    # Perch inference
    out = infer_fn(inputs=tf.convert_to_tensor(x))
    logits = out["label"].numpy()  # (bn*N_WINDOWS, 14795)
    emb = out["embedding"].numpy()  # (bn*N_WINDOWS, 1536)

    # Store
    bw = bn * N_WINDOWS
    emb_out[write_row : write_row + bw] = emb

    if MAPPED_OUR is not None:
        scores_out[write_row : write_row + bw][:, MAPPED_OUR] = logits[:bw, MAPPED_PERCH]
    else:
        # No mapping — store first N_CLASSES logits as proxy
        scores_out[write_row : write_row + bw] = logits[:bw, :N_CLASSES]

    # Metadata
    for fi, path in enumerate(batch):
        meta = parse_filename(path.name)
        ws, we = write_row + fi * N_WINDOWS, write_row + (fi + 1) * N_WINDOWS
        row_ids[ws:we] = [f"{path.stem}_{(i + 1) * 5}" for i in range(N_WINDOWS)]
        filenames[ws:we] = path.name
        sites[ws:we] = meta["site"]
        hours[ws:we] = meta["hour_utc"]

    write_row += bw
    del x, out, logits, emb, windows_list
    gc.collect()

    elapsed = time.perf_counter() - t_total_start
    files_done = (bi + 1) * BATCH_FILES
    eta = elapsed / max(files_done, 1) * (N_FILES - files_done)
    print(
        f"  batch {bi + 1}/{n_batches} ({min(files_done, N_FILES)}/{N_FILES} files)"
        f"  elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
print(f"\nSaving to {OUT_DIR} ...", flush=True)

np.save(OUT_DIR / "perch_sc_scores.npy", scores_out)
np.save(OUT_DIR / "perch_sc_embeddings.npy", emb_out)

meta_df = pd.DataFrame(
    {
        "row_id": row_ids,
        "filename": filenames,
        "site": sites,
        "hour_utc": hours,
    }
)
meta_df.to_parquet(OUT_DIR / "perch_sc_meta.parquet", index=False)

t_total = time.perf_counter() - t_total_start
print(
    f"Done. {N_FILES} files in {t_total:.0f}s ({t_total / N_FILES:.1f}s/file)",
    flush=True,
)
print(f"  scores shape : {scores_out.shape}", flush=True)
print(f"  emb shape    : {emb_out.shape}", flush=True)
print(f"  Output dir   : {OUT_DIR}", flush=True)
