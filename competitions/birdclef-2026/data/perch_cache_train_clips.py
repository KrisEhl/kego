"""Cache Perch v4 embeddings for all 35,549 training clips.

Reads train.csv, runs Perch on each clip (batched), saves embeddings +
competition-space probs. Used to train per-class LogReg probes with full
training data coverage (206 species vs 53 from soundscapes only).

Outputs:
    data/birdclef/birdclef-2026/perch_train_cache.npz
        - clip_ids:   (N,) str    "{taxon_id}/{stem}"
        - embeddings: (N, 1280)   float32  Perch embedding vectors
        - comp_probs: (N, 234)    float32  sigmoid probs in competition space
        - labels:     (N, 234)    float32  binary GT (primary + secondary)
        - species:    (234,)      str      competition species in order

Runtime: ~3h CPU single-threaded, ~45min with BATCH_SIZE=32 (batched TF).
Perch is CPU-only (TF SavedModel, no GPU kernel for this op).

Model loading (in priority order):
    1. PERCH_MODEL_DIR env var → local SavedModel directory
    2. tfhub v4 (default) → "https://tfhub.dev/google/bird-vocalization-classifier/4"
       (uses cluster tfhub cache if already downloaded)

Usage (cluster — uses tfhub v4 cache from previous run):
    KEGO_PATH_DATA=/home/kristian/projects/kego/data \\
        ~/.local/bin/uv run python competitions/birdclef-2026/perch_cache_train_clips.py

Usage (local Mac — uses local SavedModel):
    KEGO_PATH_DATA=/Users/kristianehlert/projects/kego/data \\
    PERCH_MODEL_DIR=/Users/kristianehlert/projects/kego/data/perch-v2 \\
        uv run python competitions/birdclef-2026/perch_cache_train_clips.py
"""

import glob
import os
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf

# ── config ────────────────────────────────────────────────────────────────────
DATA = Path(os.environ.get("KEGO_PATH_DATA", "data")) / "birdclef" / "birdclef-2026"
PERCH_MODEL_DIR = os.environ.get("PERCH_MODEL_DIR", "")  # optional local override
PERCH_TFHUB_URL = "https://tfhub.dev/google/bird-vocalization-classifier/4"
TRAIN_CSV = DATA / "train.csv"
TRAIN_AUDIO_DIR = DATA / "train_audio"
TAXONOMY_CSV = DATA / "taxonomy.csv"
OUT_NPZ = DATA / "perch_train_cache.npz"

SR = 32_000
CLIP_SAMPLES = SR * 5
BATCH_SIZE = 32  # clips per infer_tf call — tune if OOM
SAVE_EVERY = 500  # checkpoint to disk every N clips

# Use a separate output file when running with Perch v2 (1536-dim) vs v4 (1280-dim)
if PERCH_MODEL_DIR and Path(PERCH_MODEL_DIR).exists():
    OUT_NPZ = DATA / "perch_train_cache_v2.npz"
    print(f"Using Perch v2 model → output: {OUT_NPZ}")

# ── load competition species ───────────────────────────────────────────────────
taxonomy = pd.read_csv(TAXONOMY_CSV)
competition_species = taxonomy["primary_label"].astype(str).tolist()
n_species = len(competition_species)
sp_to_idx = {sp: i for i, sp in enumerate(competition_species)}
print(f"Competition species: {n_species}")

# ── load Perch model + label list ─────────────────────────────────────────────
if PERCH_MODEL_DIR and Path(PERCH_MODEL_DIR).exists():
    print(f"\nLoading Perch from local SavedModel: {PERCH_MODEL_DIR}")
    t0 = time.time()
    perch_model = tf.saved_model.load(PERCH_MODEL_DIR)
    perch_label_csv = Path(PERCH_MODEL_DIR) / "assets" / "label.csv"
    perch_labels_df = pd.read_csv(perch_label_csv)
    perch_labels = perch_labels_df.iloc[:, 0].astype(str).tolist()
else:
    import tensorflow_hub as hub

    print(f"\nLoading Perch v4 from tfhub: {PERCH_TFHUB_URL}")
    t0 = time.time()
    perch_model = hub.load(PERCH_TFHUB_URL)
    # find label file in tfhub cache
    matches = glob.glob("/tmp/tfhub_modules/**/label.csv", recursive=True)
    if not matches:
        raise FileNotFoundError(
            "Perch label.csv not found in tfhub cache. "
            "Run perch_cache_soundscapes.py first to populate it."
        )
    perch_label_csv_path = matches[0]
    with open(perch_label_csv_path) as f:
        # first line may be header "ebird2021,comment" — skip if so
        lines = [l.strip() for l in f if l.strip()]
    perch_labels = [l.split(",")[0] for l in lines if not l.startswith("ebird")]

print(f"Loaded in {time.time() - t0:.1f}s")

n_perch = len(perch_labels)
perch_to_idx = {lbl: i for i, lbl in enumerate(perch_labels)}
print(f"Perch labels: {n_perch}")

comp_to_perch = np.array(
    [perch_to_idx.get(sp, -1) for sp in competition_species], dtype=np.int32
)
perch_coverage = comp_to_perch >= 0
print(f"Direct Perch mapping: {perch_coverage.sum()} / {n_species} competition species")
print(f"Loaded in {time.time() - t0:.1f}s")

# smoke test + get embedding dim
_test = tf.zeros([1, CLIP_SAMPLES], dtype=tf.float32)
_logits, _emb = perch_model.infer_tf(_test)
EMB_DIM = _emb.shape[-1]
print(f"Embedding dim: {EMB_DIM}  |  logit dim: {_logits.shape[-1]}")


def logits_to_comp_probs(raw_logits: np.ndarray) -> np.ndarray:
    """Map (B, n_perch) logits → (B, n_species) sigmoid probs."""
    comp_probs = np.zeros((len(raw_logits), n_species), dtype=np.float32)
    comp_probs[:, perch_coverage] = 1.0 / (
        1.0 + np.exp(-raw_logits[:, comp_to_perch[perch_coverage]])
    )
    return comp_probs


# ── load train.csv and build label matrix ─────────────────────────────────────
print("\nLoading train.csv ...")
train_df = pd.read_csv(TRAIN_CSV)
print(f"  {len(train_df)} clips")


# Build binary label matrix from primary_label + secondary_labels
def parse_labels(row) -> list[str]:
    labels = [str(row["primary_label"]).strip()]
    sec = str(row.get("secondary_labels", "") or "")
    if sec and sec != "nan":
        labels += [
            s.strip()
            for s in sec.replace("[", "").replace("]", "").replace("'", "").split()
            if s.strip()
        ]
    return labels


# ── resume support ────────────────────────────────────────────────────────────
done_ids: set[str] = set()
all_clip_ids: list[str] = []
all_embeddings: list[np.ndarray] = []
all_comp_probs: list[np.ndarray] = []
all_labels: list[np.ndarray] = []

if OUT_NPZ.exists():
    print(f"\nResuming from existing cache: {OUT_NPZ}")
    cache = np.load(OUT_NPZ, allow_pickle=True)
    all_clip_ids = list(cache["clip_ids"].astype(str))
    all_embeddings = list(cache["embeddings"])
    all_comp_probs = list(cache["comp_probs"])
    all_labels = list(cache["labels"])
    done_ids = set(all_clip_ids)
    print(f"  Already cached: {len(done_ids)} clips")

# ── inference loop (batched) ──────────────────────────────────────────────────
print(f"\nRunning Perch inference (batch_size={BATCH_SIZE}) ...")
t_start = time.time()
n_total = len(train_df)
pending = train_df[
    ~train_df.apply(
        lambda r: f"{r['primary_label']}/{Path(str(r['filename'])).stem}", axis=1
    ).isin(done_ids)
].reset_index(drop=True)
print(f"  Clips to process: {len(pending)} / {n_total}")

batch_audio: list[np.ndarray] = []
batch_clip_ids: list[str] = []
batch_label_vecs: list[np.ndarray] = []
n_processed = 0
n_errors = 0


def flush_batch():
    global batch_audio, batch_clip_ids, batch_label_vecs
    if not batch_audio:
        return
    # pad/trim all to CLIP_SAMPLES
    padded = []
    for y in batch_audio:
        if len(y) >= CLIP_SAMPLES:
            padded.append(y[:CLIP_SAMPLES])
        else:
            padded.append(np.pad(y, (0, CLIP_SAMPLES - len(y))))
    x = tf.constant(np.stack(padded), dtype=tf.float32)
    raw_logits, embs = perch_model.infer_tf(x)
    raw_logits = raw_logits.numpy()
    embs = embs.numpy()
    cp = logits_to_comp_probs(raw_logits)
    for j in range(len(batch_clip_ids)):
        all_clip_ids.append(batch_clip_ids[j])
        all_embeddings.append(embs[j])
        all_comp_probs.append(cp[j])
        all_labels.append(batch_label_vecs[j])
    batch_audio.clear()
    batch_clip_ids.clear()
    batch_label_vecs.clear()


for i, row in pending.iterrows():
    filename = str(row["filename"])
    primary = str(row["primary_label"])
    clip_id = f"{primary}/{Path(filename).stem}"

    # audio path: train_audio/{filename} (filename already includes subdir)
    audio_path = TRAIN_AUDIO_DIR / filename
    if not audio_path.exists():
        # some datasets store without subdir
        audio_path = TRAIN_AUDIO_DIR / Path(filename).name
    if not audio_path.exists():
        n_errors += 1
        continue

    try:
        y, _ = librosa.load(audio_path, sr=SR, mono=True, duration=5.0)
    except Exception as e:
        n_errors += 1
        if n_errors <= 5:
            print(f"  WARN: {audio_path}: {e}")
        continue

    # build binary label vector
    label_vec = np.zeros(n_species, dtype=np.float32)
    for lbl in parse_labels(row):
        if lbl in sp_to_idx:
            label_vec[sp_to_idx[lbl]] = 1.0

    batch_audio.append(y)
    batch_clip_ids.append(clip_id)
    batch_label_vecs.append(label_vec)
    n_processed += 1

    if len(batch_audio) >= BATCH_SIZE:
        flush_batch()

    # progress + periodic save
    done_total = len(done_ids) + n_processed
    if n_processed % 100 == 0:
        elapsed = time.time() - t_start
        rate = n_processed / max(elapsed, 1)
        eta = (len(pending) - n_processed) / max(rate, 1e-9)
        print(
            f"  [{done_total}/{n_total}] {rate:.1f} clips/s | "
            f"ETA {eta / 60:.0f}min | errors {n_errors}",
            flush=True,
        )

    if n_processed % SAVE_EVERY == 0 and n_processed > 0:
        flush_batch()
        np.savez_compressed(
            OUT_NPZ,
            clip_ids=np.array(all_clip_ids),
            embeddings=np.stack(all_embeddings),
            comp_probs=np.stack(all_comp_probs),
            labels=np.stack(all_labels),
            species=np.array(competition_species),
        )
        print(f"  Checkpoint saved ({len(all_clip_ids)} clips so far)", flush=True)

flush_batch()

# ── final save ────────────────────────────────────────────────────────────────
elapsed_total = time.time() - t_start
print(f"\nDone in {elapsed_total / 60:.1f} min | errors: {n_errors}")

np.savez_compressed(
    OUT_NPZ,
    clip_ids=np.array(all_clip_ids),
    embeddings=np.stack(all_embeddings),
    comp_probs=np.stack(all_comp_probs),
    labels=np.stack(all_labels),
    species=np.array(competition_species),
)
print(f"Saved {len(all_clip_ids)} clips → {OUT_NPZ}")
print(f"  embeddings: {np.stack(all_embeddings).shape}")

# quick coverage report
labels_arr = np.stack(all_labels)
positives_per_species = labels_arr.sum(axis=0)
covered = (positives_per_species > 0).sum()
print(f"  Species with ≥1 positive clip: {covered} / {n_species}")
