"""
Generate Perch pseudo-labels for BirdCLEF 2026 training clips.

Runs Google's bird-vocalization-classifier (Perch v4) on all training clips,
producing a soft label matrix of shape (n_clips, 234) saved as a .npz file.

Perch covers 158/162 bird species in the competition (97.5%). Non-bird taxa
(frogs, mammals, insects) will have zero pseudo-labels and rely on ground truth.

Usage:
    uv run python competitions/birdclef-2026/pseudo_label_perch.py

Output:
    data/birdclef/birdclef-2026/perch_labels.npz
        - labels: float32 array (n_clips, 234) — raw Perch logits mapped to competition species
        - filenames: str array (n_clips,) — relative filenames matching train.csv
        - species: str array (234,) — competition species in label order
        - perch_coverage: bool array (234,) — True if species is in Perch's label set
"""

import csv
import os
import time
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# ── paths ──────────────────────────────────────────────────────────────────────
DATA = Path(os.environ.get("KEGO_PATH_DATA", "data")) / "birdclef" / "birdclef-2026"
TRAIN_CSV = DATA / "train.csv"
TAXONOMY_CSV = DATA / "taxonomy.csv"
AUDIO_DIR = DATA / "train_audio"
OUT_PATH = DATA / "perch_labels.npz"

PERCH_URL = "https://tfhub.dev/google/bird-vocalization-classifier/4"
PERCH_LABEL_PATH = (
    "/tmp/tfhub_modules/5dcbb82658655292c50ca88ce1e6f1073b17d0d9/assets/label.csv"
)

SR = 32_000
CLIP_SAMPLES = SR * 5  # 5-second window

# ── load competition species list (ordered) ────────────────────────────────────
with open(TAXONOMY_CSV) as f:
    taxonomy = list(csv.DictReader(f))
competition_species = [t["primary_label"] for t in taxonomy]
n_species = len(competition_species)
species_to_idx = {s: i for i, s in enumerate(competition_species)}
print(f"Competition species: {n_species}")

# ── load Perch label list ──────────────────────────────────────────────────────
perch_labels = open(PERCH_LABEL_PATH).read().splitlines()
perch_to_idx = {lbl: i for i, lbl in enumerate(perch_labels)}
print(f"Perch species: {len(perch_labels)}")

# ── build mapping: competition_idx → perch_idx (-1 if not covered) ─────────────
comp_to_perch = np.full(n_species, -1, dtype=np.int32)
perch_coverage = np.zeros(n_species, dtype=bool)
for comp_idx, sp in enumerate(competition_species):
    if sp in perch_to_idx:
        comp_to_perch[comp_idx] = perch_to_idx[sp]
        perch_coverage[comp_idx] = True

n_covered = perch_coverage.sum()
print(f"Perch coverage: {n_covered}/{n_species} species")
uncovered = [competition_species[i] for i in range(n_species) if not perch_coverage[i]]
print(
    f"Uncovered ({len(uncovered)}): {uncovered[:10]}{'...' if len(uncovered) > 10 else ''}"
)

# ── load Perch model ───────────────────────────────────────────────────────────
print("\nLoading Perch model...")
t0 = time.time()
model = hub.load(PERCH_URL)
print(f"Loaded in {time.time() - t0:.1f}s")


def perch_infer(audio_np: np.ndarray) -> np.ndarray:
    """Run Perch on a (1, samples) float32 array. Returns logits (10932,)."""
    x = tf.constant(audio_np[np.newaxis], dtype=tf.float32)
    logits, _embedding = model.infer_tf(x)
    return logits[0].numpy()  # (10932,)


def process_clip(audio_path: Path) -> np.ndarray:
    """
    Load audio, split into 5s windows, run Perch on each, return
    mean logits mapped to competition label space: shape (n_species,).
    """
    try:
        y, _ = librosa.load(audio_path, sr=SR, mono=True)
    except Exception as e:
        print(f"  WARN: failed to load {audio_path}: {e}")
        return np.zeros(n_species, dtype=np.float32)

    # Collect 5s windows (non-overlapping; include partial last window if ≥2.5s)
    windows = []
    for start in range(0, len(y), CLIP_SAMPLES):
        chunk = y[start : start + CLIP_SAMPLES]
        if len(chunk) < CLIP_SAMPLES // 2:
            break
        if len(chunk) < CLIP_SAMPLES:
            chunk = np.pad(chunk, (0, CLIP_SAMPLES - len(chunk)))
        windows.append(chunk)

    if not windows:
        return np.zeros(n_species, dtype=np.float32)

    # Run Perch on each window and average logits
    all_logits = np.stack([perch_infer(w) for w in windows])  # (n_windows, 10932)
    mean_logits = all_logits.mean(axis=0)  # (10932,)

    # Map to competition label space
    comp_logits = np.zeros(n_species, dtype=np.float32)
    covered_mask = comp_to_perch >= 0
    comp_logits[covered_mask] = mean_logits[comp_to_perch[covered_mask]]
    return comp_logits


# ── load training clip list ────────────────────────────────────────────────────
with open(TRAIN_CSV) as f:
    train_rows = list(csv.DictReader(f))
print(f"\nTraining clips: {len(train_rows)}")

# ── main loop ─────────────────────────────────────────────────────────────────
filenames = []
all_labels = []
n_total = len(train_rows)
t_start = time.time()

for i, row in enumerate(train_rows):
    fname = row["filename"]
    audio_path = AUDIO_DIR / fname

    labels = process_clip(audio_path)
    filenames.append(fname)
    all_labels.append(labels)

    if (i + 1) % 100 == 0 or i == 0:
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed
        eta = (n_total - i - 1) / rate
        print(
            f"  [{i + 1}/{n_total}] {rate:.1f} clips/s  ETA {eta / 60:.1f} min",
            flush=True,
        )

label_matrix = np.stack(all_labels, axis=0).astype(np.float32)
print(f"\nLabel matrix shape: {label_matrix.shape}")
print(f"Non-zero entries: {(label_matrix != 0).sum()} / {label_matrix.size}")

# ── save ───────────────────────────────────────────────────────────────────────
np.savez(
    OUT_PATH,
    labels=label_matrix,
    filenames=np.array(filenames),
    species=np.array(competition_species),
    perch_coverage=perch_coverage,
)
print(f"\nSaved → {OUT_PATH}")
print(f"Total time: {(time.time() - t_start) / 60:.1f} min")
