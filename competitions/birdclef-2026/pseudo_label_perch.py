"""
Generate Perch pseudo-labels for BirdCLEF 2026 train soundscapes.

Runs Google's bird-vocalization-classifier (Perch v4) on all 10,658 train
soundscapes in 5-second windows, producing a pseudo-label CSV that can be
merged with the labeled XC training data for retraining.

This matches the BirdCLEF 2024 1st-place pipeline (arpoyda):
  - Perch runs on UNLABELED soundscapes, not on labeled XC clips
  - High-confidence predictions (logit > THRESHOLD) become pseudo-labels
  - Pseudo-labeled segments are added to training alongside ground-truth XC clips

Perch v4 covers 10,932 species. Of our 234 competition species, 158 are bird
species with matching eBird codes; the remaining 76 (frogs, insects, mammals)
are not in Perch's label set and will have no pseudo-labels.

Usage (on GPU server):
    KEGO_PATH_DATA=/home/kristian/projects/kego/data \\
        uv run python competitions/birdclef-2026/pseudo_label_perch.py

Outputs:
    data/birdclef/birdclef-2026/perch_pseudo_labels.csv
        Columns: soundscape_filename, start_sec, end_sec, primary_label
        One row per (soundscape, 5s window) that has ≥1 confident prediction.
        primary_label: semicolon-separated predicted species (e.g. "ashgre1;trokin")

    data/birdclef/birdclef-2026/perch_pseudo_labels_soft.npz
        - filenames: str array of "soundscape_filename:start_sec" keys
        - labels: float32 (n_windows, 234) raw sigmoid probabilities (all windows)
        - species: str array (234,) species in label order
"""

import csv
import os
import time
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# ── config ─────────────────────────────────────────────────────────────────────
DATA = Path(os.environ.get("KEGO_PATH_DATA", "data")) / "birdclef" / "birdclef-2026"
SOUNDSCAPE_DIR = DATA / "train_soundscapes"
TAXONOMY_CSV = DATA / "taxonomy.csv"
OUT_CSV = DATA / "perch_pseudo_labels.csv"
OUT_SOFT = DATA / "perch_pseudo_labels_soft.npz"

PERCH_URL = "https://tfhub.dev/google/bird-vocalization-classifier/4"
PERCH_LABEL_PATH = (
    "/tmp/tfhub_modules/5dcbb82658655292c50ca88ce1e6f1073b17d0d9/assets/label.csv"
)

SR = 32_000
CLIP_SAMPLES = SR * 5  # 5-second window
LOGIT_THRESHOLD = 0.0  # sigmoid(0) = 0.5 — only high-confidence predictions

# ── load competition species list ──────────────────────────────────────────────
with open(TAXONOMY_CSV) as f:
    taxonomy = list(csv.DictReader(f))
competition_species = [t["primary_label"] for t in taxonomy]
n_species = len(competition_species)
print(f"Competition species: {n_species}")

# ── load Perch label list and build mapping ────────────────────────────────────
perch_labels = open(PERCH_LABEL_PATH).read().splitlines()
perch_to_idx = {lbl: i for i, lbl in enumerate(perch_labels)}

# comp_to_perch[i] = Perch index for competition species i, or -1 if not in Perch
comp_to_perch = np.array(
    [perch_to_idx.get(sp, -1) for sp in competition_species], dtype=np.int32
)
perch_coverage = comp_to_perch >= 0
n_covered = int(perch_coverage.sum())
print(f"Perch coverage: {n_covered}/{n_species} competition species")

# ── find soundscape files ──────────────────────────────────────────────────────
soundscape_files = sorted(SOUNDSCAPE_DIR.glob("*.ogg"))
print(f"Train soundscapes: {len(soundscape_files)}")

# ── load Perch model ───────────────────────────────────────────────────────────
print("\nLoading Perch model...")
t0 = time.time()
model = hub.load(PERCH_URL)
print(f"Loaded in {time.time() - t0:.1f}s\n")


def process_soundscape(path: Path) -> tuple[list[dict], list[np.ndarray]]:
    """
    Process a single soundscape file.

    Returns:
        rows: list of dicts for high-confidence 5s windows
        soft_labels: list of (234,) arrays for ALL windows (for soft .npz)
    """
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
    except Exception as e:
        print(f"  WARN: failed to load {path.name}: {e}", flush=True)
        return [], []

    rows, soft_labels = [], []
    n_windows = len(y) // CLIP_SAMPLES

    for w in range(n_windows):
        chunk = y[w * CLIP_SAMPLES : (w + 1) * CLIP_SAMPLES]
        start_sec = w * 5

        # Run Perch
        x = tf.constant(chunk[np.newaxis], dtype=tf.float32)
        logits, _ = model.infer_tf(x)
        raw_logits = logits[0].numpy()  # (10932,)

        # Map to competition label space
        comp_logits = np.full(n_species, -999.0, dtype=np.float32)
        comp_logits[perch_coverage] = raw_logits[comp_to_perch[perch_coverage]]

        # Soft labels (sigmoid, for npz)
        comp_probs = np.zeros(n_species, dtype=np.float32)
        comp_probs[perch_coverage] = 1.0 / (1.0 + np.exp(-comp_logits[perch_coverage]))
        soft_labels.append(comp_probs)

        # Hard pseudo-labels (logit > threshold)
        predicted_idxs = np.where((comp_logits > LOGIT_THRESHOLD) & perch_coverage)[0]
        if len(predicted_idxs) > 0:
            predicted_species = [competition_species[i] for i in predicted_idxs]
            rows.append(
                {
                    "soundscape_filename": path.name,
                    "start_sec": start_sec,
                    "end_sec": start_sec + 5,
                    "primary_label": ";".join(predicted_species),
                    "n_species": len(predicted_species),
                }
            )

    return rows, soft_labels


# ── main loop ─────────────────────────────────────────────────────────────────
all_rows = []
all_soft = []
all_keys = []
n_total = len(soundscape_files)
t_start = time.time()

for i, sc_path in enumerate(soundscape_files):
    rows, soft = process_soundscape(sc_path)
    all_rows.extend(rows)

    n_windows = len(soft)
    for w, prob in enumerate(soft):
        all_soft.append(prob)
        all_keys.append(f"{sc_path.name}:{w * 5}")

    if (i + 1) % 200 == 0 or i == 0:
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed
        eta = (n_total - i - 1) / rate
        print(
            f"  [{i + 1}/{n_total}] {rate:.1f} files/s  "
            f"pseudo-labeled windows: {len(all_rows)}  "
            f"ETA {eta / 60:.1f} min",
            flush=True,
        )

# ── write CSV ──────────────────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "soundscape_filename",
            "start_sec",
            "end_sec",
            "primary_label",
            "n_species",
        ],
    )
    writer.writeheader()
    writer.writerows(all_rows)

total_windows = len(soundscape_files) * 12  # 60s / 5s
print(f"\nPseudo-labeled rows: {len(all_rows)} / {total_windows} windows")
print(f"Coverage: {100 * len(all_rows) / total_windows:.1f}% of soundscape windows")
print(f"Saved CSV → {OUT_CSV}")

# ── write soft label npz ───────────────────────────────────────────────────────
label_matrix = np.stack(all_soft, axis=0) if all_soft else np.zeros((0, n_species))
np.savez(
    OUT_SOFT,
    labels=label_matrix,
    filenames=np.array(all_keys),
    species=np.array(competition_species),
    perch_coverage=perch_coverage,
)
print(f"Saved soft labels → {OUT_SOFT}  shape={label_matrix.shape}")
print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")
