"""
Cache Perch v4 embeddings and logits for the 66 labeled train soundscapes.

Outputs:
    data/birdclef/birdclef-2026/perch_labeled_cache.npz
        - filenames: (N,) str  "BC2026_SN01_20250101_060000:5" (soundscape:end_sec)
        - embeddings: (N, 1024) float32  Perch embedding vectors
        - logits: (N, 10932) float32  raw Perch logits (all species)
        - labels: (N, 234) float32  sigmoid probs mapped to competition species

    data/birdclef/birdclef-2026/perch_label_list.txt
        10,932-line file, one Perch species code per line (index = Perch class idx)

These are used to train per-class LogisticRegression probes for the Perch Track A pipeline.

Usage:
    cd /home/kristian/projects/kego
    LD_LIBRARY_PATH=$(python -c "import nvidia.cublas.lib; import os; print(os.path.dirname(nvidia.cublas.lib.__file__))"):$LD_LIBRARY_PATH \\
        KEGO_PATH_DATA=/home/kristian/projects/kego/data \\
        .venv/bin/python competitions/birdclef-2026/perch_cache_soundscapes.py

    # Or CPU-only (slower, ~30 min):
    CUDA_VISIBLE_DEVICES="" KEGO_PATH_DATA=... .venv/bin/python ...
"""

import csv
import os
import time
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# ── config ────────────────────────────────────────────────────────────────────
DATA = Path(os.environ.get("KEGO_PATH_DATA", "data")) / "birdclef" / "birdclef-2026"
SOUNDSCAPE_LABELS_CSV = DATA / "train_soundscapes_labels.csv"
SOUNDSCAPE_DIR = DATA / "train_soundscapes"
TAXONOMY_CSV = DATA / "taxonomy.csv"
OUT_NPZ = DATA / "perch_labeled_cache.npz"
OUT_LABELS_TXT = DATA / "perch_label_list.txt"

PERCH_URL = "https://tfhub.dev/google/bird-vocalization-classifier/4"
PERCH_LABEL_PATH = "/tmp/tfhub_modules/5dcbb82658655292c50ca88ce1e6f1073b17d0d9/assets/label.csv"

SR = 32_000
CLIP_SAMPLES = SR * 5

# ── load competition species ───────────────────────────────────────────────────
with open(TAXONOMY_CSV) as f:
    taxonomy = list(csv.DictReader(f))
competition_species = [t["primary_label"] for t in taxonomy]
n_species = len(competition_species)
print(f"Competition species: {n_species}")

# ── find the 66 labeled soundscapes ───────────────────────────────────────────
with open(SOUNDSCAPE_LABELS_CSV) as f:
    sc_labels = list(csv.DictReader(f))

labeled_stems = sorted({row["filename"].replace(".ogg", "") for row in sc_labels})
soundscape_files = []
for stem in labeled_stems:
    p = SOUNDSCAPE_DIR / f"{stem}.ogg"
    if p.exists():
        soundscape_files.append(p)
    else:
        print(f"  WARN: {p} not found")

print(f"Labeled soundscapes found: {len(soundscape_files)} / {len(labeled_stems)}")

# ── load Perch model ──────────────────────────────────────────────────────────
print("\nLoading Perch v4 from tfhub...")
t0 = time.time()
model = hub.load(PERCH_URL)
print(f"Loaded in {time.time() - t0:.1f}s")

# ── load Perch label list ─────────────────────────────────────────────────────
# Labels file is created by TF Hub after first load
if not Path(PERCH_LABEL_PATH).exists():
    # Try to find it
    import glob

    matches = glob.glob("/tmp/tfhub_modules/**/label.csv", recursive=True)
    if matches:
        PERCH_LABEL_PATH = matches[0]
        print(f"Found Perch labels at: {PERCH_LABEL_PATH}")
    else:
        raise FileNotFoundError("Perch label file not found. Run pseudo_label_perch.py first to populate tfhub cache.")

perch_labels = open(PERCH_LABEL_PATH).read().splitlines()
n_perch = len(perch_labels)
print(f"Perch label count: {n_perch}")

# Save label list
with open(OUT_LABELS_TXT, "w") as f:
    f.write("\n".join(perch_labels))
print(f"Saved Perch label list → {OUT_LABELS_TXT}")

# ── build competition → Perch mapping ─────────────────────────────────────────
perch_to_idx = {lbl: i for i, lbl in enumerate(perch_labels)}
comp_to_perch = np.array([perch_to_idx.get(sp, -1) for sp in competition_species], dtype=np.int32)
perch_coverage = comp_to_perch >= 0
print(f"Competition species with direct Perch mapping: {perch_coverage.sum()} / {n_species}")

# ── build genus → Perch indices mapping (for proxy species) ───────────────────
# Used later for genus-level proxy predictions for unmapped Amphibia/Insecta species
genus_to_perch_indices: dict[str, list[int]] = {}
for i, lbl in enumerate(perch_labels):
    genus = lbl.split("_")[0] if "_" in lbl else lbl[:6]
    genus_to_perch_indices.setdefault(genus, []).append(i)

comp_genera = [t.get("genus", "") for t in taxonomy]  # may be empty if not in CSV
print(f"Genera with Perch coverage: {sum(1 for g in comp_genera if g in genus_to_perch_indices)}")

# ── inference ─────────────────────────────────────────────────────────────────
all_keys = []
all_embeddings = []
all_logits = []
all_comp_probs = []

t_start = time.time()
for i, sc_path in enumerate(soundscape_files):
    try:
        y, _ = librosa.load(sc_path, sr=SR, mono=True)
    except Exception as e:
        print(f"  WARN: failed to load {sc_path.name}: {e}")
        continue

    n_windows = len(y) // CLIP_SAMPLES
    for w in range(n_windows):
        chunk = y[w * CLIP_SAMPLES : (w + 1) * CLIP_SAMPLES]
        start_sec = w * 5
        end_sec = start_sec + 5

        x = tf.constant(chunk[np.newaxis], dtype=tf.float32)
        raw_logits, emb = model.infer_tf(x)
        raw_logits = raw_logits[0].numpy()  # (10932,)
        emb = emb[0].numpy()  # (1024,) — Perch embedding dim

        # Competition-space probs
        comp_logits = np.full(n_species, -999.0, dtype=np.float32)
        comp_logits[perch_coverage] = raw_logits[comp_to_perch[perch_coverage]]
        comp_probs = np.zeros(n_species, dtype=np.float32)
        comp_probs[perch_coverage] = 1.0 / (1.0 + np.exp(-comp_logits[perch_coverage]))

        key = f"{sc_path.stem}:{end_sec}"
        all_keys.append(key)
        all_embeddings.append(emb)
        all_logits.append(raw_logits)
        all_comp_probs.append(comp_probs)

    elapsed = time.time() - t_start
    rate = (i + 1) / elapsed
    eta = (len(soundscape_files) - i - 1) / rate
    print(
        f"  [{i + 1}/{len(soundscape_files)}] {sc_path.name}: {n_windows} windows | "
        f"{rate:.1f} files/s | ETA {eta / 60:.1f} min",
        flush=True,
    )

# ── save ──────────────────────────────────────────────────────────────────────
embeddings_arr = np.stack(all_embeddings, axis=0)  # (N, 1024)
logits_arr = np.stack(all_logits, axis=0)  # (N, 10932)
comp_probs_arr = np.stack(all_comp_probs, axis=0)  # (N, 234)

np.savez_compressed(
    OUT_NPZ,
    filenames=np.array(all_keys),
    embeddings=embeddings_arr,
    logits=logits_arr,
    labels=comp_probs_arr,
    species=np.array(competition_species),
    perch_coverage=perch_coverage,
)

elapsed_total = time.time() - t_start
print(f"\nDone in {elapsed_total / 60:.1f} min")
print(f"Saved {len(all_keys)} windows → {OUT_NPZ}")
print(f"  embeddings: {embeddings_arr.shape}  logits: {logits_arr.shape}")
