"""
Train per-class LogisticRegression probes on top of Perch v4 embeddings.

Uses the 66 labeled train soundscapes (792 windows) as training data.
Probes learn a linear mapping from Perch embedding space → competition species,
calibrated to the Pantanal soundscape distribution.

Pipeline:
    Perch embeddings (792 × 1280)
        → PCA (64 dims)
            → LogReg per class (C=0.25, GroupKFold by site)
                → OOF predictions (792 × 234)

Outputs:
    data/birdclef/birdclef-2026/perch_probes.pkl
        {
          "pca": sklearn PCA object (fit on all 792 windows),
          "probes": dict[str, LogisticRegression | None],  # species → probe (None = no probe)
          "probe_species": list[str],  # species with trained probes
          "oof_probs": np.ndarray (792, 234),  # out-of-fold probe predictions
          "perch_probs": np.ndarray (792, 234),  # raw Perch competition-space sigmoid
          "gt_labels": np.ndarray (792, 234),  # ground truth binary labels
          "filenames": np.ndarray (792,),
          "species": np.ndarray (234,),
        }

Usage:
    cd /home/kristian/projects/kego
    KEGO_PATH_DATA=/home/kristian/projects/kego/data .venv/bin/python3 \
        competitions/birdclef-2026/train_perch_probes.py
"""

import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

DATA = Path(os.environ.get("KEGO_PATH_DATA", "data")) / "birdclef" / "birdclef-2026"
CACHE_NPZ = DATA / "perch_labeled_cache.npz"
LABELS_CSV = DATA / "train_soundscapes_labels.csv"
TAXONOMY_CSV = DATA / "taxonomy.csv"
OUT_PKL = DATA / "perch_probes.pkl"

PCA_DIMS = 64
LOGREG_C = 0.25
MIN_POSITIVES = 8  # minimum positive windows to train a probe
N_FOLDS = 5

# ── load cache ────────────────────────────────────────────────────────────────
print("Loading Perch cache...")
cache = np.load(CACHE_NPZ, allow_pickle=True)
filenames = cache["filenames"].astype(str)  # (792,)  "stem:end_sec"
embeddings = cache["embeddings"]  # (792, 1280)
perch_probs = cache["labels"]  # (792, 234)  Perch sigmoid probs
species = cache["species"].astype(str)  # (234,)
n_windows, n_species = perch_probs.shape
print(f"  {n_windows} windows, {n_species} species, emb dim {embeddings.shape[1]}")

species_to_idx = {sp: i for i, sp in enumerate(species)}

# ── extract site and stem per window ─────────────────────────────────────────
stems = np.array([fn.split(":")[0] for fn in filenames])
end_secs = np.array([int(fn.split(":")[1]) for fn in filenames])


def extract_site(stem: str) -> str:
    m = re.search(r"_(S\d+)_", stem)
    return m.group(1) if m else "S00"


sites = np.array([extract_site(s) for s in stems])
unique_sites = sorted(set(sites))
print(f"  Sites: {unique_sites}")

# ── build ground truth labels from train_soundscapes_labels.csv ──────────────
print("\nBuilding ground truth labels...")

df = pd.read_csv(LABELS_CSV)


# parse "HH:MM:SS" → seconds
def parse_time(t: str) -> int:
    h, m, s = t.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


df["start_sec"] = df["start"].apply(parse_time)
df["end_sec_gt"] = df["end"].apply(parse_time)
df["stem"] = df["filename"].str.replace(".ogg", "", regex=False)
df["labels_list"] = df["primary_label"].astype(str).str.split(";")

# build lookup: stem → list of (start_sec, end_sec, labels)
from collections import defaultdict

stem_to_gt: dict[str, list[tuple[int, int, list[str]]]] = defaultdict(list)
for _, row in df.iterrows():
    stem_to_gt[row["stem"]].append(
        (row["start_sec"], row["end_sec_gt"], row["labels_list"])
    )

# for each window in cache, collect ground truth labels
gt_labels = np.zeros((n_windows, n_species), dtype=np.float32)
n_labeled = 0
for i, (stem, end_sec) in enumerate(zip(stems, end_secs)):
    start_sec = end_sec - 5
    for gt_start, gt_end, label_list in stem_to_gt.get(stem, []):
        # overlap check: window [start_sec, end_sec) overlaps [gt_start, gt_end)
        if start_sec < gt_end and end_sec > gt_start:
            for lbl in label_list:
                lbl = lbl.strip()
                if lbl in species_to_idx:
                    gt_labels[i, species_to_idx[lbl]] = 1.0
                    n_labeled += 1

n_annotated_windows = (gt_labels.sum(axis=1) > 0).sum()
print(f"  Windows with ≥1 label: {n_annotated_windows} / {n_windows}")
print(f"  Total (window, species) positives: {int(gt_labels.sum())}")
positives_per_species = gt_labels.sum(axis=0)
species_with_labels = (positives_per_species > 0).sum()
print(f"  Species with ≥1 positive window: {species_with_labels} / {n_species}")

# ── PCA on embeddings ─────────────────────────────────────────────────────────
print(f"\nFitting PCA({PCA_DIMS}) on {n_windows} windows...")
scaler = StandardScaler()
emb_scaled = scaler.fit_transform(embeddings)
pca = PCA(n_components=PCA_DIMS, random_state=42)
X = pca.fit_transform(emb_scaled)
print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# ── train per-class LogReg probes ─────────────────────────────────────────────
print(
    f"\nTraining LogReg probes (C={LOGREG_C}, {N_FOLDS}-fold by site, min_pos={MIN_POSITIVES})..."
)

# encode sites as group labels
site_to_int = {s: i for i, s in enumerate(unique_sites)}
groups = np.array([site_to_int[s] for s in sites])

probes: dict[str, LogisticRegression | None] = {}
oof_probs = np.zeros((n_windows, n_species), dtype=np.float32)
n_trained = 0
t0 = time.time()

gkf = GroupKFold(n_splits=N_FOLDS)

for sp_idx, sp in enumerate(species):
    y = gt_labels[:, sp_idx]
    n_pos = int(y.sum())

    if n_pos < MIN_POSITIVES:
        probes[sp] = None
        # fall back to Perch probs
        oof_probs[:, sp_idx] = perch_probs[:, sp_idx]
        continue

    # train with GroupKFold to get OOF predictions
    oof_sp = np.zeros(n_windows, dtype=np.float32)
    trained_models = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr = y[train_idx]

        if y_tr.sum() < 2:
            # not enough positives in this fold's train set
            oof_sp[val_idx] = perch_probs[val_idx, sp_idx]
            continue

        clf = LogisticRegression(
            C=LOGREG_C,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
        clf.fit(X_tr, y_tr)
        oof_sp[val_idx] = clf.predict_proba(X_val)[:, 1]
        trained_models.append(clf)

    oof_probs[:, sp_idx] = oof_sp

    # train final probe on all data
    if len(trained_models) > 0:
        clf_final = LogisticRegression(
            C=LOGREG_C,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
        clf_final.fit(X, y)
        probes[sp] = clf_final
        n_trained += 1
    else:
        probes[sp] = None
        oof_probs[:, sp_idx] = perch_probs[:, sp_idx]

elapsed = time.time() - t0
print(f"  Trained {n_trained} probes in {elapsed:.1f}s")

# ── evaluate OOF probes vs Perch probs ───────────────────────────────────────
from sklearn.metrics import average_precision_score

probe_aps, perch_aps = [], []
species_with_probes = []
for sp_idx, sp in enumerate(species):
    y = gt_labels[:, sp_idx]
    if y.sum() < 2:
        continue
    try:
        ap_probe = average_precision_score(y, oof_probs[:, sp_idx])
        ap_perch = average_precision_score(y, perch_probs[:, sp_idx])
        probe_aps.append(ap_probe)
        perch_aps.append(ap_perch)
        species_with_probes.append(sp)
    except Exception:
        pass

print(f"\nOOF evaluation on {len(probe_aps)} species with ≥2 positives:")
print(f"  Perch probs  mean AP: {np.mean(perch_aps):.4f}")
print(f"  Probe OOF    mean AP: {np.mean(probe_aps):.4f}")
print(f"  Delta: {np.mean(probe_aps) - np.mean(perch_aps):+.4f}")

# top improved species
deltas = [
    (species_with_probes[i], probe_aps[i] - perch_aps[i]) for i in range(len(probe_aps))
]
deltas.sort(key=lambda x: -x[1])
print("\n  Top 10 species improved by probe:")
for sp, d in deltas[:10]:
    print(f"    {sp}: {d:+.3f}")
print("  Bottom 5 (probe hurt):")
for sp, d in deltas[-5:]:
    print(f"    {sp}: {d:+.3f}")

# ── save ──────────────────────────────────────────────────────────────────────
out = {
    "pca": pca,
    "scaler": scaler,
    "probes": probes,
    "probe_species": [sp for sp in species if probes.get(sp) is not None],
    "oof_probs": oof_probs,
    "perch_probs": perch_probs,
    "gt_labels": gt_labels,
    "filenames": filenames,
    "species": species,
    "pca_dims": PCA_DIMS,
    "logreg_c": LOGREG_C,
}

with open(OUT_PKL, "wb") as f:
    pickle.dump(out, f)

probe_count = sum(1 for v in probes.values() if v is not None)
print(f"\nSaved {probe_count} probes → {OUT_PKL}")
print(
    f"  {n_species - probe_count} species fall back to Perch probs (< {MIN_POSITIVES} positives)"
)
