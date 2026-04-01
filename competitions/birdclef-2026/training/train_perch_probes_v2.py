"""Train per-class LogReg probes on all 35,549 training clips (Perch v4 embeddings).

Supersedes train_perch_probes.py which only used 792 soundscape windows (53 species).
This version uses the full training set → probes for 206/234 species.

Pipeline:
    perch_train_cache.npz (35549 × 1280 embeddings, binary labels from train.csv)
        → StandardScaler → PCA(64 dims)
            → LogReg per class (C=1.0, StratifiedKFold-5)
                → OOF predictions (35549 × 234)

Outputs:
    data/birdclef/birdclef-2026/perch_probes_v2.pkl  (same format as perch_probes.pkl)
        {
          "pca":           sklearn PCA,
          "scaler":        sklearn StandardScaler,
          "probes":        dict[str, LogisticRegression | None],
          "species":       list[str] (234 in competition order),
          "oof_probs":     (35549, 234) float32,
          "perch_probs":   (35549, 234) float32,
          "gt_labels":     (35549, 234) float32,
          "clip_ids":      (35549,) str,
        }

Usage (cluster):
    KEGO_PATH_DATA=/home/kristian/projects/kego/data \\
        ~/.local/bin/uv run python competitions/birdclef-2026/train_perch_probes_v2.py

Usage (local):
    KEGO_PATH_DATA=/Users/kristianehlert/projects/kego/data \\
        uv run python competitions/birdclef-2026/train_perch_probes_v2.py
"""

import os
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DATA = Path(os.environ.get("KEGO_PATH_DATA", "data")) / "birdclef" / "birdclef-2026"
# Prefer v2 cache (Perch v2, 1536-dim) if available; fall back to v4 cache (1280-dim)
CACHE_NPZ = DATA / "perch_train_cache_v2.npz"
if not CACHE_NPZ.exists():
    CACHE_NPZ = DATA / "perch_train_cache.npz"
    print(f"v2 cache not found, using: {CACHE_NPZ}")
OUT_PKL = DATA / "perch_probes_v2.pkl"

PCA_DIMS = 64
LOGREG_C = 1.0  # more data → less regularisation needed vs C=0.25 before
MIN_POSITIVES = 10  # minimum positive clips to train a probe
N_FOLDS = 5

# ── load cache ────────────────────────────────────────────────────────────────
print(f"Loading cache: {CACHE_NPZ}")
cache = np.load(CACHE_NPZ, allow_pickle=True)
clip_ids = cache["clip_ids"].astype(str)  # (35549,)
embeddings = cache["embeddings"]  # (35549, 1280)
perch_probs = cache["comp_probs"]  # (35549, 234)  Perch sigmoid probs
gt_labels = cache["labels"]  # (35549, 234)  binary GT
species = cache["species"].astype(str)  # (234,)
n_clips, n_species = gt_labels.shape
print(f"  {n_clips} clips, {n_species} species, emb dim {embeddings.shape[1]}")

positives_per_species = gt_labels.sum(axis=0)
covered = (positives_per_species >= MIN_POSITIVES).sum()
print(f"  Species with ≥{MIN_POSITIVES} positives: {covered} / {n_species}")

# ── PCA ───────────────────────────────────────────────────────────────────────
print(f"\nFitting StandardScaler + PCA({PCA_DIMS}) on {n_clips} clips...")
t0 = time.time()
scaler = StandardScaler()
X = scaler.fit_transform(embeddings)
pca = PCA(n_components=PCA_DIMS, random_state=42)
X = pca.fit_transform(X)
print(
    f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}  ({time.time() - t0:.1f}s)"
)

# ── train probes ──────────────────────────────────────────────────────────────
print(
    f"\nTraining probes (C={LOGREG_C}, {N_FOLDS}-fold stratified, min_pos={MIN_POSITIVES})..."
)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
probes: dict[str, LogisticRegression | None] = {}
oof_probs = perch_probs.copy()  # default: fall back to Perch probs
n_trained = 0
t0 = time.time()

for sp_idx, sp in enumerate(species):
    y = gt_labels[:, sp_idx]
    n_pos = int(y.sum())

    if n_pos < MIN_POSITIVES:
        probes[sp] = None
        continue

    oof_sp = np.zeros(n_clips, dtype=np.float32)
    trained_folds = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        if y_tr.sum() < 2:
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
        trained_folds.append(clf)

    oof_probs[:, sp_idx] = oof_sp

    if trained_folds:
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

    if (sp_idx + 1) % 50 == 0:
        elapsed = time.time() - t0
        print(
            f"  [{sp_idx + 1}/{n_species}] {n_trained} probes trained  ({elapsed:.0f}s)",
            flush=True,
        )

elapsed = time.time() - t0
print(f"Trained {n_trained} probes in {elapsed:.1f}s")

# ── OOF evaluation ────────────────────────────────────────────────────────────
print("\nOOF evaluation...")
probe_aps, perch_aps = [], []
eval_species = []
for sp_idx, sp in enumerate(species):
    y = gt_labels[:, sp_idx]
    if y.sum() < 2 or (1 - y).sum() < 2:
        continue
    try:
        ap_probe = average_precision_score(y, oof_probs[:, sp_idx])
        ap_perch = average_precision_score(y, perch_probs[:, sp_idx])
        probe_aps.append(ap_probe)
        perch_aps.append(ap_perch)
        eval_species.append(sp)
    except Exception:
        pass

print(f"  Evaluated on {len(probe_aps)} species")
print(f"  Perch probs  mean AP: {np.mean(perch_aps):.4f}")
print(f"  Probe OOF    mean AP: {np.mean(probe_aps):.4f}")
print(f"  Delta:               {np.mean(probe_aps) - np.mean(perch_aps):+.4f}")

deltas = sorted(
    zip(eval_species, np.array(probe_aps) - np.array(perch_aps)), key=lambda x: -x[1]
)
print("\n  Top 10 improved:")
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
    "species": list(species),
    "oof_probs": oof_probs,
    "perch_probs": perch_probs,
    "gt_labels": gt_labels,
    "clip_ids": clip_ids,
    "pca_dims": PCA_DIMS,
    "logreg_c": LOGREG_C,
}
with open(OUT_PKL, "wb") as f:
    pickle.dump(out, f)

probe_count = sum(1 for v in probes.values() if v is not None)
print(f"\nSaved {probe_count} probes → {OUT_PKL}")
print(f"  {n_species - probe_count} species fall back to Perch probs")
