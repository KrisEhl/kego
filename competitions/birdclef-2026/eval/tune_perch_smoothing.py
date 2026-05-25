"""Tune 5-way acoustic archetype smoothing alphas for the Perch pipeline.

Grid-searches over the 5 alpha values using GroupKFold(5) OOF on the 59
labeled soundscapes in the perch-meta cache.  Metric: cmAP (class-mean AP).

Usage (run from repo root):
    KEGO_PATH_DATA=/home/kristian/projects/kego/data \\
    uv run python competitions/birdclef-2026/eval/tune_perch_smoothing.py

Output: best alpha config printed to stdout.
"""

import itertools
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

DATA_ROOT = Path(os.getenv("KEGO_PATH_DATA", "data")) / "birdclef" / "birdclef-2026"
CACHE_DIR = Path(os.getenv("KEGO_PATH_DATA", "data")) / "perch-meta"
N_WINDOWS = 12
SEED = 42


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------


def _load() -> tuple:
    taxonomy = pd.read_csv(DATA_ROOT / "taxonomy.csv")
    train_meta = pd.read_csv(DATA_ROOT / "train.csv")
    sc_raw = pd.read_csv(DATA_ROOT / "train_soundscapes_labels.csv")
    sample_sub = pd.read_csv(DATA_ROOT / "sample_submission.csv")

    PRIMARY_LABELS = sample_sub.columns[1:].tolist()
    label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}
    N_CLASSES = len(PRIMARY_LABELS)

    def parse_labels(x):
        if pd.isna(x):
            return []
        return [t.strip() for t in str(x).split(";") if t.strip()]

    def union_labels(s):
        return sorted(set(lbl for x in s for lbl in parse_labels(x)))

    FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")

    def parse_fname(name):
        m = FNAME_RE.match(name)
        if not m:
            return {"site": None, "hour_utc": -1}
        _, site, _, hms = m.groups()
        return {"site": site, "hour_utc": int(hms[:2])}

    sc_clean = (
        sc_raw.drop_duplicates()
        .groupby(["filename", "start", "end"])["primary_label"]
        .apply(union_labels)
        .reset_index(name="label_list")
    )
    sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
    sc_clean["row_id"] = (
        sc_clean["filename"].str.replace(".ogg", "", regex=False) + "_" + sc_clean["end_sec"].astype(str)
    )
    meta_cols = sc_clean["filename"].apply(parse_fname).apply(pd.Series)
    sc_clean = pd.concat([sc_clean, meta_cols], axis=1)

    wpf = sc_clean.groupby("filename").size()
    full_files = sorted(wpf[wpf == N_WINDOWS].index.tolist())
    full_truth = (
        sc_clean[sc_clean["filename"].isin(full_files)].sort_values(["filename", "end_sec"]).reset_index(drop=False)
    )

    # Ground truth matrix
    Y_SC = np.zeros((len(sc_clean), N_CLASSES), dtype=np.uint8)
    for i, labels in enumerate(sc_clean["label_list"]):
        for lbl in labels:
            if lbl in label_to_idx:
                Y_SC[i, label_to_idx[lbl]] = 1

    # Load Perch cache
    meta_full = pd.read_parquet(CACHE_DIR / "full_perch_meta.parquet")
    arr = np.load(CACHE_DIR / "full_perch_arrays.npz")
    scores_raw = arr["scores_full_raw"].astype(np.float32)

    # Align ground truth
    full_truth_aligned = full_truth.set_index("row_id").loc[meta_full["row_id"]].reset_index(drop=False)
    Y_FULL = Y_SC[full_truth_aligned["index"].to_numpy()]

    # Taxonomy class map
    class_name_map = taxonomy.set_index("primary_label")["class_name"].to_dict()

    # Perch BC label mapping (to identify mapped vs unmapped)
    bc_labels = None
    mapping_path = DATA_ROOT / "taxonomy.csv"  # no Perch labels locally
    # We don't have bc_labels locally, so use a simple heuristic:
    # treat all active Aves as "direct" (conservative) for local tuning
    ACTIVE_CLASSES = [PRIMARY_LABELS[i] for i in np.where(Y_SC.sum(axis=0) > 0)[0]]
    idx_smooth_amphibia = np.array(
        [label_to_idx[c] for c in ACTIVE_CLASSES if class_name_map.get(c) == "Amphibia"],
        dtype=np.int32,
    )
    idx_smooth_insecta = np.array(
        [label_to_idx[c] for c in ACTIVE_CLASSES if class_name_map.get(c) == "Insecta"],
        dtype=np.int32,
    )
    _RARE_TAXA = {"Mammalia", "Reptilia"}
    idx_smooth_rare = np.array(
        [label_to_idx[c] for c in ACTIVE_CLASSES if class_name_map.get(c) in _RARE_TAXA],
        dtype=np.int32,
    )
    idx_smooth_aves = np.array(
        [label_to_idx[c] for c in ACTIVE_CLASSES if class_name_map.get(c) == "Aves"],
        dtype=np.int32,
    )

    return (
        scores_raw,
        Y_FULL,
        meta_full,
        sc_clean,
        Y_SC,
        idx_smooth_amphibia,
        idx_smooth_insecta,
        idx_smooth_rare,
        idx_smooth_aves,
        label_to_idx,
        PRIMARY_LABELS,
        N_CLASSES,
    )


# ---------------------------------------------------------------------------
# Prior fusion (simplified — just smoothing experiment, no site/hour prior)
# ---------------------------------------------------------------------------


def smooth_cols(scores: np.ndarray, cols: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0 or len(cols) == 0:
        return scores
    s = scores.copy()
    view = s.reshape(-1, N_WINDOWS, s.shape[1])
    x = view[:, :, cols]
    prev = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    nxt = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    view[:, :, cols] = (1.0 - alpha) * x + 0.5 * alpha * (prev + nxt)
    return s


def gauss_smooth(scores: np.ndarray) -> np.ndarray:
    weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    pad = len(weights) // 2
    view = scores.reshape(-1, N_WINDOWS, scores.shape[1])
    out = view.copy()
    for i in range(view.shape[0]):
        xp = np.pad(view[i], ((pad, pad), (0, 0)), mode="edge")
        frame = np.zeros_like(view[i])
        for k, w in enumerate(weights):
            frame += w * xp[k : k + N_WINDOWS]
        out[i] = frame
    return out.reshape(-1, scores.shape[1])


def cmap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    aps = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() > 0:
            aps.append(average_precision_score(y_true[:, c], y_score[:, c]))
    return float(np.mean(aps)) if aps else 0.0


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------


def evaluate_alphas(
    scores_raw,
    Y_FULL,
    meta_full,
    idx_smooth_amphibia,
    idx_smooth_insecta,
    idx_smooth_rare,
    idx_smooth_aves,
    alpha_amphibia: float,
    alpha_insecta: float,
    alpha_aves: float,
    alpha_rare: float,
) -> float:
    """Apply smoothing and compute OOF cmAP using GroupKFold(5) on site."""
    groups = meta_full["site"].to_numpy()
    gkf = GroupKFold(n_splits=5)

    oof = np.zeros_like(scores_raw)
    for _, va_idx in gkf.split(scores_raw, groups=groups):
        s = scores_raw[va_idx].copy()
        s = smooth_cols(s, idx_smooth_amphibia, alpha_amphibia)
        s = smooth_cols(s, idx_smooth_insecta, alpha_insecta)
        s = smooth_cols(s, idx_smooth_aves, alpha_aves)
        s = smooth_cols(s, idx_smooth_rare, alpha_rare)
        oof[va_idx] = s

    oof_smoothed = gauss_smooth(oof)
    import scipy.special

    oof_prob = scipy.special.expit(oof_smoothed)
    return cmap(Y_FULL, oof_prob)


def main():
    print("Loading data...")
    (
        scores_raw,
        Y_FULL,
        meta_full,
        sc_clean,
        Y_SC,
        idx_smooth_amphibia,
        idx_smooth_insecta,
        idx_smooth_rare,
        idx_smooth_aves,
        label_to_idx,
        PRIMARY_LABELS,
        N_CLASSES,
    ) = _load()

    print(f"scores_raw : {scores_raw.shape}")
    print(f"Y_FULL     : {Y_FULL.shape}  positives: {Y_FULL.sum()}")
    print(f"Amphibia   : {len(idx_smooth_amphibia)} classes")
    print(f"Insecta    : {len(idx_smooth_insecta)} classes")
    print(f"Aves       : {len(idx_smooth_aves)} classes")
    print(f"Rare       : {len(idx_smooth_rare)} classes")

    # Baseline: no smoothing
    import scipy.special

    baseline_prob = scipy.special.expit(gauss_smooth(scores_raw))
    baseline_cmap = cmap(Y_FULL, baseline_prob)
    print(f"\nBaseline (Gauss only):  cmAP = {baseline_cmap:.6f}")

    # Current best: binary split (Amphibia+Insecta=0.35, Aves=0.0)
    texture_idx = np.concatenate([idx_smooth_amphibia, idx_smooth_insecta])
    s = smooth_cols(scores_raw, texture_idx, 0.35)
    binary_prob = scipy.special.expit(gauss_smooth(s))
    binary_cmap = cmap(Y_FULL, binary_prob)
    print(f"Binary (0.35/0.0):      cmAP = {binary_cmap:.6f}")

    # Grid search
    alpha_vals_texture = [0.25, 0.30, 0.35, 0.40, 0.45]
    alpha_vals_aves = [0.00, 0.05, 0.10, 0.15, 0.20]
    alpha_vals_rare = [0.00, 0.05, 0.10]

    # To avoid combinatorial explosion, fix amphibia and insecta independently first
    # then tune aves and rare
    best_val = 0.0
    best_cfg = {}
    rows = []

    combos = list(
        itertools.product(
            [0.35, 0.40, 0.45],  # amphibia
            [0.30, 0.35, 0.40],  # insecta
            [0.00, 0.10, 0.15, 0.20],  # aves
            [0.00, 0.05, 0.10],  # rare
        )
    )
    print(f"\nRunning grid search over {len(combos)} configs...")

    for a_amp, a_ins, a_aves, a_rare in tqdm(combos):
        val = evaluate_alphas(
            scores_raw,
            Y_FULL,
            meta_full,
            idx_smooth_amphibia,
            idx_smooth_insecta,
            idx_smooth_rare,
            idx_smooth_aves,
            a_amp,
            a_ins,
            a_aves,
            a_rare,
        )
        rows.append(
            {
                "amphibia": a_amp,
                "insecta": a_ins,
                "aves": a_aves,
                "rare": a_rare,
                "oof_cmap": val,
            }
        )
        if val > best_val:
            best_val = val
            best_cfg = {
                "amphibia": a_amp,
                "insecta": a_ins,
                "aves": a_aves,
                "rare": a_rare,
            }

    df = pd.DataFrame(rows).sort_values("oof_cmap", ascending=False)
    print("\nTop 10 configs:")
    print(df.head(10).to_string(index=False))
    print(f"\nBest: {best_cfg}  cmAP = {best_val:.6f}")
    print(f"vs baseline {baseline_cmap:.6f}  delta = {best_val - baseline_cmap:+.6f}")
    print(f"vs binary   {binary_cmap:.6f}  delta = {best_val - binary_cmap:+.6f}")


if __name__ == "__main__":
    main()
