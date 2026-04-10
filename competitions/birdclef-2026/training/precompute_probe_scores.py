"""Precompute OOF probe-augmented logit scores for the 708 labeled soundscape windows.

Replicates the full probe pipeline from kaggle_perch_v2_inference.ipynb:
  1. Prior fusion (Bayesian site/hour priors) → oof_base
  2. PCA(128) on embeddings + per-class logistic regressions → probe predictions
  3. oof_probe = (1 - PROBE_ALPHA) * oof_base + PROBE_ALPHA * probe_pred (per class)

Output: oof_probe_scores.npy — (708, 234) float32 logit scores
These mirror the quality of `final_scores` in the inference notebook before
ResidualSSMv3 is applied. Train ResidualSSMv3 with BCE(probe_scores + correction,
labels) so training base matches inference base.

Usage:
    uv run python competitions/birdclef-2026/training/precompute_probe_scores.py
    # Saves data/perch-meta/oof_probe_scores.npy and full_probe_scores.npy

    # With adapted embeddings (from train_perch_adapter.py):
    uv run python competitions/birdclef-2026/training/precompute_probe_scores.py \\
        --emb-file data/perch-meta/full_emb_adapted.npy \\
        --output-suffix adapted
    # Saves data/perch-meta/oof_probe_scores_adapted.npy and full_probe_scores_adapted.npy
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

DATA_ROOT = Path(os.environ.get("KEGO_PATH_DATA", "data"))
COMPETITION_DATA = DATA_ROOT / "birdclef" / "birdclef-2026"
PERCH_META_DIR = DATA_ROOT / "perch-meta"

# Frozen V18 probe params
PROBE_PCA_DIM = 128
PROBE_MIN_POS = 5
PROBE_C = 0.75
PROBE_ALPHA = 0.45  # blend weight for probe prediction
N_WINDOWS = 12
N_CLASSES = 234

# Prior fusion lambdas (from V18 inference notebook)
LAMBDA_EVENT = 0.45
LAMBDA_TEXTURE = 1.1
LAMBDA_PROXY_TEXTURE = 0.9

# Temporal smoothing alphas
SMOOTH_AMPHIBIA = 0.45
SMOOTH_INSECTA = 0.35
SMOOTH_AVES_DIRECT = 0.00
SMOOTH_AVES_UNCERTAIN = 0.00
SMOOTH_RARE = 0.05


# ---------------------------------------------------------------------------
# Prior fusion (copied faithfully from inference notebook)
# ---------------------------------------------------------------------------


def fit_prior_tables(prior_df: pd.DataFrame, Y_prior: np.ndarray) -> dict:
    prior_df = prior_df.reset_index(drop=True)
    global_p = Y_prior.mean(axis=0).astype(np.float32)

    site_keys = sorted(prior_df["site"].dropna().astype(str).unique())
    hour_keys = sorted(prior_df["hour_utc"].dropna().astype(int).unique())

    site_to_i, site_n, site_p = {}, [], []
    for s in site_keys:
        mask = prior_df["site"].astype(str).values == s
        site_to_i[s] = len(site_n)
        site_n.append(mask.sum())
        site_p.append(Y_prior[mask].mean(axis=0))
    site_n = np.array(site_n, dtype=np.float32)
    site_p = (
        np.stack(site_p).astype(np.float32)
        if site_p
        else np.zeros((0, Y_prior.shape[1]), np.float32)
    )

    hour_to_i, hour_n, hour_p = {}, [], []
    for h in hour_keys:
        mask = prior_df["hour_utc"].astype(int).values == h
        hour_to_i[h] = len(hour_n)
        hour_n.append(mask.sum())
        hour_p.append(Y_prior[mask].mean(axis=0))
    hour_n = np.array(hour_n, dtype=np.float32)
    hour_p = (
        np.stack(hour_p).astype(np.float32)
        if hour_p
        else np.zeros((0, Y_prior.shape[1]), np.float32)
    )

    sh_to_i, sh_n_list, sh_p_list = {}, [], []
    for (s, h), idx in prior_df.groupby(["site", "hour_utc"]).groups.items():
        sh_to_i[(str(s), int(h))] = len(sh_n_list)
        idx_arr = np.array(list(idx))
        sh_n_list.append(len(idx_arr))
        sh_p_list.append(Y_prior[idx_arr].mean(axis=0))
    sh_n = np.array(sh_n_list, dtype=np.float32)
    sh_p = (
        np.stack(sh_p_list).astype(np.float32)
        if sh_p_list
        else np.zeros((0, Y_prior.shape[1]), np.float32)
    )

    return dict(
        global_p=global_p,
        site_to_i=site_to_i,
        site_n=site_n,
        site_p=site_p,
        hour_to_i=hour_to_i,
        hour_n=hour_n,
        hour_p=hour_p,
        sh_to_i=sh_to_i,
        sh_n=sh_n,
        sh_p=sh_p,
    )


def prior_logits_fn(
    sites: np.ndarray, hours: np.ndarray, tables: dict, eps: float = 1e-4
) -> np.ndarray:
    n = len(sites)
    p = np.repeat(tables["global_p"][None, :], n, axis=0).astype(np.float32, copy=True)

    si = np.fromiter((tables["site_to_i"].get(str(s), -1) for s in sites), np.int32, n)
    hi = np.fromiter(
        (tables["hour_to_i"].get(int(h), -1) if int(h) >= 0 else -1 for h in hours),
        np.int32,
        n,
    )
    shi = np.fromiter(
        (
            tables["sh_to_i"].get((str(s), int(h)), -1) if int(h) >= 0 else -1
            for s, h in zip(sites, hours)
        ),
        np.int32,
        n,
    )

    valid = hi >= 0
    if valid.any():
        nh = tables["hour_n"][hi[valid]][:, None]
        p[valid] = (
            nh / (nh + 8.0) * tables["hour_p"][hi[valid]]
            + (1.0 - nh / (nh + 8.0)) * p[valid]
        )
    valid = si >= 0
    if valid.any():
        ns = tables["site_n"][si[valid]][:, None]
        p[valid] = (
            ns / (ns + 8.0) * tables["site_p"][si[valid]]
            + (1.0 - ns / (ns + 8.0)) * p[valid]
        )
    valid = shi >= 0
    if valid.any():
        nsh = tables["sh_n"][shi[valid]][:, None]
        p[valid] = (
            nsh / (nsh + 4.0) * tables["sh_p"][shi[valid]]
            + (1.0 - nsh / (nsh + 4.0)) * p[valid]
        )

    np.clip(p, eps, 1.0 - eps, out=p)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


def smooth_cols(scores: np.ndarray, cols: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0 or len(cols) == 0:
        return scores.copy()
    s = scores.copy()
    view = s.reshape(-1, N_WINDOWS, s.shape[1])
    x = view[:, :, cols]
    prev = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    nxt = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    view[:, :, cols] = (1.0 - alpha) * x + 0.5 * alpha * (prev + nxt)
    return s


def fuse_scores_full(
    base: np.ndarray,
    sites: np.ndarray,
    hours: np.ndarray,
    tables: dict,
    idx_mapped_active_event: np.ndarray,
    idx_mapped_active_texture: np.ndarray,
    idx_selected_proxy_active_texture: np.ndarray,
    idx_selected_prioronly_active_event: np.ndarray,
    idx_selected_prioronly_active_texture: np.ndarray,
    idx_unmapped_inactive: np.ndarray,
    idx_smooth_amphibia: np.ndarray,
    idx_smooth_insecta: np.ndarray,
    idx_smooth_aves_direct: np.ndarray,
    idx_smooth_aves_uncertain: np.ndarray,
    idx_smooth_rare: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores = base.copy()
    prior = prior_logits_fn(sites, hours, tables)

    if len(idx_mapped_active_event):
        scores[:, idx_mapped_active_event] += (
            LAMBDA_EVENT * prior[:, idx_mapped_active_event]
        )
    if len(idx_mapped_active_texture):
        scores[:, idx_mapped_active_texture] += (
            LAMBDA_TEXTURE * prior[:, idx_mapped_active_texture]
        )
    if len(idx_selected_proxy_active_texture):
        scores[:, idx_selected_proxy_active_texture] += (
            LAMBDA_PROXY_TEXTURE * prior[:, idx_selected_proxy_active_texture]
        )
    if len(idx_selected_prioronly_active_event):
        scores[:, idx_selected_prioronly_active_event] = (
            LAMBDA_EVENT * prior[:, idx_selected_prioronly_active_event]
        )
    if len(idx_selected_prioronly_active_texture):
        scores[:, idx_selected_prioronly_active_texture] = (
            LAMBDA_TEXTURE * prior[:, idx_selected_prioronly_active_texture]
        )
    if len(idx_unmapped_inactive):
        scores[:, idx_unmapped_inactive] = -8.0

    scores = smooth_cols(scores, idx_smooth_amphibia, SMOOTH_AMPHIBIA)
    scores = smooth_cols(scores, idx_smooth_insecta, SMOOTH_INSECTA)
    scores = smooth_cols(scores, idx_smooth_aves_direct, SMOOTH_AVES_DIRECT)
    scores = smooth_cols(scores, idx_smooth_aves_uncertain, SMOOTH_AVES_UNCERTAIN)
    scores = smooth_cols(scores, idx_smooth_rare, SMOOTH_RARE)
    return scores.astype(np.float32), prior.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature helpers (faithful copy from inference notebook)
# ---------------------------------------------------------------------------


def seq_features_1d(v: np.ndarray) -> tuple:
    x = v.reshape(-1, N_WINDOWS)
    prev = np.concatenate([x[:, :1], x[:, :-1]], axis=1).reshape(-1)
    nxt = np.concatenate([x[:, 1:], x[:, -1:]], axis=1).reshape(-1)
    mean_v = np.repeat(x.mean(1), N_WINDOWS)
    max_v = np.repeat(x.max(1), N_WINDOWS)
    min_v = np.repeat(x.min(1), N_WINDOWS)
    range_v = max_v - min_v
    std_v = np.repeat(x.std(1), N_WINDOWS)
    diff_from_mean = v - mean_v
    window_pos = np.tile(np.arange(N_WINDOWS) / N_WINDOWS, x.shape[0])
    delta_prev = v - prev
    return (
        prev,
        nxt,
        mean_v,
        max_v,
        min_v,
        range_v,
        std_v,
        diff_from_mean,
        window_pos,
        delta_prev,
    )


def cosine_sim_to_prototype(Z: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    Z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    p_norm = prototype / (np.linalg.norm(prototype) + 1e-8)
    return (Z_norm @ p_norm).astype(np.float32)


def build_class_features(
    Z: np.ndarray,
    raw_col: np.ndarray,
    prior_col: np.ndarray,
    base_col: np.ndarray,
    proto_sim_col: np.ndarray | None = None,
    family_mean_col: np.ndarray | None = None,
) -> np.ndarray:
    p, n, m, mx, mn, rng, std_v, diff_mean, w_pos, d_prev = seq_features_1d(base_col)
    parts = [
        Z,
        raw_col[:, None],
        prior_col[:, None],
        base_col[:, None],
        p[:, None],
        n[:, None],
        m[:, None],
        mx[:, None],
        mn[:, None],
        rng[:, None],
        std_v[:, None],
        diff_mean[:, None],
        w_pos[:, None],
        d_prev[:, None],
    ]
    if proto_sim_col is not None:
        parts.append(proto_sim_col[:, None])
    if family_mean_col is not None:
        parts.append(family_mean_col[:, None])
    return np.concatenate(parts, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute OOF and full probe scores from Perch embeddings"
    )
    parser.add_argument(
        "--emb-file",
        type=str,
        default=None,
        help=(
            "Path to adapted embeddings .npy file (overrides emb_full from the NPZ). "
            "Use with adapted embeddings from train_perch_adapter.py."
        ),
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help=(
            "Suffix appended to output filenames. E.g. 'adapted' → "
            "oof_probe_scores_adapted.npy / full_probe_scores_adapted.npy. "
            "Default: no suffix (overwrites standard files)."
        ),
    )
    parser.add_argument(
        "--npz-file",
        type=str,
        default="full_perch_arrays.npz",
        help=(
            "Filename (in PERCH_META_DIR) of the Perch cache NPZ to load. "
            "Default: 'full_perch_arrays.npz'. "
            "Use 'full_perch_arrays_59.npz' to compute probe scores for the 59-soundscape set."
        ),
    )
    parser.add_argument(
        "--apply-extra-npz",
        type=str,
        default=None,
        help=(
            "If provided, fit probes on --npz-file data, then ALSO apply to the extra soundscapes "
            "in this NPZ file that are not present in --npz-file. The combined output is saved "
            "as full_probe_scores_<npz_suffix>_extended.npy. Example use: fit on 59sc, extend to 66sc: "
            "--npz-file full_perch_arrays_59.npz --apply-extra-npz full_perch_arrays.npz"
        ),
    )
    args = parser.parse_args()

    print(f"Data root: {DATA_ROOT}")

    # -----------------------------------------------------------------------
    # Load perch cache
    # -----------------------------------------------------------------------
    npz = np.load(PERCH_META_DIR / args.npz_file)
    scores_full_raw = npz["scores_full_raw"].astype(np.float32)

    if args.emb_file is not None:
        emb_path = Path(args.emb_file)
        if not emb_path.is_absolute():
            # Try PERCH_META_DIR first, then DATA_ROOT
            candidate = PERCH_META_DIR / emb_path
            if candidate.exists():
                emb_path = candidate
            else:
                emb_path = DATA_ROOT / args.emb_file
        emb_full = np.load(emb_path).astype(np.float32)
        print(f"Loaded adapted embeddings from {emb_path}: {emb_full.shape}")
    else:
        emb_full = npz["emb_full"].astype(np.float32)
    meta_file = args.npz_file.replace("full_perch_arrays", "full_perch_meta").replace(
        ".npz", ".parquet"
    )
    meta_full = pd.read_parquet(PERCH_META_DIR / meta_file)
    print(f"Loaded perch cache: {emb_full.shape}, {scores_full_raw.shape}")

    # -----------------------------------------------------------------------
    # Load taxonomy and compute MAPPED_MASK
    # -----------------------------------------------------------------------
    taxonomy = pd.read_csv(COMPETITION_DATA / "taxonomy.csv")
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
    primary_labels = taxonomy["primary_label"].tolist()
    label_to_idx = {s: i for i, s in enumerate(primary_labels)}
    n_classes = len(primary_labels)
    assert n_classes == N_CLASSES, f"Expected {N_CLASSES} classes, got {n_classes}"

    # Load Perch labels for MAPPED_MASK computation
    labels_csv_path = PERCH_META_DIR / "perch_labels.csv"
    if not labels_csv_path.exists():
        print(f"ERROR: {labels_csv_path} not found. Run:")
        print("  kaggle models instances versions download \\")
        print("    google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1 \\")
        print("    -p /tmp/perch && tar -xzf /tmp/perch/*.tar.gz assets/labels.csv")
        print("  cp assets/labels.csv data/perch-meta/perch_labels.csv")
        sys.exit(1)

    bc_labels = (
        pd.read_csv(labels_csv_path)
        .reset_index()
        .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
    )
    no_label_index = len(bc_labels)
    taxonomy["scientific_name"] = taxonomy["scientific_name"].astype(str)
    mapping = taxonomy.merge(
        bc_labels[["scientific_name", "bc_index"]], on="scientific_name", how="left"
    )
    mapping["bc_index"] = mapping["bc_index"].fillna(no_label_index).astype(int)
    label_to_bc = mapping.set_index("primary_label")["bc_index"]
    bc_indices = np.array(
        [int(label_to_bc.loc[c]) for c in primary_labels], dtype=np.int32
    )
    mapped_mask = bc_indices != no_label_index
    print(f"Mapped   : {mapped_mask.sum()} / {n_classes}")
    unmapped_pos = np.where(~mapped_mask)[0].astype(np.int32)

    # -----------------------------------------------------------------------
    # Load soundscape labels → sc_clean + Y_SC
    # -----------------------------------------------------------------------
    fname_re = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")

    def parse_labels(x):
        if pd.isna(x):
            return []
        return [t.strip() for t in str(x).split(";") if t.strip()]

    def union_labels(series):
        return sorted({lbl for x in series for lbl in parse_labels(x)})

    def parse_soundscape_filename(name):
        m = fname_re.match(name)
        if not m:
            return {"site": None, "hour_utc": -1}
        _, site, _, hms = m.groups()
        return {"site": site, "hour_utc": int(hms[:2])}

    soundscape_lbls = pd.read_csv(COMPETITION_DATA / "train_soundscapes_labels.csv")
    sc_clean = (
        soundscape_lbls.groupby(["filename", "start", "end"])["primary_label"]
        .apply(union_labels)
        .reset_index(name="label_list")
    )
    sc_clean["end_sec"] = (
        pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
    )
    meta_cols = sc_clean["filename"].apply(parse_soundscape_filename).apply(pd.Series)
    sc_clean = pd.concat([sc_clean, meta_cols], axis=1)

    Y_SC = np.zeros((len(sc_clean), n_classes), dtype=np.uint8)
    for i, lbls in enumerate(sc_clean["label_list"]):
        for lbl in lbls:
            if lbl in label_to_idx:
                Y_SC[i, label_to_idx[lbl]] = 1

    print(f"sc_clean rows: {len(sc_clean)}, Y_SC shape: {Y_SC.shape}")

    # -----------------------------------------------------------------------
    # Build multi-hot Y_FULL for the 708 windows in meta_full
    # -----------------------------------------------------------------------
    meta_full["window_sec"] = meta_full["row_id"].str.extract(r"_(\d+)$").astype(int)
    meta_idx_map: dict[tuple, int] = {}
    for pos, row in meta_full.iterrows():
        meta_idx_map[(row["filename"], int(row["window_sec"]))] = int(pos)

    Y_FULL = np.zeros((len(meta_full), n_classes), dtype=np.float32)
    for _, row in soundscape_lbls.iterrows():

        def time_to_sec(t: str) -> int:
            h, m, s = t.split(":")
            return int(h) * 3600 + int(m) * 60 + int(s)

        window_sec = time_to_sec(str(row["start"])) + 5
        key = (row["filename"], window_sec)
        if key not in meta_idx_map:
            continue
        pos = meta_idx_map[key]
        for lbl in str(row["primary_label"]).split(";"):
            lbl = lbl.strip()
            if lbl in label_to_idx:
                Y_FULL[pos, label_to_idx[lbl]] = 1.0
    print(
        f"Y_FULL positives: {Y_FULL.sum():.0f}, active classes: {(Y_FULL.sum(0) > 0).sum()}"
    )

    # -----------------------------------------------------------------------
    # Compute class type indices for fuse_scores
    # -----------------------------------------------------------------------
    class_name_map = taxonomy.set_index("primary_label")["class_name"].to_dict()
    texture_taxa = {"Amphibia", "Insecta"}
    rare_taxa = {"Mammalia", "Reptilia"}
    active_classes = [primary_labels[i] for i in np.where(Y_SC.sum(axis=0) > 0)[0]]

    idx_active_texture = np.array(
        [
            label_to_idx[c]
            for c in active_classes
            if class_name_map.get(c) in texture_taxa
        ],
        dtype=np.int32,
    )
    idx_active_event = np.array(
        [
            label_to_idx[c]
            for c in active_classes
            if class_name_map.get(c) not in texture_taxa
        ],
        dtype=np.int32,
    )
    idx_mapped_active_texture = idx_active_texture[mapped_mask[idx_active_texture]]
    idx_mapped_active_event = idx_active_event[mapped_mask[idx_active_event]]
    idx_unmapped_active_texture = idx_active_texture[~mapped_mask[idx_active_texture]]
    idx_unmapped_active_event = idx_active_event[~mapped_mask[idx_active_event]]
    idx_unmapped_inactive = np.array(
        [i for i in unmapped_pos if primary_labels[i] not in active_classes],
        dtype=np.int32,
    )

    # Proxy: unmapped non-sonotype species with a genus match in Perch
    unmapped_df = mapping[mapping["bc_index"] == no_label_index].copy()
    unmapped_non_sonotype = unmapped_df[
        ~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)
    ].copy()
    proxy_map: dict[str, list[int]] = {}
    for _, row in unmapped_non_sonotype.iterrows():
        genus = str(row["scientific_name"]).split()[0]
        hits = bc_labels[
            bc_labels["scientific_name"].str.match(rf"^{re.escape(genus)}\s", na=False)
        ]
        if len(hits) > 0:
            proxy_map[str(row["primary_label"])] = hits["bc_index"].astype(int).tolist()
    selected_proxy_pos = np.array(
        [label_to_idx[c] for c in proxy_map if c in label_to_idx], dtype=np.int32
    )
    idx_selected_proxy_active_texture = np.intersect1d(
        selected_proxy_pos, idx_active_texture
    )
    idx_selected_prioronly_active_texture = np.setdiff1d(
        idx_unmapped_active_texture, selected_proxy_pos
    )
    idx_selected_prioronly_active_event = np.setdiff1d(
        idx_unmapped_active_event, selected_proxy_pos
    )

    # Smoothing indices
    idx_smooth_amphibia = np.array(
        [
            label_to_idx[c]
            for c in active_classes
            if class_name_map.get(c) == "Amphibia"
        ],
        dtype=np.int32,
    )
    idx_smooth_insecta = np.array(
        [label_to_idx[c] for c in active_classes if class_name_map.get(c) == "Insecta"],
        dtype=np.int32,
    )
    idx_smooth_rare = np.array(
        [label_to_idx[c] for c in active_classes if class_name_map.get(c) in rare_taxa],
        dtype=np.int32,
    )
    idx_smooth_aves_direct = np.array(
        [
            label_to_idx[c]
            for c in active_classes
            if class_name_map.get(c) == "Aves" and mapped_mask[label_to_idx[c]]
        ],
        dtype=np.int32,
    )
    idx_smooth_aves_uncertain = np.array(
        [
            label_to_idx[c]
            for c in active_classes
            if class_name_map.get(c) == "Aves" and not mapped_mask[label_to_idx[c]]
        ],
        dtype=np.int32,
    )

    fuse_kwargs = dict(
        idx_mapped_active_event=idx_mapped_active_event,
        idx_mapped_active_texture=idx_mapped_active_texture,
        idx_selected_proxy_active_texture=idx_selected_proxy_active_texture,
        idx_selected_prioronly_active_event=idx_selected_prioronly_active_event,
        idx_selected_prioronly_active_texture=idx_selected_prioronly_active_texture,
        idx_unmapped_inactive=idx_unmapped_inactive,
        idx_smooth_amphibia=idx_smooth_amphibia,
        idx_smooth_insecta=idx_smooth_insecta,
        idx_smooth_aves_direct=idx_smooth_aves_direct,
        idx_smooth_aves_uncertain=idx_smooth_aves_uncertain,
        idx_smooth_rare=idx_smooth_rare,
    )
    print(f"Active texture: {len(idx_active_texture)}, event: {len(idx_active_event)}")

    # -----------------------------------------------------------------------
    # OOF prior fusion → oof_base
    # -----------------------------------------------------------------------
    gkf = GroupKFold(n_splits=5)
    groups = meta_full["site"].to_numpy()
    oof_base = np.zeros_like(scores_full_raw, dtype=np.float32)
    oof_prior_arr = np.zeros_like(scores_full_raw, dtype=np.float32)

    print("Computing OOF prior fusion...")
    for fold_i, (_, va_idx) in enumerate(gkf.split(scores_full_raw, groups=groups)):
        va_idx = np.sort(va_idx)
        val_sites = set(meta_full.iloc[va_idx]["site"].tolist())
        prior_m = ~sc_clean["site"].isin(val_sites).values
        tables = fit_prior_tables(
            sc_clean.loc[prior_m].reset_index(drop=True), Y_SC[prior_m]
        )
        fused, prior_vals = fuse_scores_full(
            scores_full_raw[va_idx],
            meta_full.iloc[va_idx]["site"].to_numpy(),
            meta_full.iloc[va_idx]["hour_utc"].to_numpy(),
            tables,
            **fuse_kwargs,
        )
        oof_base[va_idx] = fused
        oof_prior_arr[va_idx] = prior_vals
        print(f"  Fold {fold_i + 1}/5 — val_sites: {sorted(val_sites)}")

    from sklearn.metrics import roc_auc_score

    keep = Y_FULL.sum(0) > 0
    auc_base = roc_auc_score(Y_FULL[:, keep], oof_base[:, keep], average="macro")
    print(f"OOF AUC (prior fusion only): {auc_base:.6f}")

    # -----------------------------------------------------------------------
    # PCA on embeddings
    # -----------------------------------------------------------------------
    print("Fitting PCA...")
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(emb_full)
    n_comp = min(PROBE_PCA_DIM, emb_scaled.shape[0] - 1, emb_scaled.shape[1])
    pca = PCA(n_components=n_comp)
    z_full = pca.fit_transform(emb_scaled).astype(np.float32)
    print(
        f"PCA: {n_comp} components, explained var = {pca.explained_variance_ratio_.sum():.4f}"
    )

    # Class prototypes in PCA space
    class_prototypes: dict[int, np.ndarray] = {}
    for ci in range(n_classes):
        pos_mask = Y_FULL[:, ci] == 1
        if pos_mask.sum() >= PROBE_MIN_POS:
            class_prototypes[ci] = z_full[pos_mask].mean(axis=0)

    # Family groups
    family_map = taxonomy.set_index("primary_label")["class_name"].to_dict()
    family_groups: dict[str, list[int]] = {}
    for ci, label in enumerate(primary_labels):
        fam = family_map.get(label, "Unknown")
        family_groups.setdefault(fam, []).append(ci)
    family_idx_map = {
        fam: np.array(idxs, dtype=np.int32) for fam, idxs in family_groups.items()
    }
    class_family = {
        ci: family_map.get(label, "Unknown") for ci, label in enumerate(primary_labels)
    }

    # -----------------------------------------------------------------------
    # OOF probe computation
    # -----------------------------------------------------------------------
    print("Computing OOF probe scores...")
    oof_probe = oof_base.copy()  # start from prior-fused base

    for fold_i, (_, va_idx) in enumerate(gkf.split(scores_full_raw, groups=groups)):
        va_idx = np.sort(va_idx)
        tr_idx = np.setdiff1d(np.arange(len(scores_full_raw)), va_idx)

        # Build probes on training fold
        pos_cnt = Y_FULL[tr_idx].sum(axis=0)
        probe_cls = np.where(pos_cnt >= PROBE_MIN_POS)[0]

        for ci in probe_cls:
            y_tr = Y_FULL[tr_idx, ci]
            if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
                continue

            proto_sim_tr = (
                cosine_sim_to_prototype(z_full[tr_idx], class_prototypes[ci])
                if ci in class_prototypes
                else None
            )
            proto_sim_va = (
                cosine_sim_to_prototype(z_full[va_idx], class_prototypes[ci])
                if ci in class_prototypes
                else None
            )

            fam = class_family.get(ci, "Unknown")
            other_fam = family_idx_map.get(fam, np.array([]))
            other_fam = other_fam[other_fam != ci]
            fam_mean_tr = (
                oof_base[tr_idx][:, other_fam].mean(axis=1)
                if len(other_fam) > 0
                else None
            )
            fam_mean_va = (
                oof_base[va_idx][:, other_fam].mean(axis=1)
                if len(other_fam) > 0
                else None
            )

            X_tr = build_class_features(
                z_full[tr_idx],
                raw_col=scores_full_raw[tr_idx, ci],
                prior_col=oof_prior_arr[tr_idx, ci],
                base_col=oof_base[tr_idx, ci],
                proto_sim_col=proto_sim_tr,
                family_mean_col=fam_mean_tr,
            )
            X_va = build_class_features(
                z_full[va_idx],
                raw_col=scores_full_raw[va_idx, ci],
                prior_col=oof_prior_arr[va_idx, ci],
                base_col=oof_base[va_idx, ci],
                proto_sim_col=proto_sim_va,
                family_mean_col=fam_mean_va,
            )

            clf = LogisticRegression(
                C=PROBE_C, max_iter=400, solver="liblinear", class_weight="balanced"
            )
            clf.fit(X_tr, y_tr)
            pred = clf.decision_function(X_va).astype(np.float32)
            oof_probe[va_idx, ci] = (1.0 - PROBE_ALPHA) * oof_base[
                va_idx, ci
            ] + PROBE_ALPHA * pred

        n_probes = len([ci for ci in probe_cls if Y_FULL[tr_idx, ci].sum() > 0])
        print(f"  Fold {fold_i + 1}/5 — {n_probes} probes trained")

    auc_probe = roc_auc_score(Y_FULL[:, keep], oof_probe[:, keep], average="macro")
    print(
        f"OOF AUC (probe-augmented): {auc_probe:.6f}  (delta vs prior: +{auc_probe - auc_base:.4f})"
    )

    # cmAP
    from sklearn.metrics import average_precision_score

    active_cls = np.where(Y_FULL.sum(0) > 0)[0]
    aps_base = [
        average_precision_score(Y_FULL[:, c], oof_base[:, c])
        for c in active_cls
        if Y_FULL[:, c].sum() > 0
    ]
    aps_probe = [
        average_precision_score(Y_FULL[:, c], oof_probe[:, c])
        for c in active_cls
        if Y_FULL[:, c].sum() > 0
    ]
    print(f"OOF cmAP (prior only):   {np.mean(aps_base):.4f}")
    print(f"OOF cmAP (probe-augmented): {np.mean(aps_probe):.4f}")

    # -----------------------------------------------------------------------
    # Full (in-sample) probe scores — fit on ALL 708 windows, predict on all
    # These match the quality of inference-time probes (trained on full dataset)
    # Used as Stage 2 training base in submit mode
    # -----------------------------------------------------------------------
    print("\nComputing full (in-sample) probe scores...")
    # Fit full prior tables on all soundscape windows
    full_tables = fit_prior_tables(sc_clean, Y_SC)
    full_base, full_prior_arr = fuse_scores_full(
        scores_full_raw,
        meta_full["site"].to_numpy(),
        meta_full["hour_utc"].to_numpy(),
        full_tables,
        **fuse_kwargs,
    )

    full_probe = full_base.copy()
    pos_cnt_full = Y_FULL.sum(axis=0)
    full_probe_cls = np.where(pos_cnt_full >= PROBE_MIN_POS)[0]
    full_probe_models: dict[int, object] = {}

    for ci in full_probe_cls:
        y = Y_FULL[:, ci]
        if y.sum() == 0 or y.sum() == len(y):
            continue

        proto_sim = (
            cosine_sim_to_prototype(z_full, class_prototypes[ci])
            if ci in class_prototypes
            else None
        )
        fam = class_family.get(ci, "Unknown")
        other_fam = family_idx_map.get(fam, np.array([]))
        other_fam = other_fam[other_fam != ci]
        fam_mean = full_base[:, other_fam].mean(axis=1) if len(other_fam) > 0 else None

        X = build_class_features(
            z_full,
            raw_col=scores_full_raw[:, ci],
            prior_col=full_prior_arr[:, ci],
            base_col=full_base[:, ci],
            proto_sim_col=proto_sim,
            family_mean_col=fam_mean,
        )
        clf = LogisticRegression(
            C=PROBE_C, max_iter=400, solver="liblinear", class_weight="balanced"
        )
        clf.fit(X, y)
        pred = clf.decision_function(X).astype(np.float32)  # in-sample prediction
        full_probe[:, ci] = (1.0 - PROBE_ALPHA) * full_base[:, ci] + PROBE_ALPHA * pred
        full_probe_models[ci] = clf

    print(f"  Full probe models fitted: {len(full_probe_models)} classes")
    aps_full = [
        average_precision_score(Y_FULL[:, c], full_probe[:, c])
        for c in active_cls
        if Y_FULL[:, c].sum() > 0
    ]
    print(f"  Full (in-sample) cmAP: {np.mean(aps_full):.4f}")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    out_path = PERCH_META_DIR / f"oof_probe_scores{suffix}.npy"
    np.save(out_path, oof_probe)
    print(f"\nSaved: {out_path}  shape={oof_probe.shape}  dtype={oof_probe.dtype}")

    full_out_path = PERCH_META_DIR / f"full_probe_scores{suffix}.npy"
    np.save(full_out_path, full_probe)
    print(f"Saved: {full_out_path}  shape={full_probe.shape}  dtype={full_probe.dtype}")

    # -----------------------------------------------------------------------
    # Optional: apply 59sc-fitted probes to extra soundscapes in a larger NPZ
    # Output: combined probe scores (N_base + N_extra, 234)
    # Use case: fit on 59sc, extend to 66sc for more Stage 2 training data.
    # The extra soundscapes get OOF-quality probe scores (out-of-distribution
    # for the 59sc-fitted probes), which matches test-time inference quality.
    # -----------------------------------------------------------------------
    if args.apply_extra_npz is not None:
        print(
            f"\n--- Applying 59sc probes to extra soundscapes in {args.apply_extra_npz} ---"
        )
        extra_npz_path = PERCH_META_DIR / args.apply_extra_npz
        extra_npz = np.load(extra_npz_path)
        extra_emb_all = extra_npz["emb_full"].astype(np.float32)
        extra_scores_all = extra_npz["scores_full_raw"].astype(np.float32)
        extra_meta_file = args.apply_extra_npz.replace(
            "full_perch_arrays", "full_perch_meta"
        ).replace(".npz", ".parquet")
        extra_meta_all = pd.read_parquet(PERCH_META_DIR / extra_meta_file)
        print(
            f"  Loaded extra NPZ: {extra_emb_all.shape}, meta: {extra_meta_all.shape}"
        )

        # Identify the extra rows not present in the base NPZ
        base_filenames = set(meta_full["filename"].tolist())
        extra_mask = ~extra_meta_all["filename"].isin(base_filenames).values
        n_extra = extra_mask.sum()
        print(
            f"  Extra soundscapes: {n_extra} windows "
            f"({extra_meta_all[extra_mask]['filename'].nunique()} files)"
        )

        extra_emb = extra_emb_all[extra_mask]
        extra_scores_raw = extra_scores_all[extra_mask]
        extra_meta = extra_meta_all[extra_mask].reset_index(drop=True)

        # Apply PCA fitted on base (59sc) to extra embeddings
        extra_emb_scaled = scaler.transform(extra_emb)
        extra_z = pca.transform(extra_emb_scaled).astype(np.float32)

        # Compute prior fusion for extra rows using full_tables (fitted on base)
        extra_base, extra_prior_arr = fuse_scores_full(
            extra_scores_raw,
            extra_meta["site"].to_numpy(),
            extra_meta["hour_utc"].to_numpy(),
            full_tables,
            **fuse_kwargs,
        )

        # Apply per-class probe models to extra rows
        extra_probe = extra_base.copy()
        for ci, clf in full_probe_models.items():
            proto_sim_extra = (
                cosine_sim_to_prototype(extra_z, class_prototypes[ci])
                if ci in class_prototypes
                else None
            )
            fam = class_family.get(ci, "Unknown")
            other_fam = family_idx_map.get(fam, np.array([]))
            other_fam = other_fam[other_fam != ci]
            fam_mean_extra = (
                extra_base[:, other_fam].mean(axis=1) if len(other_fam) > 0 else None
            )
            X_extra = build_class_features(
                extra_z,
                raw_col=extra_scores_raw[:, ci],
                prior_col=extra_prior_arr[:, ci],
                base_col=extra_base[:, ci],
                proto_sim_col=proto_sim_extra,
                family_mean_col=fam_mean_extra,
            )
            pred_extra = clf.decision_function(X_extra).astype(np.float32)
            extra_probe[:, ci] = (1.0 - PROBE_ALPHA) * extra_base[
                :, ci
            ] + PROBE_ALPHA * pred_extra

        # Combine base (59sc) + extra in the order they appear in the larger NPZ
        # Reorder to match the larger NPZ's row order
        combined_probe = np.zeros((len(extra_meta_all), n_classes), dtype=np.float32)
        base_row_map = {fn: [] for fn in base_filenames}
        for idx, row in meta_full.iterrows():
            base_row_map[row["filename"]].append(int(idx))

        for ei, row in extra_meta_all.iterrows():
            if not extra_mask[ei]:
                # Base row: find matching row in full_probe
                fn = row["filename"]
                ws = row["row_id"].split("_")[-1]
                for base_idx in base_row_map.get(fn, []):
                    if meta_full.iloc[base_idx]["row_id"].split("_")[-1] == ws:
                        combined_probe[ei] = full_probe[base_idx]
                        break
            else:
                # Extra row: use extra_probe (need to find index in extra_probe)
                extra_idx = (extra_meta["row_id"] == row["row_id"]).values.argmax()
                combined_probe[ei] = extra_probe[extra_idx]

        print(f"  Combined probe scores shape: {combined_probe.shape}")
        # Derive suffix from the extra NPZ filename
        extra_stem = (
            args.apply_extra_npz.replace("full_perch_arrays", "")
            .replace(".npz", "")
            .strip("_")
            or "extended"
        )
        base_stem = (
            args.npz_file.replace("full_perch_arrays", "")
            .replace(".npz", "")
            .strip("_")
            or "base"
        )
        combined_out_path = (
            PERCH_META_DIR
            / f"full_probe_scores{suffix}_{base_stem}_probes_{extra_stem}_data.npy"
        )
        np.save(combined_out_path, combined_probe)
        print(f"Saved combined: {combined_out_path}  shape={combined_probe.shape}")


if __name__ == "__main__":
    main()
