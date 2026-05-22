"""Per-class temperature calibration for the Perch v2 inference pipeline.

Loads jaejohn/perch-meta (708 rows = 59 soundscapes × 12 windows) and runs
5-fold GroupKFold OOF probe predictions, then optimises per-class temperature
T_c to maximise class-AP on the held-out folds.

Outputs:
  - console: OOF cmAP before/after calibration
  - per_class_temps.npy: (234,) float32 temperature vector → bake into notebook
  - per_class_temps.json: same in JSON format for inspection

Usage:
    uv run python competitions/birdclef-2026/eval/calibrate_perch_temps.py \
        --perch-meta data/perch-meta \
        --competition-data data/birdclef/birdclef-2026 \
        [--out-dir competitions/birdclef-2026/eval]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Feature helpers (must match kaggle_perch_v2_inference.ipynb exactly)
# ---------------------------------------------------------------------------

N_WINDOWS = 12


def seq_features_1d(v):
    x = v.reshape(-1, N_WINDOWS)
    prev = np.concatenate([x[:, :1], x[:, :-1]], axis=1).reshape(-1)
    nxt = np.concatenate([x[:, 1:], x[:, -1:]], axis=1).reshape(-1)
    mean_v = np.repeat(x.mean(1), N_WINDOWS)
    max_v = np.repeat(x.max(1), N_WINDOWS)
    min_v = np.repeat(x.min(1), N_WINDOWS)
    range_v = max_v - min_v
    return prev, nxt, mean_v, max_v, min_v, range_v


def cosine_sim_to_prototype(Z, prototype):
    Z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    p_norm = prototype / (np.linalg.norm(prototype) + 1e-8)
    return Z_norm @ p_norm


def build_class_features(
    Z, raw_col, prior_col, base_col, proto_sim_col=None, family_mean_col=None
):
    p, n, m, mx, mn, rng = seq_features_1d(base_col)
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
    ]
    if proto_sim_col is not None:
        parts.append(proto_sim_col[:, None])
    if family_mean_col is not None:
        parts.append(family_mean_col[:, None])
    return np.concatenate(parts, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Prior fusion (must match notebook exactly)
# ---------------------------------------------------------------------------

LAMBDA_EVENT = 0.4
LAMBDA_TEXTURE = 1.0


def _prior_lambda(class_name: str) -> float:
    if class_name in ("Aves",):
        return LAMBDA_EVENT
    return LAMBDA_TEXTURE


def fit_prior_tables(prior_df, Y_prior, class_names):
    prior_df = prior_df.reset_index(drop=True)
    global_p = Y_prior.mean(axis=0).astype(np.float32)
    lambdas = np.array([_prior_lambda(c) for c in class_names], dtype=np.float32)

    site_keys = sorted(prior_df["site"].dropna().astype(str).unique())
    hour_keys = sorted(prior_df["hour_utc"].dropna().astype(int).unique())

    def _table(keys, col):
        to_i, n_list, p_list = {}, [], []
        for k in keys:
            mask = prior_df[col].astype(str if isinstance(k, str) else int).values == k
            to_i[k] = len(n_list)
            n_list.append(mask.sum())
            p_list.append(Y_prior[mask].mean(axis=0))
        return (
            to_i,
            np.array(n_list, dtype=np.float32),
            (
                np.stack(p_list).astype(np.float32)
                if p_list
                else np.zeros((0, Y_prior.shape[1]), np.float32)
            ),
        )

    site_to_i, site_n, site_p = _table(site_keys, "site")
    hour_to_i, hour_n, hour_p = _table(hour_keys, "hour_utc")

    sh_to_i, sh_n_list, sh_p_list = {}, [], []
    for (s, h), idx in prior_df.groupby(["site", "hour_utc"]).groups.items():
        sh_to_i[(str(s), int(h))] = len(sh_n_list)
        idx = np.array(list(idx))
        sh_n_list.append(len(idx))
        sh_p_list.append(Y_prior[idx].mean(axis=0))
    sh_n = np.array(sh_n_list, dtype=np.float32)
    sh_p = (
        np.stack(sh_p_list).astype(np.float32)
        if sh_p_list
        else np.zeros((0, Y_prior.shape[1]), np.float32)
    )

    return dict(
        global_p=global_p,
        lambdas=lambdas,
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


def _shrink(p_local, n_local, p_global, lam):
    w = n_local / (n_local + lam)
    return w * p_local + (1 - w) * p_global


def fuse_scores(scores_raw, sites, hours, tables):
    n = len(scores_raw)
    N_CLS = scores_raw.shape[1]
    gp = tables["global_p"]
    lam = tables["lambdas"]

    base = scores_raw.copy().astype(np.float32)
    prior = np.zeros((n, N_CLS), dtype=np.float32)

    for i in range(n):
        s, h = str(sites[i]), int(hours[i])
        p_site = (
            _shrink(
                tables["site_p"][tables["site_to_i"][s]],
                tables["site_n"][tables["site_to_i"][s]],
                gp,
                lam,
            )
            if s in tables["site_to_i"]
            else gp
        )
        p_hour = (
            _shrink(
                tables["hour_p"][tables["hour_to_i"][h]],
                tables["hour_n"][tables["hour_to_i"][h]],
                gp,
                lam,
            )
            if h in tables["hour_to_i"]
            else gp
        )
        key = (s, h)
        p_sh = (
            _shrink(
                tables["sh_p"][tables["sh_to_i"][key]],
                tables["sh_n"][tables["sh_to_i"][key]],
                gp,
                lam,
            )
            if key in tables["sh_to_i"]
            else gp
        )
        combined_prior = (p_site + p_hour + p_sh) / 3.0
        prior[i] = combined_prior
        base[i] = base[i] + combined_prior

    return base, prior


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perch-meta", default="data/perch-meta")
    parser.add_argument("--competition-data", default="data/birdclef/birdclef-2026")
    parser.add_argument("--out-dir", default="competitions/birdclef-2026/eval")
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--probe-min-pos", type=int, default=8)
    parser.add_argument("--probe-c", type=float, default=0.25)
    parser.add_argument("--probe-alpha", type=float, default=0.40)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    perch_dir = Path(args.perch_meta)
    comp_dir = Path(args.competition_data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Load competition metadata
    taxonomy = pd.read_csv(comp_dir / "taxonomy.csv")
    sample_sub = pd.read_csv(comp_dir / "sample_submission.csv")
    PRIMARY_LABELS = sample_sub.columns[1:].tolist()
    N_CLASSES = len(PRIMARY_LABELS)
    label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}

    taxonomy_ = taxonomy.copy()
    taxonomy_["scientific_name"] = taxonomy_["scientific_name"].astype(str)
    CLASS_NAME_MAP = taxonomy_.set_index("primary_label")["class_name"].to_dict()
    class_names_ordered = [CLASS_NAME_MAP.get(l, "Aves") for l in PRIMARY_LABELS]

    # Load labeled soundscape data for ground-truth
    sc_labels = pd.read_csv(comp_dir / "train_soundscapes_labels.csv")

    # Load perch-meta cache
    meta_full = pd.read_parquet(perch_dir / "full_perch_meta.parquet")
    arrs = np.load(perch_dir / "full_perch_arrays.npz")
    scores_full_raw = arrs["scores_full_raw"].astype(np.float32)
    emb_full = arrs["emb_full"].astype(np.float32)

    print(
        f"Perch meta: {meta_full.shape}, scores: {scores_full_raw.shape}, emb: {emb_full.shape}"
    )
    print(f"Species: {N_CLASSES}, Labeled soundscape rows: {len(sc_labels)}")

    # Build Y_FULL ground-truth labels aligned to meta_full rows.
    # train_soundscapes_labels.csv: columns = filename, start, end, primary_label
    # start/end are "HH:MM:SS" strings; primary_label is ";"-separated taxon IDs
    def _parse_end_sec(time_str: str) -> int:
        """'00:00:05' → 5"""
        h, m, s = time_str.strip().split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)

    sc_label_map = {}
    for _, row in sc_labels.iterrows():
        fname = row["filename"]
        end_sec = _parse_end_sec(str(row["end"]))
        species_list = [
            s.strip()
            for s in str(row["primary_label"]).split(";")
            if s.strip() and s.strip() in label_to_idx
        ]
        key = (fname, end_sec)
        sc_label_map[key] = species_list

    # meta_full row_id format: "{stem}_{end_second}"
    Y_FULL = np.zeros((len(meta_full), N_CLASSES), dtype=np.float32)
    for i, row in meta_full.iterrows():
        fname = row["filename"]
        end_sec = int(row["row_id"].rsplit("_", 1)[1])
        key = (fname, end_sec)
        for sp in sc_label_map.get(key, []):
            Y_FULL[i, label_to_idx[sp]] = 1.0

    n_pos = Y_FULL.sum(axis=0)
    print(
        f"Y_FULL: {Y_FULL.shape}, positive labels: {n_pos.sum():.0f}, "
        f"classes with ≥1 pos: {(n_pos >= 1).sum()}"
    )

    # Fit prior tables on all labeled soundscape data
    tables = fit_prior_tables(meta_full, Y_FULL, class_names_ordered)
    oof_base, oof_prior = fuse_scores(
        scores_full_raw,
        meta_full["site"].to_numpy(),
        meta_full["hour_utc"].to_numpy(),
        tables,
    )

    # PCA on embeddings
    emb_scaler = StandardScaler()
    emb_scaled = emb_scaler.fit_transform(emb_full)
    n_comp = min(args.pca_dim, emb_scaled.shape[0] - 1, emb_scaled.shape[1])
    emb_pca = PCA(n_components=n_comp)
    Z_FULL = emb_pca.fit_transform(emb_scaled).astype(np.float32)
    print(
        f"PCA: {n_comp} components, {emb_pca.explained_variance_ratio_.sum():.4f} var explained"
    )

    # Class prototypes + family mapping
    CLASS_PROTOTYPES = {}
    for ci in range(N_CLASSES):
        pos_mask = Y_FULL[:, ci] == 1
        if pos_mask.sum() >= args.probe_min_pos:
            CLASS_PROTOTYPES[ci] = Z_FULL[pos_mask].mean(axis=0)

    FAMILY_GROUPS: dict = {}
    for ci, label in enumerate(PRIMARY_LABELS):
        fam = CLASS_NAME_MAP.get(label, "Unknown")
        FAMILY_GROUPS.setdefault(fam, []).append(ci)
    FAMILY_IDX_MAP = {
        f: np.array(idxs, dtype=np.int32) for f, idxs in FAMILY_GROUPS.items()
    }
    CLASS_FAMILY = {
        ci: CLASS_NAME_MAP.get(label, "Unknown")
        for ci, label in enumerate(PRIMARY_LABELS)
    }

    # Train full probe models (for baseline cmAP)
    pos_counts = Y_FULL.sum(axis=0)
    probe_idx = np.where(pos_counts >= args.probe_min_pos)[0].astype(np.int32)
    probe_models = {}
    for ci in probe_idx:
        y = Y_FULL[:, ci]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        proto_sim = (
            cosine_sim_to_prototype(Z_FULL, CLASS_PROTOTYPES[ci])
            if ci in CLASS_PROTOTYPES
            else None
        )
        fam = CLASS_FAMILY[ci]
        other_fam = FAMILY_IDX_MAP.get(fam, np.array([]))
        other_fam = other_fam[other_fam != ci]
        fam_mean = oof_base[:, other_fam].mean(axis=1) if len(other_fam) > 0 else None
        X = build_class_features(
            Z_FULL,
            scores_full_raw[:, ci],
            oof_prior[:, ci],
            oof_base[:, ci],
            proto_sim_col=proto_sim,
            family_mean_col=fam_mean,
        )
        clf = LogisticRegression(
            C=args.probe_c, max_iter=400, solver="liblinear", class_weight="balanced"
        )
        clf.fit(X, y)
        probe_models[ci] = clf

    # In-sample probe predictions (for reference — these will be overconfident)
    insample_scores = oof_base.copy()
    for ci, clf in probe_models.items():
        proto_sim = (
            cosine_sim_to_prototype(Z_FULL, CLASS_PROTOTYPES[ci])
            if ci in CLASS_PROTOTYPES
            else None
        )
        fam = CLASS_FAMILY[ci]
        other_fam = FAMILY_IDX_MAP.get(fam, np.array([]))
        other_fam = other_fam[other_fam != ci]
        fam_mean = oof_base[:, other_fam].mean(axis=1) if len(other_fam) > 0 else None
        X = build_class_features(
            Z_FULL,
            scores_full_raw[:, ci],
            oof_prior[:, ci],
            oof_base[:, ci],
            proto_sim_col=proto_sim,
            family_mean_col=fam_mean,
        )
        pred = clf.decision_function(X).astype(np.float32)
        insample_scores[:, ci] = (1 - args.probe_alpha) * oof_base[
            :, ci
        ] + args.probe_alpha * pred

    # OOF probe predictions (GroupKFold by FILENAME, for calibration)
    # Group by file (not site) to get balanced folds: 59 files / 5 folds ≈ 12 files each
    print(f"\nRunning {args.n_splits}-fold OOF probe predictions (grouped by file)...")
    gkf = GroupKFold(n_splits=args.n_splits)
    groups = meta_full["filename"].to_numpy()
    oof_probe_scores = oof_base.copy()

    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(Z_FULL, groups=groups)):
        print(f"  Fold {fold_i}: train={len(tr_idx)}, val={len(va_idx)}")
        Y_tr = Y_FULL[tr_idx]
        pos_cnt = Y_tr.sum(axis=0)
        probe_idx_fold = np.where(pos_cnt >= args.probe_min_pos)[0]

        # Re-fit PCA and scaler on training fold
        scaler_f = StandardScaler()
        pca_f = PCA(n_components=n_comp)
        Z_tr = pca_f.fit_transform(scaler_f.fit_transform(emb_full[tr_idx])).astype(
            np.float32
        )
        Z_va = pca_f.transform(scaler_f.transform(emb_full[va_idx])).astype(np.float32)

        for ci in probe_idx_fold:
            y_tr = Y_FULL[tr_idx, ci]
            if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
                continue

            proto = Z_tr[y_tr == 1].mean(axis=0) if (y_tr == 1).sum() > 0 else None
            proto_sim_tr = (
                cosine_sim_to_prototype(Z_tr, proto) if proto is not None else None
            )
            proto_sim_va = (
                cosine_sim_to_prototype(Z_va, proto) if proto is not None else None
            )

            fam = CLASS_FAMILY[ci]
            other_fam = FAMILY_IDX_MAP.get(fam, np.array([]))
            other_fam = other_fam[other_fam != ci]
            fam_tr = (
                oof_base[tr_idx][:, other_fam].mean(axis=1)
                if len(other_fam) > 0
                else None
            )
            fam_va = (
                oof_base[va_idx][:, other_fam].mean(axis=1)
                if len(other_fam) > 0
                else None
            )

            X_tr = build_class_features(
                Z_tr,
                scores_full_raw[tr_idx, ci],
                oof_prior[tr_idx, ci],
                oof_base[tr_idx, ci],
                proto_sim_col=proto_sim_tr,
                family_mean_col=fam_tr,
            )
            X_va = build_class_features(
                Z_va,
                scores_full_raw[va_idx, ci],
                oof_prior[va_idx, ci],
                oof_base[va_idx, ci],
                proto_sim_col=proto_sim_va,
                family_mean_col=fam_va,
            )
            clf = LogisticRegression(
                C=args.probe_c,
                max_iter=400,
                solver="liblinear",
                class_weight="balanced",
            )
            clf.fit(X_tr, y_tr)
            raw_pred = clf.decision_function(X_va).astype(np.float32)
            oof_probe_scores[va_idx, ci] = (1 - args.probe_alpha) * oof_base[
                va_idx, ci
            ] + args.probe_alpha * raw_pred

    # Baseline OOF cmAP (no temperature)
    valid_cls = [c for c in range(N_CLASSES) if Y_FULL[:, c].sum() > 0]

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    aps_base = [
        average_precision_score(Y_FULL[:, c], sigmoid(oof_probe_scores[:, c]))
        for c in valid_cls
    ]
    cmap_base = float(np.mean(aps_base))
    print(f"\nOOF cmAP (no temperature, {len(valid_cls)} classes): {cmap_base:.4f}")

    # Per-class temperature calibration via coarse grid search
    # Only consider sharpening (T<=1.0) to avoid previous dead end where
    # unconstrained optimizer drove T->7.4 (extreme softening).
    # Grid: {0.5, 0.6, 0.7, 0.8, 0.9, 1.0} — pick T that maximises per-class AP.
    TEMP_GRID = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32)
    print(f"Grid-search per-class temperatures over {TEMP_GRID.tolist()}...")
    temps = np.ones(N_CLASSES, dtype=np.float32)
    n_calibrated = 0
    for ci in valid_cls:
        y_ci = Y_FULL[:, ci]
        scores_ci = oof_probe_scores[:, ci]
        best_ap, best_t = -1.0, 1.0
        for t in TEMP_GRID:
            probs = sigmoid(scores_ci / t)
            ap = average_precision_score(y_ci, probs)
            if ap > best_ap:
                best_ap, best_t = ap, float(t)
        temps[ci] = best_t
        if best_t < 1.0:
            n_calibrated += 1

    aps_cal = [
        average_precision_score(
            Y_FULL[:, c], sigmoid(oof_probe_scores[:, c] / temps[c])
        )
        for c in valid_cls
    ]
    cmap_cal = float(np.mean(aps_cal))
    print(
        f"OOF cmAP (with per-class T grid, {n_calibrated} classes sharpened): {cmap_cal:.4f}"
    )
    print(f"Delta: {cmap_cal - cmap_base:+.4f}")
    print(
        "T_c distribution: ",
        {float(t): int((temps[valid_cls] == t).sum()) for t in TEMP_GRID},
    )

    # Also check in-sample cmAP for reference
    aps_insample = [
        average_precision_score(Y_FULL[:, c], sigmoid(insample_scores[:, c]))
        for c in valid_cls
    ]
    print(f"In-sample cmAP (no OOF, reference): {np.mean(aps_insample):.4f}")

    # Save temperatures
    np.save(out_dir / "per_class_temps.npy", temps)
    temps_dict = {PRIMARY_LABELS[i]: float(temps[i]) for i in range(N_CLASSES)}
    with open(out_dir / "per_class_temps.json", "w") as f:
        json.dump(temps_dict, f, indent=2)
    print(f"\nSaved: {out_dir}/per_class_temps.npy  ({temps.shape})")
    print(f"Saved: {out_dir}/per_class_temps.json")

    # Show top-10 most/least temperature-adjusted classes
    deltas = [aps_cal[i] - aps_base[i] for i in range(len(valid_cls))]
    top_gain = sorted(zip(deltas, valid_cls), reverse=True)[:10]
    print("\nTop-10 classes by AP gain from calibration:")
    for delta, ci in top_gain:
        print(
            f"  {PRIMARY_LABELS[ci]:20s}  T={temps[ci]:.3f}  AP: {aps_base[valid_cls.index(ci)]:.3f} → {aps_cal[valid_cls.index(ci)]:.3f}  ({delta:+.3f})"
        )


if __name__ == "__main__":
    main()
