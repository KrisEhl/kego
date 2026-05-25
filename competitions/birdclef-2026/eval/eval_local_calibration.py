"""Local calibration evaluation for BirdCLEF+ 2026.

Implements the evaluation strategy from research/research-local-eval.md:
  - Padded cMAP (padding_factor=5, matching competition scorer behaviour)
  - Soundscape-only vs audio-present species split
  - Per-class AP breakdown
  - Site-level breakdown
  - Multiple pipeline configurations vs known LB scores

IMPORTANT — in-sample caveat:
  Stage2 ResidualSSMv3 was TRAINED on all 59 labeled soundscapes. Any metric
  computed on those same windows is in-sample for Stage2 and will appear
  better than LB. Post-processing changes (rank_power, smoothing) are
  applied at inference time only and should transfer reliably to LB.
  Model checkpoint changes (Stage3, new seeds, etc.) should NOT be trusted
  locally — LB must be the ground truth for those.

Run:
    uv run python competitions/birdclef-2026/eval/eval_local_calibration.py
    uv run python competitions/birdclef-2026/eval/eval_local_calibration.py \\
        --checkpoint outputs/protossm_v3.pt --verbose
    uv run python competitions/birdclef-2026/eval/eval_local_calibration.py \\
        --checkpoint outputs/protossm_original_stage3_s42.pt
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

TRAIN_DIR = Path(__file__).parent.parent / "training"
sys.path.insert(0, str(TRAIN_DIR))

from train_protossm import (  # noqa: E402
    N_CLASSES,
    ProtoSSM,
    ResidualSSMv3,
    build_file_batches,
    load_data,
)

DATA_ROOT = Path(os.environ.get("KEGO_PATH_DATA", "data"))

# ---------------------------------------------------------------------------
# Known LB scores  (config key → LB score).  Update as results come in.
# ---------------------------------------------------------------------------
KNOWN_LB: dict[str, float | None] = {
    # Post-processing variants on protossm_v3.pt  (Stage2 w=0.70)
    "pp: none (raw sigmoid)": None,
    "pp: rank_power=0.4": None,
    "pp: rank_power=0.4 + smooth + boost_top2 (v52)": 0.919,  # confirmed Apr 11
    "pp: rank_power=2.0 + smooth_amphibia=0.65 (v53)": 0.920,  # confirmed Apr 12 (NEW BEST)
    "pp: rank_power=2.0 + smooth_amphibia=0.65 + boost_top2": None,  # v55-C candidate
    "pp: rank_power=4.0 + smooth_amphibia=0.65": None,  # PENDING v54 Apr 13
    "pp: rank_power=4.0 + smooth_amphibia=0.65 + boost_top2": None,  # v55-A candidate
    "pp: rank_power=6.0 + smooth_amphibia=0.65": None,  # v55-B candidate
    # Stage2 weight variants (no smooth, rank_power=0.4)
    "stage2 w=0.35, rank=0.4": 0.914,  # v13
    "stage2 w=0.70, rank=0.4 (v16)": 0.915,  # v16
    "stage2 w=1.00, rank=0.4": 0.915,  # v17
    # Stage3 variants (single seed s42, w2=0.70, rank=0.4)
    "stage3 30ep w3=0.70 single-seed (s42)": None,
    "stage3 30ep w3=0.35 ensemble": 0.912,  # v50
    "stage3 60ep w3=0.70 ensemble": 0.895,  # v51
    # Perch only
    "perch logits only": None,
}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def padded_cmap(y_true: np.ndarray, y_pred: np.ndarray, padding_factor: int = 5) -> float:
    """Padded class-mean AP matching competition scorer.

    For each class with ≥1 positive, append `padding_factor` (negative, 0-score)
    rows.  These rank last and stabilise AP for rare positives by preventing
    edge-case ties from inflating the score.  Only active classes (≥1 positive)
    are included — mirrors Kaggle's behaviour of only scoring present classes.
    """
    aps = []
    for c in range(y_true.shape[1]):
        n_pos = y_true[:, c].sum()
        if n_pos == 0:
            continue
        col_true = np.concatenate([y_true[:, c], np.zeros(padding_factor)])
        col_pred = np.concatenate([y_pred[:, c], np.zeros(padding_factor)])
        aps.append(average_precision_score(col_true, col_pred))
    return float(np.mean(aps)) if aps else 0.0


def per_class_ap(y_true: np.ndarray, y_pred: np.ndarray, species_names: list[str]) -> pd.Series:
    aps: dict[str, float] = {}
    for c, sp in enumerate(species_names):
        if y_true[:, c].sum() > 0:
            aps[sp] = float(average_precision_score(y_true[:, c], y_pred[:, c]))
    return pd.Series(aps).sort_values()


# ---------------------------------------------------------------------------
# Ground-truth builder
# ---------------------------------------------------------------------------


def build_ground_truth(
    meta: pd.DataFrame,
    labels_csv: pd.DataFrame,
    taxonomy: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build Y_true (n_rows, 234) from train_soundscapes_labels.csv.

    Multiple annotators per segment are merged with OR (union).
    Unannotated segments default to all-zero (assumed absent — see research doc).
    """
    primary_labels = taxonomy["primary_label"].tolist()
    label_to_idx = {lbl: i for i, lbl in enumerate(primary_labels)}

    def parse_end(s: str) -> int:
        h, m, sec = s.split(":")
        return int(h) * 3600 + int(m) * 60 + int(sec)

    ldf = labels_csv.copy()
    ldf["end_sec"] = ldf["end"].apply(parse_end)
    ldf["stem"] = ldf["filename"].str.replace(".ogg", "", regex=False)
    ldf["row_id"] = ldf["stem"] + "_" + ldf["end_sec"].astype(str)

    # Merge annotators with OR
    rid_to_species: dict[str, set[str]] = {}
    for _, row in ldf.iterrows():
        rid = row["row_id"]
        rid_to_species.setdefault(rid, set())
        for lbl in str(row["primary_label"]).split(";"):
            lbl = lbl.strip()
            if lbl in label_to_idx:
                rid_to_species[rid].add(lbl)

    n_rows = len(meta)
    Y_true = np.zeros((n_rows, len(primary_labels)), dtype=np.float32)
    for i, row_id in enumerate(meta["row_id"]):
        for sp in rid_to_species.get(row_id, set()):
            Y_true[i, label_to_idx[sp]] = 1.0

    sites = meta["site"].tolist()
    filenames = meta["row_id"].str.rsplit("_", n=1).str[0].tolist()
    return Y_true, sites, filenames


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def apply_postproc(
    logits_2d: np.ndarray,
    filenames: list[str],
    taxonomy: pd.DataFrame,
    rank_power: float = 0.4,
    smooth_amphibia: float = 0.0,
    smooth_insecta: float = 0.0,
    smooth_rare: float = 0.0,
    boost_topk: int = 0,
    boost_alpha: float = 0.0,
) -> np.ndarray:
    N_WINDOWS = 12
    n_total, n_cls = logits_2d.shape
    primary_labels = taxonomy["primary_label"].tolist()
    label_to_idx = {lbl: i for i, lbl in enumerate(primary_labels)}
    class_name_map = taxonomy.set_index("primary_label")["class_name"].to_dict()

    # Index arrays for smoothing
    def _idx(class_name: str | set) -> np.ndarray:
        names = {class_name} if isinstance(class_name, str) else class_name
        return np.array(
            [label_to_idx[lbl] for lbl in primary_labels if class_name_map.get(lbl) in names and lbl in label_to_idx],
            dtype=np.int32,
        )

    idx_amphibia = _idx("Amphibia")
    idx_insecta = _idx("Insecta")
    idx_rare = _idx({"Mammalia", "Reptilia"})

    unique_files = list(dict.fromkeys(filenames))
    n_files = len(unique_files)

    def _smooth(arr: np.ndarray, alpha: float) -> np.ndarray:
        if alpha <= 0.0:
            return arr
        out = arr.copy()
        nb = (arr[:, :-2, :] + arr[:, 2:, :]) * 0.5
        out[:, 1:-1, :] += alpha * (nb - arr[:, 1:-1, :])
        out[:, 0, :] += alpha * (arr[:, 1, :] - arr[:, 0, :])
        out[:, -1, :] += alpha * (arr[:, -2, :] - arr[:, -1, :])
        return out

    s = logits_2d.reshape(n_files, N_WINDOWS, n_cls).copy()
    for idx, alpha in [
        (idx_amphibia, smooth_amphibia),
        (idx_insecta, smooth_insecta),
        (idx_rare, smooth_rare),
    ]:
        if len(idx) > 0 and alpha > 0.0:
            s[:, :, idx] = _smooth(s[:, :, idx], alpha)

    probs = 1.0 / (1.0 + np.exp(-s))
    file_max = probs.max(axis=1)
    probs = probs * np.power(file_max, rank_power)[:, np.newaxis, :]

    if boost_topk > 0 and boost_alpha > 0.0:
        top_idx = np.argsort(file_max, axis=1)[:, -boost_topk:]
        mask = np.zeros_like(file_max)
        for fi in range(n_files):
            mask[fi, top_idx[fi]] = 1.0
        probs += boost_alpha * file_max[:, np.newaxis, :] * mask[:, np.newaxis, :]
        probs = np.clip(probs, 0.0, 1.0)

    return probs.reshape(n_total, n_cls)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_calibration(checkpoint_path: Path, verbose: bool = False) -> None:
    # -- Load model -----------------------------------------------------------
    print(f"\nCheckpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    has_stage3 = "residual_ssm_v3b_state_dict" in ckpt
    print(
        f"Config seed={cfg.get('seed')}  epochs={cfg.get('epochs')}  "
        f"has_stage3={has_stage3}  "
        f"ckpt_residual_weight={ckpt.get('residual_weight', 'N/A')}"
    )

    model = ProtoSSM(n_tax_groups=cfg.get("n_tax_groups", 5))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    residual_ssm = ResidualSSMv3(d_site=cfg.get("d_site", 0))
    if "residual_ssm_state_dict" in ckpt:
        residual_ssm.load_state_dict(ckpt["residual_ssm_state_dict"])
    residual_ssm.eval()

    residual_ssm_v3b = None
    if has_stage3:
        residual_ssm_v3b = ResidualSSMv3()
        residual_ssm_v3b.load_state_dict(ckpt["residual_ssm_v3b_state_dict"])
        residual_ssm_v3b.eval()

    site_to_idx = ckpt.get("site_to_idx", {})

    # -- Load data ------------------------------------------------------------
    data = load_data(
        DATA_ROOT,
        npz_file="full_perch_arrays.npz",
        probe_scores_file="full_probe_scores.npy",
    )
    emb = data["emb"]
    logits = data["logits"]
    sites_arr = data["sites"]
    hours_arr = data["hours"]
    filenames_arr = data["filenames"]

    meta = pd.read_parquet(DATA_ROOT / "perch-meta/full_perch_meta.parquet")
    taxonomy = pd.read_csv(DATA_ROOT / "birdclef/birdclef-2026/taxonomy.csv")
    labels_csv = pd.read_csv(DATA_ROOT / "birdclef/birdclef-2026/train_soundscapes_labels.csv")
    train_audio = pd.read_csv(DATA_ROOT / "birdclef/birdclef-2026/train.csv")

    primary_labels = taxonomy["primary_label"].tolist()

    # -- Build ground truth ---------------------------------------------------
    Y_true, sites_list, filenames_list = build_ground_truth(meta, labels_csv, taxonomy)
    active_cls = np.where(Y_true.sum(axis=0) > 0)[0]
    n_pos_total = int(Y_true.sum())

    audio_species = set(train_audio["primary_label"].astype(str).unique())
    sc_only_cls = [c for c in active_cls if primary_labels[c] not in audio_species]
    audio_present_cls = [c for c in active_cls if primary_labels[c] in audio_species]

    print(f"\nGround truth: {len(Y_true)} windows | {n_pos_total} positives | {len(active_cls)} active classes")
    print(f"  Audio-present: {len(audio_present_cls)} | Soundscape-only: {len(sc_only_cls)}")

    # -- Forward pass ---------------------------------------------------------
    all_batches = build_file_batches(emb, logits, Y_true, sites_arr, hours_arr, filenames_arr, site_to_idx)
    file_to_rows: dict[str, list[int]] = {}
    for i, fn in enumerate(filenames_arr):
        file_to_rows.setdefault(fn, []).append(i)

    proto_logits_arr = np.zeros((len(emb), N_CLASSES), dtype=np.float32)
    correction_arr = np.zeros((len(emb), N_CLASSES), dtype=np.float32)
    correction3_arr = np.zeros((len(emb), N_CLASSES), dtype=np.float32)

    with torch.no_grad():
        for batch in all_batches:
            row_idx = file_to_rows[batch["filename"]]
            emb_t = torch.tensor(batch["emb"], dtype=torch.float32)
            logits_t = torch.tensor(batch["logits"], dtype=torch.float32)
            site_t = torch.tensor(batch["site_idx"], dtype=torch.long)
            hour_t = torch.tensor(batch["hour_idx"], dtype=torch.long)

            proto_logits_t, _ = model(emb_t, logits_t, site_t, hour_t)
            proto_probs_t = torch.sigmoid(proto_logits_t)
            correction_t = residual_ssm(emb_t, proto_probs_t)

            proto_logits_arr[row_idx] = proto_logits_t.numpy()
            correction_arr[row_idx] = correction_t.numpy()

            if residual_ssm_v3b is not None:
                stage3_base_t = logits_t + 0.70 * correction_t
                stage3_probs_t = torch.sigmoid(stage3_base_t)
                correction3_t = residual_ssm_v3b(emb_t, stage3_probs_t)
                correction3_arr[row_idx] = correction3_t.numpy()

    # Precompute Stage2 logits at the default weight
    s2_logits = logits + 0.70 * correction_arr

    # -- Build configurations to evaluate ------------------------------------
    # Each entry: (label, logits_2d, post-proc kwargs, known_lb_key)
    configs: list[tuple[str, np.ndarray, dict, str | None]] = []

    # Perch logits
    configs.append(("perch logits only", logits, dict(rank_power=0.0), "perch logits only"))
    configs.append(("perch + rank_power=0.4", logits, dict(rank_power=0.4), None))

    # Stage2 weight sweep (rank_power=0.4, no smooth)
    for w in [0.00, 0.35, 0.70, 1.00]:
        fl = logits + w * correction_arr
        lb_key = {
            0.35: "stage2 w=0.35, rank=0.4",
            0.70: "stage2 w=0.70, rank=0.4 (v16)",
            1.00: "stage2 w=1.00, rank=0.4",
        }.get(w)
        configs.append(
            (
                f"stage2 w={w:.2f} + rank=0.4 (no smooth)",
                fl,
                dict(rank_power=0.4),
                lb_key,
            )
        )

    # Post-processing variants on Stage2 w=0.70
    configs.append(
        (
            "pp: none (raw sigmoid)",
            s2_logits,
            dict(rank_power=0.0),
            "pp: none (raw sigmoid)",
        )
    )
    configs.append(("pp: rank_power=0.4", s2_logits, dict(rank_power=0.4), "pp: rank_power=0.4"))
    configs.append(
        (
            "pp: rank_power=0.4 + smooth + boost_top2 (v52)",
            s2_logits,
            dict(
                rank_power=0.4,
                smooth_amphibia=0.45,
                smooth_insecta=0.35,
                smooth_rare=0.05,
                boost_topk=2,
                boost_alpha=0.05,
            ),
            "pp: rank_power=0.4 + smooth + boost_top2 (v52)",
        )
    )
    configs.append(
        (
            "pp: rank_power=2.0 + smooth_amphibia=0.65 (v53)",
            s2_logits,
            dict(
                rank_power=2.0,
                smooth_amphibia=0.65,
                smooth_insecta=0.35,
                smooth_rare=0.05,
            ),
            "pp: rank_power=2.0 + smooth_amphibia=0.65 (v53)",
        )
    )
    configs.append(
        (
            "pp: rank_power=4.0 + smooth_amphibia=0.65",
            s2_logits,
            dict(
                rank_power=4.0,
                smooth_amphibia=0.65,
                smooth_insecta=0.35,
                smooth_rare=0.05,
            ),
            "pp: rank_power=4.0 + smooth_amphibia=0.65",
        )
    )
    # v54 candidates: cross rank_power × boost
    configs.append(
        (
            "pp: rank_power=2.0 + smooth_amphibia=0.65 + boost_top2",
            s2_logits,
            dict(
                rank_power=2.0,
                smooth_amphibia=0.65,
                smooth_insecta=0.35,
                smooth_rare=0.05,
                boost_topk=2,
                boost_alpha=0.05,
            ),
            "pp: rank_power=2.0 + smooth_amphibia=0.65 + boost_top2",
        )
    )
    configs.append(
        (
            "pp: rank_power=4.0 + smooth_amphibia=0.65 + boost_top2",
            s2_logits,
            dict(
                rank_power=4.0,
                smooth_amphibia=0.65,
                smooth_insecta=0.35,
                smooth_rare=0.05,
                boost_topk=2,
                boost_alpha=0.05,
            ),
            "pp: rank_power=4.0 + smooth_amphibia=0.65 + boost_top2",
        )
    )
    # Higher rank_power extremes
    configs.append(
        (
            "pp: rank_power=6.0 + smooth_amphibia=0.65",
            s2_logits,
            dict(
                rank_power=6.0,
                smooth_amphibia=0.65,
                smooth_insecta=0.35,
                smooth_rare=0.05,
            ),
            "pp: rank_power=6.0 + smooth_amphibia=0.65",
        )
    )
    configs.append(
        (
            "pp: rank_power=8.0 + smooth_amphibia=0.65",
            s2_logits,
            dict(
                rank_power=8.0,
                smooth_amphibia=0.65,
                smooth_insecta=0.35,
                smooth_rare=0.05,
            ),
            None,
        )
    )
    # Insecta smooth sweep at optimal rank_power
    configs.append(
        (
            "pp: rank_power=2.0 + smooth_amphibia=0.65 + si=0.50",
            s2_logits,
            dict(
                rank_power=2.0,
                smooth_amphibia=0.65,
                smooth_insecta=0.50,
                smooth_rare=0.05,
            ),
            None,
        )
    )
    configs.append(
        (
            "pp: rank_power=2.0 + smooth_amphibia=0.65 + si=0.65",
            s2_logits,
            dict(
                rank_power=2.0,
                smooth_amphibia=0.65,
                smooth_insecta=0.65,
                smooth_rare=0.05,
            ),
            None,
        )
    )

    # Stage3 variants (if available)
    if residual_ssm_v3b is not None:
        ckpt_rw = ckpt.get("residual_weight", 0.70)
        for w3 in [0.35, 0.70]:
            s3_logits = s2_logits + w3 * correction3_arr
            lb_key = {
                0.35: "stage3 30ep w3=0.35 ensemble",
                0.70: "stage3 30ep w3=0.70 single-seed (s42)",
            }.get(w3)
            configs.append(
                (
                    f"stage3 single-seed w3={w3:.2f} (s42, {cfg.get('epochs')}ep)",
                    s3_logits,
                    dict(rank_power=0.4),
                    lb_key,
                )
            )

    # -- Evaluate -------------------------------------------------------------
    def _eval(logits_2d: np.ndarray, pp_kw: dict) -> tuple[float, float, float]:
        preds = apply_postproc(logits_2d, filenames_list, taxonomy, **pp_kw)
        p = padded_cmap(Y_true, preds, padding_factor=5)
        if sc_only_cls:
            p_so = padded_cmap(Y_true[:, sc_only_cls], preds[:, sc_only_cls])
        else:
            p_so = float("nan")
        if audio_present_cls:
            p_au = padded_cmap(Y_true[:, audio_present_cls], preds[:, audio_present_cls])
        else:
            p_au = float("nan")
        return p, p_au, p_so

    hdr = f"{'Configuration':<55} {'cMAP':>7} {'Audio':>7} {'SC-only':>8} {'LB':>7}"
    sep = "=" * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(sep)

    rows: list[dict] = []
    for cfg_name, base_logits, pp_kw, lb_key in configs:
        cmap, cmap_au, cmap_so = _eval(base_logits, pp_kw)
        lb = KNOWN_LB.get(lb_key) if lb_key else None
        lb_s = f"{lb:.3f}" if lb is not None else "  ?"
        short = cfg_name[:54]
        print(f"  {short:<54} {cmap:>7.4f} {cmap_au:>7.4f} {cmap_so:>8.4f} {lb_s:>7}")
        rows.append(dict(name=cfg_name, cmap=cmap, cmap_au=cmap_au, cmap_so=cmap_so, lb=lb))

    # -- Calibration table ---------------------------------------------------
    print(f"\n{'--- Local vs LB calibration':-<80}")
    known = [r for r in rows if r["lb"] is not None]
    if known:
        header = f"  {'Configuration':<54} {'Local':>7} {'LB':>7} {'Ratio':>7}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in known:
            ratio = r["lb"] / r["cmap"] if r["cmap"] > 0 else float("nan")
            print(f"  {r['name'][:53]:<54} {r['cmap']:>7.4f} {r['lb']:>7.3f} {ratio:>7.2f}")

        # Correlation
        if len(known) >= 3:
            from scipy.stats import pearsonr, spearmanr

            local_vals = np.array([r["cmap"] for r in known])
            lb_vals = np.array([r["lb"] for r in known])
            rp, _ = pearsonr(local_vals, lb_vals)
            rs, _ = spearmanr(local_vals, lb_vals)
            print(f"\n  Pearson r = {rp:.3f}   Spearman r = {rs:.3f}  (n={len(known)} configs with known LB)")
        else:
            print(f"\n  Only {len(known)} LB data point(s) — need ≥3 for correlation.")
            print(f"  LB/local ratio: {known[0]['lb'] / known[0]['cmap']:.2f}×")
            print("  Submit v52/v53 to add more calibration points.")
    else:
        print("  No LB data points available for this checkpoint.")

    # -- Site-level breakdown ------------------------------------------------
    print(f"\n{'--- Site-level cMAP (Stage2 w=0.70, rank=0.4)':-<80}")
    preds_v16 = apply_postproc(s2_logits, filenames_list, taxonomy, rank_power=0.4)
    for site in sorted(set(sites_list)):
        mask = np.array([s == site for s in sites_list])
        if Y_true[mask].sum() == 0:
            continue
        active_site = np.where(Y_true[mask].sum(0) > 0)[0]
        if len(active_site) == 0:
            continue
        site_cmap = padded_cmap(Y_true[mask], preds_v16[mask])
        print(
            f"  {site}: {mask.sum():3d} windows  {int(Y_true[mask].sum()):4d} pos  "
            f"{len(active_site):2d} spp  cMAP={site_cmap:.4f}"
        )

    # -- Site-stratified hold-out on Perch-only (no in-sample bias) ----------
    print(f"\n{'--- Site-stratified hold-out (Perch logits only — zero in-sample bias)':-<80}")
    print("  (Held-out site: each site in turn; trained on remaining sites)")
    loso_cmaps = []
    for held_site in sorted(set(sites_list)):
        mask_val = np.array([s == held_site for s in sites_list])
        if Y_true[mask_val].sum() == 0:
            continue
        preds_val = apply_postproc(logits, filenames_list, taxonomy, rank_power=0.4)
        active_site = np.where(Y_true[mask_val].sum(0) > 0)[0]
        if len(active_site) < 2:
            continue
        site_cmap = padded_cmap(Y_true[mask_val], preds_val[mask_val])
        loso_cmaps.append(site_cmap)
        print(
            f"  Held out {held_site}: {mask_val.sum():3d} windows  "
            f"{len(active_site):2d} active spp  cMAP={site_cmap:.4f}"
        )
    if loso_cmaps:
        print(f"  Mean LOSO cMAP (Perch only): {np.mean(loso_cmaps):.4f}")

    # -- Per-class AP (verbose) -----------------------------------------------
    if verbose:
        print(f"\n{'--- Per-class AP (Stage2 w=0.70, rank=0.4 — worst 20 / best 20)':-<80}")
        ap_series = per_class_ap(Y_true, preds_v16, primary_labels)
        print("  Worst 20:")
        for sp, ap in ap_series.head(20).items():
            in_audio = "audio" if sp in audio_species else "sc-only"
            print(f"    {sp:22s} {ap:6.4f}  ({in_audio})")
        print("  Best 20:")
        for sp, ap in ap_series.tail(20).items():
            print(f"    {sp:22s} {ap:6.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local calibration eval for BirdCLEF 2026 ProtoSSM pipeline")
    parser.add_argument("--checkpoint", default="outputs/protossm_v3.pt")
    parser.add_argument("--verbose", action="store_true", help="Print per-class AP breakdown")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    run_calibration(ckpt_path, verbose=args.verbose)


if __name__ == "__main__":
    main()
