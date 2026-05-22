"""Leave-One-Site-Out (LOSO) cross-validation for ProtoSSM Stage2.

Evaluates Stage2 (ResidualSSMv3) quality with held-out site evaluation.
Stage1 is loaded from a fixed checkpoint (trained on all 59 soundscapes).
For each of the 9 labeled sites, Stage2 is trained on 8 sites and evaluated
on the held-out site. This avoids the in-sample bias of standard evaluation.

Why this is useful:
- Standard eval: Stage2 trained on all 59 sc, evaluated on same 59 sc → inflated
- LOSO: Stage2 trained on 8-site subset, evaluated on unseen site → realistic

Primary use case: ranking Stage2 hyperparameter changes (d_model, hour encoding,
L2 penalty) and Stage1 seed quality without submitting to Kaggle.

Note: Stage1 is still in-sample (trained on all 59 soundscapes), so proto_probs
for the held-out site are in-sample for Stage1 but OOF for Stage2. This is a
better signal than pure in-sample eval for Stage2 quality comparisons.

Usage:
    # Evaluate existing checkpoint with LOSO Stage2:
    uv run python competitions/birdclef-2026/eval/eval_loso.py \\
        --checkpoint outputs/protossm_v3.pt \\
        --stage2-epochs 30 \\
        --residual-weight 0.35

    # Evaluate a Stage1 seed checkpoint:
    uv run python competitions/birdclef-2026/eval/eval_loso.py \\
        --checkpoint outputs/stage1_seed_s209.pt \\
        --stage2-epochs 30

    # Multiple seeds for comparison:
    for seed in 200 201 209 218; do
        uv run python competitions/birdclef-2026/eval/eval_loso.py \\
            --checkpoint outputs/stage1_seed_s${seed}.pt \\
            --stage2-epochs 30 --seed $seed
    done
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

TRAIN_DIR = Path(__file__).parent.parent / "training"
sys.path.insert(0, str(TRAIN_DIR))

from train_protossm import (  # noqa: E402
    D_RESIDUAL,
    DROPOUT_RESIDUAL,
    N_CLASSES,
    ProtoSSM,
    ResidualSSMv3,
    build_file_batches,
    load_data,
    predict_batches_logits,
    train_residual_ssm_v3,
)

DATA_ROOT = Path(os.environ.get("KEGO_PATH_DATA", "data"))

PADDING_FACTOR = 5  # cmAP padding factor (matches Kaggle scoring)


def compute_cmap_padded(
    preds: np.ndarray,
    labels: np.ndarray,
    active_cls: np.ndarray,
    padding_factor: int = PADDING_FACTOR,
) -> float:
    """Compute class-mean AP with padding (matches Kaggle cmAP metric).

    For each active class, pads the positive set to
    len(positives) * padding_factor if too few positives, then
    computes AP. This matches the competition metric behavior.
    """
    aps = []
    for c in active_cls:
        y_true = labels[:, c]
        y_score = preds[:, c]
        n_pos = int(y_true.sum())
        if n_pos == 0:
            continue
        # Padding: add padding_factor * n_pos zero-score negatives
        # (to stabilize AP for rare classes with few positives)
        n_pad = padding_factor * n_pos
        y_true_pad = np.concatenate([y_true, np.zeros(n_pad)])
        y_score_pad = np.concatenate([y_score, np.full(n_pad, -1e9)])
        ap = average_precision_score(y_true_pad, y_score_pad)
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def run_loso(
    checkpoint_path: Path,
    stage2_epochs: int,
    residual_weight: float,
    seed: int,
    npz_file: str,
    probe_scores_file: str = "full_probe_scores__59sc.npy",
    stage2_d_model: int = D_RESIDUAL,
    stage2_d_hour: int = 0,
    stage2_l2: float = 0.0,
    diel_priors_file: str = "none",
    diel_alpha: float = 0.0,
    verbose: bool = True,
) -> dict:
    """Run LOSO evaluation. Returns per-fold results dict.

    probe_scores_file: filename (relative to DATA_ROOT/perch-meta/) for Stage2
    base logits. Use "full_probe_scores__59sc.npy" (default) to match the real
    inference pipeline (probe-augmented logits, ~0.926 in-sample cmAP).
    Use "none" to fall back to raw Perch logits (original LOSO behavior, ~0.545).

    CRITICAL: Use probe_scores to get reliable LOSO ranking — Perch-logit LOSO
    showed seed 218 > seed 42, but LB showed seed 218 < seed 42 (0.914 vs 0.920).
    Probe-score LOSO matches the actual inference Stage2 base.

    diel_priors_file: filename for diel activity priors (234×24) in DATA_ROOT/perch-meta/.
    Applied as additive log-odds correction: final_logits += diel_alpha * (logit(prior[c,h]) - logit(global[c])).
    diel_alpha=0.0 disables (default).
    """

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    data = load_data(DATA_ROOT, npz_file=npz_file, probe_scores_file="none.npy")
    emb = data["emb"]  # (708, 1536)
    logits = data["logits"]  # (708, 234) raw Perch logits
    labels = data["labels"]  # (708, 234)
    sites = data["sites"]  # (708,) str
    hours = data["hours"]
    filenames = data["filenames"]

    all_sites = sorted(set(sites.tolist()))
    site_to_idx = {s: i + 1 for i, s in enumerate(all_sites)}
    print(f"Sites: {all_sites}")

    # file → row indices
    file_to_rows: dict[str, list[int]] = {}
    for i, fn in enumerate(filenames):
        file_to_rows.setdefault(fn, []).append(i)

    all_batches = build_file_batches(
        emb, logits, labels, sites, hours, filenames, site_to_idx
    )
    # Site of each batch
    batch_sites = [sites[file_to_rows[b["filename"]][0]] for b in all_batches]
    print(f"Total batches: {len(all_batches)}")

    # -------------------------------------------------------------------------
    # Load Stage1 from checkpoint
    # -------------------------------------------------------------------------
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    n_tax_groups = cfg.get("n_tax_groups", 5)

    model = ProtoSSM(n_tax_groups=n_tax_groups)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"Stage1 loaded from {checkpoint_path}")

    # Stage1 in-sample proto_probs (used as Stage2 input, all 708 windows)
    proto_logits_all = predict_batches_logits(model, all_batches)  # (708, 234)
    proto_probs_all = 1.0 / (1.0 + np.exp(-proto_logits_all))

    # Stage2 base: use probe-augmented logits matching real inference pipeline.
    # Probe scores (XC probes + site/hour priors) are seed-independent and match
    # the Stage2 training base used in production. Fall back to Perch logits only
    # if probe_scores_file is "none".
    probe_scores_path = DATA_ROOT / "perch-meta" / probe_scores_file
    if probe_scores_file != "none" and probe_scores_path.exists():
        stage2_base = np.load(probe_scores_path).astype(np.float32)
        print(
            f"Stage2 base: probe scores from {probe_scores_file} (shape {stage2_base.shape})"
        )
    else:
        stage2_base = logits  # (708, 234) — fallback: raw Perch logits
        print(
            "Stage2 base: raw Perch logits (probe scores not found or --probe-scores none)"
        )

    # Diel activity priors (optional): log-odds correction per (species, hour)
    diel_bias = None  # (234, 24) — additive logit correction, or None
    if diel_alpha > 0.0 and diel_priors_file != "none":
        dp_path = DATA_ROOT / "perch-meta" / diel_priors_file
        if dp_path.exists():
            diel_priors = np.load(dp_path).astype(np.float32)  # (234, 24)
            global_mean = diel_priors.mean(axis=1, keepdims=True)  # (234, 1)

            # logit(p) = log(p / (1-p)), clipped to avoid inf
            def _logit(p: np.ndarray) -> np.ndarray:
                p = np.clip(p, 1e-6, 1 - 1e-6)
                return np.log(p / (1.0 - p))

            diel_bias = _logit(diel_priors) - _logit(global_mean)  # (234, 24)
            print(f"Diel priors loaded: {diel_priors.shape}  alpha={diel_alpha}")
            print(f"  diel_bias range [{diel_bias.min():.3f}, {diel_bias.max():.3f}]")
        else:
            print(
                f"WARNING: diel priors file not found: {dp_path}  (skipping diel correction)"
            )

    # -------------------------------------------------------------------------
    # LOSO loop
    # -------------------------------------------------------------------------
    fold_results = []

    for hold_site in all_sites:
        train_batches = [b for b, s in zip(all_batches, batch_sites) if s != hold_site]
        val_batches = [b for b, s in zip(all_batches, batch_sites) if s == hold_site]

        if not val_batches:
            continue

        # Gather val row indices
        val_row_idx = []
        for b in val_batches:
            val_row_idx.extend(file_to_rows[b["filename"]])
        val_row_idx = np.array(val_row_idx)

        val_labels = labels[val_row_idx]
        active_cls = np.where(val_labels.sum(0) > 0)[0]

        if len(active_cls) == 0:
            print(f"  Hold {hold_site}: 0 active classes — skipping")
            continue

        if verbose:
            print(
                f"\n  Hold {hold_site}: train={len(train_batches)} sc, "
                f"val={len(val_batches)} sc, active_cls={len(active_cls)}"
            )

        # Train Stage2 on training sites only
        torch.manual_seed(seed)
        np.random.seed(seed)
        residual_ssm = ResidualSSMv3(
            dropout=DROPOUT_RESIDUAL,
            d_model=stage2_d_model,
            d_hour=stage2_d_hour,
        )
        residual_ssm = train_residual_ssm_v3(
            residual_ssm=residual_ssm,
            emb=emb,
            proto_logits=stage2_base,
            proto_probs=proto_probs_all,
            labels=labels,
            all_batches=train_batches,
            file_to_rows=file_to_rows,
            epochs=stage2_epochs,
            val_batches=None,  # no ES — fixed epochs like submit mode
            correction_l2=stage2_l2,
            verbose=False,
        )
        residual_ssm.eval()

        # Eval on held-out site
        with torch.no_grad():
            correction_val = np.zeros((len(val_row_idx), N_CLASSES), dtype=np.float32)
            for b in val_batches:
                rows = file_to_rows[b["filename"]]
                emb_t = torch.tensor(emb[rows], dtype=torch.float32)
                pp_t = torch.tensor(proto_probs_all[rows], dtype=torch.float32)
                corr = residual_ssm(emb_t, pp_t, hour_idx=b.get("hour_idx")).numpy()
                # map rows back to val_row_idx positions
                for ri, r in enumerate(rows):
                    pos = np.where(val_row_idx == r)[0]
                    if len(pos) > 0:
                        correction_val[pos[0]] = corr[ri]

        final_logits_val = stage2_base[val_row_idx] + residual_weight * correction_val

        # Diel prior correction: additive log-odds adjustment per (window, species)
        if diel_bias is not None:
            val_hours = hours[val_row_idx].astype(int)  # (n_val,)
            diel_corr_val = diel_bias[:, val_hours].T  # (n_val, 234)
            final_logits_val = final_logits_val + diel_alpha * diel_corr_val

        final_probs_val = 1.0 / (1.0 + np.exp(-final_logits_val))

        # cmAP with padding
        fold_cmap = compute_cmap_padded(final_probs_val, val_labels, active_cls)

        # Baseline: Perch logits only (no Stage2)
        perch_probs_val = 1.0 / (1.0 + np.exp(-stage2_base[val_row_idx]))
        baseline_cmap = compute_cmap_padded(perch_probs_val, val_labels, active_cls)

        fold_results.append(
            {
                "site": hold_site,
                "n_sc": len(val_batches),
                "n_active": len(active_cls),
                "cmap": fold_cmap,
                "baseline": baseline_cmap,
                "delta": fold_cmap - baseline_cmap,
            }
        )
        print(
            f"  Hold {hold_site}: cmAP={fold_cmap:.4f}  "
            f"(baseline={baseline_cmap:.4f}, delta={fold_cmap - baseline_cmap:+.4f})  "
            f"[{len(val_batches)} sc, {len(active_cls)} cls]"
        )

    # -------------------------------------------------------------------------
    # Aggregate
    # -------------------------------------------------------------------------
    weights = np.array([r["n_active"] for r in fold_results], dtype=float)
    cmaps = np.array([r["cmap"] for r in fold_results])
    baselines = np.array([r["baseline"] for r in fold_results])
    deltas = np.array([r["delta"] for r in fold_results])

    loso_weighted = float(np.average(cmaps, weights=weights))
    loso_simple = float(np.mean(cmaps))
    baseline_weighted = float(np.average(baselines, weights=weights))
    delta_weighted = float(np.average(deltas, weights=weights))

    print(f"\n{'=' * 60}")
    print(f"LOSO cmAP (weighted by active classes): {loso_weighted:.4f}")
    print(f"LOSO cmAP (simple average):             {loso_simple:.4f}")
    print(f"Baseline Perch logits (weighted):        {baseline_weighted:.4f}")
    print(f"Stage2 delta (weighted):                 {delta_weighted:+.4f}")
    print(f"{'=' * 60}")

    return {
        "folds": fold_results,
        "loso_weighted": loso_weighted,
        "loso_simple": loso_simple,
        "baseline_weighted": baseline_weighted,
        "delta_weighted": delta_weighted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LOSO cross-validation for ProtoSSM Stage2"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/protossm_v3.pt",
        help="Path to checkpoint with Stage1 weights (model_state_dict)",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=30,
        help="Number of Stage2 training epochs per fold (default: 30)",
    )
    parser.add_argument(
        "--residual-weight",
        type=float,
        default=0.35,
        help="Stage2 blend weight for evaluation (default: 0.35)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for Stage2 training (default: 42)",
    )
    parser.add_argument(
        "--npz-file",
        type=str,
        default="full_perch_arrays_59.npz",
        help="NPZ file with 59-soundscape Perch arrays (default: full_perch_arrays_59.npz)",
    )
    parser.add_argument(
        "--probe-scores",
        type=str,
        default="full_probe_scores__59sc.npy",
        help="Probe scores filename (relative to DATA_ROOT/perch-meta/). "
        "Default: full_probe_scores__59sc.npy (matches Stage2 training base). "
        "Use 'none' for raw Perch logits (original behavior, unreliable for seed ranking).",
    )
    parser.add_argument(
        "--stage2-d-model",
        type=int,
        default=D_RESIDUAL,
        help=f"Stage2 SSM d_model (default: {D_RESIDUAL})",
    )
    parser.add_argument(
        "--stage2-d-hour",
        type=int,
        default=0,
        help="Stage2 hour embedding dim (0=disabled, default: 0)",
    )
    parser.add_argument(
        "--stage2-l2",
        type=float,
        default=0.0,
        help="Stage2 correction L2 penalty weight (default: 0.0)",
    )
    parser.add_argument(
        "--diel-priors",
        type=str,
        default="none",
        help="Diel priors filename (relative to DATA_ROOT/perch-meta/). "
        "Default: 'none' (disabled). Use 'diel_priors.npy' to enable.",
    )
    parser.add_argument(
        "--diel-alpha",
        type=float,
        default=0.0,
        help="Diel priors blend weight (log-odds scale). 0=disabled, try 0.3-1.0 (default: 0.0).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override KEGO_PATH_DATA",
    )
    args = parser.parse_args()

    if args.data_dir:
        os.environ["KEGO_PATH_DATA"] = args.data_dir
        global DATA_ROOT
        DATA_ROOT = Path(args.data_dir)

    print(
        f"[eval_loso] checkpoint={args.checkpoint}, stage2_epochs={args.stage2_epochs}"
    )
    print(f"            residual_weight={args.residual_weight}, seed={args.seed}")
    print(
        f"            d_model={args.stage2_d_model}, d_hour={args.stage2_d_hour}, l2={args.stage2_l2}"
    )
    print(f"            probe_scores={args.probe_scores}")
    print(f"            diel_priors={args.diel_priors}, diel_alpha={args.diel_alpha}")

    run_loso(
        checkpoint_path=Path(args.checkpoint),
        stage2_epochs=args.stage2_epochs,
        residual_weight=args.residual_weight,
        seed=args.seed,
        npz_file=args.npz_file,
        probe_scores_file=args.probe_scores,
        stage2_d_model=args.stage2_d_model,
        stage2_d_hour=args.stage2_d_hour,
        stage2_l2=args.stage2_l2,
        diel_priors_file=args.diel_priors,
        diel_alpha=args.diel_alpha,
    )


if __name__ == "__main__":
    main()
