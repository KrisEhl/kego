"""Pseudo-label Stage2 training on unlabeled soundscapes.

Uses precomputed Perch embeddings+logits for all 10,659 train soundscapes.
Generates hard pseudo-labels from the current model's Perch logits (or Stage1+Stage2
final predictions), then retrains Stage2 on labeled + pseudo-labeled batches.

Why pseudo-labels:
- Only 59 labeled soundscapes (8 sites). Test soundscapes span 23 sites.
- 10,659 total train soundscapes including ~10,600 unlabeled from 14 new sites.
- 2025 winner technique: +0.058 LB via iterative pseudo-labeling.
- Unlabeled sites (S01,S02,S04-S07,S10-S12,S14,S16,S17,S20,S21) match test distribution.

Anti-circularity design:
- Pseudo-labels generated from PERCH logits only (frozen, not trained by this pipeline).
- Threshold: sigmoid(perch_logit) > --pseudo-threshold (default 0.3).
- Hard binary labels (0/1) avoid soft-label BCE circular feedback.
- Labeled 59sc batches use ground-truth labels (unchanged).

Usage:
    # Round 1: generate pseudo-labels and retrain Stage2
    KEGO_PATH_DATA=/home/kristian/projects/kego/data uv run python \\
        competitions/birdclef-2026/training/train_pseudo_stage2.py \\
        --checkpoint outputs/protossm_v3.pt \\
        --output outputs/protossm_pseudo_r1.pt \\
        --stage2-epochs 30 \\
        --pseudo-threshold 0.3

    # Round 2: use round-1 model as teacher
    KEGO_PATH_DATA=/home/kristian/projects/kego/data uv run python \\
        competitions/birdclef-2026/training/train_pseudo_stage2.py \\
        --checkpoint outputs/protossm_pseudo_r1.pt \\
        --output outputs/protossm_pseudo_r2.pt \\
        --stage2-epochs 30 \\
        --pseudo-threshold 0.3 \\
        --use-stage2-preds  # use Stage1+Stage2 preds instead of Perch only

Required:
    $KEGO_PATH_DATA/birdclef/birdclef-2026/perch_soundscape_cache/
        perch_sc_scores.npy      — float32 (N_windows, 234) Perch logits for all SC
        perch_sc_embeddings.npy  — float32 (N_windows, 1536)
        perch_sc_meta.parquet    — row_id, filename, site, hour_utc
    $KEGO_PATH_DATA/perch-meta/full_perch_arrays_59.npz — 59 labeled soundscapes
    --checkpoint                 — Stage1+Stage2 checkpoint
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TRAIN_DIR = Path(__file__).parent
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
CACHE_DIR = DATA_ROOT / "birdclef" / "birdclef-2026" / "perch_soundscape_cache"
N_WINDOWS = 12


def load_pseudo_batches(
    cache_dir: Path,
    pseudo_threshold: float,
    stage1_model: torch.nn.Module | None,
    stage2_model: torch.nn.Module | None,
    residual_weight: float,
    use_stage2_preds: bool,
    use_stage1_preds: bool,
    labeled_filenames: set[str],
    site_to_idx: dict[str, int],
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load precomputed Perch cache and build pseudo-labeled batches.

    Returns:
        pseudo_batches: list of batch dicts for UNLABELED soundscapes only
        emb_all, logits_all, pseudo_labels_all, sites_all, hours_all, filenames_all
        (these include ALL soundscapes for file_to_rows construction)
    """
    print(f"Loading Perch cache from {cache_dir} ...", flush=True)
    scores = np.load(cache_dir / "perch_sc_scores.npy")  # (N_total, 234)
    emb = np.load(cache_dir / "perch_sc_embeddings.npy")  # (N_total, 1536)
    meta = pd.read_parquet(cache_dir / "perch_sc_meta.parquet")

    filenames_arr = meta["filename"].values  # (N_total,)
    sites_arr = meta["site"].values  # (N_total,)
    hours_arr = meta["hour_utc"].values.astype(np.int16)  # (N_total,)

    print(
        f"  Total windows: {len(scores)} ({len(scores) // N_WINDOWS} files)", flush=True
    )

    # Generate pseudo-labels
    if use_stage1_preds and stage1_model is not None:
        print("  Generating pseudo-labels from Stage1 proto_probs ...", flush=True)
        unique_files = list(dict.fromkeys(filenames_arr))
        tmp_batches = []
        for fn in unique_files:
            mask = filenames_arr == fn
            idx = np.where(mask)[0]
            tmp_batches.append(
                {
                    "emb": emb[idx],
                    "logits": scores[idx],
                    "labels": np.zeros((len(idx), N_CLASSES), dtype=np.float32),
                    "site_idx": site_to_idx.get(sites_arr[idx[0]], 0),
                    "hour_idx": int(hours_arr[idx[0]]),
                    "filename": fn,
                }
            )
        proto_logits = predict_batches_logits(stage1_model, tmp_batches)
        pseudo_probs = 1.0 / (1.0 + np.exp(-proto_logits))
        print(f"  Stage1-only pseudo-labels (threshold={pseudo_threshold})", flush=True)
    elif use_stage2_preds and stage1_model is not None and stage2_model is not None:
        print(
            "  Generating pseudo-labels from Stage1+Stage2 predictions ...", flush=True
        )
        # Build temporary batches for Stage1 forward pass (all files)
        unique_files = list(dict.fromkeys(filenames_arr))
        tmp_batches = []
        for fn in unique_files:
            mask = filenames_arr == fn
            idx = np.where(mask)[0]
            tmp_batches.append(
                {
                    "emb": emb[idx],
                    "logits": scores[idx],
                    "labels": np.zeros((len(idx), N_CLASSES), dtype=np.float32),
                    "site_idx": site_to_idx.get(sites_arr[idx[0]], 0),
                    "hour_idx": int(hours_arr[idx[0]]),
                    "filename": fn,
                }
            )
        proto_logits = predict_batches_logits(
            stage1_model, tmp_batches
        )  # (N_total, 234)
        proto_probs = 1.0 / (1.0 + np.exp(-proto_logits))

        # Stage2 corrections
        stage2_model.eval()
        corrections = np.zeros_like(scores)
        row = 0
        for b in tmp_batches:
            n = len(b["emb"])
            with torch.no_grad():
                corr = stage2_model(
                    torch.tensor(b["emb"], dtype=torch.float32),
                    torch.tensor(proto_probs[row : row + n], dtype=torch.float32),
                    hour_idx=b.get("hour_idx"),
                ).numpy()
            corrections[row : row + n] = corr
            row += n

        final_logits = scores + residual_weight * corrections
        pseudo_probs = 1.0 / (1.0 + np.exp(-final_logits))
        print(f"  Stage2 pseudo-labels (threshold={pseudo_threshold})", flush=True)
    else:
        # Use raw Perch logits only (non-circular)
        pseudo_probs = 1.0 / (1.0 + np.exp(-scores))
        print(f"  Perch-only pseudo-labels (threshold={pseudo_threshold})", flush=True)

    pseudo_labels = (pseudo_probs > pseudo_threshold).astype(np.float32)
    n_pos = pseudo_labels.sum(axis=1).mean()
    print(f"  Mean positives per window: {n_pos:.2f}", flush=True)

    # Build pseudo-labeled batches for UNLABELED files only
    pseudo_batches = []
    unique_files = list(dict.fromkeys(filenames_arr))
    n_labeled_skipped = 0
    n_pseudo = 0
    for fn in unique_files:
        if fn in labeled_filenames:
            n_labeled_skipped += 1
            continue  # skip labeled soundscapes (will use ground-truth labels)
        mask = filenames_arr == fn
        idx = np.where(mask)[0]
        pseudo_batches.append(
            {
                "emb": emb[idx],
                "logits": scores[idx],
                "labels": pseudo_labels[idx],
                "site_idx": site_to_idx.get(sites_arr[idx[0]], 0),
                "hour_idx": int(hours_arr[idx[0]]),
                "filename": fn,
            }
        )
        n_pseudo += 1

    print(
        f"  Pseudo-labeled batches: {n_pseudo} files "
        f"(skipped {n_labeled_skipped} labeled files)",
        flush=True,
    )

    return pseudo_batches


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo-label Stage2 training")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/protossm_v3.pt",
        help="Stage1+Stage2 checkpoint (teacher model)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/protossm_pseudo_r1.pt",
        help="Output checkpoint path",
    )
    parser.add_argument(
        "--npz-file",
        type=str,
        default="full_perch_arrays_59.npz",
        help="Labeled 59sc NPZ file",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=30,
        help="Stage2 training epochs (default: 30)",
    )
    parser.add_argument(
        "--pseudo-threshold",
        type=float,
        default=0.3,
        help="Probability threshold for hard pseudo-labels (default: 0.3)",
    )
    parser.add_argument(
        "--residual-weight",
        type=float,
        default=0.35,
        help="Stage2 blend weight (default: 0.35)",
    )
    parser.add_argument(
        "--max-pseudo-batches",
        type=int,
        default=None,
        help="Cap pseudo-labeled batches (for testing, default: all)",
    )
    parser.add_argument(
        "--pseudo-per-epoch",
        type=int,
        default=708,
        help="Pseudo batches sampled per epoch (default: 708 = same as labeled sc count). "
        "Prevents 10K batches × 30 epochs from making training ~150× slower.",
    )
    parser.add_argument(
        "--use-stage2-preds",
        action="store_true",
        help="Use Stage1+Stage2 preds for pseudo-labels (default: Perch logits only)",
    )
    parser.add_argument(
        "--use-stage1-preds",
        action="store_true",
        help="Use Stage1 proto_probs for pseudo-labels — best calibrated (~4.6 pos/window at t=0.5)",
    )
    parser.add_argument(
        "--probe-scores",
        type=str,
        default="full_probe_scores__59sc.npy",
        help="Probe scores filename for labeled 59sc Stage2 base (relative to DATA_ROOT/perch-meta/). "
        "Default: full_probe_scores__59sc.npy. Use 'none' for raw Perch logits (not recommended).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed (default: 42)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load labeled 59sc data
    # -------------------------------------------------------------------------
    print(f"\n[1] Loading labeled data from {args.npz_file} ...", flush=True)
    data = load_data(DATA_ROOT, npz_file=args.npz_file, probe_scores_file="none.npy")
    emb_labeled = data["emb"]  # (708, 1536)
    labels_labeled = data["labels"]  # (708, 234)
    sites_labeled = data["sites"]  # (708,)
    hours_labeled = data["hours"]
    filenames_labeled = data["filenames"]

    # Stage2 base for labeled soundscapes: probe_scores (matching real inference pipeline)
    # Probe scores are seed-independent XC probes + Bayesian site/hour priors (~0.926 cmAP).
    # Using raw Perch logits here would create a training-inference mismatch for Stage2.
    probe_scores_path = DATA_ROOT / "perch-meta" / args.probe_scores
    if args.probe_scores != "none" and probe_scores_path.exists():
        logits_labeled = np.load(probe_scores_path).astype(np.float32)  # (708, 234)
        print(
            f"  Stage2 base: probe scores from {args.probe_scores} (shape {logits_labeled.shape})",
            flush=True,
        )
    else:
        logits_labeled = data["logits"]  # (708, 234) raw Perch logits — fallback
        print(
            "  Stage2 base: raw Perch logits (probe scores not found or --probe-scores none)",
            flush=True,
        )

    all_sites = sorted(set(sites_labeled.tolist()))
    site_to_idx = {s: i + 1 for i, s in enumerate(all_sites)}
    print(f"  Labeled sites: {all_sites}", flush=True)
    print(f"  Labeled windows: {len(emb_labeled)}", flush=True)

    labeled_batches = build_file_batches(
        emb_labeled,
        logits_labeled,
        labels_labeled,
        sites_labeled,
        hours_labeled,
        filenames_labeled,
        site_to_idx,
    )
    labeled_filenames = {b["filename"] for b in labeled_batches}
    print(f"  Labeled soundscapes: {len(labeled_batches)}", flush=True)

    file_to_rows_labeled: dict[str, list[int]] = {}
    for i, fn in enumerate(filenames_labeled):
        file_to_rows_labeled.setdefault(fn, []).append(i)

    # -------------------------------------------------------------------------
    # Load Stage1 + Stage2 checkpoint
    # -------------------------------------------------------------------------
    print(f"\n[2] Loading checkpoint from {checkpoint_path} ...", flush=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    n_tax_groups = cfg.get("n_tax_groups", 5)

    stage1_model = ProtoSSM(n_tax_groups=n_tax_groups)
    stage1_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    stage1_model.eval()
    print("  Stage1 loaded.", flush=True)

    d_model_saved = cfg.get("d_residual", D_RESIDUAL)
    stage2_model_for_preds = ResidualSSMv3(d_model=d_model_saved)
    if "residual_ssm_state_dict" in ckpt:
        stage2_model_for_preds.load_state_dict(ckpt["residual_ssm_state_dict"])
        print("  Stage2 loaded.", flush=True)
    else:
        print(
            "  WARNING: no Stage2 in checkpoint — using random init for pseudo-label gen.",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # Build pseudo-labeled batches from precomputed cache
    # -------------------------------------------------------------------------
    if not CACHE_DIR.exists():
        print(
            f"\nERROR: Cache dir {CACHE_DIR} does not exist.\n"
            "Run precompute_perch_soundscapes.py first.",
            flush=True,
        )
        sys.exit(1)

    print(
        f"\n[3] Building pseudo-labeled batches (threshold={args.pseudo_threshold}) ...",
        flush=True,
    )
    pseudo_batches = load_pseudo_batches(
        cache_dir=CACHE_DIR,
        pseudo_threshold=args.pseudo_threshold,
        stage1_model=stage1_model,
        stage2_model=stage2_model_for_preds,
        residual_weight=args.residual_weight,
        use_stage2_preds=args.use_stage2_preds,
        use_stage1_preds=args.use_stage1_preds,
        labeled_filenames=labeled_filenames,
        site_to_idx=site_to_idx,
    )

    if args.max_pseudo_batches is not None:
        rng = np.random.default_rng(args.seed)
        idx_sel = rng.choice(
            len(pseudo_batches),
            size=min(args.max_pseudo_batches, len(pseudo_batches)),
            replace=False,
        )
        pseudo_batches = [pseudo_batches[i] for i in sorted(idx_sel)]
        print(f"  Capped to {len(pseudo_batches)} pseudo-labeled batches", flush=True)

    # Subsample pseudo batches to pseudo_per_epoch for manageable training time.
    # All labeled batches are always included. Pseudo batches are randomly sampled.
    # Result: each "epoch" sees 708 labeled + pseudo_per_epoch pseudo soundscapes.
    # Training time ≈ 2× standard Stage2 (vs 150× if all 10K pseudo used every epoch).
    pseudo_per_epoch = min(args.pseudo_per_epoch, len(pseudo_batches))
    rng_sample = np.random.default_rng(args.seed + 999)
    if pseudo_per_epoch < len(pseudo_batches):
        sample_idx = rng_sample.choice(
            len(pseudo_batches), size=pseudo_per_epoch, replace=False
        )
        pseudo_batches_train = [pseudo_batches[i] for i in sorted(sample_idx)]
        print(
            f"  Subsampled {pseudo_per_epoch}/{len(pseudo_batches)} pseudo batches for training",
            flush=True,
        )
    else:
        pseudo_batches_train = pseudo_batches

    # -------------------------------------------------------------------------
    # Get Stage1 in-sample predictions (only for training subset)
    # -------------------------------------------------------------------------
    print("\n[4] Computing Stage1 proto_probs on labeled 59sc ...", flush=True)
    proto_logits_labeled = predict_batches_logits(stage1_model, labeled_batches)
    proto_probs_labeled = 1.0 / (1.0 + np.exp(-proto_logits_labeled))

    print(
        f"[5] Computing Stage1 proto_probs on {len(pseudo_batches_train)} pseudo batches ...",
        flush=True,
    )
    proto_logits_pseudo = predict_batches_logits(stage1_model, pseudo_batches_train)
    proto_probs_pseudo = 1.0 / (1.0 + np.exp(-proto_logits_pseudo))

    n_labeled = len(emb_labeled)
    n_pseudo_windows = len(pseudo_batches_train) * N_WINDOWS

    emb_combined = np.concatenate(
        [
            emb_labeled,
            np.concatenate([b["emb"] for b in pseudo_batches_train], axis=0),
        ],
        axis=0,
    )
    logits_combined = np.concatenate(
        [
            logits_labeled,
            np.concatenate([b["logits"] for b in pseudo_batches_train], axis=0),
        ],
        axis=0,
    )
    labels_combined = np.concatenate(
        [
            labels_labeled,
            np.concatenate([b["labels"] for b in pseudo_batches_train], axis=0),
        ],
        axis=0,
    )
    proto_probs_combined = np.concatenate(
        [proto_probs_labeled, proto_probs_pseudo], axis=0
    )

    print(
        f"  Combined: {len(emb_combined)} windows "
        f"({n_labeled} labeled + {n_pseudo_windows} pseudo-labeled)",
        flush=True,
    )

    # Build file_to_rows for combined array
    file_to_rows_combined: dict[str, list[int]] = {}
    row = 0
    for b in labeled_batches:
        fn = b["filename"]
        n = len(b["emb"])
        file_to_rows_combined[fn] = list(range(row, row + n))
        row += n
    for b in pseudo_batches_train:
        fn = b["filename"]
        n = len(b["emb"])
        file_to_rows_combined[fn] = list(range(row, row + n))
        row += n

    all_batches_combined = labeled_batches + pseudo_batches_train

    # -------------------------------------------------------------------------
    # Train new Stage2 on combined data
    # -------------------------------------------------------------------------
    print(
        f"\n[6] Training Stage2 on {len(all_batches_combined)} soundscapes "
        f"({args.stage2_epochs} epochs) ...",
        flush=True,
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    new_residual_ssm = ResidualSSMv3(dropout=DROPOUT_RESIDUAL)
    new_residual_ssm = train_residual_ssm_v3(
        residual_ssm=new_residual_ssm,
        emb=emb_combined,
        proto_logits=logits_combined,
        proto_probs=proto_probs_combined,
        labels=labels_combined,
        all_batches=all_batches_combined,
        file_to_rows=file_to_rows_combined,
        epochs=args.stage2_epochs,
        val_batches=None,
        verbose=True,
    )

    # -------------------------------------------------------------------------
    # Save checkpoint
    # -------------------------------------------------------------------------
    print(f"\n[7] Saving checkpoint to {output_path} ...", flush=True)
    save_ckpt = {
        "model_state_dict": stage1_model.state_dict(),
        "residual_ssm_state_dict": new_residual_ssm.state_dict(),
        "config": {
            **cfg,
            "pseudo_round": cfg.get("pseudo_round", 0) + 1,
            "pseudo_threshold": args.pseudo_threshold,
            "n_pseudo_batches": len(pseudo_batches),
            "use_stage2_preds": args.use_stage2_preds,
        },
        # Copy inference-required keys from source checkpoint
        "site_to_idx": ckpt.get("site_to_idx", {}),
        "species_names": ckpt.get("species_names", []),
        "fold_model_states": ckpt.get("fold_model_states", []),
        "oof_scores": ckpt.get("oof_scores", None),
    }
    torch.save(save_ckpt, output_path)
    print(f"Saved: {output_path}", flush=True)
    print(
        f"\n=== Done. Round {save_ckpt['config']['pseudo_round']} complete. ===\n"
        f"To evaluate: OMP_NUM_THREADS=4 uv run python competitions/birdclef-2026/eval/eval_loso.py "
        f"--checkpoint {output_path} --stage2-epochs 30",
        flush=True,
    )


if __name__ == "__main__":
    main()
