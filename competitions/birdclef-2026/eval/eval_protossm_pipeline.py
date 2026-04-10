"""Evaluate the full ProtoSSM v3 + ResidualSSMv3 pipeline on OOF soundscape data.

Mirrors the exact inference pipeline:
  Perch logits → ProtoSSM logits → sigmoid → ResidualSSMv3 correction →
  final_scores = perch_logits + 0.35 * correction → sigmoid → cmAP

This is the same space as Kaggle inference, so this score should correlate with LB.

Usage:
    uv run python competitions/birdclef-2026/eval/eval_protossm_pipeline.py
    uv run python competitions/birdclef-2026/eval/eval_protossm_pipeline.py \
        --checkpoint outputs/protossm_v3.pt \
        --residual-weight 0.35
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

# Add training dir to path so we can import model classes and data loader
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


def run_eval(
    checkpoint_path: Path,
    residual_weight: float,
    stage3_weight: float,
    npz_file: str | None = None,
) -> None:
    print(f"Checkpoint  : {checkpoint_path}")
    print(f"Stage 2 w   : {residual_weight}")
    print(f"Stage 3 w   : {stage3_weight}")

    # -------------------------------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------------------------------
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    n_tax_groups = cfg.get("n_tax_groups", 5)

    print(f"Checkpoint config: {cfg}")

    model = ProtoSSM(n_tax_groups=n_tax_groups)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    residual_ssm = ResidualSSMv3()
    if "residual_ssm_state_dict" in ckpt:
        residual_ssm.load_state_dict(ckpt["residual_ssm_state_dict"])
        print("ResidualSSMv3 (Stage 2) loaded from checkpoint.")
    else:
        print("WARNING: no residual_ssm_state_dict in checkpoint — correction = 0.")
    residual_ssm.eval()

    residual_ssm_v3b = None
    if "residual_ssm_v3b_state_dict" in ckpt:
        residual_ssm_v3b = ResidualSSMv3()
        residual_ssm_v3b.load_state_dict(ckpt["residual_ssm_v3b_state_dict"])
        residual_ssm_v3b.eval()
        ckpt_rw = ckpt.get("residual_weight", residual_weight)
        print(
            f"ResidualSSMv3b (Stage 3) loaded (checkpoint residual_weight={ckpt_rw})."
        )
    else:
        print("No Stage 3 (residual_ssm_v3b_state_dict) in checkpoint.")

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("\nLoading data...")
    eval_npz = npz_file or "full_perch_arrays.npz"
    # Probe scores file must match the NPZ being evaluated.
    # 59sc NPZ → use checkpoint's probe_scores_file (trained on 59sc).
    # 66sc NPZ (default) → always use full_probe_scores.npy (792 rows).
    if npz_file is not None and "59" in npz_file:
        probe_scores_file = ckpt.get("config", {}).get(
            "probe_scores_file", "full_probe_scores.npy"
        )
    else:
        probe_scores_file = "full_probe_scores.npy"
    data = load_data(DATA_ROOT, npz_file=eval_npz, probe_scores_file=probe_scores_file)
    emb = data["emb"]
    logits = data["logits"]  # (708, 234) Perch logits
    labels = data["labels"]  # (708, 234)
    sites = data["sites"]
    hours = data["hours"]
    filenames = data["filenames"]

    site_to_idx: dict[str, int] = {}
    all_sites = sorted(set(sites.tolist()))
    site_to_idx = {s: i + 1 for i, s in enumerate(all_sites)}

    # Use site_to_idx from checkpoint if available (must match training vocab)
    if "site_to_idx" in ckpt:
        site_to_idx = ckpt["site_to_idx"]
        print(f"Using checkpoint site_to_idx ({len(site_to_idx)} sites)")

    all_batches = build_file_batches(
        emb, logits, labels, sites, hours, filenames, site_to_idx
    )
    print(f"File batches: {len(all_batches)}")

    file_to_rows: dict[str, list[int]] = {}
    for i, fn in enumerate(filenames):
        if fn not in file_to_rows:
            file_to_rows[fn] = []
        file_to_rows[fn].append(i)

    # -------------------------------------------------------------------------
    # Run pipeline
    # -------------------------------------------------------------------------
    proto_logits_arr = np.zeros((len(emb), N_CLASSES), dtype=np.float32)
    correction_arr = np.zeros((len(emb), N_CLASSES), dtype=np.float32)
    correction3_arr = np.zeros((len(emb), N_CLASSES), dtype=np.float32)

    with torch.no_grad():
        for batch in all_batches:
            row_idx = file_to_rows[batch["filename"]]
            emb_t = torch.tensor(batch["emb"], dtype=torch.float32)
            logits_perch_t = torch.tensor(batch["logits"], dtype=torch.float32)
            site_t = torch.tensor(batch["site_idx"], dtype=torch.long)
            hour_t = torch.tensor(batch["hour_idx"], dtype=torch.long)

            proto_logits_t, _ = model(emb_t, logits_perch_t, site_t, hour_t)
            proto_probs_t = torch.sigmoid(proto_logits_t)

            correction_t = residual_ssm(emb_t, proto_probs_t)
            proto_logits_arr[row_idx] = proto_logits_t.numpy()
            correction_arr[row_idx] = correction_t.numpy()

            # Stage 3: input = sigmoid(stage2_base + rw * correction)
            if residual_ssm_v3b is not None:
                stage3_base_t = logits_perch_t + residual_weight * correction_t
                stage3_probs_t = torch.sigmoid(stage3_base_t)
                correction3_t = residual_ssm_v3b(emb_t, stage3_probs_t)
                correction3_arr[row_idx] = correction3_t.numpy()

    proto_probs_arr = 1.0 / (1.0 + np.exp(-proto_logits_arr))

    # Stage 2 pipeline: Perch + rw * correction2
    final_logits = logits + residual_weight * correction_arr
    final_probs = 1.0 / (1.0 + np.exp(-final_logits))

    # Stage 3 pipeline: Stage 2 + w3 * correction3
    final3_logits = final_logits + stage3_weight * correction3_arr
    final3_probs = 1.0 / (1.0 + np.exp(-final3_logits))

    # -------------------------------------------------------------------------
    # Compute cmAP on all labeled windows
    # -------------------------------------------------------------------------
    active_cls = np.where(labels.sum(0) > 0)[0]
    print(f"\nActive classes (≥1 positive label): {len(active_cls)}")

    def cmAP(preds: np.ndarray, name: str) -> float:
        aps = []
        for c in active_cls:
            if labels[:, c].sum() > 0:
                ap = average_precision_score(labels[:, c], preds[:, c])
                aps.append(ap)
        score = float(np.mean(aps))
        print(f"  {name:45s}: {score:.4f}  ({len(aps)} classes)")
        return score

    print("\n--- cmAP on all labeled soundscape windows ---")
    cmAP(1.0 / (1.0 + np.exp(-logits)), "Perch logits (baseline)")
    cmAP(proto_probs_arr, "ProtoSSM probs (Stage 1 only)")
    cmAP(final_probs, f"Perch + {residual_weight:.2f}×ResidualSSMv3 (Stage 2)")
    if residual_ssm_v3b is not None:
        cmAP(final3_probs, f"Stage 2 + {stage3_weight:.2f}×ResidualSSMv3b (Stage 3)")

    # Blend weight sensitivity for Stage 2
    print("\n--- Stage 2 blend weight sensitivity ---")
    for w in [0.0, 0.35, 0.50, 0.60, 0.70, 0.80, 1.00]:
        p = 1.0 / (1.0 + np.exp(-(logits + w * correction_arr)))
        cmAP(p, f"stage2 w={w:.2f}")

    # Stage 3 blend weight sensitivity (if present)
    if residual_ssm_v3b is not None:
        print(
            f"\n--- Stage 3 blend weight sensitivity (stage2_w={residual_weight:.2f}) ---"
        )
        for w3 in [0.0, 0.25, 0.35, 0.50, 0.70, 1.00]:
            p = 1.0 / (1.0 + np.exp(-(final_logits + w3 * correction3_arr)))
            cmAP(p, f"stage3 w={w3:.2f}")

    # Correction stats
    print("\nStage 2 correction stats:")
    print(
        f"  mean abs: {np.abs(correction_arr).mean():.4f}  range: [{correction_arr.min():.3f}, {correction_arr.max():.3f}]"
    )
    if residual_ssm_v3b is not None:
        print("Stage 3 correction stats:")
        print(
            f"  mean abs: {np.abs(correction3_arr).mean():.4f}  range: [{correction3_arr.min():.3f}, {correction3_arr.max():.3f}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ProtoSSM v3 + ResidualSSMv3 pipeline on labeled soundscapes"
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/protossm_v3.pt",
        help="Path to checkpoint file (default: outputs/protossm_v3.pt)",
    )
    parser.add_argument(
        "--residual-weight",
        type=float,
        default=0.70,
        help="Weight for Stage 2 ResidualSSMv3 correction (default: 0.70)",
    )
    parser.add_argument(
        "--stage3-weight",
        type=float,
        default=0.70,
        help="Weight for Stage 3 ResidualSSMv3b correction (default: 0.70)",
    )
    parser.add_argument(
        "--npz-file",
        default=None,
        help="NPZ file to load (default: full_perch_arrays.npz = 66sc). "
        "Use 'full_perch_arrays_59.npz' to evaluate on 59sc only.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override KEGO_PATH_DATA",
    )
    args = parser.parse_args()

    if args.data_dir:
        os.environ["KEGO_PATH_DATA"] = args.data_dir
        # Reimport DATA_ROOT after env change
        global DATA_ROOT
        DATA_ROOT = Path(args.data_dir)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    run_eval(checkpoint_path, args.residual_weight, args.stage3_weight, args.npz_file)


if __name__ == "__main__":
    main()
