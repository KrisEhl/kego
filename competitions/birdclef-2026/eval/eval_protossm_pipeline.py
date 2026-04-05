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


def run_eval(checkpoint_path: Path, residual_weight: float) -> None:
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Res weight : {residual_weight}")

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
        print("ResidualSSMv3 loaded from checkpoint.")
    else:
        print("WARNING: no residual_ssm_state_dict in checkpoint — correction = 0.")
    residual_ssm.eval()

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("\nLoading data...")
    data = load_data(DATA_ROOT)
    emb = data["emb"]  # (708, 1536)
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

    proto_probs_arr = 1.0 / (1.0 + np.exp(-proto_logits_arr))

    # final = Perch logits + weight * correction → sigmoid
    final_logits = logits + residual_weight * correction_arr
    final_probs = 1.0 / (1.0 + np.exp(-final_logits))

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
    cmAP(final_probs, f"Perch + {residual_weight:.2f}×ResidualSSMv3 (full pipeline)")

    # Also show what different blend weights look like
    print("\n--- Blend weight sensitivity ---")
    for w in [0.0, 0.15, 0.25, 0.35, 0.50]:
        p = 1.0 / (1.0 + np.exp(-(logits + w * correction_arr)))
        cmAP(p, f"weight={w:.2f}")

    # Print correction stats
    print("\nCorrection stats:")
    print(f"  mean abs correction : {np.abs(correction_arr).mean():.4f}")
    print(f"  max abs correction  : {np.abs(correction_arr).max():.4f}")
    print(f"  Perch logits range  : {logits.min():.3f} to {logits.max():.3f}")
    print(
        f"  Correction range    : {correction_arr.min():.3f} to {correction_arr.max():.3f}"
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
        default=0.35,
        help="Weight for ResidualSSMv3 correction (default: 0.35)",
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

    run_eval(checkpoint_path, args.residual_weight)


if __name__ == "__main__":
    main()
