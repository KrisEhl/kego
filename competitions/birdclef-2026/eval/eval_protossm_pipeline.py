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
import pandas as pd
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

    _d_site = cfg.get("d_site", 0)
    residual_ssm = ResidualSSMv3(d_site=_d_site)
    if "residual_ssm_state_dict" in ckpt:
        residual_ssm.load_state_dict(ckpt["residual_ssm_state_dict"])
        print(f"ResidualSSMv3 (Stage 2) loaded from checkpoint. d_site={_d_site}")
    else:
        print("WARNING: no residual_ssm_state_dict in checkpoint — correction = 0.")
    residual_ssm.eval()

    # Site profiles (for site-aware model)
    site_profiles_eval: dict = {}
    if "site_profiles" in ckpt and _d_site > 0:
        site_profiles_eval = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in ckpt["site_profiles"].items()
        }
        print(f"Site profiles loaded: {len(site_profiles_eval)} sites")

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

            # Site profile lookup for site-aware model
            _site_name = {v: k for k, v in ckpt.get("site_to_idx", {}).items()}.get(
                batch["site_idx"]
            )
            _site_prof = site_profiles_eval.get(_site_name) if _site_name else None
            correction_t = residual_ssm(emb_t, proto_probs_t, _site_prof)
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

    # -------------------------------------------------------------------------
    # Post-processing evaluation (mirrors inference notebook)
    # -------------------------------------------------------------------------
    _tax_path = DATA_ROOT / "birdclef" / "birdclef-2026" / "taxonomy.csv"
    if _tax_path.exists():
        taxonomy_ = pd.read_csv(_tax_path)
        PRIMARY_LABELS = taxonomy_["primary_label"].tolist()
        label_to_idx = {lbl: i for i, lbl in enumerate(PRIMARY_LABELS)}
        class_name_map = taxonomy_.set_index("primary_label")["class_name"].to_dict()

        active_labels = [PRIMARY_LABELS[c] for c in active_cls]
        _TEXTURE_TAXA = {"Amphibia", "Insecta"}
        _RARE_TAXA = {"Mammalia", "Reptilia"}

        # Build archetype index arrays (only for active classes)
        _idx_amphibia = np.array(
            [
                label_to_idx[c]
                for c in active_labels
                if class_name_map.get(c) == "Amphibia"
            ],
            dtype=np.int32,
        )
        _idx_insecta = np.array(
            [
                label_to_idx[c]
                for c in active_labels
                if class_name_map.get(c) == "Insecta"
            ],
            dtype=np.int32,
        )
        _idx_rare = np.array(
            [
                label_to_idx[c]
                for c in active_labels
                if class_name_map.get(c) in _RARE_TAXA
            ],
            dtype=np.int32,
        )
        _idx_aves = np.array(
            [label_to_idx[c] for c in active_labels if class_name_map.get(c) == "Aves"],
            dtype=np.int32,
        )

        n_files = len(set(filenames.tolist()))
        N_WINDOWS = 12

        def _delta_shift_smooth(arr_3d: np.ndarray, alpha: float) -> np.ndarray:
            if alpha <= 0.0:
                return arr_3d
            out = arr_3d.copy()
            neighbors = (arr_3d[:, :-2, :] + arr_3d[:, 2:, :]) * 0.5
            out[:, 1:-1, :] += alpha * (neighbors - arr_3d[:, 1:-1, :])
            out[:, 0, :] += alpha * (arr_3d[:, 1, :] - arr_3d[:, 0, :])
            out[:, -1, :] += alpha * (arr_3d[:, -2, :] - arr_3d[:, -1, :])
            return out

        def apply_postproc(
            base_logits: np.ndarray,
            smooth_amphibia: float = 0.45,
            smooth_insecta: float = 0.35,
            smooth_aves: float = 0.0,
            smooth_rare: float = 0.05,
            rank_power: float = 0.4,
            boost_alpha: float = 0.05,
            boost_topk: int = 2,
        ) -> np.ndarray:
            n_total, n_cls = base_logits.shape
            # Reshape to (n_files, N_WINDOWS, n_cls) — assumes rows sorted by file
            s = base_logits.reshape(n_files, N_WINDOWS, n_cls).copy()
            # Temporal smoothing
            for idx_arr, alpha in [
                (_idx_amphibia, smooth_amphibia),
                (_idx_insecta, smooth_insecta),
                (_idx_aves, smooth_aves),
                (_idx_rare, smooth_rare),
            ]:
                if len(idx_arr) > 0 and alpha > 0.0:
                    s[:, :, idx_arr] = _delta_shift_smooth(s[:, :, idx_arr], alpha)
            # Sigmoid
            probs = 1.0 / (1.0 + np.exp(-s))  # (n_files, N_WINDOWS, n_cls)
            # Rank-aware scaling: file_max^rank_power
            file_max = probs.max(axis=1)  # (n_files, n_cls)
            probs = probs * np.power(file_max, rank_power)[:, np.newaxis, :]
            # File-level confidence boost top-k
            if boost_topk > 0 and boost_alpha > 0.0:
                top_idx = np.argsort(file_max, axis=1)[:, -boost_topk:]
                boost_mask = np.zeros_like(file_max)
                for fi in range(n_files):
                    boost_mask[fi, top_idx[fi]] = 1.0
                probs = (
                    probs
                    + boost_alpha
                    * file_max[:, np.newaxis, :]
                    * boost_mask[:, np.newaxis, :]
                )
                probs = np.clip(probs, 0.0, 1.0)
            return probs.reshape(n_total, n_cls)

        print("\n--- Post-processing cmAP (v52 config: A4+A2+rank-aware) ---")
        pp_default = apply_postproc(final_logits)
        cmAP(pp_default, "v52 default (smooth+rank+boost_top2)")

        print("\n--- Post-processing parameter sweep ---")
        # Rank power sweep
        for rp in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0]:
            p = apply_postproc(final_logits, rank_power=rp)
            cmAP(p, f"rank_power={rp:.1f}")
        # Boost topk sweep
        for topk in [0, 1, 2, 3, 5]:
            p = apply_postproc(final_logits, boost_topk=topk)
            cmAP(p, f"boost_topk={topk}")
        # Boost alpha sweep
        for ba in [0.0, 0.03, 0.05, 0.08, 0.12]:
            p = apply_postproc(final_logits, boost_alpha=ba)
            cmAP(p, f"boost_alpha={ba:.2f}")
        # Smoothing: amphibia alpha sweep
        for sa in [0.0, 0.20, 0.35, 0.45, 0.60]:
            p = apply_postproc(final_logits, smooth_amphibia=sa)
            cmAP(p, f"smooth_amphibia={sa:.2f}")
        # No smoothing at all (rank+boost only)
        p = apply_postproc(
            final_logits, smooth_amphibia=0.0, smooth_insecta=0.0, smooth_rare=0.0
        )
        cmAP(p, "rank+boost only (no smooth)")

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
