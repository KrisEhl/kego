"""Diagonal CORAL embedding alignment: labeled soundscapes → unlabeled distribution.

Stage2 is trained on embeddings from 8 labeled sites (S03,S08,...). At inference
it processes test soundscapes from unseen sites. Diagonal CORAL rescales labeled
embeddings per-dimension to match the unlabeled distribution — Stage2 then trains
and generalises in the same space it sees at test time (no transform at inference).

Outputs (saved to DATA_ROOT/perch-meta/):
    coral_transform.npz    — mu_src, std_src, mu_tgt, std_tgt (1536,) each
    coral_emb_aligned.npy  — float32 (708, 1536) aligned labeled embeddings

Usage:
    KEGO_PATH_DATA=/home/kristian/projects/kego/data uv run python \\
        competitions/birdclef-2026/training/precompute_coral.py
"""

import os
from pathlib import Path

import numpy as np

DATA_ROOT = Path(os.environ.get("KEGO_PATH_DATA", "data"))
CACHE_DIR = DATA_ROOT / "birdclef" / "birdclef-2026" / "perch_soundscape_cache"
META_DIR = DATA_ROOT / "perch-meta"
NPZ_FILE = "full_perch_arrays_59.npz"

EPS = 1e-8  # prevent div-by-zero for near-constant dimensions


def main() -> None:
    # -------------------------------------------------------------------------
    # Load source: labeled 59-soundscape windows
    # -------------------------------------------------------------------------
    npz_path = META_DIR / NPZ_FILE
    print(f"Loading labeled embeddings from {npz_path} ...", flush=True)
    npz = np.load(npz_path)
    emb_src = npz["emb_full"].astype(np.float32)  # (708, 1536)
    print(f"  Source shape: {emb_src.shape}")

    # -------------------------------------------------------------------------
    # Load target: ALL unlabeled soundscape windows from cache
    # -------------------------------------------------------------------------
    print(f"Loading cache embeddings from {CACHE_DIR} ...", flush=True)
    emb_tgt = np.load(CACHE_DIR / "perch_sc_embeddings.npy").astype(np.float32)
    print(f"  Cache shape: {emb_tgt.shape}")

    # Use only unlabeled windows as target (exclude labeled sites from target stats
    # so we don't blend labeled-site characteristics back in).  We identify labeled
    # windows via the meta parquet and exclude them from target stats.
    try:
        import pandas as pd

        meta = pd.read_parquet(CACHE_DIR / "perch_sc_meta.parquet")
        labeled_sites = {"S03", "S08", "S13", "S15", "S18", "S19", "S22", "S23"}
        unlabeled_mask = ~meta["site"].isin(labeled_sites).values
        emb_tgt_unlabeled = emb_tgt[unlabeled_mask]
        print(
            f"  Unlabeled-only target: {emb_tgt_unlabeled.shape}  "
            f"({unlabeled_mask.sum()} / {len(unlabeled_mask)} windows)"
        )
        target_emb = emb_tgt_unlabeled
    except Exception as e:
        print(f"  WARNING: could not filter by site ({e}), using all cache windows as target")
        target_emb = emb_tgt

    # -------------------------------------------------------------------------
    # Diagonal CORAL statistics
    # -------------------------------------------------------------------------
    print("Computing CORAL statistics ...", flush=True)
    mu_src = emb_src.mean(axis=0)  # (1536,)
    std_src = emb_src.std(axis=0)  # (1536,)
    mu_tgt = target_emb.mean(axis=0)  # (1536,)
    std_tgt = target_emb.std(axis=0)  # (1536,)

    print(
        f"  Source: μ range [{mu_src.min():.3f}, {mu_src.max():.3f}]  "
        f"σ range [{std_src.min():.4f}, {std_src.max():.3f}]"
    )
    print(
        f"  Target: μ range [{mu_tgt.min():.3f}, {mu_tgt.max():.3f}]  "
        f"σ range [{std_tgt.min():.4f}, {std_tgt.max():.3f}]"
    )

    # Distribution gap metric (cosine of mean vectors, Frobenius of cov diff)
    mu_diff_norm = float(np.linalg.norm(mu_tgt - mu_src))
    std_ratio_mean = float((std_tgt / (std_src + EPS)).mean())
    print(f"  ||μ_tgt - μ_src|| = {mu_diff_norm:.4f}")
    print(f"  mean(σ_tgt / σ_src) = {std_ratio_mean:.4f}")

    # -------------------------------------------------------------------------
    # Align labeled embeddings: z = (x - μ_src) / σ_src * σ_tgt + μ_tgt
    # -------------------------------------------------------------------------
    print("Aligning labeled embeddings ...", flush=True)
    emb_aligned = (emb_src - mu_src) / (std_src + EPS) * (std_tgt + EPS) + mu_tgt
    emb_aligned = emb_aligned.astype(np.float32)

    # Sanity check
    mu_aligned = emb_aligned.mean(axis=0)
    std_aligned = emb_aligned.std(axis=0)
    mu_err = float(np.abs(mu_aligned - mu_tgt).mean())
    std_ratio = float((std_aligned / (std_tgt + EPS)).mean())
    print(f"  Alignment check: mean |μ_aligned - μ_tgt| = {mu_err:.6f}  mean(σ_aligned / σ_tgt) = {std_ratio:.4f}")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    print(f"\nSaving to {META_DIR} ...", flush=True)
    np.savez(
        META_DIR / "coral_transform.npz",
        mu_src=mu_src,
        std_src=std_src,
        mu_tgt=mu_tgt,
        std_tgt=std_tgt,
    )
    np.save(META_DIR / "coral_emb_aligned.npy", emb_aligned)

    print("  coral_transform.npz : mu/std arrays (1536,) each")
    print(f"  coral_emb_aligned.npy: {emb_aligned.shape}  range [{emb_aligned.min():.3f}, {emb_aligned.max():.3f}]")
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
