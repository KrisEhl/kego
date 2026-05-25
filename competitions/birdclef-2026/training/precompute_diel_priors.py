"""Compute diel (hour-of-day) and site activity priors from all 10,658 unlabeled soundscapes.

Uses precomputed Perch logits from perch_soundscape_cache to estimate
P(species active | hour) and P(species active | site) across all training sites.
These are much more reliable than the narrow Bayesian tables built from 59 labeled sc.

Outputs (saved to DATA_ROOT/perch-meta/):
    diel_priors.npy           — float32 (234, 24) mean sigmoid(logit) per species per hour
    site_priors_all.npy       — float32 (234, n_sites) mean sigmoid(logit) per species per site
    diel_site_priors.npy      — float32 (234, n_sites, 24) joint site×hour priors
    diel_priors_meta.json     — hour counts, site names, global mean

Usage:
    KEGO_PATH_DATA=/home/kristian/projects/kego/data uv run python \\
        competitions/birdclef-2026/training/precompute_diel_priors.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

DATA_ROOT = Path(os.environ.get("KEGO_PATH_DATA", "data"))
CACHE_DIR = DATA_ROOT / "birdclef" / "birdclef-2026" / "perch_soundscape_cache"
OUT_DIR = DATA_ROOT / "perch-meta"
N_CLASSES = 234
N_HOURS = 24


def main() -> None:
    print(f"Loading Perch cache from {CACHE_DIR} ...", flush=True)
    scores = np.load(CACHE_DIR / "perch_sc_scores.npy")  # (N, 234) logits
    meta = pd.read_parquet(CACHE_DIR / "perch_sc_meta.parquet")

    hours = meta["hour_utc"].values.astype(np.int16)  # (N,)
    sites = meta["site"].values  # (N,) str
    all_sites = sorted(set(sites.tolist()))
    site_to_idx = {s: i for i, s in enumerate(all_sites)}
    n_sites = len(all_sites)

    print(f"  Total windows: {len(scores)}", flush=True)
    print(f"  Sites ({n_sites}): {all_sites}", flush=True)
    print(f"  Hours present: {sorted(set(hours.tolist()))}", flush=True)

    # Sigmoid probabilities
    probs = expit(scores.astype(np.float32))  # (N, 234)

    # --- Diel priors: mean prob per (species, hour) ---
    print("\nComputing diel priors (234 × 24) ...", flush=True)
    diel_sum = np.zeros((N_CLASSES, N_HOURS), dtype=np.float64)
    diel_count = np.zeros((N_HOURS,), dtype=np.int64)
    for h in range(N_HOURS):
        mask = hours == h
        if mask.sum() > 0:
            diel_sum[:, h] = probs[mask].sum(axis=0)
            diel_count[h] = mask.sum()

    diel_priors = np.zeros((N_CLASSES, N_HOURS), dtype=np.float32)
    for h in range(N_HOURS):
        if diel_count[h] > 0:
            diel_priors[:, h] = (diel_sum[:, h] / diel_count[h]).astype(np.float32)
        else:
            diel_priors[:, h] = probs.mean(axis=0)  # global fallback

    # --- Site priors: mean prob per (species, site) ---
    print("Computing site priors (234 × n_sites) ...", flush=True)
    site_priors = np.zeros((N_CLASSES, n_sites), dtype=np.float32)
    for s, sidx in site_to_idx.items():
        mask = sites == s
        if mask.sum() > 0:
            site_priors[:, sidx] = probs[mask].mean(axis=0)

    # --- Joint site×hour priors ---
    print("Computing joint site×hour priors (234 × n_sites × 24) ...", flush=True)
    joint_sum = np.zeros((N_CLASSES, n_sites, N_HOURS), dtype=np.float64)
    joint_count = np.zeros((n_sites, N_HOURS), dtype=np.int64)
    for i in range(len(probs)):
        h = int(hours[i])
        sidx = site_to_idx[sites[i]]
        joint_sum[:, sidx, h] += probs[i]
        joint_count[sidx, h] += 1

    joint_priors = np.zeros((N_CLASSES, n_sites, N_HOURS), dtype=np.float32)
    global_mean = probs.mean(axis=0).astype(np.float32)
    for sidx in range(n_sites):
        for h in range(N_HOURS):
            if joint_count[sidx, h] > 0:
                joint_priors[:, sidx, h] = (joint_sum[:, sidx, h] / joint_count[sidx, h]).astype(np.float32)
            else:
                # Fall back to site mean, then diel mean, then global
                site_cnt = joint_count[sidx, :].sum()
                if site_cnt > 0:
                    joint_priors[:, sidx, h] = site_priors[:, sidx]
                else:
                    joint_priors[:, sidx, h] = diel_priors[:, h]

    # --- Save ---
    print(f"\nSaving to {OUT_DIR} ...", flush=True)
    np.save(OUT_DIR / "diel_priors.npy", diel_priors)
    np.save(OUT_DIR / "site_priors_all.npy", site_priors)
    np.save(OUT_DIR / "diel_site_priors.npy", joint_priors)

    meta_out = {
        "hour_counts": {int(h): int(diel_count[h]) for h in range(N_HOURS)},
        "site_names": all_sites,
        "site_to_idx": site_to_idx,
        "global_mean_range": [float(global_mean.min()), float(global_mean.max())],
        "n_windows": int(len(scores)),
        "n_sites": n_sites,
    }
    with open(OUT_DIR / "diel_priors_meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"  diel_priors.npy       : {diel_priors.shape}  range [{diel_priors.min():.3f}, {diel_priors.max():.3f}]")
    print(f"  site_priors_all.npy   : {site_priors.shape}")
    print(f"  diel_site_priors.npy  : {joint_priors.shape}")
    print("\nTop 5 most active species at hour 7 (dawn):")
    dawn = diel_priors[:, 7]
    for i in dawn.argsort()[-5:][::-1]:
        print(f"  species {i:3d}: {dawn[i]:.4f}")
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
