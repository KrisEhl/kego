"""Precompute per-site mean Perch embeddings from training soundscapes.

Samples up to MAX_FILES_PER_SITE soundscapes per site, runs Perch inference,
and computes per-site mean embeddings (1536 dims). Used as static context for
the site-aware ResidualSSMv3 (Track C).

Outputs (saved to $KEGO_PATH_DATA/perch-meta/):
  per_site_profiles.npz:
    - profiles: float32 (n_sites, 1536) -- per-site mean embedding
    - site_names: list of site strings (len n_sites)
    - global_mean: float32 (1536,) -- global mean across all sites

Usage:
    KEGO_PATH_DATA=/home/kristian/projects/kego/data \
    PERCH_MODEL_DIR=/home/kristian/perch_v2_cpu \
    ~/.local/bin/uv run python competitions/birdclef-2026/training/precompute_site_profiles.py
"""

import gc
import os
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf

DATA_ROOT = Path(os.environ.get("KEGO_PATH_DATA", "data"))
SC_DIR = DATA_ROOT / "birdclef" / "birdclef-2026" / "train_soundscapes"
OUT_DIR = DATA_ROOT / "perch-meta"
MODEL_DIR = Path(os.environ.get("PERCH_MODEL_DIR", str(Path.home() / "perch_v2_cpu")))

SR = 32_000
WINDOW_SAMPLES = SR * 5  # 5-second windows
FILE_SAMPLES = SR * 60  # 60-second file
N_WINDOWS = 12
MAX_FILES_PER_SITE = 30  # cap per site for speed (~2h total)
SEED = 42

_SC_RE = re.compile(r"BC2026_Train_(\d+)_(S\w+)_(\d{8})_(\d{6})\.ogg", re.IGNORECASE)


def load_perch():
    """Load Perch v2 TF SavedModel and return the infer function."""
    import tensorflow as tf

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    print(f"Loading Perch from {MODEL_DIR} ...")
    t0 = time.time()
    bc = tf.saved_model.load(str(MODEL_DIR))
    infer_fn = bc.signatures["serving_default"]
    print(f"Perch loaded in {time.time() - t0:.1f}s")
    return infer_fn


def perch_embed_file(infer_fn, path: Path) -> np.ndarray:
    """Return (N_WINDOWS, 1536) float32 Perch embeddings for a 60s soundscape."""
    import tensorflow as tf

    audio, fs = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if fs != SR:
        import resampy

        audio = resampy.resample(audio, fs, SR)
    audio = audio[:FILE_SAMPLES]
    if len(audio) < FILE_SAMPLES:
        audio = np.pad(audio, (0, FILE_SAMPLES - len(audio)))
    windows = audio.reshape(N_WINDOWS, WINDOW_SAMPLES)
    # Batch all windows in one call for efficiency
    wins_tf = tf.constant(windows, dtype=tf.float32)  # (12, WINDOW_SAMPLES)
    out = infer_fn(inputs=wins_tf)
    emb = out["embedding"].numpy()  # (12, 1536)
    return emb.astype(np.float32)


def main():
    # Discover all soundscape files grouped by site
    all_files = sorted(SC_DIR.glob("*.ogg"))
    print(f"Total soundscapes: {len(all_files)}")

    site_to_files: dict[str, list[Path]] = {}
    for p in all_files:
        m = _SC_RE.search(p.name)
        if m:
            site = m.group(2)
            site_to_files.setdefault(site, []).append(p)

    sites = sorted(site_to_files.keys())
    print(f"Sites found: {sites}")

    # Reproducible sampling per site
    rng = np.random.default_rng(SEED)
    sampled: dict[str, list[Path]] = {}
    for site in sites:
        files = site_to_files[site]
        n = min(len(files), MAX_FILES_PER_SITE)
        chosen = rng.choice(len(files), n, replace=False)
        sampled[site] = [files[i] for i in sorted(chosen)]
        print(f"  {site}: {len(files)} total, {n} sampled")

    total_files = sum(len(v) for v in sampled.values())
    print(f"Total files to process: {total_files}")

    model = load_perch()

    profiles = {}
    t0_all = time.time()
    done = 0
    for site in sites:
        site_embs = []
        for path in sampled[site]:
            try:
                emb = perch_embed_file(model, path)  # (12, 1536)
                site_embs.append(emb.mean(axis=0))  # (1536,) mean over windows
            except Exception as e:
                print(f"  WARNING: {path.name} failed: {e}")
            done += 1
            if done % 20 == 0:
                elapsed = time.time() - t0_all
                eta = elapsed / done * (total_files - done)
                print(f"  [{done}/{total_files}] {elapsed / 60:.1f}min elapsed, {eta / 60:.1f}min ETA")
        if site_embs:
            profiles[site] = np.stack(site_embs, axis=0).mean(axis=0)  # (1536,)
            print(f"  {site}: {len(site_embs)} files → mean emb computed")
        gc.collect()

    # Build output arrays
    site_names = sites
    n_sites = len(site_names)
    profiles_arr = np.zeros((n_sites, 1536), dtype=np.float32)
    for i, site in enumerate(site_names):
        if site in profiles:
            profiles_arr[i] = profiles[site]

    global_mean = profiles_arr.mean(axis=0)

    # Fill missing sites with global mean
    for i, site in enumerate(site_names):
        if site not in profiles:
            print(f"  WARNING: {site} has no embeddings, using global mean")
            profiles_arr[i] = global_mean

    out_path = OUT_DIR / "per_site_profiles.npz"
    np.savez(
        out_path,
        profiles=profiles_arr,
        global_mean=global_mean,
        site_names=np.array(site_names),
    )
    print(f"\nSaved: {out_path}")
    print(f"Profiles shape: {profiles_arr.shape}")
    print(f"Sites: {site_names}")
    print(f"Total time: {(time.time() - t0_all) / 60:.1f}min")


if __name__ == "__main__":
    main()
