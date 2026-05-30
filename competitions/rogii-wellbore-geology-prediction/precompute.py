"""Precompute per-well feature tensors and save as .npz cache.

Run once before training — eliminates CSV parsing and feature engineering
on every training run.

    uv run python competitions/rogii-wellbore-geology-prediction/precompute.py

Cache lives at:  outputs/cache/{train,test}/{well_id}.npz
Each file contains:
    feat            float32 (N, N_FEAT)  — input features
    target          float32 (N,)         — TVT deviation from PS anchor (train only)
    ps              int                  — prediction-start row index
    ps_tvt          float                — TVT at PS
    row_idx         int64 (N,)           — original CSV row indices
    tvt_input_is_nan bool (N,)           — True for post-PS rows (evaluation zone)
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

DATA_DIR = (
    Path(os.environ.get("KEGO_PATH_DATA", _PROJECT_ROOT / "data")) / "rogii" / "rogii-wellbore-geology-prediction"
)
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
CACHE_DIR = Path(__file__).parent / "outputs" / "cache"

# Keep in sync with train_seq.SEQ_FEATURES
SEQ_FEATURES = [
    "gr_norm",
    "z_delta",
    "dx",
    "dy",
    "delta_md",
    "is_post_ps",
    "tvt_dev_known",
]
N_FEAT = len(SEQ_FEATURES)


def _list_well_ids(directory: Path) -> list[str]:
    return sorted(
        m.group(1) for f in directory.iterdir() if (m := re.match(r"^([0-9a-f]+)__horizontal_well\.csv$", f.name))
    )


def precompute_well(wid: str, directory: Path, out_dir: Path) -> None:
    out_path = out_dir / f"{wid}.npz"
    if out_path.exists():
        return  # already cached

    h = pd.read_csv(directory / f"{wid}__horizontal_well.csv")
    ps = int(h["TVT_input"].notna().sum())
    if ps == 0 or ps >= len(h):
        return

    ps_tvt = float(h.iloc[ps - 1]["TVT"] if "TVT" in h.columns else h.iloc[ps - 1]["TVT_input"])
    ps_z = float(h.iloc[ps - 1]["Z"])
    ps_x = float(h.iloc[ps - 1]["X"])
    ps_y = float(h.iloc[ps - 1]["Y"])

    gr = h["GR"].ffill().bfill().fillna(h["GR"].median())
    gr_norm = ((gr - float(gr.median())) / (float(gr.std()) + 1e-6)).values

    n = len(h)
    feat = np.zeros((n, N_FEAT), dtype=np.float32)
    feat[:, 0] = gr_norm
    feat[:, 1] = h["Z"].values - ps_z
    feat[:, 2] = h["X"].values - ps_x
    feat[:, 3] = h["Y"].values - ps_y
    feat[:, 4] = h["MD"].diff().fillna(1.0).values
    feat[:, 5] = np.where(np.arange(n) >= ps, 1.0, 0.0)

    if "TVT" in h.columns:
        tvt_dev = h["TVT"].values - ps_tvt
    else:
        tvt_dev = np.where(h["TVT_input"].notna(), h["TVT_input"].values - ps_tvt, 0.0)
    feat[:, 6] = np.where(np.arange(n) < ps, tvt_dev, 0.0)

    target = (h["TVT"].values - ps_tvt).astype(np.float32) if "TVT" in h.columns else None

    kwargs: dict = dict(
        feat=feat,
        ps=np.array(ps, dtype=np.int64),
        ps_tvt=np.array(ps_tvt, dtype=np.float64),
        row_idx=np.arange(n, dtype=np.int64),
        tvt_input_is_nan=h["TVT_input"].isna().values,
    )
    if target is not None:
        kwargs["target"] = target

    np.savez_compressed(out_path, **kwargs)


def main() -> None:
    for split, directory in [("train", TRAIN_DIR), ("test", TEST_DIR)]:
        out_dir = CACHE_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)
        well_ids = _list_well_ids(directory)
        print(f"Precomputing {len(well_ids)} {split} wells → {out_dir}", flush=True)
        t0 = time.time()
        for i, wid in enumerate(well_ids):
            precompute_well(wid, directory, out_dir)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(well_ids)}  {time.time() - t0:.0f}s", flush=True)
        elapsed = time.time() - t0
        cached = sum(1 for _ in out_dir.glob("*.npz"))
        total_mb = sum(f.stat().st_size for f in out_dir.glob("*.npz")) / 1e6
        print(f"  Done: {cached} files, {total_mb:.0f} MB, {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
