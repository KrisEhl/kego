"""Validate the ensemble+blend STACK: build the XGB+CatBoost NNLS ensemble OOF, then
re-tune the consensus blend ON THE ENSEMBLE BASE (audit 2026-05-31 flagged that w=0.125
was tuned on single-XGB OOF — applying it to the ensemble is unvalidated + may double-count
since pf/beam/dense are already model inputs and the NNLS sum ~1.07 inflates drift while the
blend shrinks it). Saves the ensemble OOF (never persisted before). GPU-free-ish (CatBoost
CPU/GPU auto). Reuses the cached feature parquet.

    uv run kego run competitions/rogii-.../tune_ensemble_blend.py [--target cluster]
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

_HERE = Path(__file__).parent
CACHE_DIR = _HERE / "outputs" / "feat_cache"
NON_FEATURES = {"well", "id", "target"}
KIN = {
    "incl_deg",
    "azi_deg",
    "dls",
    "build_rate",
    "cos_incl",
    "sin_incl",
    "tvt_dip_full_d",
    "tvt_dip_late_d",
    "azi_delta",
    "apparent_dip_dir",
    "b_dip_full",
    "b_dip_late",
    "b_dip_early",
    "b_dip_slope",
    "plane_dip_x",
    "plane_dip_y",
}
_mlflow_run_id = os.environ.get("KEGO_MLFLOW_RUN_ID", "")


def log_metric_live(key, value, step=None):
    if _mlflow_run_id:
        from mlflow.tracking import MlflowClient

        MlflowClient().log_metric(_mlflow_run_id, key, float(value), step=step)


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def _device():
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> None:
    print("KEGO_PARAM experiment ensemble_blend_validate", flush=True)
    ver = hashlib.md5((_HERE / "rogii_features.py").read_bytes()).hexdigest()[:10]
    cache = CACHE_DIR / f"train_{ver}_773w.parquet"
    if not cache.exists():
        raise SystemExit(f"cache {cache} missing")
    df = pd.read_parquet(cache)
    feat = [c for c in df.columns if c not in NON_FEATURES and not c.startswith("div_") and c not in KIN]
    X = df[feat].to_numpy(np.float32)
    X[~np.isfinite(X)] = np.nan
    y = df["target"].to_numpy(np.float64)
    groups = df["well"].to_numpy()
    pf = df["pf_ancc_delta"].to_numpy(np.float64)
    beam = df["beam_med_d"].to_numpy(np.float64)
    dense = df["tvt_dense_d"].to_numpy(np.float64)
    dev = _device()
    print(f"KEGO_PARAM feat {len(feat)}  device {dev}", flush=True)
    log_metric_live("progress_pct", 5)

    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor

    oof_x = np.full(len(y), np.nan)
    oof_c = np.full(len(y), np.nan)
    for k, (tr, va) in enumerate(GroupKFold(n_splits=4).split(X, y, groups)):
        mx = XGBRegressor(
            n_estimators=3000,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            early_stopping_rounds=80,
            tree_method="hist",
            device=dev,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        mx.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        oof_x[va] = mx.predict(X[va])
        mc = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.03,
            depth=7,
            l2_leaf_reg=3.0,
            loss_function="RMSE",
            od_type="Iter",
            od_wait=80,
            task_type="GPU" if dev == "cuda" else "CPU",
            random_seed=42,
            verbose=False,
        )
        mc.fit(X[tr], y[tr], eval_set=(X[va], y[va]), use_best_model=True)
        oof_c[va] = mc.predict(X[va])
        print(f"fold {k} done", flush=True)
        log_metric_live("progress_pct", 5 + 70 * (k + 1) / 4)

    w, _ = nnls(np.column_stack([oof_x, oof_c]).astype(np.float64), y)
    ens = w[0] * oof_x + w[1] * oof_c
    np.savez(
        _HERE / "outputs" / "oof_ensemble.npz",
        ens=ens,
        oof_x=oof_x,
        oof_c=oof_c,
        y=y,
        groups=groups.astype(str),
        pf=pf,
        beam=beam,
        dense=dense,
        w=w,
    )
    base_x = _rmse(y, oof_x)
    base_e = _rmse(y, ens)
    print(
        f"single XGB OOF={base_x:.4f}  ensemble OOF={base_e:.4f}  NNLS w=[{w[0]:.3f},{w[1]:.3f}] sum={w.sum():.3f}",
        flush=True,
    )
    print(f"KEGO_METRIC ensemble_oof {base_e:.6f}", flush=True)

    consensus = np.median(np.column_stack([pf, beam, dense]), axis=1)
    ws = np.round(np.arange(0.0, 0.2501, 0.0125), 4)
    print("=== consensus-blend sweep ON THE ENSEMBLE BASE ===", flush=True)
    best_w, best_r = 0.0, base_e
    for wv in ws:
        r = _rmse(y, ens * (1 - wv) + consensus * wv)
        if wv in (0.0, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2):
            print(f"  w={wv:.4f}: rmse={r:.4f}  gain_vs_ens={base_e - r:+.4f}", flush=True)
        if r < best_r:
            best_r, best_w = r, float(wv)
    print(f"BEST blend on ensemble: w={best_w} rmse={best_r:.4f} (gain vs ens {base_e - best_r:+.4f})", flush=True)

    # Disjoint-half transfer of the consensus blend on the ENSEMBLE base.
    def _transfer(n_splits=12):
        wells = np.unique(groups)
        out = []
        for s in range(n_splits):
            rng = np.random.default_rng(2000 + s)
            tune = set(wells[rng.permutation(len(wells))[: len(wells) // 2]].tolist())
            mA = np.array([g in tune for g in groups])
            mB = ~mA
            bA = _rmse(y[mA], ens[mA])
            bw, bg = 0.0, 0.0
            for wv in ws:
                g = bA - _rmse(y[mA], ens[mA] * (1 - wv) + consensus[mA] * wv)
                if g > bg:
                    bg, bw = g, wv
            out.append(_rmse(y[mB], ens[mB]) - _rmse(y[mB], ens[mB] * (1 - bw) + consensus[mB] * bw))
        return np.array(out)

    g = _transfer()
    print(
        f"TRANSFER (blend on ensemble): mean={g.mean():+.4f} min={g.min():+.4f} n_neg={int((g < 0).sum())}/{len(g)}",
        flush=True,
    )
    log_metric_live("post_ps_rmse", best_r)
    log_metric_live("progress_pct", 100)
    params = {
        "single_xgb_oof": round(base_x, 4),
        "ensemble_oof": round(base_e, 4),
        "nnls_w": [round(float(w[0]), 4), round(float(w[1]), 4)],
        "best_blend_w_on_ensemble": best_w,
        "best_blend_rmse": round(best_r, 4),
        "blend_gain_vs_ensemble": round(base_e - best_r, 4),
        "transfer_mean": round(float(g.mean()), 4),
        "transfer_n_neg": int((g < 0).sum()),
    }
    (_HERE / "outputs" / "ensemble_blend.json").write_text(json.dumps(params, indent=2) + "\n")
    print(f"ENSEMBLE_BLEND {params}", flush=True)


if __name__ == "__main__":
    main()
