"""Re-tune the PF-blend post-processing on the ACTUAL 198-feature model.

Audit (2026-05-31) finding: the v23 pp gain is ENTIRELY the PF-blend
(d = d·(1−w) + pf·w). The tau-attenuation is inert here (only ~0.8% of rows
have md_since < tau; median md_since ≈ 2458 ft) and per-well savgol is a
boundary-overfit +0.003. The blend was also tuned on the 212-feat OOF while the
LB kernel runs 198-feat (no div_*). This script rebuilds the OOF WITHOUT the
divergence features (= the kernel that will be submitted) and sweeps only the
blend weight, so the shipped w is calibrated against the actual submitted model.

GPU-free; loads the cached feature parquet — no feature rebuild.
    uv run kego run competitions/rogii-.../tune_pp_blend.py
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

_HERE = Path(__file__).parent
CACHE_DIR = _HERE / "outputs" / "feat_cache"
NON_FEATURES = {"well", "id", "target"}
_mlflow_run_id = os.environ.get("KEGO_MLFLOW_RUN_ID", "")


def log_metric_live(key, value, step=None):
    if _mlflow_run_id:
        from mlflow.tracking import MlflowClient

        MlflowClient().log_metric(_mlflow_run_id, key, float(value), step=step)


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def main() -> None:
    print("KEGO_PARAM experiment pp_blend_retune_198", flush=True)
    ver = hashlib.md5((_HERE / "rogii_features.py").read_bytes()).hexdigest()[:10]
    cache = CACHE_DIR / f"train_{ver}_773w.parquet"
    if not cache.exists():
        raise SystemExit(f"cache {cache} missing — run train_seq_feats.py once to build it")
    print(f"Loading cached features {cache.name}", flush=True)
    df = pd.read_parquet(cache)

    # Drop divergence features → exactly the 198-feat model the LB kernel runs.
    feat = [c for c in df.columns if c not in NON_FEATURES and not c.startswith("div_")]
    n_div = sum(c.startswith("div_") for c in df.columns)
    print(f"KEGO_PARAM no_divergence 1  ({len(feat)} feat, dropped {n_div} div_)", flush=True)
    X = df[feat].to_numpy(np.float32)
    X[~np.isfinite(X)] = np.nan
    y = df["target"].to_numpy(np.float64)
    groups = df["well"].to_numpy()
    pf = df["pf_ancc_delta"].to_numpy(np.float64)  # the PF drift estimate to blend toward
    log_metric_live("progress_pct", 5)

    # 4-fold XGB OOF on the 198-feat model (CPU; kernel config). ~8 min.
    from xgboost import XGBRegressor

    oof = np.full(len(df), np.nan)
    for k, (tr, va) in enumerate(GroupKFold(n_splits=4).split(X, y, groups)):
        m = XGBRegressor(
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
            device="cpu",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        oof[va] = m.predict(X[va])
        print(f"fold {k} done", flush=True)
        log_metric_live("progress_pct", 5 + 70 * (k + 1) / 4)

    np.savez(_HERE / "outputs" / "oof_198.npz", oof=oof, pf=pf, y=y, groups=groups.astype(str))
    base = _rmse(y, oof)
    print(f"BASELINE 198-feat OOF post_ps_rmse = {base:.4f}", flush=True)
    print(f"KEGO_METRIC baseline_oof_198 {base:.6f}", flush=True)
    log_metric_live("progress_pct", 80)

    # Fine blend-only sweep — d = oof*(1-w) + pf*w.
    best_w, best_r = 0.0, base
    for w in np.round(np.arange(0.0, 0.2001, 0.0125), 4):
        r = _rmse(y, oof * (1 - w) + pf * w)
        print(f"  blend w={w:<6}: rmse={r:.4f}  gain={base - r:+.4f}", flush=True)
        if r < best_r:
            best_r, best_w = r, float(w)

    print("\n=== BLEND RE-TUNE RESULT (198-feat) ===", flush=True)
    print(f"baseline      {base:.4f}", flush=True)
    print(f"+blend(w={best_w}) {best_r:.4f}  gain {base - best_r:+.4f}", flush=True)
    print(f"KEGO_METRIC pp_blend_oof {best_r:.6f}", flush=True)
    print(f"KEGO_METRIC pp_blend_gain {base - best_r:.6f}", flush=True)
    log_metric_live("post_ps_rmse", best_r)
    log_metric_live("progress_pct", 100)

    params = {
        "model": "198-feat (no divergence)",
        "baseline": round(base, 4),
        "pp_oof": round(best_r, 4),
        "gain": round(base - best_r, 4),
        "w_pf": best_w,
        "method": "PF-blend only (attenuation + savgol dropped per audit)",
    }
    (_HERE / "outputs" / "pp_blend_198.json").write_text(json.dumps(params, indent=2) + "\n")
    print(f"PP_BLEND_PARAMS {params}", flush=True)


if __name__ == "__main__":
    main()
