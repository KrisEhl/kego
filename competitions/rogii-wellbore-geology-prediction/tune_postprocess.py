"""Tune the reference's drift-attenuation + PF-blend + Savitzky-Golay POST-PROCESSING
on a 4-fold XGB OOF (GPU-free; loads the cached feature parquet — no rebuild).

The audit (2026-05-31) identified this as the highest-EV untried lever toward the
reference's 8.905: it operates on the FINAL drift predictions, so it's structural
(above the ~0.05 CV-noise floor), and the reference Optuna-tunes it on OOF.

    pp(d) = per_well_savgol[ (d*(1-w) + pf*w) * (1 - exp(-md_since/tau)) * alpha ]

Stage A grids (alpha, tau, w) vectorised on the OOF array (fast). Stage B adds
per-well savgol windows on the Stage-A best. Reports baseline vs best OOF post-PS RMSE.

kego run compatible:  uv run kego run competitions/rogii-.../tune_postprocess.py
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
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
    print("KEGO_PARAM experiment postprocess_tune", flush=True)
    ver = hashlib.md5((_HERE / "rogii_features.py").read_bytes()).hexdigest()[:10]
    cache = CACHE_DIR / f"train_{ver}_773w.parquet"
    if not cache.exists():
        raise SystemExit(f"cache {cache} missing — run train_seq_feats.py once to build it")
    print(f"Loading cached features {cache.name}", flush=True)
    df = pd.read_parquet(cache)
    feat = [c for c in df.columns if c not in NON_FEATURES]
    X = df[feat].to_numpy(np.float32)
    X[~np.isfinite(X)] = np.nan
    y = df["target"].to_numpy(np.float64)
    groups = df["well"].to_numpy()
    md_since = df["md_since"].to_numpy(np.float64)
    pf = df["pf_ancc_delta"].to_numpy(np.float64)  # PF drift estimate
    log_metric_live("progress_pct", 10)

    # 4-fold XGB OOF (CPU; v13c config). This is the only slow part (~8min).
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
        log_metric_live("progress_pct", 10 + 60 * (k + 1) / 4)

    # Save OOF + aux so future pp tuning skips the ~8min xgb retrain.
    np.savez(_HERE / "outputs" / "oof_v23.npz", oof=oof, md_since=md_since, pf=pf, y=y, groups=groups)
    base = _rmse(y, oof)
    print(f"BASELINE OOF post_ps_rmse = {base:.4f}", flush=True)
    print(f"KEGO_METRIC baseline_oof {base:.6f}", flush=True)

    # ---- Stage A: vectorised grid over (alpha, tau, w_pf) — attenuation + PF-blend ----
    def atten(tau):  # 1 - exp(-md_since/tau): drift shrinks to 0 near the anchor (md_since→0)
        return 1.0 - np.exp(-md_since / tau)

    best = (base, 1.0, np.inf, 0.0)  # (rmse, alpha, tau, w)
    for w in np.round(np.arange(0.0, 0.51, 0.05), 3):
        blend = oof * (1 - w) + pf * w
        for tau in np.unique(np.round(np.logspace(np.log10(20), np.log10(600), 16))):
            bt = blend * atten(tau)
            for alpha in np.round(np.arange(0.70, 1.001, 0.025), 4):
                r = _rmse(y, bt * alpha)
                if r < best[0]:
                    best = (r, float(alpha), float(tau), float(w))
    r_a, alpha, tau, w = best
    print(f"Stage A best: rmse={r_a:.4f} alpha={alpha} tau={tau} w_pf={w} (vs base {base:.4f})", flush=True)
    log_metric_live("progress_pct", 80)

    # ---- Stage B: per-well Savitzky-Golay on the Stage-A best ----
    dA = (oof * (1 - w) + pf * w) * atten(tau) * alpha
    well_rows = {wid: np.where(groups == wid)[0] for wid in np.unique(groups)}
    r_b, sg_best = r_a, 0
    for win in [0, 7, 11, 15, 21, 31]:
        if win == 0:
            continue
        dS = dA.copy()
        for rows in well_rows.values():
            n = len(rows)
            wl = min(win, n if n % 2 else n - 1)
            if wl >= 5:  # need wl > polyorder(3)
                dS[rows] = savgol_filter(dA[rows], wl, 3)
        r = _rmse(y, dS)
        print(f"  savgol win={win}: rmse={r:.4f}", flush=True)
        if r < r_b:
            r_b, sg_best = r, win

    print("\n=== POST-PROCESS RESULT ===", flush=True)
    print(f"baseline           {base:.4f}", flush=True)
    print(f"+attenuation/blend {r_a:.4f}  (alpha={alpha} tau={tau:.0f} w_pf={w})", flush=True)
    print(f"+savgol(win={sg_best}) {r_b:.4f}", flush=True)
    print(f"TOTAL gain         {base - r_b:+.4f}", flush=True)
    print(f"KEGO_METRIC pp_oof {r_b:.6f}", flush=True)
    print(f"KEGO_METRIC pp_gain {base - r_b:.6f}", flush=True)
    log_metric_live("post_ps_rmse", r_b)
    log_metric_live("progress_pct", 100)

    # Persist best params (robust capture for applying pp in the kernel + judging overfit).
    import json

    params = {
        "baseline": round(base, 4),
        "stageA": round(r_a, 4),
        "pp_oof": round(r_b, 4),
        "gain": round(base - r_b, 4),
        "alpha": alpha,
        "tau": round(tau, 1),
        "w_pf": w,
        "savgol_win": sg_best,
    }
    (_HERE / "outputs" / "pp_best.json").write_text(json.dumps(params, indent=2) + "\n")
    print(f"PP_PARAMS {params}", flush=True)


if __name__ == "__main__":
    main()
