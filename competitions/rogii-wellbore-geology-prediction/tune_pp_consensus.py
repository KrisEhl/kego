"""Improve the PROVEN LB lever (the PF-blend) OOF-validated.

LB BEST 10.105 came from blending the XGB drift 10% toward pf_ancc_delta. Audit
(2026-05-31): the blend is the ONLY lever with demonstrated LB traction (-0.433),
and its OOF is *directionally* predictive (unlike depth6/features where OOF is
inverted). So improving the blend on OOF is the disciplined next move.

This tests blending the 195-feat XGB drift toward a ROBUST CONSENSUS of independent
drift estimators (median/mean of pf_ancc_delta, beam_med_d, tvt_dense_d, form_mean_d,
sc_ens_d) vs pf_ancc-alone, with a w-sweep. A consensus is structurally more robust
(less reliant on one estimator that may be lucky on the 4 public wells) -> better
for the PRIVATE LB. Recomputes the 4-fold OOF on the current cache for clean alignment.

    uv run kego run competitions/rogii-.../tune_pp_consensus.py
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
# Independent drift estimators (all "estimate - last_tvt", same units as the XGB drift target).
ESTIMATORS = ["pf_ancc_delta", "beam_med_d", "tvt_dense_d", "form_mean_d", "sc_ens_d"]
_mlflow_run_id = os.environ.get("KEGO_MLFLOW_RUN_ID", "")


def log_metric_live(key, value, step=None):
    if _mlflow_run_id:
        from mlflow.tracking import MlflowClient

        MlflowClient().log_metric(_mlflow_run_id, key, float(value), step=step)


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def main() -> None:
    print("KEGO_PARAM experiment pp_consensus_blend", flush=True)
    ver = hashlib.md5((_HERE / "rogii_features.py").read_bytes()).hexdigest()[:10]
    cache = CACHE_DIR / f"train_{ver}_773w.parquet"
    if not cache.exists():
        raise SystemExit(f"cache {cache} missing — run train_seq_feats.py once to build it")
    print(f"Loading cached features {cache.name}", flush=True)
    df = pd.read_parquet(cache)
    feat = [c for c in df.columns if c not in NON_FEATURES and not c.startswith("div_") and c not in KIN]
    X = df[feat].to_numpy(np.float32)
    X[~np.isfinite(X)] = np.nan
    y = df["target"].to_numpy(np.float64)
    groups = df["well"].to_numpy()
    print(f"KEGO_PARAM feat {len(feat)}  (195-feat kernel config)", flush=True)
    log_metric_live("progress_pct", 5)

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

    base = _rmse(y, oof)
    print(f"BASELINE OOF (no blend) = {base:.4f}", flush=True)
    print(f"KEGO_METRIC baseline_oof {base:.6f}", flush=True)

    E = {e: df[e].to_numpy(np.float64) for e in ESTIMATORS}
    stack3 = np.column_stack([E["pf_ancc_delta"], E["beam_med_d"], E["tvt_dense_d"]])
    stack5 = np.column_stack([E[e] for e in ESTIMATORS])
    # Save OOF + estimators so future blend analysis skips the ~8min retrain.
    np.savez(
        _HERE / "outputs" / "oof_consensus.npz",
        oof=oof,
        y=y,
        groups=groups.astype(str),
        pf=E["pf_ancc_delta"],
        beam=E["beam_med_d"],
        dense=E["tvt_dense_d"],
    )
    # Candidate blend targets (the thing we shrink the XGB drift toward).
    targets = {
        "pf_ancc (current 10.105)": E["pf_ancc_delta"],
        "mean(pf,beam,dense)": stack3.mean(axis=1),
        "median(pf,beam,dense)": np.median(stack3, axis=1),
        "mean(all5)": stack5.mean(axis=1),
        "median(all5)": np.median(stack5, axis=1),
    }
    ws = np.round(np.arange(0.025, 0.3001, 0.025), 4)
    results = {}
    for name, tgt in targets.items():
        best_w, best_r = 0.0, base
        row = []
        for w in ws:
            r = _rmse(y, oof * (1 - w) + tgt * w)
            row.append((float(w), r))
            if r < best_r:
                best_r, best_w = r, float(w)
        results[name] = {"best_w": best_w, "best_rmse": round(best_r, 4), "gain": round(base - best_r, 4)}
        print(f"{name:28s} best w={best_w:.3f}  rmse={best_r:.4f}  gain={base - best_r:+.4f}", flush=True)
        # full curve for the two most interesting
        if "median(pf,beam,dense)" in name or "pf_ancc" in name:
            print("   " + "  ".join(f"w{w:.3f}={r:.4f}" for w, r in row), flush=True)
    log_metric_live("progress_pct", 95)

    # ---- Disjoint-half transfer test (the README's blessed methodology): tune w on half the
    # wells, eval the gain on the disjoint half. A blend that OVERFITS its w-optimum shows
    # negative/unstable transfer; a robust one transfers positively. pf-alone vs median(3). ----
    def _transfer(target, n_splits=12):
        wells = np.unique(groups)
        out = []
        for s in range(n_splits):
            rng = np.random.default_rng(1000 + s)
            perm = rng.permutation(len(wells))
            tune_set = set(wells[perm[: len(wells) // 2]].tolist())
            mA = np.array([g in tune_set for g in groups])
            mB = ~mA
            baseA = _rmse(y[mA], oof[mA])
            best_w, best_g = 0.0, 0.0
            for w in ws:
                g = baseA - _rmse(y[mA], oof[mA] * (1 - w) + target[mA] * w)
                if g > best_g:
                    best_g, best_w = g, w
            gB = _rmse(y[mB], oof[mB]) - _rmse(y[mB], oof[mB] * (1 - best_w) + target[mB] * best_w)
            out.append(gB)
        return np.array(out)

    print("\n=== DISJOINT-HALF TRANSFER (tune w on half wells, eval gain on disjoint half) ===", flush=True)
    for nm, tgt in [("pf_ancc-alone", E["pf_ancc_delta"]), ("median(pf,beam,dense)", np.median(stack3, axis=1))]:
        g = _transfer(tgt)
        print(
            f"{nm:22s} mean={g.mean():+.4f}  min={g.min():+.4f}  n_negative={int((g < 0).sum())}/{len(g)}", flush=True
        )

    best_name = min(results, key=lambda k: results[k]["best_rmse"])
    print("\n=== CONSENSUS-BLEND RESULT ===", flush=True)
    print(f"baseline (no blend)  {base:.4f}", flush=True)
    for name, r in results.items():
        tag = "  <-- BEST" if name == best_name else ""
        print(f"{name:28s} {r['best_rmse']:.4f} (w={r['best_w']:.3f}, {r['gain']:+.4f}){tag}", flush=True)
    print(f"KEGO_METRIC consensus_best_oof {results[best_name]['best_rmse']:.6f}", flush=True)
    print(f"KEGO_METRIC consensus_best_gain {results[best_name]['gain']:.6f}", flush=True)
    log_metric_live("post_ps_rmse", results[best_name]["best_rmse"])
    log_metric_live("progress_pct", 100)
    (_HERE / "outputs" / "pp_consensus.json").write_text(
        json.dumps({"baseline": round(base, 4), "best": best_name, "results": results}, indent=2) + "\n"
    )
    print(f"PP_CONSENSUS best={best_name} {results[best_name]}", flush=True)


if __name__ == "__main__":
    main()
