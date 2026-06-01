# ruff: noqa: F821
# (names build_dataset/FormationPlaneKNN/DenseANCCImputer/_warmup_numba are
#  defined in rogii_features.py, prepended by build_kernel.sh — undefined here
#  because this file is a fragment, not a standalone module.)
# ============================================================================
# Kaggle inference entrypoint — APPENDED to rogii_features.py by build_kernel.sh
# to form the single-file `inference_kernel.py` (Kaggle script kernels are
# single-file). Trains the seqfeats pipeline (divergence features dropped in _xy
# -> ~195 feat, matching the LB-10.538 anchor's no-divergence set) in-kernel on
# CPU and writes submission.csv. Reproduces train_seq_feats.py: GroupKFold(4),
# 4 models averaged, then PF-blend post-processing (w=0.10, -0.054 ft OOF).
#
# enable_gpu=false on Kaggle → device="cpu". Set ROGII_KERNEL_DEBUG=1 for a
# fast local end-to-end smoke (few wells, 50 trees).
# ============================================================================
import glob as _glob
import os as _os
from pathlib import Path as _Path

import numpy as _np
import pandas as _pd
from scipy.optimize import nnls as _nnls
from sklearn.model_selection import GroupKFold as _GroupKFold
from xgboost import XGBRegressor as _XGBRegressor

_NON_FEATURES = {"well", "id", "target"}
# v25 kinematic/dip cols — dropped (neutral; keep kernel at the 195-feat LB-10.105 config).
_KIN_COLS = {
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
_DEBUG = _os.environ.get("ROGII_KERNEL_DEBUG") == "1"

# Post-processing: blend the XGB drift toward median(pf_ancc_delta, beam_med_d, tvt_dense_d)
# at w=0.125. MECHANISM (audit-corrected): beam_med_d/tvt_dense_d are individually WEAK blend
# targets — the median is NOT a consensus of 3 good estimators; it acts as a ROBUST CLIPPER of
# pf_ancc's extreme drift corrections (the 2 conservative estimators bound pf's over-corrections).
# JUSTIFICATION: the disjoint-half transfer test (tune w on half the wells, eval on the disjoint
# half — the methodology that blessed the original blend) shows median3 transfers robustly
# (mean +0.026, 0/12 splits negative) while pf-alone OVERFITS its w-optimum (mean +0.017, 4/12
# negative). The blend is the one lever with proven LB traction (10.538->10.105) and OOF-directional.
# All three are input drift estimates (MD/geometry/known-zone GR + type-well log) -> no leakage.
_PP = {"w": 0.125}
_CONSENSUS = ("pf_ancc_delta", "beam_med_d", "tvt_dense_d")


def _postprocess(drift, consensus):
    return drift * (1 - _PP["w"]) + consensus * _PP["w"]


def _find_data_dir() -> _Path:
    """Locate the competition data dir (train/ + test/) on Kaggle or locally."""
    explicit = [
        "/kaggle/input/competitions/rogii-wellbore-geology-prediction",
        "/kaggle/input/rogii-wellbore-geology-prediction",
    ]
    for c in explicit:
        if _os.path.isdir(_os.path.join(c, "test")):
            return _Path(c)
    for c in _glob.glob("/kaggle/input/*"):
        if _os.path.isdir(_os.path.join(c, "test")) and _os.path.isdir(_os.path.join(c, "train")):
            return _Path(c)
    local = _os.path.join(_os.environ.get("KEGO_PATH_DATA", "data"), "rogii", "rogii-wellbore-geology-prediction")
    return _Path(local)


def _xy(df: "_pd.DataFrame"):
    # Drop div_* (divergence) features: the LB-10.538 anchor was trained WITHOUT them
    # and a 5-seed A/B found them neutral. Also drop the v25 kinematic/dip cols (also
    # neutral, mean Δ +0.015). Keeps the model == the 195-feat config that scored LB 10.105
    # (depth7 + pf_ancc blend w=0.10) — the current banked best. (depth6 explored, audit-FAILed
    # the submission as sub-LB-noise; not applied. See README "Strategy after the 10.105 audit".)
    feat = [
        c
        for c in df.columns
        if c not in _NON_FEATURES
        and not c.startswith("div_")
        and c not in _KIN_COLS
        and not c.startswith("gr_dwt")  # DWT neutral (v35)
        and c != "ncc_tw_delta"  # typewell-NCC neutral (v39); drop to match the 195-feat training config
    ]
    X = df[feat].to_numpy(_np.float32)
    X[~_np.isfinite(X)] = _np.nan
    return feat, X


# v42: PF particle multiplier — the variance-reduced pf_ancc (the strongest feature) is a
# real OOF win that scales monotonically (mult1 10.50 > mult2 10.39 > mult4 10.27, all 3-seed,
# leakage-audited). The kernel MUST set this (run_pf_ancc reads ROGII_PF_MULT) or it ships the
# mult1 loser. Kaggle code limit ~9h; mult4 in-kernel build est ~2h (4× the ~12min mult1 build)
# + CPU GBDT ~1h -> fits. Start conservative (2) to validate the path+budget, then raise to 4.
_PF_MULT = "2"


def _main() -> None:
    _os.environ.setdefault("ROGII_PF_MULT", _PF_MULT)
    print(f"ROGII_PF_MULT={_os.environ.get('ROGII_PF_MULT')}", flush=True)
    data = _find_data_dir()
    train_dir, test_dir = data / "train", data / "test"
    out_dir = _Path("/kaggle/working") if _os.path.isdir("/kaggle/working") else _Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"data={data}  out={out_dir}  debug={_DEBUG}", flush=True)

    _warmup_numba()
    hw_paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    train_wids = [p.stem.replace("__horizontal_well", "") for p in hw_paths]
    if _DEBUG:
        train_wids = train_wids[:12]
        hw_paths = [train_dir / f"{w}__horizontal_well.csv" for w in train_wids]

    print(f"Fitting imputers on {len(train_wids)} wells...", flush=True)
    FI = FormationPlaneKNN(train_wids, train_dir)
    DI = DenseANCCImputer(train_wids, train_dir)

    print("Building train features...", flush=True)
    df = build_dataset(hw_paths, True, FI, DI, n_jobs=4)
    feat, X = _xy(df)
    y = df["target"].to_numpy(_np.float64)
    groups = df["well"].to_numpy()
    print(f"train: {len(df):,} rows x {len(feat)} feat", flush=True)

    n_est = 50 if _DEBUG else 3000
    # v31: NNLS ensemble of XGB + CatBoost (-0.150 OOF vs single XGB; 3-seed confirmed; structural).
    # CatBoost wrapped defensively: if absent in the Kaggle env (enable_internet=false), fall back
    # to single XGB so a missing dep can't blank the score (submission stays valid).
    _cb_ok = False
    try:
        from catboost import CatBoostRegressor as _CatBoost

        _cb_ok = True
    except Exception as _e:
        print(f"CatBoost unavailable ({_e}) -> single-XGB fallback", flush=True)

    def _mk_xgb():
        return _XGBRegressor(
            n_estimators=n_est,
            learning_rate=0.03,
            # depth7 = the on-board v31 ensemble config. depth6 (v36, -0.036 OOF, 3-seed) was
            # audit-FAILED for shipping: -0.036 is within ~1 std, 3-seed < the project's >=5-seed
            # rule, and v37 is the smoking gun (a 3-seed -0.035 reg win flipped to +0.056 at 5 seeds).
            # depth6 is DIRECTIONALLY better (pooled v27+v36 6/6, sign-test p=0.031) -> re-apply +
            # ship ONLY after >=5 paired seeds confirm it (needs the cluster).
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

    oof_x = _np.full(len(y), _np.nan)
    oof_c = _np.full(len(y), _np.nan)
    xgb_models, cb_models = [], []
    for tr, va in _GroupKFold(n_splits=4).split(X, y, groups):
        mx = _mk_xgb()
        mx.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        oof_x[va] = mx.predict(X[va])
        xgb_models.append(mx)
        if _cb_ok:
            mc = _CatBoost(
                iterations=n_est,
                learning_rate=0.03,
                depth=7,  # on-board v31 config; depth6 pending >=5-seed confirmation (see XGB note above)
                l2_leaf_reg=3.0,
                loss_function="RMSE",
                od_type="Iter",
                od_wait=80,
                task_type="CPU",
                random_seed=42,
                verbose=False,
            )
            mc.fit(X[tr], y[tr], eval_set=(X[va], y[va]), use_best_model=True)
            oof_c[va] = mc.predict(X[va])
            cb_models.append(mc)
    if _cb_ok:
        A = _np.column_stack([oof_x, oof_c]).astype(_np.float64)
        w, _ = _nnls(A, y.astype(_np.float64))
        print(f"NNLS weights: xgb={w[0]:.4f} cb={w[1]:.4f}", flush=True)
    else:
        w = None
    print(f"trained {len(xgb_models)} xgb + {len(cb_models)} cb fold models", flush=True)

    test_paths = sorted(test_dir.glob("*__horizontal_well.csv"))
    df_te = build_dataset(test_paths, False, FI, DI, n_jobs=4)
    # Index test by TRAIN feature columns (same order); reindex fills any column
    # the test build didn't produce with NaN (xgboost handles NaN). Critical so the
    # model sees exactly the columns it was trained on, even on the hidden test set.
    Xte = df_te.reindex(columns=feat).to_numpy(_np.float32)
    Xte[~_np.isfinite(Xte)] = _np.nan
    xgb_te = _np.mean([m.predict(Xte) for m in xgb_models], axis=0)
    if w is not None:
        cb_te = _np.mean([m.predict(Xte) for m in cb_models], axis=0)
        drift = w[0] * xgb_te + w[1] * cb_te
    else:
        drift = xgb_te
    # NO consensus blend here: v32 validation showed the blend does NOT transfer on the ENSEMBLE base
    # (disjoint-half transfer mean -0.046, 8/12 splits negative) — the ensemble already incorporates
    # pf/beam/dense (inputs) + NNLS de-shrink conflicts with the blend's shrink. So ensemble-ONLY.
    # (The blend stays on the single-XGB kernels (LB 10.105 / consensus v3) where it DOES transfer.)
    sub = _pd.DataFrame({"id": df_te["id"], "tvt": df_te["last_known_tvt"].to_numpy() + drift})
    sub.to_csv(out_dir / "submission.csv", index=False)
    print(f"Wrote {len(sub):,} rows -> {out_dir / 'submission.csv'}", flush=True)


if __name__ == "__main__":
    _main()
