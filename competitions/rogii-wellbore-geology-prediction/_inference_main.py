# ruff: noqa: F821
# (names build_dataset/FormationPlaneKNN/DenseANCCImputer/_warmup_numba are
#  defined in rogii_features.py, prepended by build_kernel.sh — undefined here
#  because this file is a fragment, not a standalone module.)
# ============================================================================
# Kaggle inference entrypoint — APPENDED to rogii_features.py by build_kernel.sh
# to form the single-file `inference_kernel.py` (Kaggle script kernels are
# single-file). Trains the 198-feature seqfeats pipeline in-kernel on CPU and
# writes submission.csv. Reproduces train_seq_feats.py exactly: GroupKFold(4),
# 4 models averaged for the test prediction (= the validated OOF-10.62 setup).
#
# enable_gpu=false on Kaggle → device="cpu". Set ROGII_KERNEL_DEBUG=1 for a
# fast local end-to-end smoke (few wells, 50 trees).
# ============================================================================
import glob as _glob
import os as _os
from pathlib import Path as _Path

import numpy as _np
import pandas as _pd
from scipy.signal import savgol_filter as _savgol
from sklearn.model_selection import GroupKFold as _GroupKFold
from xgboost import XGBRegressor as _XGBRegressor

_NON_FEATURES = {"well", "id", "target"}
_DEBUG = _os.environ.get("ROGII_KERNEL_DEBUG") == "1"

# Post-processing tuned on the v23 4-fold OOF (CPU): -0.041 ft. tau=39 drift-attenuation
# (suppresses drift within ~39ft of the anchor — physical prior) is the main lever;
# w_pf blend + per-well savgol are minor. md_since/pf are inputs → no leakage.
_PP = {"alpha": 1.0, "tau": 39.0, "w_pf": 0.1, "savgol_win": 31}


def _postprocess(drift, md_since, pf, well_ids):
    d = drift * (1 - _PP["w_pf"]) + pf * _PP["w_pf"]
    d = d * (1 - _np.exp(-md_since / _PP["tau"])) * _PP["alpha"]
    out = d.copy()
    for wid in _np.unique(well_ids):
        rows = _np.where(well_ids == wid)[0]
        n = len(rows)
        wl = min(_PP["savgol_win"], n if n % 2 else n - 1)
        if wl >= 5:
            out[rows] = _savgol(d[rows], wl, 3)
    return out


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
    feat = [c for c in df.columns if c not in _NON_FEATURES]
    X = df[feat].to_numpy(_np.float32)
    X[~_np.isfinite(X)] = _np.nan
    return feat, X


def _main() -> None:
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
    models = []
    for tr, va in _GroupKFold(n_splits=4).split(X, y, groups):
        m = _XGBRegressor(
            n_estimators=n_est,
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
        models.append(m)
    print(f"trained {len(models)} fold models", flush=True)

    test_paths = sorted(test_dir.glob("*__horizontal_well.csv"))
    df_te = build_dataset(test_paths, False, FI, DI, n_jobs=4)
    # Index test by TRAIN feature columns (same order); reindex fills any column
    # the test build didn't produce with NaN (xgboost handles NaN). Critical so the
    # model sees exactly the columns it was trained on, even on the hidden test set.
    Xte = df_te.reindex(columns=feat).to_numpy(_np.float32)
    Xte[~_np.isfinite(Xte)] = _np.nan
    drift = _np.mean([m.predict(Xte) for m in models], axis=0)
    # v23 post-processing (tau=39 drift-attenuation + PF-blend + per-well savgol): -0.041 OOF
    drift = _postprocess(
        drift,
        df_te["md_since"].to_numpy(_np.float64),
        df_te["pf_ancc_delta"].to_numpy(_np.float64),
        df_te["well"].to_numpy(),
    )
    sub = _pd.DataFrame({"id": df_te["id"], "tvt": df_te["last_known_tvt"].to_numpy() + drift})
    sub.to_csv(out_dir / "submission.csv", index=False)
    print(f"Wrote {len(sub):,} rows -> {out_dir / 'submission.csv'}", flush=True)


if __name__ == "__main__":
    _main()
