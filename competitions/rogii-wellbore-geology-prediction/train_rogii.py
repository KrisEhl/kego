"""Train Rogii wellbore geology prediction — TVT regression.

kego run compatible:
    uv run kego run competitions/rogii-wellbore-geology-prediction/train_rogii.py
    uv run kego run ... --model lightgbm --folds 5 --seed 42
    uv run kego run ... --fold 0 --target cluster   # single-fold fan-out

Stdout protocol (parsed by kego runner → MLflow):
    KEGO_METRIC <name> <value>
    KEGO_PARAM  <name> <value>

Figures are logged directly via mlflow.tracking.MlflowClient when both
KEGO_MLFLOW_RUN_ID and MLFLOW_TRACKING_URI are present in the environment.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

# Live MLflow logging — visible in UI while the job is running
_mlflow_run_id = os.environ.get("KEGO_MLFLOW_RUN_ID", "")


def log_metric_live(key: str, value: float, step: int | None = None) -> None:
    if _mlflow_run_id:
        from mlflow.tracking import MlflowClient

        MlflowClient().log_metric(_mlflow_run_id, key, value, step=step)


DATA_DIR = (
    Path(os.environ.get("KEGO_PATH_DATA", _PROJECT_ROOT / "data")) / "rogii" / "rogii-wellbore-geology-prediction"
)
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = Path(__file__).parent / "outputs"
TARGET = "TVT"


# ── Model registry ─────────────────────────────────────────────────────────────
# To add a new architecture: define a factory and append an entry.
#
#   factory(seed, debug) → sklearn-compatible estimator with .fit() / .predict()
#   early_stopping       → True for GBTs that accept eval_set + callbacks in fit()
#   early_stopping_rounds


def _lgbm_factory(seed: int, debug: bool) -> Any:
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=50 if debug else 1000,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def _xgb_factory(seed: int, debug: bool) -> Any:
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=50 if debug else 2000,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )


def _cat_factory(seed: int, debug: bool) -> Any:
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=50 if debug else 3000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=seed,
        early_stopping_rounds=50,
        verbose=0,
        allow_writing_files=False,
    )


# kind drives the fit() call (each library has a different early-stopping API)
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "lightgbm": {"factory": _lgbm_factory, "kind": "lightgbm"},
    "xgboost": {"factory": _xgb_factory, "kind": "xgboost"},
    "catboost": {"factory": _cat_factory, "kind": "catboost"},
}

# --model ensemble trains all three per fold and averages predictions
ENSEMBLE_MEMBERS = ["lightgbm", "xgboost", "catboost"]


def _fit_one(cfg: dict[str, Any], seed: int, debug: bool, X_tr, y_tr, X_val, y_val) -> Any:
    """Fit a single model, dispatching to each library's early-stopping API."""
    model = cfg["factory"](seed, debug)
    kind = cfg["kind"]
    if kind == "lightgbm":
        import lightgbm as lgb

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
    elif kind == "xgboost":
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    elif kind == "catboost":
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    else:
        model.fit(X_tr, y_tr)
    return model


# ── Feature columns ────────────────────────────────────────────────────────────
# Only columns available in both train and test wells (no geology surfaces).
FEATURE_COLS: list[str] = [
    "MD",
    "X",
    "Y",
    "Z",
    "GR",
    "tvt_input_interp",
    "tvt_input_is_known",
    "tvt_anchor",
    "delta_md_from_ps",
    "typewell_tvt_nn",
    "typewell_pattern_tvt_nn",
    "prePS_tvt_nn",
    "prePS_tvt_dev",
    # Multi-scale Pearson NCC (typewell GR only — valid for test wells)
    "ncc_tvt_hw8",
    "ncc_tvt_hw15",
    "ncc_tvt_hw25",
    "ncc_score_hw8",
    "ncc_score_hw15",
    "ncc_score_hw25",
    "ncc_tvt_blend",
    # Anchor-zone statistics (from XGB Starter public notebook — cv=15.01 ft)
    "known_tvt_std",
    "known_tvt_range",
    "known_gr_mean",
    "known_gr_std",
    # Slope features: TVT rate of change near PS (geological dip)
    "slope_tvt_md",
    "baseline_tvt_slope",
    # GR residual vs typewell at PS anchor TVT
    "tw_gr_at_ps_tvt",
    "gr_minus_tw_at_ps",
    "md_frac",
    "well_md_range",
    "gr_roll_mean_10",
    "gr_roll_std_10",
    "gr_roll_mean_50",
    "gr_roll_std_50",
    "well_gr_mean",
    "well_gr_std",
]
FORMATION_FEATURE_COLS: list[str] = [
    "formation_knn_tvt",
    "formation_knn_dev",
    "formation_knn_bias",
]


def _feature_cols(use_formation_knn: bool) -> list[str]:
    return [*FEATURE_COLS, *FORMATION_FEATURE_COLS] if use_formation_knn else FEATURE_COLS


# ── Data loading ───────────────────────────────────────────────────────────────


def _list_well_ids(directory: Path) -> list[str]:
    return sorted(
        m.group(1) for f in directory.iterdir() if (m := re.match(r"^([0-9a-f]+)__horizontal_well\.csv$", f.name))
    )


def _tvt_nn_1d(gr_query: np.ndarray, ref_gr: np.ndarray, ref_tvt: np.ndarray) -> np.ndarray:
    """Vectorised 1-D nearest-neighbour TVT lookup keyed on GR value."""
    right_idx = np.searchsorted(ref_gr, gr_query).clip(0, len(ref_gr) - 1)
    left_idx = (right_idx - 1).clip(0)
    left_dist = np.abs(gr_query - ref_gr[left_idx])
    right_dist = np.abs(gr_query - ref_gr[right_idx])
    return np.where(left_dist <= right_dist, ref_tvt[left_idx], ref_tvt[right_idx])


def _typewell_tvt_nn(gr: pd.Series, typewell: pd.DataFrame) -> pd.Series:
    """Nearest-neighbour TVT lookup in typewell keyed on GR value."""
    tw = typewell.dropna(subset=["TVT", "GR"]).sort_values("GR")
    if tw.empty:
        return pd.Series(np.nan, index=gr.index)
    chosen = _tvt_nn_1d(gr.fillna(gr.median()).to_numpy(), tw["GR"].to_numpy(), tw["TVT"].to_numpy())
    return pd.Series(chosen, index=gr.index)


def _typewell_pattern_tvt_nn(gr: pd.Series, typewell: pd.DataFrame) -> pd.Series:
    """Pattern-based TVT lookup: match (GR, roll_mean_10, roll_mean_50, roll_std_50)
    against the same 4-D descriptor computed for the typewell.
    More robust than point-wise GR matching — uses the local GR pattern.
    """
    from sklearn.neighbors import KNeighborsRegressor

    tw = typewell.dropna(subset=["TVT", "GR"]).sort_values("TVT").reset_index(drop=True)
    if len(tw) < 10:
        return pd.Series(np.nan, index=gr.index)

    # Build 4-D descriptor for typewell
    tg = tw["GR"]
    tw_feats = np.column_stack(
        [
            tg.values,
            tg.rolling(10, min_periods=1, center=True).mean().values,
            tg.rolling(50, min_periods=1, center=True).mean().values,
            tg.rolling(50, min_periods=1, center=True).std().fillna(0).values,
        ]
    )

    # Build 4-D descriptor for horizontal well
    gf = gr.ffill().bfill().fillna(gr.median())
    h_feats = np.column_stack(
        [
            gf.values,
            gf.rolling(10, min_periods=1, center=True).mean().values,
            gf.rolling(50, min_periods=1, center=True).mean().values,
            gf.rolling(50, min_periods=1, center=True).std().fillna(0).values,
        ]
    )

    knn = KNeighborsRegressor(n_neighbors=3, metric="euclidean")
    knn.fit(tw_feats, tw["TVT"].values)
    return pd.Series(knn.predict(h_feats), index=gr.index)


def _prePS_tvt_nn(gr: pd.Series, tvt_input: pd.Series) -> pd.Series:
    """Nearest-neighbour TVT using the well's own pre-PS anchor region as reference.
    Pre-PS rows have exact TVT_input — richer and well-specific vs the generic typewell.
    """
    known = tvt_input.notna() & gr.notna()
    if known.sum() < 2:
        return pd.Series(np.nan, index=gr.index)
    ref = pd.Series(tvt_input[known].values, index=gr[known].values).sort_index()
    ref_gr = ref.index.to_numpy(dtype=float)
    ref_tvt = ref.values
    chosen = _tvt_nn_1d(gr.fillna(gr.median()).to_numpy(), ref_gr, ref_tvt)
    return pd.Series(chosen, index=gr.index)


def _ncc_tvt_vectorised(
    gr_filled: np.ndarray,
    tw_gr: np.ndarray,
    tw_tvt: np.ndarray,
    hw: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Pearson NCC via matrix multiply: O(N_h × N_tw × wsize) total.

    For each horizontal row i, find the typewell position j where the GR window
    [j-hw:j+hw] has the highest Pearson correlation with h_gr[i-hw:i+hw].

    Returns (tvt_pred, ncc_score) of shape (N_h,).
    """
    wsize = 2 * hw + 1
    n_h = len(gr_filled)
    n_tw = len(tw_gr)
    n_valid_tw = n_tw - wsize + 1
    if n_valid_tw < 1:
        return np.full(n_h, np.nan), np.zeros(n_h)

    # Pad arrays at edges so every row gets a full window
    h_pad = np.pad(gr_filled, hw, mode="edge")
    tw_pad = np.pad(tw_gr, hw, mode="edge")

    # Extract all windows: (N_h, wsize) and (N_tw, wsize)
    h_wins = np.lib.stride_tricks.sliding_window_view(h_pad, wsize)[:n_h]  # (N_h, W)
    tw_wins = np.lib.stride_tricks.sliding_window_view(tw_pad, wsize)[:n_tw]  # (N_tw, W)

    # Normalize each window to zero mean, unit std (Pearson normalisation)
    h_means = h_wins.mean(axis=1, keepdims=True)
    h_stds = h_wins.std(axis=1, keepdims=True).clip(min=1e-6)
    h_norm = np.nan_to_num((h_wins - h_means) / h_stds, nan=0.0, posinf=0.0, neginf=0.0)  # (N_h, W)

    tw_means = tw_wins.mean(axis=1, keepdims=True)
    tw_stds = tw_wins.std(axis=1, keepdims=True).clip(min=1e-6)
    tw_norm = np.nan_to_num((tw_wins - tw_means) / tw_stds, nan=0.0, posinf=0.0, neginf=0.0)  # (N_tw, W)

    # NCC matrix: (N_h, N_tw). einsum avoids spurious BLAS fp warnings seen with matmul on finite inputs.
    ncc_mat = np.einsum("ij,kj->ik", h_norm, tw_norm, optimize=True) / wsize  # Pearson r
    ncc_mat = np.nan_to_num(ncc_mat, nan=-1.0, posinf=-1.0, neginf=-1.0)

    # Best typewell position for each horizontal row
    best_j = np.argmax(ncc_mat, axis=1)
    tvt_pred = tw_tvt[np.minimum(best_j + hw, n_tw - 1)]
    ncc_score = ncc_mat[np.arange(n_h), best_j]

    return tvt_pred, ncc_score


def _ncc_tvt_all_scales(
    gr: pd.Series,
    typewell: pd.DataFrame,
    ps_tvt: float,
    search_range_ft: float = 150.0,
) -> dict[str, np.ndarray]:
    """NCC TVT predictions at hw=8, 15, 25 with softmax-weighted blend.

    Constrains the typewell search to ps_tvt ± search_range_ft — critical because
    the post-PS zone is only ±15 ft std; searching the full typewell finds spurious matches.
    """
    tw = (
        typewell.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["TVT", "GR"])
        .sort_values("TVT")
        .reset_index(drop=True)
    )
    n = len(gr)
    nan_result = {
        k: np.full(n, np.nan)
        for k in [
            "ncc_tvt_hw8",
            "ncc_tvt_hw15",
            "ncc_tvt_hw25",
            "ncc_score_hw8",
            "ncc_score_hw15",
            "ncc_score_hw25",
            "ncc_tvt_blend",
        ]
    }
    if tw.empty:
        return nan_result

    # Restrict typewell to search window around ps_tvt
    mask = (tw["TVT"] >= ps_tvt - search_range_ft) & (tw["TVT"] <= ps_tvt + search_range_ft)
    tw_w = tw[mask].reset_index(drop=True)
    if len(tw_w) < 5:
        return nan_result

    tw_gr = tw_w["GR"].values.astype(np.float64)
    tw_tvt = tw_w["TVT"].values
    gr_clean = gr.replace([np.inf, -np.inf], np.nan)
    gr_median = float(gr_clean.median())
    if not np.isfinite(gr_median):
        gr_median = 0.0
    gr_filled = gr_clean.ffill().bfill().fillna(gr_median).values.astype(np.float64)

    results: dict[str, np.ndarray] = {}
    scores, preds = [], []
    for hw in [8, 15, 25]:
        tvt, score = _ncc_tvt_vectorised(gr_filled, tw_gr, tw_tvt, hw)
        results[f"ncc_tvt_hw{hw}"] = tvt
        results[f"ncc_score_hw{hw}"] = score
        scores.append(score)
        preds.append(tvt)

    # Softmax blend across scales
    scores_arr = np.stack(scores)
    preds_arr = np.stack(preds)
    w = np.exp(np.clip(scores_arr, -10, 10))
    w /= w.sum(axis=0, keepdims=True).clip(min=1e-9)
    results["ncc_tvt_blend"] = (preds_arr * w).sum(axis=0)
    return results


def _beam_search_tvt(
    gr_filled: np.ndarray,
    tw_gr: np.ndarray,
    tw_tvt: np.ndarray,
    ps_tvt: float,
    ps_row: int = 0,
    sigma_emission: float = 20.0,
    max_step_ft: float = 2.0,
    search_range_ft: float = 150.0,
    gr_valid: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorised forward HMM for TVT trajectory prediction.

    V[j] = best accumulated log-prob of reaching grid position j at current step.
    Greedy decode: tvt_pred[i] = tvt_grid[argmax V] after incorporating emission i.
    Uses scipy maximum_filter1d for O(N_grid) transition step per row.

    Returns tvt_pred of shape (N_h,); all rows initialised to ps_tvt.
    """
    from scipy.ndimage import maximum_filter1d

    n_h = len(gr_filled)
    mask = (tw_tvt >= ps_tvt - search_range_ft) & (tw_tvt <= ps_tvt + search_range_ft)
    tvt_grid = tw_tvt[mask]
    gr_grid = tw_gr[mask]
    n_grid = len(tvt_grid)
    if n_grid < 3:
        return np.full(n_h, ps_tvt)

    dt = float(np.median(np.diff(tvt_grid))) if n_grid > 1 else 0.5
    max_step = max(1, int(round(max_step_ft / dt)))

    # Initialise: start at ps_tvt position with log-prob 0
    start_idx = int(np.argmin(np.abs(tvt_grid - ps_tvt)))
    V = np.full(n_grid, -1e9)
    V[start_idx] = 0.0

    tvt_pred = np.full(n_h, ps_tvt)

    valid = gr_valid if gr_valid is not None else np.ones(n_h, dtype=bool)

    # Run over ALL rows — pre-PS GR calibrates the beam so it enters the post-PS
    # zone already aligned at the correct typewell position (key from public solutions)
    for i in range(n_h):
        V = maximum_filter1d(V, size=2 * max_step + 1, mode="constant", cval=-1e9)
        if valid[i]:
            diff = gr_filled[i] - gr_grid
            V = V - 0.5 * (diff / sigma_emission) ** 2
        if i >= ps_row:  # only record post-PS predictions
            tvt_pred[i] = tvt_grid[int(np.argmax(V))]

    return tvt_pred


def _beam_search_all_sigmas(
    gr: pd.Series,
    typewell: pd.DataFrame,
    ps_tvt: float,
    ps_row: int = 0,
) -> dict[str, np.ndarray]:
    """Run beam search with 3 sigma variants and return TVT predictions."""
    tw = typewell.dropna(subset=["TVT", "GR"]).sort_values("TVT").reset_index(drop=True)
    n = len(gr)
    if tw.empty:
        return {
            k: np.full(n, ps_tvt) for k in ["beam_tvt_loose", "beam_tvt_medium", "beam_tvt_tight", "beam_tvt_blend"]
        }

    # Keep original NaN mask — only emit when actual GR data exists
    gr_values = gr.values.astype(np.float64)
    gr_valid = ~np.isnan(gr_values)
    gr_filled = np.where(gr_valid, gr_values, 0.0)  # 0 placeholder for missing (emission skipped)
    tw_gr = tw["GR"].values.astype(np.float64)
    tw_tvt = tw["TVT"].values

    results = {}
    for name, sigma in [("loose", 50.0), ("medium", 20.0), ("tight", 10.0)]:
        results[f"beam_tvt_{name}"] = _beam_search_tvt(
            gr_filled, tw_gr, tw_tvt, ps_tvt, ps_row=ps_row, sigma_emission=sigma, gr_valid=gr_valid
        )
    results["beam_tvt_blend"] = np.mean(
        [results["beam_tvt_loose"], results["beam_tvt_medium"], results["beam_tvt_tight"]], axis=0
    )
    return results


def load_dataset(directory: Path, max_wells: int | None = None) -> pd.DataFrame:
    """Load and concatenate all horizontal wells from directory."""
    well_ids = _list_well_ids(directory)
    if max_wells is not None:
        rng = np.random.default_rng(42)
        well_ids = list(rng.choice(well_ids, size=min(max_wells, len(well_ids)), replace=False))

    frames: list[pd.DataFrame] = []
    n_wells = len(well_ids)
    t_start = time.time()
    for i, wid in enumerate(well_ids):
        # Progress through the feature-build phase (otherwise invisible — looks "stuck")
        if i and (i % 100 == 0):
            elapsed = time.time() - t_start
            eta = elapsed / i * (n_wells - i)
            pct = 100 * i / n_wells
            print(f"  load_dataset {i}/{n_wells} ({pct:.0f}%)  {elapsed:.0f}s  ETA {eta:.0f}s", flush=True)
            log_metric_live("load_pct", pct)
        h = pd.read_csv(directory / f"{wid}__horizontal_well.csv")
        t = pd.read_csv(directory / f"{wid}__typewell.csv")
        h["well_id"] = wid
        h["_row_idx"] = np.arange(len(h))  # original 0-based row index within each well's CSV
        h["_typewell_tvt_nn"] = _typewell_tvt_nn(h["GR"], t)
        h["_typewell_pattern_tvt_nn"] = _typewell_pattern_tvt_nn(h["GR"], t)
        h["_prePS_tvt_nn"] = _prePS_tvt_nn(h["GR"], h["TVT_input"])

        # Multi-scale NCC and beam search — constrained to ±150 ft around PS anchor
        ps = int(h["TVT_input"].notna().sum())
        ps_tvt = float(h.iloc[ps - 1]["TVT_input"]) if ps > 0 else 0.0
        ncc = _ncc_tvt_all_scales(h["GR"], t, ps_tvt=ps_tvt)
        for k, v in ncc.items():
            h[f"_{k}"] = v
        # Typewell GR at ps_tvt for GR-residual feature
        tw_s = t.dropna(subset=["TVT", "GR"]).sort_values("TVT")
        if not tw_s.empty and ps > 0:
            idx = int(np.argmin(np.abs(tw_s["TVT"].values - ps_tvt)))
            h["_tw_gr_at_ps"] = float(tw_s["GR"].values[idx])
        else:
            h["_tw_gr_at_ps"] = np.nan
        frames.append(h)

    return pd.concat(frames, ignore_index=True)


# ── Feature engineering ────────────────────────────────────────────────────────


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TVT_input: linear interpolation within each well, extrapolate at ends
    tvt_interp = df["TVT_input"].copy()
    for wid, grp_idx in df.groupby("well_id", sort=False).groups.items():
        s = df.loc[grp_idx, "TVT_input"]
        s_filled = s.interpolate(method="linear", limit_direction="both").ffill().bfill()
        tvt_interp.loc[grp_idx] = s_filled.fillna(0).to_numpy()
    df["tvt_input_interp"] = tvt_interp
    df["tvt_input_is_known"] = df["TVT_input"].notna().astype(np.float32)

    # Typewell-derived TVT priors
    df["typewell_tvt_nn"] = df["_typewell_tvt_nn"]
    df["typewell_pattern_tvt_nn"] = df["_typewell_pattern_tvt_nn"]

    # Pre-PS self-reference: match post-PS GR against this well's own pre-PS GR↔TVT
    df["prePS_tvt_nn"] = df["_prePS_tvt_nn"]

    # PS anchor features: TVT at the last known point, and MD distance from it
    for wid, grp_idx in df.groupby("well_id", sort=False).groups.items():
        grp = df.loc[grp_idx]
        known = grp["TVT_input"].notna()
        anchor_tvt = grp.loc[known, "TVT_input"].iloc[-1] if known.any() else 0.0
        anchor_md = grp.loc[known, "MD"].iloc[-1] if known.any() else grp["MD"].iloc[0]
        df.loc[grp_idx, "tvt_anchor"] = anchor_tvt
        df.loc[grp_idx, "delta_md_from_ps"] = grp["MD"] - anchor_md

    # Pre-PS deviation: GR-predicted TVT relative to the anchor
    df["prePS_tvt_dev"] = df["prePS_tvt_nn"] - df["tvt_anchor"]

    # Multi-scale NCC features (from load_dataset)
    for col in [
        "ncc_tvt_hw8",
        "ncc_tvt_hw15",
        "ncc_tvt_hw25",
        "ncc_score_hw8",
        "ncc_score_hw15",
        "ncc_score_hw25",
        "ncc_tvt_blend",
    ]:
        df[col] = df[f"_{col}"].fillna(df["tvt_anchor"])

    # Anchor-zone statistics (pre-PS TVT + GR summary — from XGB Starter notebook)
    for wid, grp_idx in df.groupby("well_id", sort=False).groups.items():
        grp = df.loc[grp_idx]
        known = grp["TVT_input"].notna()
        if known.any():
            tvt_k = grp.loc[known, "TVT_input"]
            gr_k = grp.loc[known, "GR"].ffill().bfill().fillna(0)
            df.loc[grp_idx, "known_tvt_std"] = float(tvt_k.std()) if len(tvt_k) > 1 else 0.0
            df.loc[grp_idx, "known_tvt_range"] = float(tvt_k.max() - tvt_k.min())
            df.loc[grp_idx, "known_gr_mean"] = float(gr_k.mean())
            df.loc[grp_idx, "known_gr_std"] = float(gr_k.std()) if len(gr_k) > 1 else 0.0
            # Slope: fit TVT vs MD on last 200 pre-PS rows
            tail = grp.loc[known].tail(200)
            if len(tail) >= 2:
                md_c = tail["MD"].values - tail["MD"].values[-1]
                tvt_c = tail["TVT_input"].values
                denom = np.dot(md_c, md_c)
                slope = float(np.dot(md_c, tvt_c) / denom) if denom > 0 else 0.0
            else:
                slope = 0.0
            df.loc[grp_idx, "slope_tvt_md"] = slope
            anchor_tvt = float(tvt_k.iloc[-1])
            anchor_md = float(grp.loc[known, "MD"].iloc[-1])
            df.loc[grp_idx, "baseline_tvt_slope"] = anchor_tvt + slope * (grp["MD"] - anchor_md)
        else:
            for col in ["known_tvt_std", "known_tvt_range", "known_gr_mean", "known_gr_std", "slope_tvt_md"]:
                df.loc[grp_idx, col] = 0.0
            df.loc[grp_idx, "baseline_tvt_slope"] = df.loc[grp_idx, "tvt_anchor"]

    # GR residual vs typewell GR at ps_tvt (constant per well, precomputed in load_dataset)
    df["tw_gr_at_ps_tvt"] = df["_tw_gr_at_ps"].fillna(df.groupby("well_id", sort=False)["GR"].transform("mean"))
    gr_filled = df["GR"].fillna(
        df.groupby("well_id", sort=False)["GR"].transform(lambda x: x.rolling(10, min_periods=1, center=True).mean())
    )
    df["gr_minus_tw_at_ps"] = gr_filled - df["tw_gr_at_ps_tvt"]

    # Relative position within each well's borehole
    md_min = df.groupby("well_id", sort=False)["MD"].transform("min")
    md_max = df.groupby("well_id", sort=False)["MD"].transform("max")
    df["md_frac"] = (df["MD"] - md_min) / (md_max - md_min + 1e-9)
    df["well_md_range"] = md_max - md_min

    # GR rolling statistics (within each well, ordered by MD as loaded)
    for win in [10, 50]:
        df[f"gr_roll_mean_{win}"] = df.groupby("well_id", sort=False)["GR"].transform(
            lambda x: x.rolling(win, min_periods=1, center=True).mean()
        )
        df[f"gr_roll_std_{win}"] = df.groupby("well_id", sort=False)["GR"].transform(
            lambda x: x.rolling(win, min_periods=1, center=True).std().fillna(0)
        )

    # Well-level GR aggregates
    df["well_gr_mean"] = df.groupby("well_id", sort=False)["GR"].transform("mean")
    df["well_gr_std"] = df.groupby("well_id", sort=False)["GR"].transform("std").fillna(0)

    # Slim the dataframe: drop dead intermediate columns and downcast features to float32.
    # Keeps per-fold memory low so 4 parallel cluster folds don't swap (~525→~210 B/row).
    keep = (
        set(FEATURE_COLS)
        | set(FORMATION_FEATURE_COLS)
        | {
            "well_id",
            "_row_idx",
            TARGET,
            "TVT_input",
            "tvt_anchor",
        }
    )
    drop_cols = [c for c in df.columns if c.startswith("_") and c != "_row_idx" and c not in keep]
    df = df.drop(columns=drop_cols)
    f32_cols = [c for c in df.columns if c in FEATURE_COLS or c in FORMATION_FEATURE_COLS]
    df[f32_cols] = df[f32_cols].astype(np.float32)

    return df


# ── Formation surface features ────────────────────────────────────────────────


def _fit_formation_surface(df: pd.DataFrame, max_points_per_well: int = 120) -> Any | None:
    """Fit TVT + Z as a spatial surface from training wells only."""
    from sklearn.neighbors import KNeighborsRegressor

    valid = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["X", "Y", "Z", TARGET, "well_id"])
    if valid.empty:
        return None

    sampled_idx: list[int] = []
    for _, idx in valid.groupby("well_id", sort=False).groups.items():
        idx_arr = np.asarray(list(idx))
        if len(idx_arr) > max_points_per_well:
            idx_arr = idx_arr[np.linspace(0, len(idx_arr) - 1, max_points_per_well, dtype=int)]
        sampled_idx.extend(idx_arr.tolist())

    ref = valid.loc[sampled_idx]
    n_neighbors = min(25, len(ref))
    if n_neighbors < 1:
        return None

    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance", metric="euclidean")
    model.fit(ref[["X", "Y"]].to_numpy(dtype=np.float64), (ref[TARGET] + ref["Z"]).to_numpy(dtype=np.float64))
    return model


def add_formation_features(df: pd.DataFrame, surface: Any | None) -> pd.DataFrame:
    """Apply a fitted formation surface and calibrate per well from known TVT_input rows."""
    df = df.copy()
    if surface is None:
        base_tvt = df["tvt_anchor"].to_numpy(dtype=np.float64)
    else:
        xy = df[["X", "Y"]].replace([np.inf, -np.inf], np.nan)
        med = xy.median(numeric_only=True).fillna(0.0)
        xy_arr = xy.fillna(med).to_numpy(dtype=np.float64)
        base_tvt = surface.predict(xy_arr) - df["Z"].to_numpy(dtype=np.float64)

    pred = pd.Series(base_tvt, index=df.index, dtype="float64")
    bias_s = pd.Series(0.0, index=df.index, dtype="float64")
    for _, grp_idx in df.groupby("well_id", sort=False).groups.items():
        grp = df.loc[grp_idx]
        known = grp["TVT_input"].notna()
        bias = float((grp.loc[known, "TVT_input"] - pred.loc[grp.loc[known].index]).mean()) if known.any() else 0.0
        if not np.isfinite(bias):
            bias = 0.0
        pred.loc[grp_idx] += bias
        bias_s.loc[grp_idx] = bias

    df["formation_knn_tvt"] = pred
    df["formation_knn_dev"] = pred - df["tvt_anchor"]
    df["formation_knn_bias"] = bias_s
    return df


# ── CV training ────────────────────────────────────────────────────────────────


def run_cv(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    target: str = TARGET,
) -> tuple[np.ndarray, list[tuple[Any, Any | None]]]:
    """Group k-fold CV grouped by well. Returns (oof_preds, trained models)."""
    y = df[target].to_numpy(dtype=np.float64)
    groups = df["well_id"].to_numpy()

    gkf = GroupKFold(n_splits=args.folds)
    all_splits = list(gkf.split(np.zeros(len(df)), y, groups))

    # --fold N: run only that fold (used for cluster fan-out via kego run --folds)
    if args.fold is not None:
        enumerated = [(args.fold, all_splits[args.fold])]
    else:
        enumerated = list(enumerate(all_splits))

    oof_preds = np.full(len(df), np.nan)
    models: list[tuple[Any, Any | None]] = []
    feature_cols = _feature_cols(args.formation_knn)

    for fold_num, (train_idx, val_idx) in enumerated:
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        formation_surface = _fit_formation_surface(train_df) if args.formation_knn else None
        if args.formation_knn:
            train_df = add_formation_features(train_df, formation_surface)
            val_df = add_formation_features(val_df, formation_surface)

        X_tr = train_df[feature_cols].reset_index(drop=True)
        X_val = val_df[feature_cols].reset_index(drop=True)
        y_tr, y_val = y[train_idx], y[val_idx]

        if args.model == "ensemble":
            # Train all members, average their val predictions
            member_models = [
                _fit_one(MODEL_CONFIGS[name], args.seed, args.debug, X_tr, y_tr, X_val, y_val)
                for name in ENSEMBLE_MEMBERS
            ]
            val_preds = np.mean([m.predict(X_val) for m in member_models], axis=0)
            fold_model: Any = member_models  # list → ensemble
        else:
            fold_model = _fit_one(cfg, args.seed, args.debug, X_tr, y_tr, X_val, y_val)
            val_preds = fold_model.predict(X_val)

        oof_preds[val_idx] = val_preds

        fold_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
        fold_r2 = float(r2_score(y_val, val_preds))
        print(f"Fold {fold_num}  RMSE={fold_rmse:.4f}  R²={fold_r2:.4f}", flush=True)
        print(f"KEGO_METRIC fold_rmse_{fold_num} {fold_rmse:.6f}", flush=True)
        print(f"KEGO_METRIC fold_r2_{fold_num} {fold_r2:.6f}", flush=True)
        log_metric_live("fold_rmse", fold_rmse, step=fold_num)
        log_metric_live("progress_pct", (fold_num + 1) / args.folds * 100, step=fold_num)

        models.append((fold_model, formation_surface))

    return oof_preds, models


# ── Test prediction ────────────────────────────────────────────────────────────


def predict_test(
    models: list[tuple[Any, Any | None]],
    deviation: bool = False,
    use_formation_knn: bool = False,
) -> pd.DataFrame:
    """Ensemble predictions on test wells, returning submission-format DataFrame.

    Submission format: id={well_id}_{row_idx}, tvt=predicted_TVT
    Only post-PS rows are scored (where TVT_input is NaN in the test file).
    """
    df_test = load_dataset(TEST_DIR)
    df_test = engineer_features(df_test)
    feature_cols = _feature_cols(use_formation_knn)

    def _predict(fold_model: Any, X) -> np.ndarray:
        # fold_model may be a single estimator or a list (ensemble) — average either way
        if isinstance(fold_model, list):
            return np.mean([m.predict(X) for m in fold_model], axis=0)
        return fold_model.predict(X)

    preds = np.mean(
        [
            _predict(
                fold_model,
                (add_formation_features(df_test, formation_surface) if use_formation_knn else df_test)[feature_cols],
            )
            for fold_model, formation_surface in models
        ],
        axis=0,
    )
    if deviation:
        preds = preds + df_test["tvt_anchor"].to_numpy()
    df_test["tvt"] = preds

    # Keep only post-PS rows (TVT_input is NaN = evaluation zone)
    post_ps = df_test[df_test["TVT_input"].isna()].copy()
    post_ps["id"] = post_ps["well_id"] + "_" + post_ps["_row_idx"].astype(str)
    return post_ps[["id", "tvt"]]


# ── Figure logging ─────────────────────────────────────────────────────────────


def log_figures(
    df_train: pd.DataFrame,
    oof_preds: np.ndarray,
    models: list[tuple[Any, Any | None]],
    feature_cols: list[str],
) -> None:
    """Log diagnostic figures to MLflow via MlflowClient (no run state changes).

    Requires KEGO_MLFLOW_RUN_ID + MLFLOW_TRACKING_URI in the environment.
    The kego runner injects KEGO_MLFLOW_RUN_ID for both local and cluster targets.
    """
    run_id = os.environ.get("KEGO_MLFLOW_RUN_ID", "")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not (run_id and tracking_uri):
        return

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    def log_artifact(path: Path) -> None:
        try:
            client.log_artifact(run_id, str(path), artifact_path="figures")
        except OSError as e:
            print(f"Warning: could not log artifact {path.name}: {e}", flush=True)

    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    mask = ~np.isnan(oof_preds)
    y_true = df_train[TARGET].to_numpy()[mask]
    y_pred = oof_preds[mask]

    # OOF: predicted vs actual scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=1, alpha=0.15, rasterized=True)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="perfect")
    ax.set_xlabel("True TVT")
    ax.set_ylabel("Predicted TVT")
    ax.set_title("OOF predictions vs actuals")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = fig_dir / "oof_scatter.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log_artifact(path)

    # Feature importance (mean over folds). For ensemble folds, use the first member.
    trained_models = [(m[0] if isinstance(m, list) else m) for m, _ in models]
    if hasattr(trained_models[0], "feature_importances_"):
        imp = np.mean([m.feature_importances_ for m in trained_models], axis=0)
        fi = pd.Series(imp, index=feature_cols).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        fi.plot.barh(ax=ax)
        ax.set_title("Feature importance (mean over folds)")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        path = fig_dir / "feature_importance.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        log_artifact(path)


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Rogii wellbore TVT regression")
    parser.add_argument("--model", default="lightgbm", choices=[*MODEL_CONFIGS, "ensemble"])
    parser.add_argument("--folds", type=int, default=4, metavar="N")
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        metavar="N",
        help="Run a single fold only (for kego run --folds fan-out on cluster)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Fast smoke test: 20 wells, 50 trees")
    parser.add_argument(
        "--deviation",
        action="store_true",
        default=True,
        help="Train on TVT - tvt_anchor (deviation from PS) instead of absolute TVT",
    )
    parser.add_argument("--no-deviation", dest="deviation", action="store_false")
    parser.add_argument(
        "--train-post-ps-only",
        action="store_true",
        default=True,
        help="Train only on post-PS rows (matches public XGB Starter). Default on.",
    )
    parser.add_argument("--train-all-rows", dest="train_post_ps_only", action="store_false")
    parser.add_argument("--formation-knn", action="store_true", help="Experimental fold-aware spatial formation KNN")
    args = parser.parse_args()

    # "ensemble" has no single cfg — run_cv handles member dispatch internally
    cfg = MODEL_CONFIGS.get(args.model, {})

    print(f"KEGO_PARAM model {args.model}", flush=True)
    print(f"KEGO_PARAM train_post_ps_only {args.train_post_ps_only}", flush=True)
    print(f"KEGO_PARAM folds {args.folds}", flush=True)
    print(f"KEGO_PARAM seed {args.seed}", flush=True)
    print(f"KEGO_PARAM deviation {args.deviation}", flush=True)
    print(f"KEGO_PARAM formation_knn {args.formation_knn}", flush=True)
    print(f"KEGO_PARAM debug {args.debug}", flush=True)

    max_wells = 20 if args.debug else None
    df_train = load_dataset(TRAIN_DIR, max_wells=max_wells)
    df_train = engineer_features(df_train)  # full-well context (rolling, slopes) before any filtering
    df_train = df_train.dropna(subset=[TARGET])

    # Train only on the post-PS zone (matches public XGB Starter @ 15.01 ft). Pre-PS rows have
    # huge deviations from the well descending to PS depth — training on them wastes model
    # capacity on dynamics irrelevant to the flat post-PS prediction task.
    if args.train_post_ps_only:
        n_before = len(df_train)
        df_train = df_train[df_train["TVT_input"].isna()].copy()
        print(f"Filtered to post-PS rows: {len(df_train):,} / {n_before:,}", flush=True)

    # Deviation mode: train on TVT - tvt_anchor; model learns ±15 ft drift, not absolute 11k+ ft
    train_target = TARGET
    if args.deviation:
        df_train["tvt_dev"] = df_train[TARGET] - df_train["tvt_anchor"]
        train_target = "tvt_dev"

    print(
        f"Loaded {len(df_train):,} rows from {df_train['well_id'].nunique()} wells"
        f"  target={'tvt_dev (deviation)' if args.deviation else 'TVT (absolute)'}",
        flush=True,
    )

    oof_preds, models = run_cv(df_train, cfg, args, target=train_target)

    # Convert deviation predictions back to absolute TVT for metric computation
    anchor = df_train["tvt_anchor"].to_numpy()
    oof_abs = oof_preds + anchor if args.deviation else oof_preds

    mask = ~np.isnan(oof_abs)
    tvt_true = df_train[TARGET].to_numpy()
    oof_rmse = float(np.sqrt(mean_squared_error(tvt_true[mask], oof_abs[mask])))
    oof_r2 = float(r2_score(tvt_true[mask], oof_abs[mask]))

    # Post-PS only — the competition metric
    post_ps_mask = mask & df_train["TVT_input"].isna().to_numpy()
    post_ps_rmse = float(np.sqrt(mean_squared_error(tvt_true[post_ps_mask], oof_abs[post_ps_mask])))

    print(f"OOF      RMSE={oof_rmse:.4f}  R²={oof_r2:.4f}", flush=True)
    print(f"Post-PS  RMSE={post_ps_rmse:.4f}", flush=True)
    print(f"KEGO_METRIC oof_rmse {oof_rmse:.6f}", flush=True)
    print(f"KEGO_METRIC oof_r2 {oof_r2:.6f}", flush=True)
    print(f"KEGO_METRIC post_ps_rmse {post_ps_rmse:.6f}", flush=True)
    log_metric_live("oof_rmse", oof_rmse)
    log_metric_live("post_ps_rmse", post_ps_rmse)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    oof_out = df_train[["well_id", "MD", TARGET]].copy()
    oof_out["oof_pred"] = oof_abs
    oof_out.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)

    # Test predictions only when all folds are present (not single-fold fan-out)
    if args.fold is None:
        test_out = predict_test(models, deviation=args.deviation, use_formation_knn=args.formation_knn)
        test_out.to_csv(OUTPUT_DIR / "submission.csv", index=False)
        print(f"Saved test predictions: {len(test_out):,} rows", flush=True)

    log_figures(df_train, oof_abs, models, _feature_cols(args.formation_knn))


if __name__ == "__main__":
    main()
