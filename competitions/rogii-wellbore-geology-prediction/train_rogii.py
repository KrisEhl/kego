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


MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "lightgbm": {
        "factory": _lgbm_factory,
        "early_stopping": True,
        "early_stopping_rounds": 50,
    },
    # "xgboost": {
    #     "factory": _xgb_factory,
    #     "early_stopping": True,
    #     "early_stopping_rounds": 50,
    # },
    # "ridge": {
    #     "factory": lambda seed, debug: Ridge(alpha=1.0),
    #     "early_stopping": False,
    # },
}


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
    # Multi-scale Pearson NCC: primary signal from public solutions (r=0.9993 with TVT)
    "ncc_tvt_hw8",
    "ncc_tvt_hw15",
    "ncc_tvt_hw25",
    "ncc_score_hw8",
    "ncc_score_hw15",
    "ncc_score_hw25",
    "ncc_tvt_blend",  # softmax-weighted blend across scales
    # Formation spatial KNN with per-well bias calibration
    "knn_tvt_pred",
    "knn_tvt_dev",  # deviation from anchor
    "md_frac",
    "well_md_range",
    "gr_roll_mean_10",
    "gr_roll_std_10",
    "gr_roll_mean_50",
    "gr_roll_std_50",
    "well_gr_mean",
    "well_gr_std",
]


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
    h_norm = (h_wins - h_means) / h_stds  # (N_h, W)

    tw_means = tw_wins.mean(axis=1, keepdims=True)
    tw_stds = tw_wins.std(axis=1, keepdims=True).clip(min=1e-6)
    tw_norm = (tw_wins - tw_means) / tw_stds  # (N_tw, W)

    # NCC matrix: (N_h, N_tw) — one BLAS dgemm call
    ncc_mat = (h_norm @ tw_norm.T) / wsize  # Pearson r

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
    tw = typewell.dropna(subset=["TVT", "GR"]).sort_values("TVT").reset_index(drop=True)
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
    gr_filled = gr.ffill().bfill().fillna(float(gr.median())).values.astype(np.float64)

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


def load_dataset(directory: Path, max_wells: int | None = None) -> pd.DataFrame:
    """Load and concatenate all horizontal wells from directory."""
    well_ids = _list_well_ids(directory)
    if max_wells is not None:
        rng = np.random.default_rng(42)
        well_ids = list(rng.choice(well_ids, size=min(max_wells, len(well_ids)), replace=False))

    frames: list[pd.DataFrame] = []
    for wid in well_ids:
        h = pd.read_csv(directory / f"{wid}__horizontal_well.csv")
        t = pd.read_csv(directory / f"{wid}__typewell.csv")
        h["well_id"] = wid
        h["_row_idx"] = np.arange(len(h))  # original 0-based row index within each well's CSV
        h["_typewell_tvt_nn"] = _typewell_tvt_nn(h["GR"], t)
        h["_typewell_pattern_tvt_nn"] = _typewell_pattern_tvt_nn(h["GR"], t)
        h["_prePS_tvt_nn"] = _prePS_tvt_nn(h["GR"], h["TVT_input"])

        # Multi-scale NCC — constrained to ±150 ft around PS anchor
        ps = int(h["TVT_input"].notna().sum())
        ps_tvt = float(h.iloc[ps - 1]["TVT_input"]) if ps > 0 else 0.0
        ncc = _ncc_tvt_all_scales(h["GR"], t, ps_tvt=ps_tvt)
        for k, v in ncc.items():
            h[f"_{k}"] = v

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

    # Formation KNN with per-well bias calibration
    # TVT ≈ -Z + formation_depth(X,Y) + bias_well  (physics formula from public solutions)
    # We use the EGFDU column (available in train) or recover from TVT_input for test
    has_egfdu = "EGFDU" in df.columns
    if has_egfdu:
        df["_knn_raw"] = -df["Z"] + df["EGFDU"]
    else:
        # Recover from anchor: EGFDU_rec = TVT_input + Z - egfdu_tw, but we just need the
        # per-well bias from the anchor rows
        df["_knn_raw"] = df["tvt_anchor"]  # fallback for test wells without EGFDU

    # Per-well bias: calibrate from anchor region where TVT_input is known
    knn_tvt = df["tvt_anchor"].copy()
    for wid, grp_idx in df.groupby("well_id", sort=False).groups.items():
        grp = df.loc[grp_idx]
        known = grp["TVT_input"].notna()
        if known.sum() > 0 and has_egfdu:
            # bias = mean(TVT_input - (-Z + EGFDU)) over known anchor rows
            bias = float((grp.loc[known, "TVT_input"] - grp.loc[known, "_knn_raw"]).mean())
            knn_tvt.loc[grp_idx] = grp["_knn_raw"] + bias
    df["knn_tvt_pred"] = knn_tvt
    df["knn_tvt_dev"] = df["knn_tvt_pred"] - df["tvt_anchor"]

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

    return df


# ── CV training ────────────────────────────────────────────────────────────────


def run_cv(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    target: str = TARGET,
) -> tuple[np.ndarray, list[Any]]:
    """Group k-fold CV grouped by well. Returns (oof_preds, trained models)."""
    X = df[FEATURE_COLS].reset_index(drop=True)  # DataFrame preserves feature names for LightGBM
    y = df[target].to_numpy(dtype=np.float64)
    groups = df["well_id"].to_numpy()

    gkf = GroupKFold(n_splits=args.folds)
    all_splits = list(gkf.split(X, y, groups))

    # --fold N: run only that fold (used for cluster fan-out via kego run --folds)
    if args.fold is not None:
        enumerated = [(args.fold, all_splits[args.fold])]
    else:
        enumerated = list(enumerate(all_splits))

    oof_preds = np.full(len(df), np.nan)
    models: list[Any] = []

    for fold_num, (train_idx, val_idx) in enumerated:
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = cfg["factory"](args.seed, args.debug)

        if cfg.get("early_stopping"):
            import lightgbm as lgb

            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(cfg.get("early_stopping_rounds", 50), verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
        else:
            model.fit(X_tr, y_tr)

        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds

        fold_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
        fold_r2 = float(r2_score(y_val, val_preds))
        print(f"Fold {fold_num}  RMSE={fold_rmse:.4f}  R²={fold_r2:.4f}", flush=True)
        print(f"KEGO_METRIC fold_rmse_{fold_num} {fold_rmse:.6f}", flush=True)
        print(f"KEGO_METRIC fold_r2_{fold_num} {fold_r2:.6f}", flush=True)

        models.append(model)

    return oof_preds, models


# ── Test prediction ────────────────────────────────────────────────────────────


def predict_test(models: list[Any], deviation: bool = False) -> pd.DataFrame:
    """Ensemble predictions on test wells, returning submission-format DataFrame.

    Submission format: id={well_id}_{row_idx}, tvt=predicted_TVT
    Only post-PS rows are scored (where TVT_input is NaN in the test file).
    """
    df_test = load_dataset(TEST_DIR)
    df_test = engineer_features(df_test)
    X_test = df_test[FEATURE_COLS]
    preds = np.mean([m.predict(X_test) for m in models], axis=0)
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
    models: list[Any],
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
    client.log_artifact(run_id, str(path), artifact_path="figures")

    # Feature importance (mean over folds)
    if hasattr(models[0], "feature_importances_"):
        imp = np.mean([m.feature_importances_ for m in models], axis=0)
        fi = pd.Series(imp, index=FEATURE_COLS).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        fi.plot.barh(ax=ax)
        ax.set_title("Feature importance (mean over folds)")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        path = fig_dir / "feature_importance.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        client.log_artifact(run_id, str(path), artifact_path="figures")


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Rogii wellbore TVT regression")
    parser.add_argument("--model", default="lightgbm", choices=list(MODEL_CONFIGS))
    parser.add_argument("--folds", type=int, default=5, metavar="N")
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
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]

    print(f"KEGO_PARAM model {args.model}", flush=True)
    print(f"KEGO_PARAM folds {args.folds}", flush=True)
    print(f"KEGO_PARAM seed {args.seed}", flush=True)
    print(f"KEGO_PARAM deviation {args.deviation}", flush=True)
    print(f"KEGO_PARAM debug {args.debug}", flush=True)

    max_wells = 20 if args.debug else None
    df_train = load_dataset(TRAIN_DIR, max_wells=max_wells)
    df_train = engineer_features(df_train)
    df_train = df_train.dropna(subset=[TARGET])

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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    oof_out = df_train[["well_id", "MD", TARGET]].copy()
    oof_out["oof_pred"] = oof_abs
    oof_out.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)

    # Test predictions only when all folds are present (not single-fold fan-out)
    if args.fold is None:
        test_out = predict_test(models, deviation=args.deviation)
        test_out.to_csv(OUTPUT_DIR / "submission.csv", index=False)
        print(f"Saved test predictions: {len(test_out):,} rows", flush=True)

    log_figures(df_train, oof_abs, models)


if __name__ == "__main__":
    main()
