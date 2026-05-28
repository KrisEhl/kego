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
    "typewell_tvt_nn",
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


def _typewell_tvt_nn(gr: pd.Series, typewell: pd.DataFrame) -> pd.Series:
    """Nearest-neighbour TVT lookup in typewell keyed on GR value."""
    tw = typewell.dropna(subset=["TVT", "GR"]).sort_values("GR")
    if tw.empty:
        return pd.Series(np.nan, index=gr.index)
    tw_gr = tw["GR"].to_numpy()
    tw_tvt = tw["TVT"].to_numpy()
    gr_vals = gr.fillna(gr.median()).to_numpy()
    right_idx = np.searchsorted(tw_gr, gr_vals).clip(0, len(tw_gr) - 1)
    left_idx = (right_idx - 1).clip(0)
    left_dist = np.abs(gr_vals - tw_gr[left_idx])
    right_dist = np.abs(gr_vals - tw_gr[right_idx])
    chosen = np.where(left_dist <= right_dist, tw_tvt[left_idx], tw_tvt[right_idx])
    return pd.Series(chosen, index=gr.index)


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
        h["_typewell_tvt_nn"] = _typewell_tvt_nn(h["GR"], t)
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

    # Typewell-derived TVT prior
    df["typewell_tvt_nn"] = df["_typewell_tvt_nn"]

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
) -> tuple[np.ndarray, list[Any]]:
    """Group k-fold CV grouped by well. Returns (oof_preds, trained models)."""
    X = df[FEATURE_COLS].reset_index(drop=True)  # DataFrame preserves feature names for LightGBM
    y = df[TARGET].to_numpy(dtype=np.float64)
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


def predict_test(models: list[Any]) -> pd.DataFrame:
    """Ensemble predictions on all test wells."""
    df_test = load_dataset(TEST_DIR)
    df_test = engineer_features(df_test)
    X_test = df_test[FEATURE_COLS]
    preds = np.mean([m.predict(X_test) for m in models], axis=0)
    df_test["TVT_predicted"] = preds
    return df_test[["well_id", "MD", "TVT_predicted"]]


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
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]

    print(f"KEGO_PARAM model {args.model}", flush=True)
    print(f"KEGO_PARAM folds {args.folds}", flush=True)
    print(f"KEGO_PARAM seed {args.seed}", flush=True)
    print(f"KEGO_PARAM debug {args.debug}", flush=True)

    max_wells = 20 if args.debug else None
    df_train = load_dataset(TRAIN_DIR, max_wells=max_wells)
    df_train = engineer_features(df_train)
    df_train = df_train.dropna(subset=[TARGET])

    print(
        f"Loaded {len(df_train):,} rows from {df_train['well_id'].nunique()} wells",
        flush=True,
    )

    oof_preds, models = run_cv(df_train, cfg, args)

    mask = ~np.isnan(oof_preds)
    oof_rmse = float(np.sqrt(mean_squared_error(df_train[TARGET].to_numpy()[mask], oof_preds[mask])))
    oof_r2 = float(r2_score(df_train[TARGET].to_numpy()[mask], oof_preds[mask]))

    print(f"OOF  RMSE={oof_rmse:.4f}  R²={oof_r2:.4f}", flush=True)
    print(f"KEGO_METRIC oof_rmse {oof_rmse:.6f}", flush=True)
    print(f"KEGO_METRIC oof_r2 {oof_r2:.6f}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    oof_out = df_train[["well_id", "MD", TARGET]].copy()
    oof_out["oof_pred"] = oof_preds
    oof_out.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)

    # Test predictions only when all folds are present (not single-fold fan-out)
    if args.fold is None:
        test_out = predict_test(models)
        test_out.to_csv(OUTPUT_DIR / "submission.csv", index=False)
        print(f"Saved test predictions: {len(test_out):,} rows", flush=True)

    log_figures(df_train, oof_preds, models)


if __name__ == "__main__":
    main()
