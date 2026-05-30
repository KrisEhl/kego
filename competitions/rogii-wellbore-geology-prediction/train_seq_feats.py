"""Train Rogii TVT with the full sequence-estimator feature suite (ported from
the public 9.85-ft solution). Drift target, post-PS rows only, GroupKFold by well.

kego run compatible:
    uv run kego run competitions/rogii-wellbore-geology-prediction/train_seq_feats.py
    uv run kego run ... --target cluster --folds 0,1,2,3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import rogii_features as rf  # noqa: E402

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

NON_FEATURES = {"well", "id", "target"}


def _detect_device() -> str:
    """cuda if a GPU is allocated/visible (Ray --gpu), else cpu. Auto-fallback keeps
    local runs working."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _xgb(seed: int, debug: bool, device: str):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=50 if debug else 3000,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5,
        early_stopping_rounds=80,
        tree_method="hist",
        device=device,  # "cuda" → fit on a 3090; "cpu" fallback
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )


def _log_curve(train_hist, val_hist, fold_num, every=25):
    """Log per-round train/val RMSE so the boosting learning curve (and any overfit
    gap) is visible in the MLflow UI. Namespaced per fold so curves don't collide."""
    n = len(val_hist)
    for i in list(range(0, n, every)) + [n - 1]:
        log_metric_live(f"train_rmse_f{fold_num}", float(train_hist[i]), step=i)
        log_metric_live(f"val_rmse_f{fold_num}", float(val_hist[i]), step=i)


def _fit_one(model_name, seed, debug, device, Xtr, ytr, Xva, yva, fold_num=0):
    """Fit one model family on a fold. All handle NaN natively. Returns fitted model.

    Logs a train+val RMSE learning curve per fold. Train is evaluated on a 100k
    subsample so the extra eval doesn't slow the fit; val (full) drives early stopping.
    """
    n = 50 if debug else 3000
    rng = np.random.default_rng(seed)
    si = rng.choice(len(Xtr), size=min(100_000, len(Xtr)), replace=False)
    if model_name == "catboost":
        from catboost import CatBoostRegressor

        m = CatBoostRegressor(
            iterations=n,
            learning_rate=0.03,
            depth=7,
            l2_leaf_reg=3.0,
            loss_function="RMSE",
            od_type="Iter",
            od_wait=80,
            task_type="GPU" if device == "cuda" else "CPU",
            random_seed=seed,
            verbose=False,
        )
        m.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
        res = m.get_evals_result()
        _log_curve(res["learn"]["RMSE"], res["validation"]["RMSE"], fold_num)
        return m
    # default: xgboost — eval train(subsample) + val(full); early stopping uses val (last entry)
    m = _xgb(seed, debug, device)
    m.fit(Xtr, ytr, eval_set=[(Xtr[si], ytr[si]), (Xva, yva)], verbose=False)
    res = m.evals_result_
    _log_curve(res["validation_0"]["rmse"], res["validation_1"]["rmse"], fold_num)
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument(
        "--all-folds",
        action="store_true",
        help="Run all folds + OOF + test in ONE job (build once). Overrides --fold (cluster default injects --fold 0).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="xgboost", choices=["xgboost", "catboost"])
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    device = _detect_device()
    print(f"KEGO_PARAM model {args.model}", flush=True)
    print(f"KEGO_PARAM folds {args.folds}", flush=True)
    print(f"KEGO_PARAM seed {args.seed}", flush=True)
    print(f"KEGO_PARAM device {device}", flush=True)
    print("KEGO_PARAM features seq_estimators", flush=True)

    rf._warmup_numba()
    print("numba warmed up", flush=True)

    hw_paths = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
    train_wids = [p.stem.replace("__horizontal_well", "") for p in hw_paths]
    if args.debug:
        rng = np.random.default_rng(42)
        train_wids = list(rng.choice(train_wids, size=min(40, len(train_wids)), replace=False))
        hw_paths = [TRAIN_DIR / f"{w}__horizontal_well.csv" for w in train_wids]

    print(f"Fitting formation imputers on {len(train_wids)} wells...", flush=True)
    FI = rf.FormationPlaneKNN(train_wids, TRAIN_DIR)
    DI = rf.DenseANCCImputer(train_wids, TRAIN_DIR)
    log_metric_live("progress_pct", 5)

    print("Building features (per-well estimators)...", flush=True)
    t0 = time.time()
    df = rf.build_dataset(hw_paths, is_train=True, FI=FI, DI=DI, n_jobs=4)
    print(f"Built {len(df):,} rows × {len(df.columns)} cols in {time.time() - t0:.0f}s", flush=True)
    log_metric_live("progress_pct", 40)

    feat_cols = [c for c in df.columns if c not in NON_FEATURES]
    # Vectorised inf/-inf -> nan ONCE on the float32 array. pandas .replace() over
    # 3.78M x 198 (~750M cells) is single-threaded and was the low-CPU bottleneck.
    X = df[feat_cols].to_numpy(np.float32)
    X[~np.isfinite(X)] = np.nan  # ~isfinite is True for +/-inf and nan; setting nan->nan is a no-op
    y = df["target"].to_numpy(np.float64)  # drift = TVT - last_known_tvt
    groups = df["well"].to_numpy()

    # --all-folds overrides the cluster-injected --fold 0 so one job builds once + runs all folds.
    run_all = args.all_folds or args.fold is None
    gkf = GroupKFold(n_splits=args.folds)
    splits = list(gkf.split(X, y, groups))
    enumerated = list(enumerate(splits)) if run_all else [(args.fold, splits[args.fold])]

    oof = np.full(len(df), np.nan)
    models = []
    for fold_num, (tr, va) in enumerated:
        model = _fit_one(args.model, args.seed, args.debug, device, X[tr], y[tr], X[va], y[va], fold_num=fold_num)
        oof[va] = model.predict(X[va])
        fold_rmse = float(np.sqrt(mean_squared_error(y[va], oof[va])))  # drift RMSE == TVT RMSE
        print(f"Fold {fold_num}  post_ps_rmse={fold_rmse:.4f}", flush=True)
        print(f"KEGO_METRIC fold_rmse_{fold_num} {fold_rmse:.6f}", flush=True)
        log_metric_live("fold_rmse", fold_rmse, step=fold_num)
        log_metric_live("progress_pct", 40 + 50 * (fold_num + 1) / args.folds)
        models.append(model)

    if run_all:
        mask = ~np.isnan(oof)
        post_ps_rmse = float(np.sqrt(mean_squared_error(y[mask], oof[mask])))
        print(f"OOF post-PS RMSE = {post_ps_rmse:.4f} ft", flush=True)
        print(f"KEGO_METRIC post_ps_rmse {post_ps_rmse:.6f}", flush=True)
        log_metric_live("post_ps_rmse", post_ps_rmse)
        log_metric_live("progress_pct", 95)

        # Test predictions: imputers fit on ALL train wells, no exclusion
        test_paths = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
        if test_paths:
            df_te = rf.build_dataset(test_paths, is_train=False, FI=FI, DI=DI, n_jobs=4)
            if len(df_te):
                Xte = df_te[feat_cols].to_numpy(np.float32)
                Xte[~np.isfinite(Xte)] = np.nan
                drift = np.mean([m.predict(Xte) for m in models], axis=0)
                sub = pd.DataFrame({"id": df_te["id"], "tvt": df_te["last_known_tvt"].to_numpy() + drift})
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                sub.to_csv(OUTPUT_DIR / "submission_seq_feats.csv", index=False)
                print(f"Saved {len(sub):,} test predictions", flush=True)
        log_metric_live("progress_pct", 100)


if __name__ == "__main__":
    main()
