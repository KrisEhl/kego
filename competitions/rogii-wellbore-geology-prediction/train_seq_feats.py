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


def _xgb(seed: int, debug: bool):
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
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="xgboost")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    print(f"KEGO_PARAM model {args.model}", flush=True)
    print(f"KEGO_PARAM folds {args.folds}", flush=True)
    print(f"KEGO_PARAM seed {args.seed}", flush=True)
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
    X = df[feat_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    y = df["target"].to_numpy(np.float64)  # drift = TVT - last_known_tvt
    groups = df["well"].to_numpy()

    gkf = GroupKFold(n_splits=args.folds)
    splits = list(gkf.split(X, y, groups))
    enumerated = [(args.fold, splits[args.fold])] if args.fold is not None else list(enumerate(splits))

    oof = np.full(len(df), np.nan)
    models = []
    for fold_num, (tr, va) in enumerated:
        model = _xgb(args.seed, args.debug)
        model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        oof[va] = model.predict(X.iloc[va])
        fold_rmse = float(np.sqrt(mean_squared_error(y[va], oof[va])))  # drift RMSE == TVT RMSE
        print(f"Fold {fold_num}  post_ps_rmse={fold_rmse:.4f}", flush=True)
        print(f"KEGO_METRIC fold_rmse_{fold_num} {fold_rmse:.6f}", flush=True)
        log_metric_live("fold_rmse", fold_rmse, step=fold_num)
        log_metric_live("progress_pct", 40 + 50 * (fold_num + 1) / args.folds)
        models.append(model)

    if args.fold is None:
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
                Xte = df_te[feat_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
                drift = np.mean([m.predict(Xte) for m in models], axis=0)
                sub = pd.DataFrame({"id": df_te["id"], "tvt": df_te["last_known_tvt"].to_numpy() + drift})
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                sub.to_csv(OUTPUT_DIR / "submission_seq_feats.csv", index=False)
                print(f"Saved {len(sub):,} test predictions", flush=True)
        log_metric_live("progress_pct", 100)


if __name__ == "__main__":
    main()
