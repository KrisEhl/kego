"""Optuna regularization tuning of the XGB member at depth6 (ladder step B; the
8.905 ref's actual edge was 'Optuna-tuned reg'). Objective = XGB-alone GroupKFold(4)
OOF post-PS RMSE on the cached 195-feat set (div_/kinematics/dwt dropped — matches the
v36 ens-d6 config). Depth fixed at 6 (v36 confirmed d6<d7, 3/3 paired). Tunes
reg_alpha/reg_lambda/min_child_weight/subsample/colsample_bytree.

Why XGB-alone (not the full ensemble) per trial: the ref tuned singles then NNLS-blended;
a 4-fold XGB fit is ~half an ensemble trial. Best reg is then validated in the ensemble
(paired vs the d6 baseline) before any kernel ship.

Multi-worker: pass the SAME --storage <path> to 2 cluster jobs (one per GPU) → they share
one Optuna JournalStorage study and split the trials. Each worker runs --n-trials.

    uv run kego run competitions/.../tune_optuna_reg.py --target cluster --n-trials 30
    # 2 workers (both cards), shared study:
    uv run kego run ... --storage outputs/optuna_reg.journal --n-trials 15  (x2, one heavy + one default)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

_HERE = Path(__file__).parent
CACHE_DIR = _HERE / "outputs" / "feat_cache"
NON_FEATURES = {"well", "id", "target"}
KINEMATIC_COLS = {
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
_mlflow_run_id = os.environ.get("KEGO_MLFLOW_RUN_ID", "")


def log_metric_live(key, value, step=None):
    if _mlflow_run_id:
        from mlflow.tracking import MlflowClient

        MlflowClient().log_metric(_mlflow_run_id, key, float(value), step=step)


def _feat_version() -> str:
    return hashlib.md5((_HERE / "rogii_features.py").read_bytes()).hexdigest()[:10]


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--seed", type=int, default=42, help="TPE sampler seed")
    p.add_argument("--storage", default="", help="Optuna JournalStorage path for multi-worker (empty = in-memory)")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    import optuna
    from xgboost import XGBRegressor

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dev = _device()
    n_trees = 50 if args.debug else 3000
    print(f"KEGO_PARAM experiment optuna_reg_d{args.depth}", flush=True)
    print(f"KEGO_PARAM device {dev}  n_trials {args.n_trials}  depth {args.depth}", flush=True)

    cache = CACHE_DIR / f"train_{_feat_version()}_773w.parquet"
    if not cache.exists():
        raise SystemExit(f"cache {cache} missing — run train_seq_feats.py once to build it")
    df = pd.read_parquet(cache)
    feat = [
        c
        for c in df.columns
        if c not in NON_FEATURES and not c.startswith("div_") and c not in KINEMATIC_COLS and not c.startswith("gr_dwt")
    ]
    X = df[feat].to_numpy(np.float32)
    X[~np.isfinite(X)] = np.nan
    y = df["target"].to_numpy(np.float64)
    groups = df["well"].to_numpy()
    if args.debug:
        # tiny slice for a fast tracking smoke
        keep = np.isin(groups, np.unique(groups)[:40])
        X, y, groups = X[keep], y[keep], groups[keep]
    print(f"KEGO_PARAM feat {len(feat)}  rows {len(y)}", flush=True)
    log_metric_live("progress_pct", 5)

    splits = list(GroupKFold(n_splits=args.folds).split(X, y, groups))
    # baseline: depth6 with the current default reg (reg_alpha=0.1, reg_lambda=1.0, mcw=5, ss=0.8, cs=0.7)
    done = {"n": 0}

    def _rmse_for(params) -> float:
        oof = np.full(len(y), np.nan)
        for tr, va in splits:
            m = XGBRegressor(
                n_estimators=n_trees,
                learning_rate=0.03,
                max_depth=args.depth,
                early_stopping_rounds=80,
                tree_method="hist",
                device=dev,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                **params,
            )
            m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
            oof[va] = m.predict(X[va])
        return float(np.sqrt(mean_squared_error(y, oof)))

    def objective(trial):
        params = {
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 30.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        r = _rmse_for(params)
        done["n"] += 1
        log_metric_live("trial_rmse", r, step=done["n"])
        log_metric_live("progress_pct", 5 + 90 * done["n"] / max(args.n_trials, 1))
        print(f"trial {done['n']}: rmse={r:.4f}  {params}", flush=True)
        return r

    if args.storage:
        storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(args.storage))
        study = optuna.create_study(
            study_name="rogii_reg_d6",
            storage=storage,
            direction="minimize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )
    else:
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed))

    # enqueue the current-default config as trial 0 so we always know the baseline at depth6
    try:
        study.enqueue_trial(
            {"reg_alpha": 0.1, "reg_lambda": 1.0, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.7},
            skip_if_exists=True,
        )
    except Exception:
        pass

    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials)
    print(f"optimize done in {time.time() - t0:.0f}s", flush=True)
    print(f"BEST rmse={study.best_value:.4f}  params={study.best_params}", flush=True)
    print(f"KEGO_METRIC post_ps_rmse {study.best_value:.6f}", flush=True)
    log_metric_live("post_ps_rmse", study.best_value)
    log_metric_live("progress_pct", 100)


if __name__ == "__main__":
    main()
