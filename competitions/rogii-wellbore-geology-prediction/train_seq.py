"""Sequence model for Rogii TVT prediction.

Processes the full GR + Z + XY trajectory as a 1D sequence.
The model sees the entire pre-PS context (known GR + TVT) and outputs
post-PS TVT deviations. Uses causal 1D CNN so post-PS positions attend
only to themselves and the pre-PS context.

kego run compatible:
    uv run kego run competitions/rogii-wellbore-geology-prediction/train_seq.py
    uv run kego run ... --target cluster --folds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

# Live MLflow logging — injected by kego runner for both local and cluster targets
_mlflow_run_id = os.environ.get("KEGO_MLFLOW_RUN_ID", "")


def log_epoch(epoch: int, train_loss: float, val_loss: float) -> None:
    """Log train/val metrics per epoch to MLflow (visible in UI while running)."""
    if _mlflow_run_id:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        client.log_metric(_mlflow_run_id, "train_loss", train_loss, step=epoch)
        client.log_metric(_mlflow_run_id, "val_loss", val_loss, step=epoch)


DATA_DIR = (
    Path(os.environ.get("KEGO_PATH_DATA", _PROJECT_ROOT / "data")) / "rogii" / "rogii-wellbore-geology-prediction"
)
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = Path(__file__).parent / "outputs"

# ── Input features (per MD step) ──────────────────────────────────────────────
# All available for both train and test wells
SEQ_FEATURES = [
    "gr_norm",  # GR normalised per well: (GR - median) / std
    "z_delta",  # Z - Z_at_PS
    "dx",  # X - X_at_PS
    "dy",  # Y - Y_at_PS
    "delta_md",  # MD step (1 ft typically)
    "is_post_ps",  # 0 = pre-PS (TVT known), 1 = post-PS (target)
    "tvt_dev_known",  # TVT deviation from anchor (0 for post-PS rows — masked)
]
N_FEAT = len(SEQ_FEATURES)


# ── Architecture ──────────────────────────────────────────────────────────────


class DilatedTCN(nn.Module):
    """Dilated temporal convolutional network.

    Causal: each position can only see current + earlier positions.
    Wide receptive field via exponentially growing dilation.
    """

    def __init__(self, n_feat: int = N_FEAT, d: int = 64, n_layers: int = 6, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(n_feat, d)
        layers = []
        for i in range(n_layers):
            dilation = 2**i
            layers.append(
                nn.Sequential(
                    nn.Conv1d(d, d, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_feat)  →  output: (B, T)
        h = self.input_proj(x)  # (B, T, d)
        h = h.permute(0, 2, 1)  # (B, d, T)
        for layer in self.layers:
            # Causal: trim right padding so each step only sees the past
            out = layer(h)
            # trim to original length
            h = h + out[..., : h.size(-1)]
        h = h.permute(0, 2, 1)  # (B, T, d)
        return self.out(h).squeeze(-1)  # (B, T)


# ── Data ──────────────────────────────────────────────────────────────────────


def _list_well_ids(directory: Path) -> list[str]:
    return sorted(
        m.group(1) for f in directory.iterdir() if (m := re.match(r"^([0-9a-f]+)__horizontal_well\.csv$", f.name))
    )


def _get_egfdu_tw(h: pd.DataFrame, t: pd.DataFrame) -> float:
    if "Geology" in t.columns:
        eg = t[t["Geology"] == "EGFDU"]["TVT"]
        if len(eg) > 0:
            return float(eg.min())
    r = h.iloc[0]
    return float(r["TVT"]) + float(r["Z"]) - float(r["EGFDU"])


_CACHE_DIR = Path(__file__).parent / "outputs" / "cache"


def _load_from_cache(wid: str, split: str) -> dict[str, Any] | None:
    path = _CACHE_DIR / split / f"{wid}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    return {
        "well_id": wid,
        "feat": d["feat"],
        "target": d["target"] if "target" in d else None,
        "ps": int(d["ps"]),
        "ps_tvt": float(d["ps_tvt"]),
        "row_idx": d["row_idx"],
        "tvt_input_is_nan": d["tvt_input_is_nan"],
    }


def load_well_tensors(
    directory: Path,
    well_ids: list[str] | None = None,
    debug: bool = False,
) -> list[dict[str, Any]]:
    """Load wells — from .npz cache if available, otherwise parse CSV on-the-fly."""
    ids = well_ids or _list_well_ids(directory)
    if debug:
        rng = np.random.default_rng(42)
        ids = list(rng.choice(ids, size=min(30, len(ids)), replace=False))

    split = "train" if "train" in str(directory) else "test"
    wells = []
    cache_hits = 0

    for wid in ids:
        cached = _load_from_cache(wid, split)
        if cached is not None:
            wells.append(cached)
            cache_hits += 1
            continue

        # Fallback: parse CSV and compute on-the-fly
        h = pd.read_csv(directory / f"{wid}__horizontal_well.csv")
        ps = int(h["TVT_input"].notna().sum())
        if ps == 0 or ps >= len(h):
            continue

        ps_tvt = float(h.iloc[ps - 1]["TVT"] if "TVT" in h.columns else h.iloc[ps - 1]["TVT_input"])
        ps_z = float(h.iloc[ps - 1]["Z"])
        ps_x = float(h.iloc[ps - 1]["X"])
        ps_y = float(h.iloc[ps - 1]["Y"])

        gr = h["GR"].ffill().bfill().fillna(h["GR"].median())
        gr_norm = ((gr - float(gr.median())) / (float(gr.std()) + 1e-6)).values

        n = len(h)
        feat = np.zeros((n, N_FEAT), dtype=np.float32)
        feat[:, 0] = gr_norm
        feat[:, 1] = h["Z"].values - ps_z
        feat[:, 2] = h["X"].values - ps_x
        feat[:, 3] = h["Y"].values - ps_y
        feat[:, 4] = h["MD"].diff().fillna(1.0).values
        feat[:, 5] = np.where(np.arange(n) >= ps, 1.0, 0.0)

        if "TVT" in h.columns:
            tvt_dev = h["TVT"].values - ps_tvt
        else:
            tvt_dev = np.where(h["TVT_input"].notna(), h["TVT_input"].values - ps_tvt, 0.0)
        feat[:, 6] = np.where(np.arange(n) < ps, tvt_dev, 0.0)

        target = (h["TVT"].values - ps_tvt).astype(np.float32) if "TVT" in h.columns else None

        wells.append(
            {
                "well_id": wid,
                "feat": feat,
                "target": target,
                "ps": ps,
                "ps_tvt": ps_tvt,
                "row_idx": np.arange(n, dtype=np.int64),
                "tvt_input_is_nan": h["TVT_input"].isna().values,
            }
        )

    if cache_hits:
        print(f"  Loaded {cache_hits}/{len(ids)} wells from .npz cache", flush=True)
    return wells


# ── Training ──────────────────────────────────────────────────────────────────


def train_fold(
    train_wells: list[dict],
    val_wells: list[dict],
    args: argparse.Namespace,
    device: torch.device,
    fold_num: int = 0,
) -> tuple[nn.Module, float]:
    """Train one fold, return (model, best_val_rmse)."""
    model = DilatedTCN(n_feat=N_FEAT, d=args.d_model, n_layers=args.n_layers, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")
    best_state = None
    patience_count = 0

    for epoch in range(args.epochs):
        model.train()
        rng = np.random.default_rng(epoch + 42)
        train_order = rng.permutation(len(train_wells))
        train_loss = 0.0
        n_samples = 0

        # Dynamic-padding mini-batch: pad each batch to its longest well.
        # Loss mask ignores padded positions so no signal bleeds from padding.
        for batch_start in range(0, len(train_order), args.batch_size):
            batch_idx = train_order[batch_start : batch_start + args.batch_size]
            valid = [train_wells[i] for i in batch_idx if train_wells[i]["target"] is not None]
            if not valid:
                continue

            max_len = max(len(w["feat"]) for w in valid)
            B = len(valid)
            feat_np = np.zeros((B, max_len, N_FEAT), dtype=np.float32)
            tgt_np = np.zeros((B, max_len), dtype=np.float32)
            mask_np = np.zeros((B, max_len), dtype=bool)  # True = valid post-PS position

            for i, w in enumerate(valid):
                n = len(w["feat"])
                ps = w["ps"]
                f = w["feat"].copy()
                # Input masking: randomly zero tvt_dev_known (feat col 6) for 50% of
                # pre-PS rows so the model can't passthrough known TVT — forces GR learning
                if args.mask_prob > 0 and ps > 0:
                    drop = rng.random(ps) < args.mask_prob
                    f[:ps][drop, 6] = 0.0
                feat_np[i, :n] = f
                tgt_np[i, :n] = w["target"]
                mask_np[i, ps:n] = True  # only post-PS rows contribute to loss

            feat_t = torch.from_numpy(feat_np).to(device)
            tgt_t = torch.from_numpy(tgt_np).to(device)
            mask_t = torch.from_numpy(mask_np).to(device)

            pred = model(feat_t)  # (B, max_len)
            loss = (F.mse_loss(pred, tgt_t, reduction="none") * mask_t).sum() / mask_t.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            n_samples += 1

        sched.step()

        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_preds, val_trues = [], []
            with torch.no_grad():
                for w in val_wells:
                    if w["target"] is None:
                        continue
                    feat = torch.from_numpy(w["feat"]).unsqueeze(0).to(device)
                    pred = model(feat).squeeze(0).cpu().numpy()
                    ps = w["ps"]
                    val_preds.extend((pred[ps:] + w["ps_tvt"]).tolist())
                    val_trues.extend((w["target"][ps:] + w["ps_tvt"]).tolist())

            val_rmse = float(np.sqrt(mean_squared_error(val_trues, val_preds)))
            train_loss_avg = train_loss / max(n_samples, 1)
            print(
                f"  epoch {epoch + 1:3d}  train_loss={train_loss_avg:.4f}  val_post_ps_rmse={val_rmse:.4f}",
                flush=True,
            )
            log_epoch(epoch + 1, train_loss_avg, val_rmse)

            if val_rmse < best_val:
                best_val = val_rmse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= args.patience:
                    print(f"  Early stop at epoch {epoch + 1}", flush=True)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Rogii sequence model")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16, help="Wells per GPU batch")
    parser.add_argument(
        "--mask_prob", type=float, default=0.5, help="Fraction of pre-PS tvt_dev_known to zero during training"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Log all params immediately — before any computation
    print(f"KEGO_PARAM folds {args.folds}", flush=True)
    print(f"KEGO_PARAM fold {args.fold}", flush=True)
    print(f"KEGO_PARAM seed {args.seed}", flush=True)
    print(f"KEGO_PARAM epochs {args.epochs}", flush=True)
    print(f"KEGO_PARAM d_model {args.d_model}", flush=True)
    print(f"KEGO_PARAM n_layers {args.n_layers}", flush=True)
    print(f"KEGO_PARAM lr {args.lr}", flush=True)
    print(f"KEGO_PARAM dropout {args.dropout}", flush=True)
    print(f"KEGO_PARAM patience {args.patience}", flush=True)
    print(f"KEGO_PARAM batch_size {args.batch_size}", flush=True)
    print(f"KEGO_PARAM mask_prob {args.mask_prob}", flush=True)
    print(f"KEGO_PARAM debug {args.debug}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print("Loading training wells...", flush=True)
    all_wells = load_well_tensors(TRAIN_DIR, debug=args.debug)
    well_ids = np.array([w["well_id"] for w in all_wells])
    print(f"Loaded {len(all_wells)} wells", flush=True)

    gkf = GroupKFold(n_splits=args.folds)
    dummy = np.zeros(len(all_wells))
    all_splits = list(gkf.split(dummy, dummy, well_ids))

    if args.fold is not None:
        splits = [(args.fold, all_splits[args.fold])]
    else:
        splits = list(enumerate(all_splits))

    oof_preds: dict[str, np.ndarray] = {}
    fold_models: list[nn.Module] = []

    for fold_num, (train_idx, val_idx) in splits:
        train_wells = [all_wells[i] for i in train_idx]
        val_wells = [all_wells[i] for i in val_idx]
        print(f"\nFold {fold_num}  train={len(train_wells)}  val={len(val_wells)}", flush=True)

        model, val_rmse = train_fold(train_wells, val_wells, args, device, fold_num=fold_num)
        fold_models.append(model)

        print(f"KEGO_METRIC fold_post_ps_rmse_{fold_num} {val_rmse:.6f}", flush=True)

        # Store OOF predictions
        model.eval()
        with torch.no_grad():
            for w in val_wells:
                feat = torch.from_numpy(w["feat"]).unsqueeze(0).to(device)
                pred = model(feat).squeeze(0).cpu().numpy()
                oof_preds[w["well_id"]] = pred + w["ps_tvt"]

    # OOF post-PS RMSE
    if args.fold is None:
        oof_true, oof_pred = [], []
        for w in all_wells:
            wid = w["well_id"]
            if wid not in oof_preds or w["target"] is None:
                continue
            ps = w["ps"]
            oof_pred.extend(oof_preds[wid][ps:].tolist())
            oof_true.extend((w["target"][ps:] + w["ps_tvt"]).tolist())

        post_ps_rmse = float(np.sqrt(mean_squared_error(oof_true, oof_pred)))
        print(f"\nOOF post-PS RMSE = {post_ps_rmse:.4f} ft", flush=True)
        print(f"KEGO_METRIC post_ps_rmse {post_ps_rmse:.6f}", flush=True)

        # Save OOF predictions
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        oof_rows = []
        for w in all_wells:
            wid = w["well_id"]
            if wid not in oof_preds:
                continue
            ps = w["ps"]
            for i, (ri, pred) in enumerate(zip(w["row_idx"][ps:], oof_preds[wid][ps:])):
                oof_rows.append({"well_id": wid, "row_idx": int(ri), "oof_pred": float(pred)})
        pd.DataFrame(oof_rows).to_csv(OUTPUT_DIR / "oof_seq.csv", index=False)

        # Test predictions (ensemble over folds)
        test_wells = load_well_tensors(TEST_DIR)
        sub_rows = []
        for w in test_wells:
            feat = torch.from_numpy(w["feat"]).unsqueeze(0).to(device)
            preds = []
            for m in fold_models:
                m.eval()
                with torch.no_grad():
                    preds.append(m(feat).squeeze(0).cpu().numpy())
            pred_mean = np.mean(preds, axis=0) + w["ps_tvt"]
            # Only post-PS rows (TVT_input is NaN)
            for ri, is_nan, pred in zip(w["row_idx"], w["tvt_input_is_nan"], pred_mean):
                if is_nan:
                    sub_rows.append({"id": f"{w['well_id']}_{ri}", "tvt": float(pred)})

        pd.DataFrame(sub_rows).to_csv(OUTPUT_DIR / "submission_seq.csv", index=False)
        print(f"Saved {len(sub_rows)} test predictions", flush=True)

        # Save model checkpoints
        ckpt_dir = OUTPUT_DIR / "seq_checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        for i, m in enumerate(fold_models):
            torch.save(m.state_dict(), ckpt_dir / f"fold_{i}.pt")


if __name__ == "__main__":
    main()
