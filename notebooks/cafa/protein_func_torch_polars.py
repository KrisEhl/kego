#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protein function prediction (GPU-accelerated) — Polars + PyTorch
Now supports external per-term weights via --weights (e.g., ia.csv with columns: EntryID,weight).

Schema (train.csv): ['database','EntryID','gene_name','sequence','term','aspect','TaxonID']

Usage examples
--------------
Train using external weights (replace internal weighting):
  python protein_func_torch_polars.py train --train train.csv --outdir ./torch_model \
      --weights ia.csv --weight_mode replace --amp

Train combining weights (multiply external with effective-number weighting):
  python protein_func_torch_polars.py train --train train.csv --outdir ./torch_model \
      --weights ia.csv --weight_mode multiply --amp

Predict:
  python protein_func_torch_polars.py predict --model ./torch_model --input test.csv --output preds.parquet
"""
import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

try:
    import polars as pl
except Exception as e:
    print(
        "ERROR: This script requires Polars. Install with: pip install polars",
        file=sys.stderr,
    )
    sys.exit(1)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

AMINO = "ACDEFGHIKLMNPQRSTVWY"  # 20 canonical AAs
AA2IDX = {aa: i + 1 for i, aa in enumerate(AMINO)}  # 0 reserved for padding/unknown
PAD_IDX = 0

REQUIRED_COLS = [
    "database",
    "EntryID",
    "gene_name",
    "sequence",
    "term",
    "aspect",
    "TaxonID",
]


# -------------------- Polars IO helpers --------------------
def read_csv(path: str) -> pl.DataFrame:
    return pl.read_csv(path)


def ensure_required_cols(df: pl.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")


def group_labels(df: pl.DataFrame) -> pl.DataFrame:
    g = (
        df.group_by("EntryID")
        .agg(
            [
                pl.col("sequence").first().alias("sequence"),
                pl.col("term").unique().alias("terms"),
                pl.col("aspect").unique().alias("aspects"),
                pl.col("TaxonID").first().alias("TaxonID"),
                pl.col("database").first().alias("database"),
                pl.col("gene_name").first().alias("gene_name"),
            ]
        )
        .select(
            [
                "EntryID",
                "sequence",
                "terms",
                "aspects",
                "TaxonID",
                "database",
                "gene_name",
            ]
        )
    )
    return g


def clean_seq(seq: str) -> str:
    s = (seq or "").strip().upper()
    return "".join([c if c in AMINO else "X" for c in s])


def seq_to_indices(seq: str, max_len: int):
    s = clean_seq(seq)
    if len(s) > max_len:
        half = max_len // 2
        s = s[:half] + s[-(max_len - half) :]
    arr = [AA2IDX.get(c, PAD_IDX) for c in s]
    if len(arr) < max_len:
        arr = arr + [PAD_IDX] * (max_len - len(arr))
    return arr[:max_len]


# -------------------- Dataset --------------------
class ProtDataset(Dataset):
    def __init__(
        self,
        sequences: List[str],
        term_lists: List[List[str]],
        label_map: Dict[str, int],
        max_len: int,
    ):
        self.seqs = sequences
        self.term_lists = term_lists
        self.lbl_map = label_map
        self.max_len = max_len
        self.C = len(label_map)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = torch.tensor(seq_to_indices(self.seqs[idx], self.max_len), dtype=torch.long)
        y = torch.zeros(self.C, dtype=torch.float32)
        for t in self.term_lists[idx] or []:
            j = self.lbl_map.get(t)
            if j is not None:
                y[j] = 1.0
        return x, y


# -------------------- Model --------------------
class CNNSeqEncoder(nn.Module):
    def __init__(
        self,
        vocab=1 + len(AMINO),
        emb_dim=64,
        max_len=1024,
        channels=(128, 128, 256),
        kernels=(5, 9, 15),
        dropout=0.1,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=PAD_IDX)
        convs = []
        in_ch = emb_dim
        for ch, k in zip(channels, kernels):
            convs.append(nn.Conv1d(in_ch, ch, kernel_size=k, padding=k // 2))
            in_ch = ch
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList([nn.BatchNorm1d(c) for c in channels])
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(channels[-1] * 2, 512)  # global max + mean pool
        self.out_dim = 512

    def forward(self, x):  # x: [B, L] long
        e = self.emb(x).transpose(1, 2)  # [B, E, L]
        h = e
        for conv, bn in zip(self.convs, self.norms):
            h = conv(h)
            h = bn(h)
            h = F.gelu(h)
        # Global pooling
        h_max = torch.amax(h, dim=2)
        h_mean = torch.mean(h, dim=2)
        h = torch.cat([h_max, h_mean], dim=1)
        h = self.dropout(h)
        z = self.proj(h)
        z = F.gelu(z)
        return z  # [B, 512]


class MultiLabelHead(nn.Module):
    def __init__(self, in_dim, num_labels):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 1024)
        self.fc2 = nn.Linear(1024, num_labels)

    def forward(self, z):
        h = F.gelu(self.fc1(z))
        h = F.dropout(h, p=0.1, training=self.training)
        logits = self.fc2(h)
        return logits


class ProtModel(nn.Module):
    def __init__(self, max_len, num_labels):
        super().__init__()
        self.encoder = CNNSeqEncoder(max_len=max_len)
        self.head = MultiLabelHead(self.encoder.out_dim, num_labels)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.head(z)
        return logits


# -------------------- Loss & Metrics --------------------
def effective_num_weights(freqs: np.ndarray, beta=0.9995) -> np.ndarray:
    f = np.asarray(freqs, dtype=np.float64)
    eff = (1 - beta) / (1 - np.power(beta, np.maximum(f, 1.0)))
    w = 1.0 / eff
    w = w / np.mean(w)
    return w.astype(np.float32)


class FocalBCEWithLogits(nn.Module):
    def __init__(self, gamma=1.5, class_weights=None):
        super().__init__()
        self.gamma = gamma
        # Use descriptive buffer name; avoid duplicate attribute registration
        if class_weights is not None:
            self.register_buffer(
                "class_weight",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weight = None

    def forward(self, logits, targets):
        # logits, targets: [B, C]
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)  # if y=1 -> p, else 1-p
        focal = (1 - pt).pow(self.gamma)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = focal * bce
        if self.class_weight is not None:
            loss = loss * self.class_weight  # broadcast [C]
        return loss.mean()


def macro_f1(y_true: np.ndarray, y_prob: np.ndarray, thresh=0.5, eps=1e-9) -> float:
    y_pred = (y_prob >= thresh).astype(np.float32)
    tp = (y_true * y_pred).sum(axis=0)
    fp = ((1 - y_true) * y_pred).sum(axis=0)
    fn = (y_true * (1 - y_pred)).sum(axis=0)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return float(np.mean(f1))


def per_freq_thresholds(y_true, y_prob, freqs_aligned, bins=5):
    f = np.asarray(freqs_aligned, dtype=np.float64)
    qs = np.quantile(f, np.linspace(0, 1, bins + 1))
    bin_ids = np.digitize(f, qs[1:-1], right=True)
    thresholds = np.zeros(bins, dtype=np.float32)
    for b in range(bins):
        cand = np.linspace(0.05, 0.95, 19)
        best_t, best_f1 = 0.5, -1
        cols = bin_ids == b
        if not np.any(cols):
            thresholds[b] = 0.5
            continue
        yt = y_true[:, cols]
        yp = y_prob[:, cols]
        for t in cand:
            f1 = macro_f1(yt, yp, thresh=t)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds[b] = best_t
    return {"qs": qs.tolist(), "thresholds": thresholds.tolist()}


def apply_per_freq_thresholds(freqs_aligned, meta):
    qs = np.array(meta["qs"])
    thresholds = np.array(meta["thresholds"])
    f = np.asarray(freqs_aligned, dtype=np.float64)
    bin_ids = np.digitize(f, qs[1:-1], right=True)
    return thresholds[bin_ids]


# -------------------- Hierarchy --------------------
def load_hierarchy(path):
    if path is None:
        return None
    df = pl.read_csv(path)
    if "child" not in df.columns or "parent" not in df.columns:
        raise ValueError("Hierarchy CSV must have columns: child,parent")
    return list(zip(df["child"].to_list(), df["parent"].to_list()))


def upward_closure(prob_vec, term_to_idx, edges):
    if edges is None or len(edges) == 0:
        return prob_vec
    parents = {c: [] for c in term_to_idx.keys()}
    for c, p in edges:
        if c in parents:
            parents[c].append(p)
    p = prob_vec.copy()
    for _ in range(3):
        changed = False
        for c, plist in parents.items():
            ci = term_to_idx.get(c, None)
            if ci is None:
                continue
            for pa in plist:
                pi = term_to_idx.get(pa, None)
                if pi is None:
                    continue
                old = p[pi]
                p[pi] = max(p[pi], p[ci])
                changed = changed or (p[pi] != old)
        if not changed:
            break
    return p


# -------------------- Weights --------------------
def read_external_weights(path: str) -> dict:
    """
    Read a CSV with at least two columns: an ID column and a weight column.
    Accepts ID column: any of {term, EntryID, go_id, id} (case-insensitive).
    Accepts weight column: any of {weight, Weight, w, importance, ia} (case-insensitive).
    Returns dict {term_id -> weight(float)}.
    """
    df = pl.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    # ID column detection
    id_col = None
    for cand in ("term", "entryid", "go_id", "id"):
        if cand in lower:
            id_col = lower[cand]
            break
    if id_col is None:
        raise ValueError(
            "Weights file must include an ID column: one of {term, EntryID, go_id, id}."
        )
    # Weight column detection
    w_col = None
    for cand in ("weight", "w", "importance", "ia"):
        if cand in lower:
            w_col = lower[cand]
            break
    if w_col is None:
        raise ValueError(
            "Weights file must include a weight column: one of {weight, w, importance, ia}."
        )
    ids = df[id_col].to_list()
    ws = [float(x) for x in df[w_col].to_list()]
    return {i: w for i, w in zip(ids, ws)}


def align_class_weights(
    labels: List[str], freqs_aligned: List[int], args
) -> np.ndarray:
    """
    Build per-class weights according to args.weight_mode:
    - none: use effective-number weighting only (default if --weights not given)
    - replace: use external weights only
    - multiply: external * effective-number
    """
    eff = effective_num_weights(np.array(freqs_aligned), beta=args.beta_effnum)
    if args.weights is None:
        return eff
    ext = read_external_weights(args.weights)
    arr = np.array([ext.get(t, 1.0) for t in labels], dtype=np.float32)
    # normalize external weights to mean 1.0 for stability
    m = float(arr.mean()) if arr.size else 1.0
    if m > 0:
        arr = arr / m
    if args.weight_mode == "replace":
        return arr
    elif args.weight_mode == "multiply":
        return arr * eff
    else:
        raise ValueError("weight_mode must be one of: replace, multiply")


# -------------------- Train & Predict --------------------
def train(args):
    df = read_csv(args.train)
    ensure_required_cols(df)
    grouped = group_labels(df)

    seqs = grouped["sequence"].to_list()
    terms_list = grouped["terms"].to_list()

    # Build label map
    freq = {}
    for tl in terms_list:
        if tl is None:
            continue
        for t in tl:
            freq[t] = freq.get(t, 0) + 1
    labels = list(freq.keys())
    term_to_idx = {t: i for i, t in enumerate(labels)}
    freqs_aligned = [freq[t] for t in labels]
    C = len(labels)
    print(f"[INFO] #proteins={len(seqs)}  #labels={C}")

    # Split
    n = len(seqs)
    rng = np.random.default_rng(args.seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    v = int(round(args.val_frac * n))
    val_idx = idx[:v]
    tr_idx = idx[v:]

    train_ds = ProtDataset(
        [seqs[i] for i in tr_idx],
        [terms_list[i] for i in tr_idx],
        term_to_idx,
        args.max_len,
    )
    val_ds = ProtDataset(
        [seqs[i] for i in val_idx],
        [terms_list[i] for i in val_idx],
        term_to_idx,
        args.max_len,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    model = ProtModel(max_len=args.max_len, num_labels=C).to(device)

    class_w = align_class_weights(labels, freqs_aligned, args)
    # Coverage log: how many label IDs matched the external weights
    if args.weights is not None:
        try:
            ext_map = read_external_weights(args.weights)
            matched = sum(1 for t in labels if t in ext_map)
            print(
                f"[INFO] External weights: matched {matched}/{len(labels)} labels (file={args.weights})"
            )
        except Exception as e:
            print(f"[WARN] Could not compute weight coverage: {e}")

    criterion = FocalBCEWithLogits(gamma=args.focal_gamma, class_weights=class_w).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_f1 = -1
    os.makedirs(args.outdir, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss)
        # Validation
        model.eval()
        all_probs = []
        all_true = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(
                    enabled=args.amp and device.type == "cuda"
                ):
                    logits = model(xb)
                    probs = torch.sigmoid(logits).float().cpu().numpy()
                all_probs.append(probs)
                all_true.append(yb.numpy())
        P = (
            np.concatenate(all_probs, axis=0)
            if len(all_probs)
            else np.zeros((0, C), dtype=np.float32)
        )
        Y = (
            np.concatenate(all_true, axis=0)
            if len(all_true)
            else np.zeros((0, C), dtype=np.float32)
        )
        f1 = macro_f1(Y, P, 0.5) if len(Y) else 0.0
        print(
            f"[EP {ep:02d}] train_loss={total/len(train_loader):.6f} val_macroF1@0.5={f1:.4f} time={time.time()-t0:.1f}s"
        )
        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                {
                    "model": model.state_dict(),
                    "num_labels": C,
                    "max_len": args.max_len,
                },
                os.path.join(args.outdir, "model.pt"),
            )

    thr_meta = per_freq_thresholds(Y, P, freqs_aligned, bins=args.bins)

    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(
            {
                "max_len": args.max_len,
                "batch_size": args.batch_size,
                "focal_gamma": args.focal_gamma,
                "beta_effnum": args.beta_effnum,
                "bins": args.bins,
                "amp": args.amp,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "weight_mode": args.weight_mode,
                "weights_file": args.weights,
            },
            f,
            indent=2,
        )
    with open(os.path.join(args.outdir, "label_map.json"), "w") as f:
        json.dump({"labels": labels, "freqs": freqs_aligned}, f, indent=2)

    with open(os.path.join(args.outdir, "thresholds.json"), "w") as f:
        json.dump(thr_meta, f, indent=2)

    if args.hierarchy:
        edges = load_hierarchy(args.hierarchy)
        if edges is not None:
            with open(os.path.join(args.outdir, "hierarchy_edges.json"), "w") as f:
                json.dump(edges, f)

    print(f"[DONE] Saved model to {args.outdir}")


def predict(args):
    # --- load config & label artifacts ---
    with open(os.path.join(args.model, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(args.model, "label_map.json")) as f:
        lm = json.load(f)
    with open(os.path.join(args.model, "thresholds.json")) as f:
        thr_meta = json.load(f)

    labels = lm["labels"]
    freqs_aligned = lm["freqs"]
    C = len(labels)
    term_to_idx = {t: i for i, t in enumerate(labels)}

    # optional hierarchy
    edges = None
    hier_path = os.path.join(args.model, "hierarchy_edges.json")
    if os.path.exists(hier_path):
        with open(hier_path) as f:
            edges = json.load(f)

    # --- load model ---
    ckpt = torch.load(os.path.join(args.model, "model.pt"), map_location="cpu")
    max_len = ckpt.get("max_len", config["max_len"])
    model = ProtModel(max_len=max_len, num_labels=C)
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # --- read input ---
    df = read_csv(args.input)
    if "sequence" not in df.columns:
        raise ValueError("Input must contain 'sequence'")
    if "EntryID" not in df.columns:
        # create stable string IDs if missing
        df = df.with_columns(pl.int_range(0, df.height).cast(pl.Utf8).alias("EntryID"))

    seqs = df["sequence"].to_list()
    eids = df["EntryID"].to_list()

    # --- prepare GPU tensors for thresholds & hierarchy ---
    thr_per_class = torch.from_numpy(
        apply_per_freq_thresholds(freqs_aligned, thr_meta)
    ).to(device)

    child_idx_t = parent_idx_t = None
    if edges:
        child_idx, parent_idx = [], []
        for c, p in edges:
            ci = term_to_idx.get(c)
            pi = term_to_idx.get(p)
            if ci is not None and pi is not None:
                child_idx.append(ci)
                parent_idx.append(pi)
        if child_idx:
            child_idx_t = torch.tensor(child_idx, dtype=torch.long, device=device)
            parent_idx_t = torch.tensor(parent_idx, dtype=torch.long, device=device)

    # --- streaming writer (CAFA TSV; no header) ---
    out_path = args.output
    # ensure directory exists
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    f = open(out_path, "w", buffering=1024 * 1024)

    # helper: write positives for one protein
    def _write(pid: str, term_idx_t: torch.Tensor, prob_t: torch.Tensor):
        if term_idx_t.numel() == 0:
            return
        # gather GO term strings on CPU once per protein
        terms_np = np.asarray(labels, dtype=object)[term_idx_t.detach().cpu().numpy()]
        probs_np = prob_t.detach().float().cpu().numpy()
        for go, p in zip(terms_np, probs_np):
            f.write(f"{pid}\t{go}\t{p:.3f}\n")

    # --- batched inference (O(batch) RAM) ---
    batch = args.batch_size
    use_amp = config.get("amp", True) and device.type == "cuda"

    with torch.inference_mode():
        for s in range(0, len(seqs), batch):
            chunk = seqs[s : s + batch]
            # tokenize -> [B, L] long
            X = torch.tensor(
                [seq_to_indices(q, max_len) for q in chunk],
                dtype=torch.long,
                device=device,
            )

            # forward -> probabilities on device
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(X)
                    P_b = torch.sigmoid(logits)
            else:
                logits = model(X)
                P_b = torch.sigmoid(logits)

            # vectorized hierarchy closure (child -> parent), a few rounds
            if child_idx_t is not None and child_idx_t.numel() > 0:
                for _ in range(3):
                    before = P_b.index_select(1, parent_idx_t)
                    after = torch.maximum(before, P_b.index_select(1, child_idx_t))
                    if torch.equal(before, after):
                        break
                    P_b[:, parent_idx_t] = after

            # threshold broadcast -> boolean mask
            pred_mask = P_b >= thr_per_class  # [B, C] bool

            # stream positives per protein
            for i, pid in enumerate(eids[s : s + pred_mask.shape[0]]):
                pos_idx = torch.nonzero(pred_mask[i], as_tuple=False).flatten()
                if pos_idx.numel() == 0:
                    continue
                _write(pid, pos_idx, P_b[i, pos_idx])

            # free batch tensors early
            del X, logits, P_b, pred_mask
            if device.type == "cuda":
                torch.cuda.synchronize()

    f.close()
    print(f"[DONE] Wrote CAFA-format predictions to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser("train")
    p_tr.add_argument("--train", type=str, default="train.csv")
    p_tr.add_argument("--hierarchy", type=str, default=None)
    p_tr.add_argument("--outdir", type=str, default="torch_model")
    p_tr.add_argument("--max_len", type=int, default=1024)
    p_tr.add_argument("--batch_size", type=int, default=256)
    p_tr.add_argument("--epochs", type=int, default=5)
    p_tr.add_argument("--val_frac", type=float, default=0.1)
    p_tr.add_argument("--seed", type=int, default=42)
    p_tr.add_argument("--lr", type=float, default=2e-3)
    p_tr.add_argument("--weight_decay", type=float, default=1e-2)
    p_tr.add_argument("--focal_gamma", type=float, default=1.5)
    p_tr.add_argument("--beta_effnum", type=float, default=0.9995)
    p_tr.add_argument("--bins", type=int, default=5)
    p_tr.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision (recommended on CUDA)",
    )
    p_tr.add_argument("--grad_clip", type=float, default=1.0)
    p_tr.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    p_tr.add_argument(
        "--weights",
        "--weight",
        dest="weights",
        type=str,
        default=None,
        help="CSV with per-term weights file (id + weight column)",
    )
    p_tr.add_argument(
        "--weight_mode",
        type=str,
        default="replace",
        choices=["replace", "multiply"],
        help="Use external weights only, or multiply with effective-number",
    )

    p_pr = sub.add_parser("predict")
    p_pr.add_argument("--model", type=str, required=True)
    p_pr.add_argument("--input", type=str, required=True)
    p_pr.add_argument("--output", type=str, required=True)
    p_pr.add_argument("--batch_size", type=int, default=256)

    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "predict":
        predict(args)


if __name__ == "__main__":
    main()
