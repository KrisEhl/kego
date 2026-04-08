"""Train a lightweight Perch Embedding Adapter for Pantanal-specific representations.

Architecture:
    PerchEmbeddingAdapter: emb_adapted = emb + α * MLP(concat(emb, logits))
    - Input: concat(emb 1536, perch_logits 234) = 1770 dims
    - 2-layer MLP: Linear → LayerNorm → GELU → Dropout → Linear (zero-init)
    - Residual: emb_out = emb_in + α * delta, α ≈ 0.1 init (log_alpha = -2.3), learned
    - ~1.8M parameters total

Training:
    - Linear head proxy (1536 → 234) as surrogate for probes
    - BCE loss with per-class positive weights (cap=5.0)
    - Adam LR=1e-3, weight_decay=1e-4
    - Max 200 epochs, early stopping patience=30 on val BCE
    - Val split: 20% of soundscapes (rng_split seed=42)
    - Batch: all windows for a soundscape at once (N_WINDOWS per file)

After training:
    - Saves adapter checkpoint: {"adapter_state_dict": ..., "log_alpha": ...}
    - Generates and saves adapted embeddings to data/perch-meta/full_emb_adapted.npy

Usage:
    uv run python competitions/birdclef-2026/training/train_perch_adapter.py \\
        --output outputs/perch_adapter.pt
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT = Path(os.environ.get("KEGO_PATH_DATA", "data"))
COMPETITION_DATA = DATA_ROOT / "birdclef" / "birdclef-2026"
PERCH_META_DIR = DATA_ROOT / "perch-meta"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

N_CLASSES = 234
D_EMB = 1536
D_LOGITS = 234
D_HIDDEN = 512
DROPOUT = 0.2
N_WINDOWS = 12

POS_WEIGHT_CAP = 5.0
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 200
PATIENCE = 30
VAL_FRAC = 0.20


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PerchEmbeddingAdapter(nn.Module):
    """Transforms frozen Perch embeddings into Pantanal-specific representations.

    Input:  concat(emb 1536, perch_logits 234) = 1770 dims
    Output: adapted emb (1536 dims). Zero-init so starts as identity.
    """

    def __init__(
        self,
        d_emb: int = D_EMB,
        d_logits: int = D_LOGITS,
        d_hidden: int = D_HIDDEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_emb + d_logits, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_emb),
        )
        # Zero-init output layer → adapter starts as identity
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
        # log_alpha = -2.3 → α ≈ 0.1 at init
        self.log_alpha = nn.Parameter(torch.tensor(-2.3))

    def forward(self, emb: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Apply adapter.

        Args:
            emb:    (N, 1536) Perch embeddings
            logits: (N, 234)  raw Perch logits

        Returns:
            (N, 1536) adapted embeddings
        """
        x = torch.cat([emb, logits], dim=-1)
        delta = self.proj(x)
        return emb + self.log_alpha.exp().clamp(max=0.5) * delta


class AdapterWithHead(nn.Module):
    """Adapter + linear classification head (used only for training signal)."""

    def __init__(self, adapter: PerchEmbeddingAdapter, n_classes: int = N_CLASSES):
        super().__init__()
        self.adapter = adapter
        self.head = nn.Linear(D_EMB, n_classes)

    def forward(self, emb: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        adapted = self.adapter(emb, logits)
        return self.head(adapted)


# ---------------------------------------------------------------------------
# Data loading (standalone, no dependency on train_protossm.py)
# ---------------------------------------------------------------------------


def load_data(data_root: Path) -> dict:
    """Load Perch embeddings, logits, labels, and metadata.

    Returns:
        dict with keys:
            emb:        (N, 1536) float32
            logits:     (N, 234)  float32
            labels:     (N, 234)  float32
            sites:      (N,)      str array
            filenames:  (N,)      str array
            species_list: list[str]
    """
    npz = np.load(PERCH_META_DIR / "full_perch_arrays.npz")
    emb = npz["emb_full"].astype(np.float32)
    scores = npz["scores_full_raw"].astype(np.float32)

    meta = pd.read_parquet(PERCH_META_DIR / "full_perch_meta.parquet")
    meta["window_sec"] = meta["row_id"].str.extract(r"_(\d+)$").astype(int)

    assert len(meta) == len(emb), f"Meta rows ({len(meta)}) != NPZ rows ({len(emb)})"

    taxonomy = pd.read_csv(COMPETITION_DATA / "taxonomy.csv")
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
    species_list = taxonomy["primary_label"].tolist()
    assert len(species_list) == N_CLASSES
    species_idx = {s: i for i, s in enumerate(species_list)}

    labels_raw = pd.read_csv(COMPETITION_DATA / "train_soundscapes_labels.csv")

    def time_to_sec(t: str) -> int:
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)

    labels_raw["start_sec"] = labels_raw["start"].apply(time_to_sec)
    labels_raw["window_sec"] = labels_raw["start_sec"] + 5

    n_rows = len(meta)
    labels = np.zeros((n_rows, N_CLASSES), dtype=np.float32)

    meta_idx_map: dict[tuple[str, int], int] = {}
    for pos, row in meta.iterrows():
        meta_idx_map[(row["filename"], int(row["window_sec"]))] = int(pos)

    for _, row in labels_raw.iterrows():
        key = (row["filename"], int(row["window_sec"]))
        if key not in meta_idx_map:
            continue
        pos = meta_idx_map[key]
        for lbl in str(row["primary_label"]).split(";"):
            lbl = lbl.strip()
            if lbl in species_idx:
                labels[pos, species_idx[lbl]] = 1.0

    print(
        f"Loaded {n_rows} windows, {labels.sum():.0f} positive labels, "
        f"{(labels.sum(axis=0) > 0).sum()} species with ≥1 positive"
    )

    return {
        "emb": emb,
        "logits": scores,
        "labels": labels,
        "sites": meta["site"].values,
        "filenames": meta["filename"].values,
    }


def build_file_batches(
    emb: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    filenames: np.ndarray,
) -> list[dict]:
    """Group rows by filename into file-level batches.

    Returns list of dicts with keys: emb, logits, labels, filename.
    """
    unique_files = list(dict.fromkeys(filenames))
    batches = []
    for fn in unique_files:
        idx = np.where(filenames == fn)[0]
        batches.append(
            {
                "emb": emb[idx],
                "logits": logits[idx],
                "labels": labels[idx],
                "filename": fn,
            }
        )
    return batches


def compute_pos_weights(
    labels: np.ndarray, cap: float = POS_WEIGHT_CAP
) -> torch.Tensor:
    """Frequency-based per-class positive weights (1/sqrt(pos_count)), capped."""
    pos_counts = labels.sum(axis=0).astype(np.float32)
    pos_counts = np.maximum(pos_counts, 1.0)
    weights = 1.0 / np.sqrt(pos_counts)
    weights = np.clip(weights, 0.0, cap)
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_adapter(
    model: AdapterWithHead,
    batches_train: list[dict],
    batches_val: list[dict],
    pos_weights: torch.Tensor,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> dict:
    """Train the adapter with a linear head proxy.

    Returns:
        history dict with train_loss and val_loss lists
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(42)

    best_val_loss = float("inf")
    best_adapter_state = None
    epochs_no_improve = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        train_losses = []

        idxs = rng.permutation(len(batches_train))
        for i in idxs:
            batch = batches_train[i]
            emb_t = torch.tensor(batch["emb"], dtype=torch.float32)
            logits_t = torch.tensor(batch["logits"], dtype=torch.float32)
            labels_t = torch.tensor(batch["labels"], dtype=torch.float32)

            optimizer.zero_grad()
            pred = model(emb_t, logits_t)
            loss = F.binary_cross_entropy_with_logits(
                pred, labels_t, pos_weight=pos_weights
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        mean_train = float(np.mean(train_losses))

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in batches_val:
                emb_t = torch.tensor(batch["emb"], dtype=torch.float32)
                logits_t = torch.tensor(batch["logits"], dtype=torch.float32)
                labels_t = torch.tensor(batch["labels"], dtype=torch.float32)
                pred = model(emb_t, logits_t)
                val_loss = F.binary_cross_entropy_with_logits(
                    pred, labels_t, pos_weight=pos_weights
                )
                val_losses.append(val_loss.item())

        mean_val = float(np.mean(val_losses))

        history["train_loss"].append(mean_train)
        history["val_loss"].append(mean_val)

        elapsed = time.time() - t0
        if epoch % 10 == 0 or epoch <= 5:
            alpha_val = model.adapter.log_alpha.exp().item()
            improved = "*" if mean_val < best_val_loss else ""
            print(
                f"  Epoch {epoch:3d}/{max_epochs}"
                f"  train={mean_train:.5f}  val={mean_val:.5f}"
                f"  α={alpha_val:.4f}  [{elapsed:.1f}s] {improved}",
                flush=True,
            )

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_adapter_state = {
                k: v.clone() for k, v in model.adapter.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"  Early stop at epoch {epoch} (best val={best_val_loss:.5f})",
                    flush=True,
                )
                break

    # Restore best adapter weights
    if best_adapter_state is not None:
        model.adapter.load_state_dict(best_adapter_state)
        print(f"  Restored best adapter (val={best_val_loss:.5f})", flush=True)

    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PerchEmbeddingAdapter for Pantanal-specific embeddings"
    )
    parser.add_argument(
        "--output",
        default="outputs/perch_adapter.pt",
        help="Output checkpoint path (default: outputs/perch_adapter.pt)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override KEGO_PATH_DATA (default: env var or 'data/')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for val split (default: 42)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=MAX_EPOCHS,
        help=f"Max training epochs (default: {MAX_EPOCHS})",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
        help=f"Early stopping patience (default: {PATIENCE})",
    )
    args = parser.parse_args()

    if args.data_dir:
        os.environ["KEGO_PATH_DATA"] = args.data_dir

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t_start = time.time()
    print(f"[PerchEmbeddingAdapter training | seed={args.seed}]", flush=True)
    print(f"Data root: {DATA_ROOT}", flush=True)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    print("\n--- Loading data ---", flush=True)
    data = load_data(DATA_ROOT)
    emb = data["emb"]
    logits = data["logits"]
    labels = data["labels"]
    filenames = data["filenames"]
    sites = data["sites"]

    n_windows = len(emb)
    print(f"Windows: {n_windows}, emb: {emb.shape}, logits: {logits.shape}", flush=True)

    # -----------------------------------------------------------------------
    # Val split: 20% of soundscapes (files)
    # -----------------------------------------------------------------------
    unique_files = list(dict.fromkeys(filenames))
    rng_split = np.random.default_rng(42)  # fixed seed for reproducibility
    n_val_files = max(1, int(len(unique_files) * VAL_FRAC))
    val_file_idx = rng_split.choice(len(unique_files), size=n_val_files, replace=False)
    val_file_set = {unique_files[i] for i in val_file_idx}
    print(
        f"Val files: {n_val_files} / {len(unique_files)} soundscapes",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Build file-level batches
    # -----------------------------------------------------------------------
    all_batches = build_file_batches(emb, logits, labels, filenames)
    train_batches = [b for b in all_batches if b["filename"] not in val_file_set]
    val_batches = [b for b in all_batches if b["filename"] in val_file_set]
    print(f"Batches: {len(train_batches)} train, {len(val_batches)} val", flush=True)

    # -----------------------------------------------------------------------
    # Compute positive weights (on train set only)
    # -----------------------------------------------------------------------
    train_row_mask = np.isin(filenames, [b["filename"] for b in train_batches])
    pos_weights = compute_pos_weights(labels[train_row_mask], cap=POS_WEIGHT_CAP)
    print(
        f"Pos weights: min={pos_weights.min():.3f}, max={pos_weights.max():.3f}, "
        f"mean={pos_weights.mean():.3f}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------
    adapter = PerchEmbeddingAdapter(
        d_emb=D_EMB, d_logits=D_LOGITS, d_hidden=D_HIDDEN, dropout=DROPOUT
    )
    model = AdapterWithHead(adapter, n_classes=N_CLASSES)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_adapter_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(
        f"Model: {n_params:,} total params, {n_adapter_params:,} adapter params",
        flush=True,
    )
    print(
        f"Initial α = {adapter.log_alpha.exp().item():.4f} "
        f"(log_alpha={adapter.log_alpha.item():.3f})",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    print(
        f"\n--- Training (max {args.max_epochs} epochs, patience={args.patience}) ---",
        flush=True,
    )
    history = train_adapter(
        model=model,
        batches_train=train_batches,
        batches_val=val_batches,
        pos_weights=pos_weights,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )

    final_alpha = adapter.log_alpha.exp().item()
    print(f"\nFinal α = {final_alpha:.4f}", flush=True)

    # -----------------------------------------------------------------------
    # Save adapter checkpoint
    # -----------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state_dict": adapter.state_dict(),
            "log_alpha": adapter.log_alpha.item(),
            "alpha": final_alpha,
            "d_emb": D_EMB,
            "d_logits": D_LOGITS,
            "d_hidden": D_HIDDEN,
            "dropout": DROPOUT,
        },
        output_path,
    )
    print(f"Adapter checkpoint saved to {output_path}", flush=True)

    # -----------------------------------------------------------------------
    # Generate and save adapted embeddings for all N windows
    # -----------------------------------------------------------------------
    print("\n--- Generating adapted embeddings ---", flush=True)
    adapter.eval()
    emb_adapted_list = []
    with torch.no_grad():
        batch_size = 256
        for start in range(0, n_windows, batch_size):
            end = min(start + batch_size, n_windows)
            emb_t = torch.tensor(emb[start:end], dtype=torch.float32)
            logits_t = torch.tensor(logits[start:end], dtype=torch.float32)
            adapted = adapter(emb_t, logits_t).numpy()
            emb_adapted_list.append(adapted)

    emb_adapted = np.concatenate(emb_adapted_list, axis=0).astype(np.float32)
    assert emb_adapted.shape == emb.shape, (
        f"Shape mismatch: {emb_adapted.shape} vs {emb.shape}"
    )

    # Compute delta stats for diagnostics
    delta = emb_adapted - emb
    print(
        f"Adapted emb shape: {emb_adapted.shape}  dtype={emb_adapted.dtype}",
        flush=True,
    )
    print(
        f"Delta stats: mean={delta.mean():.6f}, std={delta.std():.6f}, "
        f"max_abs={np.abs(delta).max():.6f}",
        flush=True,
    )

    adapted_path = PERCH_META_DIR / "full_emb_adapted.npy"
    np.save(adapted_path, emb_adapted)
    print(f"Adapted embeddings saved to {adapted_path}", flush=True)

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s ({t_total / 60:.1f}min)", flush=True)
    print(f"α = {final_alpha:.4f}", flush=True)
    print(
        f"Best val BCE: {min(history['val_loss']):.5f} "
        f"at epoch {history['val_loss'].index(min(history['val_loss'])) + 1}",
        flush=True,
    )


if __name__ == "__main__":
    main()
