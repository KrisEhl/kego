"""Train ProtoSSM v2 on Perch embeddings from 59 labeled soundscapes.

Architecture: 4-layer bidirectional Selective SSM (Mamba-style) with prototype
classification head, gated Perch fusion, and optional ResidualSSM second pass.
Trained on 708 windows (59 files × 12) from labeled soundscapes.

Usage:
    # Local train mode (80 epochs, 5-fold CV, best weights):
    uv run python competitions/birdclef-2026/training/train_protossm.py --mode train

    # Submit mode (40 epochs, full dataset, for Kaggle kernel):
    uv run python competitions/birdclef-2026/training/train_protossm.py --mode submit

    # Custom output path:
    uv run python competitions/birdclef-2026/training/train_protossm.py \\
        --mode train --output outputs/protossm_v2.pt

Output (outputs/protossm_v1.pt by default):
    {
        'model_state_dict': OrderedDict,
        'oof_scores':       (708, 234) float32  -- only in train mode,
        'config':           dict,
        'species_names':    list[str],
        'site_to_idx':      dict[str, int],
    }
"""

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT = Path(os.environ.get("KEGO_PATH_DATA", "data"))
COMPETITION_DATA = DATA_ROOT / "birdclef" / "birdclef-2026"
PERCH_META_DIR = DATA_ROOT / "perch-meta"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Architecture (V18 config)
D_INPUT = 1536
D_MODEL = 320
D_STATE = 32
N_SSM_LAYERS = 4
N_PROTOTYPES = 2
CROSS_ATTN_HEADS = 8
META_DIM = 24
N_SITES = 20
DROPOUT = 0.12
N_WINDOWS = 12
N_CLASSES = 234

# ResidualSSM second pass (v2 addition)
USE_RESIDUAL_SSM = True
D_RESIDUAL = 128
D_STATE_RESIDUAL = 16
DROPOUT_RESIDUAL = 0.20

# Training — shared
LR = 8e-4
WEIGHT_DECAY = 1e-3
FOCAL_GAMMA = 2.5
KD_WEIGHT = 0.15
AUX_WEIGHT = 0.1
LABEL_SMOOTH = 0.03
MIXUP_ALPHA = 0.4
MIXUP_WARMUP = 5
SWA_LR = 4e-4
POS_WEIGHT_CAP = 25.0
N_FOLDS = 5

TRAIN_EPOCHS = 80
TRAIN_PATIENCE = 20

SUBMIT_EPOCHS = 40
SUBMIT_PATIENCE = 8

SWA_START_FRAC = 0.65


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(data_root: Path) -> dict:
    """Load Perch embeddings, logits, labels, and metadata.

    Returns:
        dict with keys:
            emb:        (708, 1536) float32
            logits:     (708, 234)  float32
            labels:     (708, 234)  float32
            sites:      (708,)      str array (site IDs like 'S08')
            hours:      (708,)      int16 array (hour UTC 0-23)
            filenames:  (708,)      str array
            window_sec: (708,)      int array (5..60)
            species_list: list[str] -- 234 primary_label strings (competition order)
            taxonomy:   DataFrame   -- (234, 5)
    """
    # Load Perch cache
    npz = np.load(PERCH_META_DIR / "full_perch_arrays.npz")
    emb = npz["emb_full"].astype(np.float32)  # (708, 1536)
    scores = npz["scores_full_raw"].astype(np.float32)  # (708, 234)

    meta = pd.read_parquet(PERCH_META_DIR / "full_perch_meta.parquet")
    meta["window_sec"] = meta["row_id"].str.extract(r"_(\d+)$").astype(int)

    assert len(meta) == len(emb), f"Meta rows ({len(meta)}) != NPZ rows ({len(emb)})"

    # Load taxonomy (defines species order and class groups)
    taxonomy = pd.read_csv(COMPETITION_DATA / "taxonomy.csv")
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
    species_list = taxonomy["primary_label"].tolist()
    assert len(species_list) == N_CLASSES
    species_idx = {s: i for i, s in enumerate(species_list)}

    # Build multi-hot label matrix from train_soundscapes_labels.csv
    labels_raw = pd.read_csv(COMPETITION_DATA / "train_soundscapes_labels.csv")

    def time_to_sec(t: str) -> int:
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)

    labels_raw["start_sec"] = labels_raw["start"].apply(time_to_sec)
    # Labels file uses start=0..55 (0-indexed window start); meta uses window_sec=5..60
    # (window_sec is the end of the 5s window, matching the row_id suffix)
    labels_raw["window_sec"] = labels_raw["start_sec"] + 5

    n_rows = len(meta)
    labels = np.zeros((n_rows, N_CLASSES), dtype=np.float32)

    # Build join index: (filename, window_sec) → meta row position
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
        "hours": meta["hour_utc"].values,
        "filenames": meta["filename"].values,
        "window_sec": meta["window_sec"].values,
        "species_list": species_list,
        "taxonomy": taxonomy,
    }


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


class SelectiveSSM(nn.Module):
    """Simplified Mamba-style selective state space model.

    Processes a sequence of length T with d_model-dim inputs.
    """

    def __init__(self, d_model: int, d_state: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Project input into x_ssm and gate z
        self.in_proj = nn.Linear(d_model, d_model * 2)

        # Depthwise conv over time axis (groups=d_model → per-channel)
        self.conv1d = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model
        )

        # Input-dependent projections
        self.dt_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)

        # Fixed diagonal A (log-parameterised, negative so eig < 0)
        A_log = torch.arange(1, d_state + 1, dtype=torch.float32).log()
        self.A_log = nn.Parameter(A_log)  # (d_state,)

        # Skip connection
        self.D = nn.Parameter(torch.ones(d_model))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (T, d_model)

        Returns:
            (T, d_model)
        """
        T, d = x.shape

        # Split into SSM input and gating branch
        xz = self.in_proj(x)  # (T, 2*d_model)
        x_ssm, z = xz.chunk(2, dim=-1)  # each (T, d_model)

        # Depthwise conv over time: (1, d_model, T) → (1, d_model, T) → (T, d_model)
        x_conv = self.conv1d(x_ssm.unsqueeze(0).permute(0, 2, 1))
        x_conv = x_conv.permute(0, 2, 1).squeeze(0)  # (T, d_model)
        x_ssm = F.silu(x_conv)

        # Input-dependent parameters
        dt = F.softplus(self.dt_proj(x_ssm))  # (T, d_model)
        B = self.B_proj(x_ssm)  # (T, d_state)
        C = self.C_proj(x_ssm)  # (T, d_state)

        # Discretize: dA = exp(-exp(A_log) * dt), dB = dt * B (per timestep)
        A = -torch.exp(self.A_log)  # (d_state,)  negative
        # dA: (T, d_model, d_state) — each model dim uses same A_log
        dA = torch.exp(dt.unsqueeze(-1) * A)  # (T, d_model, d_state)

        # Sequential scan over T timesteps
        h = torch.zeros(d, self.d_state, device=x.device)  # (d_model, d_state)
        ys = []
        for t in range(T):
            dBt = dt[t].unsqueeze(-1) * B[t]  # (d_model, d_state)
            h = h * dA[t] + dBt  # (d_model, d_state)
            yt = (h * C[t]).sum(-1)  # (d_model,)
            ys.append(yt)

        y = torch.stack(ys, dim=0)  # (T, d_model)

        # Skip connection
        y = y + x_ssm * self.D

        # Gating
        output = y * F.silu(z)
        return self.dropout(output)


class BidirectionalSSMBlock(nn.Module):
    """One bidirectional SSM block: forward + backward passes, merged residually."""

    def __init__(self, d_model: int, d_state: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm_fwd = SelectiveSSM(d_model, d_state, dropout)
        self.ssm_bwd = SelectiveSSM(d_model, d_state, dropout)
        # Merge 2*d_model → d_model
        self.merge = nn.Linear(d_model * 2, d_model)
        self.merge_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (T, d_model)

        Returns:
            (T, d_model)
        """
        residual = x
        x_norm = self.norm(x)

        fwd_out = self.ssm_fwd(x_norm)  # (T, d_model)
        bwd_out = self.ssm_bwd(x_norm.flip(0)).flip(0)  # (T, d_model)

        merged = self.merge(torch.cat([fwd_out, bwd_out], dim=-1))  # (T, d_model)
        return self.merge_norm(merged + residual)


class ResidualSSM(nn.Module):
    """2-layer BiSSM on first-pass residuals for temporal correction.

    Takes (logits - perch_logits) as input, learns corrections via a lightweight
    BiSSM sequence model. Output projection is zero-initialized so training starts
    as identity (no correction at epoch 0).
    """

    def __init__(
        self,
        n_classes: int,
        d_residual: int = D_RESIDUAL,
        d_state: int = D_STATE_RESIDUAL,
        dropout: float = DROPOUT_RESIDUAL,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(n_classes, d_residual),
            nn.LayerNorm(d_residual),
            nn.Dropout(dropout),
        )
        self.ssm1 = BidirectionalSSMBlock(d_residual, d_state, dropout)
        self.ssm2 = BidirectionalSSMBlock(d_residual, d_state, dropout)
        self.out_proj = nn.Linear(d_residual, n_classes)
        # Zero-init: starts as no-op, gradient pulls toward useful corrections
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            residual: (T, n_classes) first-pass logits minus perch_logits

        Returns:
            (T, n_classes) correction delta to add to logits
        """
        h = self.in_proj(residual)  # (T, d_residual)
        h = self.ssm1(h)  # (T, d_residual)
        h = self.ssm2(h)  # (T, d_residual)
        return self.out_proj(h)  # (T, n_classes)


class TemporalCrossAttention(nn.Module):
    """Multi-head self-attention over the T=12 time axis."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (T, d_model)

        Returns:
            (T, d_model)
        """
        residual = x
        x_norm = self.norm(x).unsqueeze(0)  # (1, T, d_model) for batch_first
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        return residual + self.dropout(attn_out.squeeze(0))


class ProtoSSM(nn.Module):
    """ProtoSSM v2 — bidirectional SSM with prototype classification head + optional ResidualSSM.

    Processes a single soundscape file's 12 windows at a time.
    """

    def __init__(
        self,
        d_input: int = D_INPUT,
        d_model: int = D_MODEL,
        d_state: int = D_STATE,
        n_ssm_layers: int = N_SSM_LAYERS,
        n_prototypes: int = N_PROTOTYPES,
        cross_attn_heads: int = CROSS_ATTN_HEADS,
        meta_dim: int = META_DIM,
        n_sites: int = N_SITES,
        dropout: float = DROPOUT,
        n_classes: int = N_CLASSES,
        n_tax_groups: int = 5,
        has_residual_ssm: bool = USE_RESIDUAL_SSM,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.n_prototypes = n_prototypes
        self.n_tax_groups = n_tax_groups
        self.has_residual_ssm = has_residual_ssm

        # Metadata embeddings (site + hour → meta_dim)
        self.site_embed = nn.Embedding(n_sites + 1, meta_dim // 2, padding_idx=0)
        self.hour_embed = nn.Embedding(24, meta_dim // 2)

        # Input projection: 1536 + meta_dim → d_model
        self.input_proj = nn.Sequential(
            nn.Linear(d_input + meta_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Stacked bidirectional SSM blocks
        self.ssm_layers = nn.ModuleList(
            [
                BidirectionalSSMBlock(d_model, d_state, dropout)
                for _ in range(n_ssm_layers)
            ]
        )

        # Temporal cross-attention after final SSM block
        self.cross_attn = TemporalCrossAttention(d_model, cross_attn_heads, dropout)

        # Prototype classification head
        # n_classes * n_prototypes learnable vectors in d_model space
        self.prototypes = nn.Parameter(torch.randn(n_classes * n_prototypes, d_model))
        nn.init.xavier_uniform_(self.prototypes.view(n_classes * n_prototypes, d_model))

        # Temperature (softplus ensures > 0) and bias per class
        self.proto_temp = nn.Parameter(torch.zeros(1))
        self.class_bias = nn.Parameter(torch.zeros(n_classes))

        # Gated fusion: α per class controls prototype vs Perch logit
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

        # Taxonomic auxiliary head (5 groups)
        self.tax_head = nn.Linear(d_model, n_tax_groups)

        # ResidualSSM second pass (v2 addition)
        if has_residual_ssm:
            self.residual_ssm = ResidualSSM(n_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        emb: torch.Tensor,
        perch_logits: torch.Tensor,
        site_idx: torch.Tensor,
        hour_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for one soundscape (T=12 windows).

        Args:
            emb:          (T, d_input) Perch embeddings
            perch_logits: (T, n_classes) raw Perch logits
            site_idx:     scalar or (1,) int — site embedding index
            hour_idx:     scalar or (1,) int — hour UTC

        Returns:
            logits: (T, n_classes)
            aux_logits: (n_tax_groups,)  taxonomic auxiliary head (mean-pooled)
        """
        T = emb.shape[0]

        # Metadata conditioning
        s_emb = self.site_embed(site_idx).expand(T, -1)  # (T, meta_dim//2)
        h_emb = self.hour_embed(hour_idx).expand(T, -1)  # (T, meta_dim//2)
        meta = torch.cat([s_emb, h_emb], dim=-1)  # (T, meta_dim)

        # Input projection
        x = self.input_proj(torch.cat([emb, meta], dim=-1))  # (T, d_model)

        # SSM layers
        for layer in self.ssm_layers:
            x = layer(x)  # (T, d_model)

        # Temporal cross-attention
        h = self.cross_attn(x)  # (T, d_model)

        # Prototype similarity
        h_norm = F.normalize(h, dim=-1)  # (T, d_model)
        p_norm = F.normalize(self.prototypes, dim=-1)  # (K*P, d_model)
        # sim: (T, K*P) → (T, K, P) → max over P → (T, K)
        sim = (h_norm @ p_norm.T).reshape(T, self.n_classes, self.n_prototypes)
        sim = sim.max(-1).values  # (T, n_classes)
        sim = sim * F.softplus(self.proto_temp) + self.class_bias

        # Gated fusion with Perch logits
        alpha = torch.sigmoid(self.fusion_alpha)  # (n_classes,)
        logits = alpha * sim + (1 - alpha) * perch_logits  # (T, n_classes)

        # ResidualSSM second pass: correct logits based on temporal residual pattern
        if self.has_residual_ssm:
            residual = logits - perch_logits  # (T, n_classes)
            delta = self.residual_ssm(residual)  # (T, n_classes)
            logits = logits + delta

        # Taxonomic aux head (mean over time)
        aux_logits = self.tax_head(h.mean(0))  # (n_tax_groups,)

        return logits, aux_logits


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.5,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Focal binary cross-entropy loss."""
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    p_t = torch.exp(-bce)
    return ((1 - p_t) ** gamma * bce).mean()


def compute_pos_weights(
    labels: np.ndarray, cap: float = POS_WEIGHT_CAP
) -> torch.Tensor:
    """Frequency-based per-class positive weights (1/sqrt(pos_count)), capped."""
    pos_counts = labels.sum(axis=0).astype(np.float32)
    pos_counts = np.maximum(pos_counts, 1.0)
    weights = 1.0 / np.sqrt(pos_counts)
    weights = np.clip(weights, 0.0, cap)
    return torch.tensor(weights, dtype=torch.float32)


def build_tax_labels(taxonomy: pd.DataFrame) -> tuple[torch.Tensor, list[str]]:
    """Build (n_classes, n_groups) one-hot matrix from taxonomy class_name.

    Returns:
        tax_matrix: (n_classes, n_tax_groups) float32
        group_names: list of group name strings
    """
    group_names = sorted(taxonomy["class_name"].unique().tolist())
    group_idx = {g: i for i, g in enumerate(group_names)}
    n_classes = len(taxonomy)
    tax_matrix = np.zeros((n_classes, len(group_names)), dtype=np.float32)
    for i, row in taxonomy.iterrows():
        tax_matrix[i, group_idx[row["class_name"]]] = 1.0
    return torch.tensor(tax_matrix), group_names


# ---------------------------------------------------------------------------
# Prototype initialization
# ---------------------------------------------------------------------------


def init_prototypes(
    model: ProtoSSM,
    emb: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Initialise prototypes as mean of projected embeddings for positive windows.

    For classes with no positive windows, leave xavier init unchanged.
    """
    model.eval()
    with torch.no_grad():
        emb_t = torch.tensor(emb, dtype=torch.float32)
        # Project 1536 → d_model via the linear part of input_proj
        # We use the full input_proj (without meta) — just zero out meta dims
        T_all = emb_t.shape[0]
        meta_zeros = torch.zeros(T_all, model.site_embed.embedding_dim * 2)
        x_in = torch.cat([emb_t, meta_zeros], dim=-1)
        proj_out = model.input_proj(x_in)  # (N, d_model)

        for cls_idx in range(model.n_classes):
            pos_mask = labels[:, cls_idx] > 0
            if pos_mask.sum() == 0:
                continue
            cls_mean = proj_out[pos_mask].mean(0)  # (d_model,)
            # Set all n_prototypes for this class to the mean
            start = cls_idx * model.n_prototypes
            end = start + model.n_prototypes
            model.prototypes.data[start:end] = cls_mean.unsqueeze(0).expand(
                model.n_prototypes, -1
            )
    model.train()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def build_file_batches(
    emb: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    sites: np.ndarray,
    hours: np.ndarray,
    filenames: np.ndarray,
    site_to_idx: dict[str, int],
) -> list[dict]:
    """Group rows by filename into file-level batches (each has T=12 windows).

    Returns list of dicts, one per soundscape file, with keys:
        emb, logits, labels, site_idx, hour_idx, filename
    """
    unique_files = list(dict.fromkeys(filenames))  # preserve order
    batches = []
    for fn in unique_files:
        mask = filenames == fn
        # Sort by window_sec — meta is already sorted but be safe
        idx = np.where(mask)[0]
        batches.append(
            {
                "emb": emb[idx],  # (12, 1536)
                "logits": logits[idx],  # (12, 234)
                "labels": labels[idx],  # (12, 234)
                "site_idx": site_to_idx.get(sites[idx[0]], 0),
                "hour_idx": int(hours[idx[0]]),
                "filename": fn,
            }
        )
    return batches


def mixup_files(
    batch_a: dict,
    batch_b: dict,
    alpha: float = MIXUP_ALPHA,
) -> dict:
    """File-level mixup: mix two file-batches with Beta(alpha) coefficient."""
    lam = np.random.beta(alpha, alpha)
    return {
        "emb": lam * batch_a["emb"] + (1 - lam) * batch_b["emb"],
        "logits": lam * batch_a["logits"] + (1 - lam) * batch_b["logits"],
        "labels": lam * batch_a["labels"] + (1 - lam) * batch_b["labels"],
        "site_idx": batch_a["site_idx"],  # keep file_a metadata
        "hour_idx": batch_a["hour_idx"],
        "filename": batch_a["filename"],
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_model(
    model: ProtoSSM,
    batches_train: list[dict],
    batches_val: list[dict] | None,
    pos_weights: torch.Tensor,
    tax_matrix: torch.Tensor,
    epochs: int,
    patience: int,
    use_mixup: bool = True,
    verbose: bool = True,
) -> tuple[ProtoSSM, dict]:
    """Train the ProtoSSM model.

    Args:
        model:          ProtoSSM instance
        batches_train:  list of file-batch dicts (train set)
        batches_val:    list of file-batch dicts (validation set), or None
        pos_weights:    (n_classes,) tensor of positive weights
        tax_matrix:     (n_classes, n_tax_groups) one-hot tensor
        epochs:         maximum epochs
        patience:       early stopping patience
        use_mixup:      whether to apply mixup augmentation
        verbose:        print progress

    Returns:
        best_model:  ProtoSSM with best-val-loss weights loaded
        history:     dict of lists (train_loss, val_loss)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, eta_min=1e-5
    )

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=SWA_LR)
    swa_start = int(epochs * SWA_START_FRAC)
    swa_active = False
    swa_count = 0

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    n_train = len(batches_train)
    rng = np.random.default_rng(42)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_losses = []

        # Shuffle training batches
        idxs = rng.permutation(n_train)

        for i in idxs:
            batch = batches_train[i]

            # File-level mixup (skip warm-up epochs)
            if use_mixup and epoch > MIXUP_WARMUP:
                j = rng.integers(n_train)
                batch = mixup_files(batch, batches_train[j])

            emb_t = torch.tensor(batch["emb"], dtype=torch.float32)
            logits_perch = torch.tensor(batch["logits"], dtype=torch.float32)
            labels_t = torch.tensor(batch["labels"], dtype=torch.float32)
            site_t = torch.tensor(batch["site_idx"], dtype=torch.long)
            hour_t = torch.tensor(batch["hour_idx"], dtype=torch.long)

            # Label smoothing
            labels_smooth = labels_t * (1 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH

            optimizer.zero_grad()

            logits_out, aux_out = model(emb_t, logits_perch, site_t, hour_t)

            # Focal BCE
            loss_focal = focal_bce_with_logits(
                logits_out, labels_smooth, gamma=FOCAL_GAMMA, pos_weight=pos_weights
            )

            # Perch KD (MSE on sigmoid outputs)
            loss_kd = F.mse_loss(torch.sigmoid(logits_out), torch.sigmoid(logits_perch))

            # Taxonomic aux BCE (mean-pooled labels → group labels)
            # labels_t: (T, n_classes), tax_matrix: (n_classes, n_tax_groups)
            tax_targets = (labels_t.mean(0) @ tax_matrix).clamp(0, 1)  # (n_tax_groups,)
            loss_aux = F.binary_cross_entropy_with_logits(aux_out, tax_targets)

            loss = loss_focal + KD_WEIGHT * loss_kd + AUX_WEIGHT * loss_aux
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        mean_train_loss = float(np.mean(train_losses))

        # SWA update
        if epoch >= swa_start:
            if not swa_active:
                swa_active = True
                if verbose:
                    print(f"  [SWA started at epoch {epoch}]")
            swa_model.update_parameters(model)
            swa_scheduler.step()
            swa_count += 1
        else:
            scheduler.step()

        # Validation
        val_loss = float("nan")
        if batches_val:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in batches_val:
                    emb_t = torch.tensor(batch["emb"], dtype=torch.float32)
                    logits_perch = torch.tensor(batch["logits"], dtype=torch.float32)
                    labels_t = torch.tensor(batch["labels"], dtype=torch.float32)
                    site_t = torch.tensor(batch["site_idx"], dtype=torch.long)
                    hour_t = torch.tensor(batch["hour_idx"], dtype=torch.long)

                    logits_out, aux_out = model(emb_t, logits_perch, site_t, hour_t)

                    loss_focal = focal_bce_with_logits(
                        logits_out, labels_t, gamma=FOCAL_GAMMA, pos_weight=pos_weights
                    )
                    loss_kd = F.mse_loss(
                        torch.sigmoid(logits_out), torch.sigmoid(logits_perch)
                    )
                    tax_targets = (labels_t.mean(0) @ tax_matrix).clamp(0, 1)
                    loss_aux = F.binary_cross_entropy_with_logits(aux_out, tax_targets)
                    v_loss = loss_focal + KD_WEIGHT * loss_kd + AUX_WEIGHT * loss_aux
                    val_losses.append(v_loss.item())

            val_loss = float(np.mean(val_losses))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        elapsed = time.time() - t0
        history["train_loss"].append(mean_train_loss)
        history["val_loss"].append(val_loss)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            val_str = f"  val={val_loss:.4f}" if not math.isnan(val_loss) else ""
            print(
                f"  Epoch {epoch:3d}/{epochs}  train={mean_train_loss:.4f}{val_str}"
                f"  [{elapsed:.1f}s]  lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if batches_val and epochs_no_improve >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch} (patience={patience})")
            break

    # Finalise SWA — ProtoSSM uses LayerNorm (no BatchNorm), so update_bn is not needed.
    if swa_count >= 3:
        if verbose:
            print(f"  Finalising SWA ({swa_count} checkpoints)...")
        best_state = {k: v.clone() for k, v in swa_model.module.state_dict().items()}
        if verbose:
            print("  Using SWA-averaged weights.")

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ---------------------------------------------------------------------------
# OOF evaluation
# ---------------------------------------------------------------------------


def predict_batches(
    model: ProtoSSM,
    batches: list[dict],
) -> np.ndarray:
    """Run model inference on a list of file-batches.

    Returns:
        (N_windows, n_classes) float32 sigmoid probabilities
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in batches:
            emb_t = torch.tensor(batch["emb"], dtype=torch.float32)
            logits_perch = torch.tensor(batch["logits"], dtype=torch.float32)
            site_t = torch.tensor(batch["site_idx"], dtype=torch.long)
            hour_t = torch.tensor(batch["hour_idx"], dtype=torch.long)
            logits_out, _ = model(emb_t, logits_perch, site_t, hour_t)
            probs = torch.sigmoid(logits_out).numpy()  # (T, n_classes)
            preds.append(probs)
    return np.concatenate(preds, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ProtoSSM v5 on Perch soundscape embeddings"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "submit"],
        default="train",
        help="train=80 epochs + 5-fold CV; submit=40 epochs on full dataset",
    )
    parser.add_argument(
        "--output",
        default="outputs/protossm_v1.pt",
        help="Output checkpoint path",
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
        help="Random seed",
    )
    args = parser.parse_args()

    if args.data_dir:
        os.environ["KEGO_PATH_DATA"] = args.data_dir

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t_start = time.time()
    print(f"[ProtoSSM v5 | mode={args.mode}]")
    print(f"Data root: {DATA_ROOT}")

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    print("\n--- Loading data ---")
    data = load_data(DATA_ROOT)
    emb = data["emb"]  # (708, 1536)
    logits = data["logits"]  # (708, 234)
    labels = data["labels"]  # (708, 234)
    sites = data["sites"]
    hours = data["hours"]
    filenames = data["filenames"]
    species_list = data["species_list"]
    taxonomy = data["taxonomy"]

    # Site vocabulary
    all_sites = sorted(set(sites.tolist()))
    site_to_idx = {s: i + 1 for i, s in enumerate(all_sites)}  # 0 = padding
    print(f"Sites: {all_sites} (vocab size {len(site_to_idx)})")

    # Positive weights and taxonomic matrix
    pos_weights = compute_pos_weights(labels, cap=POS_WEIGHT_CAP)
    tax_matrix, group_names = build_tax_labels(taxonomy)
    print(f"Taxonomic groups: {group_names}")

    # File-level batches (each file = 12 windows in sequence order)
    # Re-sort meta so window_sec order is preserved per file
    # (already sorted in the NPZ — verified above)
    all_batches = build_file_batches(
        emb, logits, labels, sites, hours, filenames, site_to_idx
    )
    print(f"File batches: {len(all_batches)} files × 12 windows")

    # -----------------------------------------------------------------------
    # Config dict for checkpoint
    # -----------------------------------------------------------------------
    config = {
        "d_input": D_INPUT,
        "d_model": D_MODEL,
        "d_state": D_STATE,
        "n_ssm_layers": N_SSM_LAYERS,
        "n_prototypes": N_PROTOTYPES,
        "cross_attn_heads": CROSS_ATTN_HEADS,
        "meta_dim": META_DIM,
        "n_sites": N_SITES,
        "dropout": DROPOUT,
        "n_windows": N_WINDOWS,
        "n_classes": N_CLASSES,
        "n_tax_groups": len(group_names),
        "tax_group_names": group_names,
        "mode": args.mode,
        "epochs": TRAIN_EPOCHS if args.mode == "train" else SUBMIT_EPOCHS,
        "seed": args.seed,
        "has_residual_ssm": USE_RESIDUAL_SSM,
        "d_residual": D_RESIDUAL,
        "d_state_residual": D_STATE_RESIDUAL,
    }

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    epochs = TRAIN_EPOCHS if args.mode == "train" else SUBMIT_EPOCHS
    patience = TRAIN_PATIENCE if args.mode == "train" else SUBMIT_PATIENCE

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "submit":
        # ----- Submit mode: train on full dataset, no CV -----
        print(f"\n--- Submit mode: training on all {len(all_batches)} files ---")
        print(f"Epochs={epochs}, patience={patience}")

        model = ProtoSSM(n_tax_groups=len(group_names))
        init_prototypes(model, emb, labels)

        model, history = train_model(
            model=model,
            batches_train=all_batches,
            batches_val=None,
            pos_weights=pos_weights,
            tax_matrix=tax_matrix,
            epochs=epochs,
            patience=patience,
            use_mixup=True,
            verbose=True,
        )

        t_total = time.time() - t_start
        print(f"\nTotal time: {t_total:.1f}s ({t_total / 60:.1f}min)")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "species_names": species_list,
                "site_to_idx": site_to_idx,
            },
            output_path,
        )
        print(f"Saved to {output_path}")

    else:
        # ----- Train mode: 5-fold GroupKFold by site -----
        print(f"\n--- Train mode: {N_FOLDS}-fold GroupKFold by site ---")
        print(f"Epochs={epochs}, patience={patience}")

        # Build file-level arrays for GroupKFold
        file_list = [b["filename"] for b in all_batches]
        file_sites = [
            b["filename"].split("_")[3] if "_" in b["filename"] else "UNK"
            for b in all_batches
        ]
        # Use the actual site from batch dict
        file_sites = [sites[filenames == fn][0] for fn in file_list]

        gkf = GroupKFold(n_splits=N_FOLDS)
        # Groups for GroupKFold: use site
        file_groups = np.array(file_sites)
        file_indices = np.arange(len(all_batches))

        # OOF scores stored per-row (708 windows total)
        oof_scores = np.zeros((len(emb), N_CLASSES), dtype=np.float32)
        # Row index for each file-batch
        file_to_rows: dict[str, list[int]] = {}
        for i, fn in enumerate(filenames):
            if fn not in file_to_rows:
                file_to_rows[fn] = []
            file_to_rows[fn].append(i)

        fold_models = []
        fold_val_aps = []

        for fold, (train_idx, val_idx) in enumerate(
            gkf.split(file_indices, groups=file_groups)
        ):
            t_fold = time.time()
            print(f"\n=== Fold {fold + 1}/{N_FOLDS} ===")
            batches_tr = [all_batches[i] for i in train_idx]
            batches_vl = [all_batches[i] for i in val_idx]
            val_sites = sorted(set(file_groups[val_idx]))
            print(f"  Train files: {len(batches_tr)} | Val files: {len(batches_vl)}")
            print(f"  Val sites: {val_sites}")

            # Pos weights from training fold only
            train_row_idx = []
            for b in batches_tr:
                train_row_idx.extend(file_to_rows[b["filename"]])
            fold_pos_weights = compute_pos_weights(
                labels[train_row_idx], cap=POS_WEIGHT_CAP
            )

            model = ProtoSSM(n_tax_groups=len(group_names))
            init_prototypes(model, emb[train_row_idx], labels[train_row_idx])

            model, history = train_model(
                model=model,
                batches_train=batches_tr,
                batches_val=batches_vl,
                pos_weights=fold_pos_weights,
                tax_matrix=tax_matrix,
                epochs=epochs,
                patience=patience,
                use_mixup=True,
                verbose=True,
            )

            # OOF predictions
            val_preds = predict_batches(model, batches_vl)
            # Place back into OOF array
            val_row_idx = []
            for b in batches_vl:
                val_row_idx.extend(file_to_rows[b["filename"]])
            oof_scores[val_row_idx] = val_preds

            # Quick per-fold cmAP on classes with positives in val set
            val_labels = labels[val_row_idx]
            active_cls = np.where(val_labels.sum(0) > 0)[0]
            if len(active_cls) > 0:
                from sklearn.metrics import average_precision_score

                aps = []
                for c in active_cls:
                    if val_labels[:, c].sum() > 0:
                        ap = average_precision_score(val_labels[:, c], val_preds[:, c])
                        aps.append(ap)
                mean_ap = float(np.mean(aps)) if aps else float("nan")
                fold_val_aps.append(mean_ap)
                print(
                    f"  Fold {fold + 1} OOF mean-AP ({len(aps)} classes): {mean_ap:.4f}"
                )

            fold_models.append(model.state_dict())
            t_fold_end = time.time() - t_fold
            print(
                f"  Fold {fold + 1} time: {t_fold_end:.1f}s ({t_fold_end / 60:.1f}min)"
            )

        if fold_val_aps:
            print(f"\nOOF cmAP across folds: {np.mean(fold_val_aps):.4f}")

        # Retrain on full dataset for the final artifact
        print("\n--- Retraining on full dataset for final artifact ---")
        model_final = ProtoSSM(n_tax_groups=len(group_names))
        init_prototypes(model_final, emb, labels)

        model_final, _ = train_model(
            model=model_final,
            batches_train=all_batches,
            batches_val=None,
            pos_weights=pos_weights,
            tax_matrix=tax_matrix,
            epochs=epochs,
            patience=patience,
            use_mixup=True,
            verbose=True,
        )

        t_total = time.time() - t_start
        print(f"\nTotal time: {t_total:.1f}s ({t_total / 60:.1f}min)")

        torch.save(
            {
                "model_state_dict": model_final.state_dict(),
                "fold_model_states": fold_models,
                "oof_scores": oof_scores,
                "config": config,
                "species_names": species_list,
                "site_to_idx": site_to_idx,
            },
            output_path,
        )
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
