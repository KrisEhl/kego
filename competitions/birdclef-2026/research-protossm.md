# ProtoSSM v5 — Technical Documentation

Source: https://www.kaggle.com/code/dingjiarun/pantanal-distill-birdclef2026-improvement-a4dc68
Author: dingjiarun
Kernel: `pantanal-distill-birdclef2026-improvement-a4dc68`
Retrieved: 2026-04-04

## Overview

This notebook extends the standard Perch v2 pipeline with **ProtoSSM** — a PyTorch bidirectional
Selective State Space Model (Mamba-style) trained on top of frozen Perch embeddings. It also adds
a **ResidualSSM** second-pass correction model. Both models are trained entirely on the 59
fully-labeled soundscape files (708 windows).

Claims: V17 improves over V16 (LB 0.924). V18 is the latest config shown in the notebook.

**Note**: Perch embeddings are 1536-dim (not 8192), 234 target species — identical to our setup.

---

## Full Pipeline

```
Audio (32 kHz, 60s)
  → 12 × 5s windows
  → Perch v2 (frozen TF SavedModel)
      → raw logits (14,795 classes) + 1536-dim embeddings

Training-time (on 59 labeled soundscapes, 708 windows):
  → Prior fusion (Bayesian site×hour) → OOF base scores
  → PCA (128-dim) + MLP/LogReg per-class probes (PROBE_MIN_POS=5)
  → ProtoSSM v5 (4-layer bidirectional SSM, d_model=320)
       trained with: focal BCE + Perch KD loss + taxonomic aux head
       + SWA + Mixup + CosineAnnealingWarmRestarts
  → Residual SSM (2-layer, d_model=128)
       trained on first-pass errors (MSE on residuals)

Test-time:
  → Perch inference (or cache load)
  → Prior fusion → MLP probes → ProtoSSM (5-shift TTA) → Ensemble blend
  → ResidualSSM correction
  → Per-taxon temperature (Aves=1.10, texture=0.95)
  → File-level confidence scaling (top-k=2)
  → Rank-aware post-processing (max^0.4)
  → Adaptive delta smoothing (α=0.20, confidence-weighted)
  → Per-class threshold sharpening (234 hardcoded thresholds from V18 OOF)
  → submission.csv
```

---

## ProtoSSM v5 Architecture (V18 config)

```python
d_input = 1536      # Perch embedding size
d_model = 320       # hidden dimension  (V17: 256)
d_state = 32        # SSM state size    (V17: 16)
n_ssm_layers = 4    # stacked bidirectional SSM blocks  (V17: 3)
n_prototypes = 2    # learnable class prototypes        (V17: 1)
cross_attn_heads = 8  # TemporalCrossAttention heads    (V17: 4)
meta_dim = 24       # site + hour embedding size        (V17: 16)
n_sites = 20        # spatial embedding vocabulary
dropout = 0.12      # (V17: 0.15)
n_windows = 12      # T=12 sequence length
n_classes = 234
```

### Layer sequence per SSM block

1. `SelectiveSSM` (Mamba-style) forward pass
2. `SelectiveSSM` backward pass (sequence flipped)
3. Merge 2×d_model → d_model via Linear + LayerNorm + residual
4. After the final SSM block: `TemporalCrossAttention` (multi-head self-attention)
   captures non-local patterns (e.g. dawn chorus onset, counter-singing)

### SelectiveSSM (simplified Mamba)

- Input split via `in_proj` into `x_ssm` and gate `z`
- Depthwise `conv1d` over time axis
- Input-dependent dt via `dt_proj` + softplus
- Selective A/B/C matrices (B_proj, C_proj are input-dependent)
- Sequential scan over T=12 steps: `h = h * dA + x * dB; y = h @ C`
- Skip connection: `output = y + x * D`

### Classification head

```
h_norm = F.normalize(h, dim=-1)             # per-window embedding
p_norm = F.normalize(prototypes, dim=-1)    # (n_classes, d_model)
sim = h_norm @ p_norm.T * softplus(proto_temp) + class_bias

# Gated fusion with raw Perch logits (key innovation):
alpha = sigmoid(fusion_alpha)               # learnable per class
logits = alpha * sim + (1 - alpha) * perch_logits
```

Each class independently learns how much to trust prototype similarity vs raw Perch logit.

---

## ProtoSSM Training (V18)

| Parameter | Value | vs V17 |
|---|---|---|
| Optimizer | AdamW | same |
| Learning rate | 8e-4 | was 1e-3 |
| Weight decay | 1e-3 | was 2e-3 |
| Scheduler | OneCycleLR + CosineAnnealingWarmRestarts (T_0=20) | new in V18 |
| Epochs | 80 (submit: 40) | was 60 |
| Patience | 20 (submit: 8) | was 15 |
| Batch | Whole dataset (59 files — tiny) | same |
| Loss | Focal BCE (γ=2.5) + 0.15 × Perch KD (MSE) + 0.1 × taxonomic aux BCE | γ 2.0→2.5 |
| Label smoothing | 0.03 | was 0.02 |
| Mixup | File-level Beta(α=0.4), skip first 5 warm-up epochs | α 0.3→0.4 |
| SWA | Start at epoch 65%, swa_lr=4e-4, average ≥3 checkpoints | start 70%→65% |
| pos_weight_cap | 25.0 | was 30.0 |
| OOF folds | 5-fold GroupKFold by site (train mode only) | was 3 |
| Prototype init | Class-mean of projected embeddings from training data | same |
| Taxonomic head | `class_name` → family group → auxiliary BCE loss (weight=0.1) | same |

### Focal BCE loss

```python
def focal_bce_with_logits(logits, targets, gamma=2.0, pos_weight=None):
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    p_t = torch.exp(-bce)
    return ((1 - p_t) ** gamma * bce).mean()
```

### Class frequency weighting

```python
freq_weight = 1 / sqrt(pos_count_per_class)
freq_weight = clip(freq_weight, max=cap)   # cap=10.0 V17, cap=25.0 pos_weight_cap V18
```

---

## Residual SSM (V18)

### Architecture

```python
# Input: concat(1536-dim embeddings, 234-dim first_pass_scores) = 1770-dim
d_input = 1536 + 234
d_model = 128         # (V17: 64)
d_state = 16          # (V17: 8)
n_ssm_layers = 2      # (V17: 1)
dropout = 0.1
```

Single-pass BiSSM on the concatenated input, output head initialized to zeros (corrections start near zero).

### Training

```python
target = labels - sigmoid(first_pass_ensemble)  # residuals in [-1, 1]
loss = MSE(correction, target)

optimizer = AdamW(lr=8e-4, weight_decay=1e-3)
scheduler = OneCycleLR(pct_start=0.1)
epochs = 40, patience = 12
val_fraction = 0.15 (random split, seed=123)
correction_weight = 0.35
```

Applied at test time: `final_scores = first_pass + 0.35 × correction`

### Wall-time guard

```python
# SKIP ResidualSSM if > 4 minutes elapsed since notebook start
if wall_min >= 4.0:
    SKIP
```

---

## MLP Probes (V18)

| Parameter | V18 | Our implementation |
|---|---|---|
| PCA dim | 128 | 32 |
| min_pos | 5 | 8 |
| C (LogReg) | 0.75 | 0.25 |
| probe_alpha | 0.45 | 0.40 |
| MLP hidden | (256, 128) | (64,) |
| MLP lr | 5e-4 | 0.001 |
| MLP alpha L2 | 0.005 | 0.01 |
| MLP max_iter | 500 | 300 |

### Class balancing (MLP — no class_weight support)

```python
# Oversample positives instead of using class_weight
repeat = max(1, n_neg // n_pos)
X_bal = vstack([X_tr, tile(X_tr[pos_idx], (repeat, 1))])
y_bal = concat([y_tr, ones(n_pos * repeat)])
```

### Feature vector (143 features)

```
PCA(128)         | embedding projection
raw_logit        | Perch raw score for this class
prior_logit      | Bayesian site×hour prior
base_logit       | prior-fused score
prev_base        | previous window's base score
next_base        | next window's base score
mean_base        | file-level mean
max_base         | file-level max
std_base         | file-level std (NEW vs our 41-feature set)
diff_mean        | base - mean_base (onset detection)
diff_prev        | base - prev_base (onset detection)
diff_next        | base - next_base (offset detection)
raw × prior      | interaction term
raw × base       | interaction term
prior × base     | interaction term
```

Supports `mlp | lgbm | logreg` backends (CFG["probe_backend"]).

---

## Post-processing Pipeline (V18)

Applied in order after ProtoSSM + ResidualSSM ensemble:

### Step 1 — Per-taxon temperature scaling
```python
T_AVES = 1.10      # divide Aves logits by 1.10 (sharpen slightly)
T_TEXTURE = 0.95   # divide Amphibia/Insecta by 0.95 (soften slightly)
probs = sigmoid(logits / class_temperatures)
```

### Step 2 — File-level confidence scaling (top_k=2)
```python
# For each file, find top-k species by max window score
# Add 0.05 × file_max to all windows for those species
# Boosts species where at least one window is confident
```

### Step 3 — Rank-aware scaling (power=0.4)
```python
file_max = probs.reshape(n_files, n_windows, n_classes).max(axis=1)
scale = file_max ** 0.4   # power transform
probs *= scale[:, None, :]
```

### Step 4 — Adaptive delta smoothing (α=0.20)
```python
# Confidence-weighted smoothing: smoother when uncertain, sharper when confident
conf = current_window_max_prob
a = base_alpha * (1.0 - conf)   # α shrinks as confidence grows
neighbor_avg = (prev + next) / 2
smoothed = (1 - a) * current + a * neighbor_avg
```

### Step 5 — Per-class threshold sharpening
```python
# 234 hardcoded thresholds from V18 OOF (mostly 0.5, some 0.25–0.70)
# Applied as: probs = (probs - threshold) / (1 - threshold) for probs >= threshold
#             probs = probs / (2 * threshold)               for probs < threshold
```

---

## Prior Fusion (same as our implementation, small V18 tweaks)

```python
lambda_event         = 0.45   # V18 (was 0.4 in our implementation)
lambda_texture       = 1.1    # V18 (was 1.0)
lambda_proxy_texture = 0.9    # V18 (was 0.8)
smooth_texture       = 0.35   # unchanged
smooth_event         = 0.15   # local-max propagation for Aves
```

Also extends **genus proxies to unmapped Aves** (not just Amphibia/Insecta):
```python
PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}  # vs our {"Amphibia", "Insecta"}
```

---

## Datasets / Attachments

- BirdCLEF 2026 competition data
- Perch v2 TF SavedModel (`google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1`)
- TensorFlow 2.20.0 custom wheels (StableHLO compatibility)
- Optional: pre-computed Perch cache (`/kaggle/input/perch-meta` — same format as ours)

---

## Key Differences vs Our perch-v2 (LB 0.912)

| Feature | Ours (LB 0.912) | ProtoSSM (LB 0.924+) |
|---|---|---|
| Temporal model | None (LogReg per class) | ProtoSSM: 4-layer BiSSM + cross-attention |
| Second pass | None | ResidualSSM |
| Probe PCA dim | 32 | 128 |
| Probe min_pos | 8 | 5 (more classes) |
| Probe features | 41–43 | 143 |
| Probe alpha | 0.40 | 0.45 |
| Aves genus proxies | No | Yes |
| lambda_event | 0.40 | 0.45 |
| lambda_texture | 1.0 | 1.1 |
| Ensemble weight | Fixed 0.60/0.40 | OOF-optimized grid search |
| TTA | None | 5 temporal window shifts |
| Post-processing | Gaussian smooth only | Rank-aware + delta smooth + per-class thresholds |
| Framework addition | sklearn only | + PyTorch (SSM models) |

---

## Replication Notes

1. **ProtoSSM trains on only 708 rows** (59 files × 12 windows) — tiny dataset. The model is
   small enough to train in ~40 epochs in submit mode without GPU.

2. **The gated fusion per class** (`fusion_alpha`) is the core innovation: each of 234 species
   learns independently how much to trust prototype similarity vs raw Perch logit.

3. **Aves genus proxies** — extending proxy lookup to unmapped Aves species is a small but
   potentially meaningful addition over our approach.

4. **Timing constraints**: submit mode uses 40 epochs + patience=8 (vs 80+20 train mode).
   ResidualSSM is skipped if >4 minutes elapsed. The notebook is tightly time-budgeted for
   the 90-min Kaggle scoring environment.

5. **Per-class thresholds are hardcoded** from a previous OOF run — they do not recompute in
   submit mode. This means V18 is "closed" in the sense that the thresholds were optimized
   externally and baked in.

6. **No GPU** — all PyTorch runs on CPU. T=12 sequence length makes the SSM scan cheap.
   The main bottleneck remains Perch inference (~85 min for ~780 test soundscapes).
