# Plan: BirdCLEF+ 2026

Competition: multilabel bird species identification from passive acoustic monitoring (PAM)
recordings in the Pantanal wetlands, South America.

- **Metric**: cmAP (class-mean average precision)
- **Prize**: $50,000
- **Deadline**: May 27, 2026
- **Data**: ~15GB audio (.ogg). 35,549 labeled training clips.
- **Task**: Given 5s soundscape chunk, predict which of 234 species are present (multilabel)

### Data facts (confirmed from EDA)

| Property | Value |
|---|---|
| Training CSV | `train.csv` |
| Training recordings | 35,549 |
| Target species | **234** (from taxonomy.csv) |
| Species with training data | **206** (28 zero-shot, mostly sonotype splits `47158son*`) |
| Primary label format | Mixed: numeric iNat taxon ID OR eBird code (e.g. `ashgre1`) |
| Filename format | Includes subdirectory: `{taxon_id}/{filename}.ogg` |
| Class breakdown | Aves 93%, Amphibia 5%, Insecta 1%, Mammalia <1%, Reptilia <1% |
| Recordings/species | min=1, median=125, max=499 |
| Secondary labels | 4,372 recordings (12%) have secondary species |
| Test row IDs | `{soundscape_stem}_{end_second}` at 5, 10, 15, ... (5s non-overlapping chunks) |

---

## Status

### Current best: LB 0.876 (v5 checkpoints + inference tricks, kernel v15)

Gap to public top notebooks (0.912) = **0.036**. Two distinct public approaches exist — see research section.

**Pending (both submitted Mar 23, awaiting scores):**
- kernel v16: B1 NoisyStudent 4-fold + inference tricks
- Perch v4 Track A kernel v8: frozen Perch + LogReg probes — ready to submit once daily slot frees up

### Results

| Version | LB | Notes |
|---|---|---|
| EfficientNet-B0 plain, 30 epochs | 0.758 | 5-fold, 128-mel |
| EfficientNet-B3 naive SED, 50 epochs | 0.750 | too few temporal frames |
| EfficientNet-B3 plain, 50 epochs | 0.776 | 5-fold, 128-mel |
| B0 NoisyStudent + GEMFreqPool + AttentionSED, 50ep | 0.782 | 224-mel, minmax norm, time_mask=100 |
| B0 baseline, early stopping patience=15 | 0.783 | same arch, time_mask=100, no dual loss |
| B0 baseline + dual loss + time_mask=30 (all 5 folds) | 0.765 | **REGRESSION** |
| EfficientNet-B1 BirdSet XCL pretrained (5-fold) | 0.782 | 256-mel, 1-channel — no gain |
| perch-v2 fold0 only (Perch stage-1 5ep, min_prob=0.1) | 0.746 | single fold |
| soundscape-v1 (53/66 soundscapes, soundscape OOF val, 4-fold) | 0.827 | +0.044 — soundscape labels are key |
| soundscape-v2 (all 66 soundscapes, XC OOF val, 4-fold) | 0.854 | +0.027 vs v1 |
| soundscape-v3 (HTK mel + warm restarts + gain aug, 4-fold) | 0.858 | +0.004 vs v2 |
| soundscape-v4 (+ CE loss) | 0.723 | **REGRESSION** — CE loss incompatible with multilabel |
| **soundscape-v3 + temporal smoothing (kernel v13)** | **0.864** | +0.006 |
| **soundscape-v5 (+ bg noise p=0.3, kernel v14)** | **0.864** | bg noise = neutral |
| **v5 + inference tricks (kernel v15)** | **0.876** | +0.012 — 50% stride + class-type smooth + file-max prior |
| **soundscape-v6-b1 + inference tricks (kernel v16)** | **pending** | B1 NoisyStudent 4-fold, val losses 0.0324–0.0337 |
| **Perch v4 Track A (kernel perch-v8)** | **pending** | frozen Perch + LogReg probes (40/60) + genus proxy + smooth |

### Local validation findings

**Neither clip OOF nor soundscape cmAP reliably predicts LB ranking.** Use LB submissions as ground truth; local metrics only useful as coarse sanity checks.

---

## Public notebook research (Mar 23) — how they score 0.90+

### Two distinct approaches

**Track A: Perch v2 frozen feature extractor — 0.892–0.912**

Google's `google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1` (14,795 species, TF SavedModel) as a frozen embedding extractor. CPU-only, <90 min. No fine-tuning.

Pipeline:
1. Audio → 12 × 5s windows → Perch v2 → 1536-dim embeddings + raw logits (14,795 classes)
2. Species mapping: `taxonomy.scientific_name` → Perch label vocab → BC_INDICES for 234 target classes
3. **Unmapped species** (Amphibia/Insecta with no direct Perch class) → genus-level proxy: `max` logit over all Perch classes sharing the same genus
4. **Class-type-aware temporal smoothing** (critical — this alone gets to 0.908):
   - Texture classes (Amphibia, Insecta — continuous callers): average-neighbor α=0.35
   - Event classes (Aves — discrete calls): **local-max propagation** α=0.15: `(1-α)*x + α*max(x, prev, next)`
5. **Per-class LogReg probes**: PCA 32-64dim of 1536-dim embeddings; GroupKFold by site on 59 fully-labeled soundscapes; blend 60/40 (Perch logits / probe)
6. **Bayesian site×hour prior** from `train_soundscapes_labels.csv` occurrence statistics
7. Temperature scaling: divide logits by T=1.10–1.15 before sigmoid

Score evolution: raw Perch ~0.87 → +texture smooth → ~0.90 → +event local-max → 0.908 → +genus proxies → **0.912**

**Key insight**: Perch knows 14,795 species; we just need to correctly map them to our 234. Genus-level proxies handle the non-Aves classes with no direct Perch class.

**Perch cache workflow**: pre-compute embeddings/logits on all soundscapes, save as parquet+npz, attach as Kaggle dataset. Inference notebook loads cache, never re-runs Perch.

**Track B: Fine-tuned CNN + inference tricks — 0.858–0.892**

Same B0 NoisyStudent backbone, but with inference improvements:
1. **50% stride overlapping windows** (2.5s stride, `2*n_chunks - 1` windows per file, averaged back to 5s slots) — biggest single inference improvement
2. **Circular TTA**: time-shift by 1.25s (CHUNK//4), average with original
3. **File-max prior**: `probs += 0.05 × file_max_per_file` (adds global species prior per soundscape)
4. **Temperature scaling** T=1.10–1.15
5. **Sharpened temporal smoothing**: `[0.10, 0.20, 0.40, 0.20, 0.10]` weights then `probs^(1/SHARPEN_POWER)`
6. **Class-type-aware smoothing** (same as Track A)
7. **Model blend**: 0.8 × fine-tuned + 0.2 × base → 0.892

**HGNetV2-B0** (`hgnetv2_b0.ssld_stage2_ft_in1k`): 256 mel, MixUp (alpha=1.0), OpenVINO for fast CPU inference.

### Key mel config (universal across all high-scorers)
- 224 mel, n_fft=2048, hop=512, fmin=0, fmax=16k, **HTK mel scale**, **slaney norm**, top_db=80
- Per-clip min-max normalization to [0, 1]

### What 0.920–0.937 teams are likely doing (not yet public)
Perch + fine-tuned CNN ensemble, or larger pretrained models with stronger soundscape fine-tuning.

---

## Next steps (ordered by expected impact)

### ✅ Step 1 — Inference notebook improvements (kernel v15, submitted Mar 23)

**Implemented (no retraining):**
- 50% stride overlapping windows (2.5s stride, averaged back to 5s slots)
- Class-type-aware smoothing: texture (Amphibia/Insecta) avg-neighbour α=0.35; event (Aves) local-max α=0.15
- File-max prior: `preds += 0.05 × per-species max` per soundscape
- Taxonomy split from `taxonomy.csv` column `class_name`

**Score: 0.876** (+0.012 vs 0.864 baseline)

Not yet done (lower priority, add if score disappoints):
- Circular TTA 1.25s shift
- Temperature scaling T=1.10

---

### 🔄 Step 2 — Larger backbone B1 NoisyStudent (soundscape-v6-b1, kernel v16 pending)

**Expected: +0.005–0.015 LB over v15 | Risk: CPU inference time**

Tag: `soundscape-v6-b1`. Backbone: `tf_efficientnet_b1.ns_jft_in1k`. Same config as v5 + inference tricks.
- All 4 folds done: val losses 0.0337, 0.0324, 0.0332, 0.0328 (slightly better than B0 v5)
- Dataset: `aldisued/birdclef2026-soundscape-v6-b1` uploaded Mar 23
- Kernel v16 submitted Mar 23 — **score pending**

---

### 🎯 Step 3 — Perch v4 pipeline (new approach)

**Expected: +0.03–0.05 LB → target ~0.90+ | Effort: high | Medium risk**

**Using Perch v4** (10,932 species, 1280-dim embeddings) uploaded as `aldisued/perch-v4-model` Kaggle dataset.

**Progress:**
- [x] `perch_cache_soundscapes.py` written and run — saved `perch_labeled_cache.npz` (792 windows × 66 soundscapes, embeddings shape (792, 1280), logits (792, 10932))
- [x] `train_perch_probes.py` written and run — 53 probes trained (≥8 pos windows), mean AP 0.059 → 0.096 (+0.037); saved `perch_probes.pkl`
- [x] Genus-level proxy built in inference notebook (eBird code prefix matching for unmapped Amphibia/Insecta)
- [x] `kaggle_perch_inference.ipynb` built — Perch v4 + probes blend (40/60) + class-type smooth + file-max prior
- [x] Uploaded `aldisued/perch-v4-model` (84MB) + `aldisued/birdclef2026-perch-v4-artifacts` (35MB) to Kaggle
- [x] Kernel `aldisued/birdclef-2026-perch-v4-inference` v8 — COMPLETE, **ready to submit** (blocked by daily limit Mar 23)
- [ ] Submit kernel v8 and record LB score

**Submit command:**
```bash
kaggle competitions submit -c birdclef-2026 -k aldisued/birdclef-2026-perch-v4-inference -v 8 -f "submission.csv" -m "Perch v4 Track A: frozen embeddings + LogReg probes (40/60 blend) + genus proxy + class-type smooth + file-max prior"
```

**Option B — Perch + CNN ensemble:** after Option A LB is known, blend Perch logits + fine-tuned CNN preds.

**Note on Perch v4 vs v2**: Public notebooks use v2 (14,795 species, 1536-dim, available natively on Kaggle). We use v4 (10,932 species, 1280-dim) uploaded as a dataset. Our probes were trained on v4 embeddings — incompatible with v2.

---

### Step 5 — Multi-year BirdCLEF data pretraining

**Expected: +0.02–0.05 LB | Training: ~8–12h | Medium risk**

BirdCLEF 2021/2022/2023/2024 datasets (~117K clips, all public on Kaggle). Pretrain → fine-tune on 2026. Large effort; defer until Perch approach is validated.

---

## Kaggle setup

- **Notebook**: `aldisued/birdclef-2026-baseline-inference`
- **Current dataset**: `aldisued/birdclef2026-soundscape-v6-b1` (kernel v16 pending) | v15 LB **0.876**
- **CRITICAL**: `enable_gpu: false` — competition GPU limit is 0 min; GPU requests cause silent failure
- Submit: `kaggle competitions submit -c birdclef-2026 -k aldisued/birdclef-2026-baseline-inference -v <int> -f "submission.csv" -m "..."`

---

## Hardware & Training Convention

Training on **2× RTX 3090** at `kristian@omarchyd` (Tailscale).

**Standard: 4 folds (`--n_folds 4`)**. Fits 2 GPUs in exactly 2 clean rounds (2+2).

Launch pattern (use subshell per fold to ensure `cd` applies; use `>>` to not overwrite logs):
```bash
ssh kristian@omarchyd "(cd /home/kristian/projects/kego && CUDA_VISIBLE_DEVICES=0 KEGO_PATH_DATA=/home/kristian/projects/kego/data nohup ~/.local/bin/uv run python competitions/birdclef-2026/train.py --baseline --soundscape-labels --htk --warm-restarts --gain-aug --fold 0 --n_folds 4 --tag <tag> >> /tmp/<tag>_fold0.log 2>&1) &"
ssh kristian@omarchyd "(cd /home/kristian/projects/kego && CUDA_VISIBLE_DEVICES=1 KEGO_PATH_DATA=/home/kristian/projects/kego/data nohup ~/.local/bin/uv run python competitions/birdclef-2026/train.py --baseline --soundscape-labels --htk --warm-restarts --gain-aug --fold 1 --n_folds 4 --tag <tag> >> /tmp/<tag>_fold1.log 2>&1) &"
```

**Checkpoint naming**: always `--tag <experiment-name>`. Saves to `outputs/{tag}_fold{N}.pt`.

### Experiment log

| Tag | Key args | Folds | LB | Notes |
|---|---|---|---|---|
| *(legacy)* | `--baseline`, patience=15, time_mask=100 | 0–4 | 0.783 | |
| `birdset-v1` | `--birdset`, patience=10 | 0–4 | 0.782 | BirdSet XCL — no gain |
| `soundscape-v1` | `--baseline --soundscape-labels`, 53 soundscapes | 0–3 | 0.827 | |
| `soundscape-v2` | `--baseline --soundscape-labels`, all 66 soundscapes | 0–3 | 0.854 | |
| `soundscape-v3` | `--baseline --soundscape-labels --htk --warm-restarts --gain-aug` | 0–3 | 0.858 | |
| `soundscape-v4` | + CE loss | 0–3 | 0.723 | **DEAD END** |
| `soundscape-v5` | + `--bg-noise-p 0.3` | 0–3 | 0.864 | bg noise = neutral; temporal smoothing was the gain |
| `soundscape-v6-b1` | `--backbone tf_efficientnet_b1.ns_jft_in1k` + bg noise | 0–3 | pending | val losses 0.0337/0.0324/0.0332/0.0328 — kernel v16 |

---

## Dead ends / lessons learned

- `enable_gpu: true` → silent scoring failure
- OOF cmAP (clip-level) ≠ LB cmAP; 66 soundscapes cover only 75/234 species → local metrics unreliable
- **Dual loss + time_mask=30**: LB 0.783 → 0.765 (regression)
- **BirdSet XCL pretraining** (B1): LB 0.782 — bird-domain pretraining from generic XC data doesn't help
- **CE loss**: LB 0.723 — `F.cross_entropy` applies softmax, trains model to suppress co-occurring species, destroys multilabel recall. BCE is mandatory.
- **Perch pseudo-labeling on soundscapes**: 99% of 127,896 windows have near-zero signal. Neutral at best (tied 0.0331 val_loss), LB 0.746 for single fold.
- **Background noise (soundscape-v5)**: LB 0.864 = neutral vs v3+temporal smoothing. Label contamination from unlabeled bird calls in bg noise likely cancels any domain benefit.
- **Naive SED head**: 0.750 (worse than plain B3 0.776) — too few temporal frames after 32× downsampling
- **B3 backbone**: 0.776 — bigger not better here; B0 is the sweet spot for CPU inference budget

---

## Perch Pseudo-Labeling — Findings (Mar 21)

Perch on **unlabeled soundscapes** does NOT help (mean prob = 0.000107, 99% windows near-zero). The 0.862 baseline gap was from `train_soundscapes_labels.csv`, not Perch.

**Perch as frozen feature extractor** (Track A above) is different — it uses Perch's own logits/embeddings directly, not as pseudo-labels for CNN training.
