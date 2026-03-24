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

**Pending:**
- kernel v21: soundscape-v7 + class-cond pooling + persistence penalty (timeout fixes applied)
- Perch v4 Track A kernel v8: ready to submit

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
| **soundscape-v6-b1 + inference tricks (kernel v16)** | **TIMEOUT** | B1 NoisyStudent 4-fold — CPU too slow |
| **v18/v19: class-cond pooling + persistence penalty + circular TTA** | **TIMEOUT** | rglob + TTA pushed over 90-min limit |
| **v21: soundscape-v7 + class-cond pooling + persistence penalty** | **pending** | mixup α=1.0, timeout fixes (TTA off, rglob→iterdir) |
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

The literature and competition pattern point to: **calibrated Perch + fine-tuned CNN ensemble, with per-class threshold learning and site/time priors**. The jump from 0.912 to 0.937 is most likely per-class calibration + co-occurrence priors + a stronger CNN (B3/B4 or EfficientNetV2) — not just a bigger backbone.

Key insights from field literature (see `research-lit.md`):
- The biggest jump is **not** a larger CNN. It is a calibrated ensemble of Perch + CNN + structured priors.
- Per-class calibration (Platt scaling / temperature per class) on the 66 labeled soundscapes is high-value, zero training cost.
- Temporal post-processing is **not a hack** — it reflects genuine biological structure. More acoustic archetypes (transient/repetitive/chorus/continuous/rare) outperform a binary texture/event split.
- PCEN as a second input channel alongside log-mel is a consistent literature recommendation for noisy field recordings.
- Co-occurrence priors (species PMI from training soundscapes) can fix borderline detections.
- Pseudo-labeling only helps when the teacher has confident positives — confirmed by our own 0.746 result.

---

## Next steps (ordered by expected impact)

### ✅ Step 1 — Inference notebook improvements (kernel v15, LB 0.876)

- 50% stride overlapping windows, class-type-aware smoothing, file-max prior
- **Score: 0.876** (+0.012 vs 0.864 baseline)

---

### ❌ Step 2 — B1 NoisyStudent backbone (soundscape-v6-b1, kernel v16) — DEAD END

**CPU timeout** — EfficientNet-B1 + 50% overlap is too slow on Kaggle CPU. B0/B3 remain the only viable backbones for CPU inference within time limits.

---

### ⚠️ Inference time budget — CRITICAL

**90-min Kaggle CPU limit is tight.** v15 (LB 0.876) was close to the edge; v18/v19 timed out due to:
1. `rglob("*")` over all `/kaggle/input` scanned 35K+ train audio paths — fixed to `iterdir()`
2. Circular TTA doubles inference time — disabled (`tta=False`), keep as opt-in

**Circular TTA**: do NOT re-enable unless timing confirmed safe with a headroom test. Once v21 LB score is known, estimate remaining budget before enabling TTA.

---

### 🔄 Step 3 — Perch v4 Track A (kernel perch-v8)

**Expected: +0.03–0.05 LB → ~0.90+ | Using Perch v4 (10,932 species, 1280-dim)**

- [x] Cache: `perch_labeled_cache.npz` (792 × 1280 emb, 792 × 10932 logits)
- [x] Probes: 53 LogReg probes, mean AP 0.059 → 0.096 (+0.037 OOF)
- [x] Kernel `aldisued/birdclef-2026-perch-v4-inference` v8 COMPLETE — **ready to submit**
- [ ] Submit and record LB score

```bash
kaggle competitions submit -c birdclef-2026 -k aldisued/birdclef-2026-perch-v4-inference -v 8 -f "submission.csv" -m "Perch v4 Track A: frozen embeddings + LogReg probes (40/60 blend) + genus proxy + class-type smooth + file-max prior"
```

**Note on Perch v4 vs v2**: Public notebooks use Perch v2 (14,795 species, 1536-dim, natively on Kaggle). We use v4 (10,932 species, 1280-dim) uploaded as a dataset. Probes are v4-trained — incompatible with v2.

**Perch v4 assets note**: `genus.csv` (2,333 bird genera) and `family.csv` (249 families) are separate label vocabularies for hierarchical outputs, not per-label mappings. `infer_tf` returns only species logits (10,932 dims). For unmapped non-bird species (76 classes: Amphibia, Insecta, Mammalia, Reptilia), Perch has no genus-level coverage — these remain best served by the rough eBird prefix proxy or direct fine-tuned CNN scores.

---

### 🎯 Step 4 — Perch + CNN ensemble blend

**Expected: +0.01–0.03 LB on top of best individual | No retraining required | High priority**

After Perch LB score is known, blend Perch Track A predictions + soundscape CNN predictions.

Plan:
1. Run both models on the 66 labeled soundscapes (OOF for CNN; direct for Perch cache)
2. Grid-search blend weight α: `α × perch + (1-α) × cnn` for α ∈ [0.1, 0.2, ..., 0.9]
3. Evaluate on soundscape cmAP — pick best α, then submit to LB
4. Implement as a new inference notebook that loads both models and blends slot-level predictions

Literature support: Perch embeddings capture global bioacoustic patterns well; fine-tuned CNN captures Pantanal-specific distribution shifts. These are genuinely complementary. Public 0.912 notebooks already use this blend.

---

### 🎯 Step 5 — Per-class calibration + site×hour prior (post-processing only)

**Expected: +0.005–0.020 LB | No retraining | Medium-high priority**

Two orthogonal improvements, both using the 66 labeled soundscapes as calibration set:

**A. Per-class temperature scaling**
Learn a per-class temperature T_c such that `sigmoid(logit / T_c)` maximizes AP on OOF predictions. Currently using global T=1.0. Literature shows per-class calibration is consistently valuable in long-tail multilabel ecology.
- Implementation: after getting predictions on 66 soundscapes (OOF), optimize T_c per class with Brent's method or grid search
- Especially useful for the 76 zero-shot / weak-shot non-bird species

**B. Site×hour Bayesian prior**
Build `P(species | site, hour)` from `train_soundscapes_labels.csv`:
- Extract site code (e.g., `SN01`) and hour from filename timestamp
- Count species occurrences per (site, hour) bin → normalize
- Apply as additive prior: `preds += β × prior(site, hour)` (β ≈ 0.05–0.15)
- This is the "Bayesian site×hour prior" used by public 0.9+ notebooks

**C. Ablation to run first: raw embeddings without PCA**
Literature notes PCA can hurt rare classes. Try `LogReg` directly on standardized 1280-dim embeddings (no PCA reduction) for the per-class probes. If better on OOF AP, retrain probes and update artifacts.

---

### Step 6 — Co-occurrence prior (medium priority)

**Expected: +0.003–0.010 LB | No retraining**

Build species-species conditional probability from labeled training soundscapes and training clips:
- `P(species_j present | species_i high-confidence)` from `train_soundscapes_labels.csv`
- Apply small correction: if species_i score > 0.5, boost correlated species_j slightly
- Condition on habitat/site to avoid spurious co-occurrences

---

### Step 7 — PCEN + log-mel 2-channel input (medium priority)

**Expected: +0.005–0.015 LB | Requires retraining**

Add a second input channel: PCEN (Per-Channel Energy Normalization) alongside log-mel. PCEN is specifically designed for PAM recordings with variable background noise — suppresses stationary noise, enhances transient events. Consistent recommendation in ecoacoustics literature.

Implementation: `torchaudio.transforms.PCEN` applied to the same mel filterbank, concatenated with log-mel as channel dim → 2-channel input to EfficientNet.

---

### Step 8 — Multi-year BirdCLEF data pretraining (lower priority)

**Expected: +0.02–0.05 LB | Training: ~8–12h | Medium risk**

BirdCLEF 2021/2022/2023/2024 datasets (~117K clips, all public on Kaggle). Pretrain → fine-tune on 2026. Large effort; defer until Perch ensemble approach is validated. Note: BirdSet XCL pretraining already failed (LB 0.782), suggesting label mismatch with Pantanal is the bottleneck, not data volume. Multi-year BirdCLEF is more geographically aligned and worth testing.

---

## Kaggle setup

- **Notebook**: `aldisued/birdclef-2026-baseline-inference`
- **Current dataset**: `aldisued/birdclef2026-soundscape-v7` | kernel v21 pending
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
| `soundscape-v6-b1` | `--backbone tf_efficientnet_b1.ns_jft_in1k` + bg noise | 0–3 | **TIMEOUT** | CPU too slow for B1 + 50% overlap — **DEAD END** |
| `soundscape-v7` | same as v5 + `--mixup-alpha 1.0` (default) | 0–3 | pending v21 | val losses 0.0329–0.0341 (mean 0.0335) |

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
- **B1 backbone (soundscape-v6-b1)**: CPU timeout — B1 + 50% overlap exceeds Kaggle time limit. Only B0/B3 viable.
- **Circular TTA**: 2× inference → timeout. Keep `tta=False`. Only re-enable with confirmed headroom.
- **`rglob("*")` in sanity check cell**: scanned 35K+ train audio files → hidden time cost. Use `iterdir()` only.

---

## Perch Pseudo-Labeling — Findings (Mar 21)

Perch on **unlabeled soundscapes** does NOT help (mean prob = 0.000107, 99% windows near-zero). The 0.862 baseline gap was from `train_soundscapes_labels.csv`, not Perch.

**Perch as frozen feature extractor** (Track A above) is different — it uses Perch's own logits/embeddings directly, not as pseudo-labels for CNN training.
