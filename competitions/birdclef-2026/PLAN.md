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

### Current best: LB 0.912 (Perch v2, kernel `aldisued/birdclef-2026-perch-v2-inference` v3, Mar 27)

**Blend v4 ready (with fix needed)**: kernel v3 (1-fold CNN) COMPLETE — but needs TF memory release added before submitting. See "Inference time budget" section for root cause.

**Active work**: Blend Perch v2 + CNN in a single kernel. kernel_sources approach (Step 4 v1) was a dead end — CNN preds from kernel_sources are all-zero (dry-run output). New approach: run both models in the same notebook.

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
| **v21: soundscape-v7 + class-cond pooling + persistence penalty** | **0.882** | mixup α=1.0, timeout fixes (TTA off, rglob→iterdir) |
| **v23: soundscape-v8 (HGNetV2-B0, wrong mel config)** | **0.858** | regression — trained with EfficientNet mel config (win=2048, hop=512, 224 mel); should use win=626, hop=313, 256 mel |
| **Perch v4 Track A (kernel perch-v8)** | **TIMEOUT** | unbatched infer_tf (1 call/slot) — timed out |
| **Perch v4 batched (kernel perch-v9)** | **0.677** | 1 infer_tf/file — completed but far below CNN (0.882). Perch v4 label mapping weak vs public Perch v2. |
| **perch-v10/v11/v13 (181 probes, full training set)** | **TIMEOUT** | Consistent timeout — test set ≥780 soundscapes × ~7s/file ≈ 91min > 90min budget. v9 (0.677) was a fluke on a fast scoring node. |
| **v25–v30: OpenVINO / ONNX Runtime** | **DEAD END** | OpenVINO: no-internet install fails. ONNX Runtime: bundled wheels work but ONNX is slower than PyTorch on Kaggle CPU — even 4 folds no TTA times out |
| **v31: PyTorch 2 best folds + circular TTA** | **0.857** | **REGRESSION** — fold diversity (4 folds) worth ~0.025 more than circular TTA. Dropping 2 folds not worth it. |
| **v32: sharpened smooth + temperature T=1.1** | **0.881** | post-processing swap — neutral (±0.001 vs 0.882) |
| **v35: 4-fold v7 + 4-fold HGNetV2-Baseline blend** | **TIMEOUT** | 8 models too slow for scoring env (COMPLETE, blank score) — reverted to 4×v7 for v36 |
| **v36: 4-fold v7 only** | **pending** | reverted to proven config; kernel COMPLETE; submit when slot available |
| **Perch v2 port (kernel perch-v2-inference v3)** | **0.912** | tied with public top. Full pipeline: Bayesian priors + LogReg probes + genus proxies |
| **soundscape-v9 (pseudo-label pretraining)** | **DEAD END** | sc_cmap 0.65–0.69 vs v7 0.976 — regression regardless of epochs/threshold |
| **Blend v1 (kernel_sources approach)** | **0.912** | BUG: CNN preds from kernel_sources = all-zero (dry-run output). 0.80×perch + 0.20×0 = same ranking → same LB |
| **Blend v2 (single kernel, 4-fold CNN)** | **TIMEOUT** | kernel v1 — 4-fold no-overlap ~44 min + Perch ~7 min = too slow in scoring env |
| **Blend v3 (single kernel, 2-fold CNN)** | **TIMEOUT** | kernel v2 — same memory pressure issue, ~29 min locally → >90 min scoring env |
| **Blend v4 attempt 1 (Mar 30 07:19 UTC)** | **TIMEOUT** | kernel v3 — COMPLETE but blank score = timed out in scoring env |
| **Blend v4 attempt 2 (Mar 30 18:48 UTC)** | **TIMEOUT** | TF threads capped + parallel I/O + memory release — still blank score |

### Local validation findings

**sc_cmap(held) = primary local metric** — correctly ranks all submitted models in LB order. 14 held-out labeled soundscapes, `val_frac=0.15`, `seed=42`, stratified by station.

| Model | sc_cmap(held) | LB |
|---|---|---|
| soundscape-v1 | 0.853 | 0.827 |
| soundscape-v2 | 0.989 | 0.854 |
| soundscape-v3 | 0.955 | 0.858 |
| soundscape-v4 (CE loss) | 0.335 | 0.723 |
| soundscape-v7 | **0.976** | **0.882** |

**Perch v2 local α calibration is NOT reliable** — raw Perch logits (jaejohn/perch-meta) give cmAP 0.35 vs CNN 0.95 on labeled soundscapes (incomparable to LB 0.912 from full probe+prior pipeline). Use LB to calibrate blend weight.

---

## Public notebook research (Mar 23) — how they score 0.90+

### Two distinct approaches

**Track A: Perch frozen feature extractor — 0.892–0.912**

Google's `google/bird-vocalization-classifier/tensorflow2/bird-vocalization-classifier` (10,932 species, TF SavedModel) as a frozen embedding extractor. CPU-only, <90 min. No fine-tuning. (**Note**: v2 and v4 both have 10,932 species — same label list. Version doesn't matter.)

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

### ❌ Step 9 — Pseudo-Label Pipeline — DEAD END (Mar 28)

**Infrastructure built** (reusable for future experiments):
- `data/pseudo_label_cnn_soft.py` — generates soft `.npz` pseudo-labels from any checkpoint ensemble
- `eval/eval_pseudo_val.py` — evaluates ensembles on 500 S01/S02 files vs pseudo-labels
- `train_cnn.py --soundscape-val-frac 0.15` — leakage-free soundscape cmAP validation (14 held-out files, stratified by station, seed=42)

**Validation findings** (sc_cmap on 14 held-out labeled soundscapes with real labels):

| Model | sc_cmap | Notes |
|---|---|---|
| v7 4-fold ensemble | **0.9762** | baseline — no pseudo-labels |
| v9 (5ep, min-prob 0.1, 126K windows) | 0.65 | −0.33 vs v7 |
| v9b (5ep, min-prob 0.3, 93K windows) | 0.62 | worse — fewer windows didn't help |
| v9c (2ep, min-prob 0.1, 126K windows) | 0.69 | best variant — fewer epochs = less damage |
| v9d (1ep, min-prob 0.1, 126K windows) | 0.69 | same as v9c — floor reached |

**Root cause**: Two-stage pretraining (Stage 1 pseudo-labels → Stage 2 XC fine-tuning) is a net regression regardless of pretraining duration. Even 1 epoch of Stage 1 on 126K noisy pseudo-label windows costs ~0.28 sc_cmap that 30 epochs of fine-tuning cannot recover. The teacher (v7, LB 0.882) is not accurate enough on these stations for the student to benefit.

**Why it failed here but works in BirdCLEF 2024/2025**: Those solutions use pseudo-labeling to handle species with very few XC training clips (<10 examples). Our 234 species are reasonably covered (median 125 clips/species). Pseudo-labels on unlabeled soundscapes add domain-shift signal but the XC training already covers the taxonomy — the added noise outweighs the benefit.

**What to try instead** (if pseudo-labels are revisited):
- Mix pseudo-labels directly into Stage 2 training (same loader as XC data, weighted down) rather than separate pretraining stage
- Use only very high-confidence windows (min-prob 0.5+, ~10K windows) as hard-label augmentation
- Use Perch embeddings as features rather than pseudo-labels for training

**Phase 4 (Perch soft labels)** — deprioritized. Same architecture issue applies; Perch is also weaker than v7 on these species (LB 0.677 vs 0.882).

---

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

**Circular TTA**: do NOT re-enable unless timing confirmed safe with a headroom test. v21 scored 0.882 — the public 0.892 uses TTA and fits within 90 min. The previous timeouts (v18/v19) were caused by `rglob` + B1 backbone, both fixed. TTA is the main untried gain — see Step 5b.

#### Blend kernel timeout analysis (Mar 29) — IMPORTANT

Both blend attempts timed out despite local estimates well under budget:

| Kernel | Local runtime | Scoring env | Result |
|---|---|---|---|
| Perch v2 standalone | ~7 min | ~30 min | ✅ scores fine |
| CNN standalone (4-fold, 50% overlap) | ~35 min | ~50 min | ✅ LB 0.882 |
| Blend v2 (Perch + 4-fold no-overlap CNN) | ~51 min | >90 min | ❌ timeout |
| Blend v3 (Perch + 2-fold no-overlap CNN) | ~29 min | >90 min | ❌ timeout |

**Root cause: TF CPU thread oversubscription** (confirmed via BirdCLEF 2025 paper + TF GitHub issues).

TF consumes ALL CPUs by default and does **not** release its thread pools after inference. When PyTorch runs after TF in the same kernel, both compete for the same 2-core Kaggle CPU → oversubscription → 2-4× slowdown. This alone explains a 45-min estimate becoming 100+ min.

Key facts from research:
- `del birdclassifier; gc.collect()` does NOT free TF's internal allocator (TF never does this — documented limitation). Python references are freed but TF's thread pools remain active.
- TFLite gives **10× speedup** over TF SavedModel: 17s/file → 1.4s/file (BirdCLEF 2025 paper, arxiv 2507.08236). 739 files × 1.4s = 17 min vs 200+ min.
- Hardware is identical for interactive vs scoring env (2× Intel Xeon, 32GB RAM) — hardware difference is NOT the cause.
- iopub_timeout (Kaggle's nbclient, default 4s) can silently kill cells with no output — mitigated by `print(flush=True)` in all inference loops.

**What we know from kernel v4 dry-run log (20 files, 315s total)**:
- TF wheel install: 71s fixed
- CNN per-file: 13.4s first (warmup), **0.8s/file** thereafter → 739 files 1-fold ≈ **10 min**

**Fixes applied in blend v4 (kernel v5)**:
1. `tf.config.threading.set_intra_op_parallelism_threads(1)` + `set_inter_op_parallelism_threads(1)` immediately after TF import (limits TF threads before they grab all CPUs)
2. `torch.set_num_threads(2)` before CNN runs (prevents PyTorch oversubscription)
3. `del birdclassifier, infer_fn` + `gc.collect()` (frees Python references, won't free TF allocator)
4. Fold-by-fold CNN loading (minimises peak RAM)
5. `MAX_CNN_FOLDS=1` (~10 min CNN, ~80 min headroom)

**Future: TFLite conversion** — would give 10× Perch speedup and unlock 4-fold CNN blend. See Step 5h.

---

### ❌ Step 3 — Perch quick probes (soundscapes only) — DEAD END (LB 0.677)

**Result: LB 0.677 — far below CNN at 0.882.**

Root cause: only 53/234 species had trained probes (only species present in 66 labeled soundscapes = 792 windows). Non-Aves species (76 classes) have no Perch coverage at all. Both Perch v2 and v4 on Kaggle have the same 10,932 species — switching versions wouldn't help.

**Batching fix worked** (v9 completed without timeout), but the score ceiling is the probe training data.

---

### ❌ Step 3b — Perch v4 standalone with full training data probes — DEAD END (timeout)

**Result: Consistent timeout. Our Perch v4 inference ≥91 min on test set > 90-min Kaggle budget.**

Root causes (now understood clearly after analyzing the public 0.912 notebook):
1. **Wrong Perch version**: We used Perch v4 (self-uploaded), which is slower than the official `perch_v2_cpu` model on Kaggle (a CPU-optimized SavedModel)
2. **Wrong probe scope**: We trained probes only for 76 "unknown" species. The right approach trains per-class probes for ALL species with enough positives (~52–130), treating Perch embeddings as the *input features* rather than the fallback
3. **Wrong architecture understanding**: We thought the public notebooks used a two-kernel cache to avoid live test inference. They don't — the inference kernel runs Perch live on test soundscapes in ~6 minutes. The cache (`jaejohn/perch-meta`) only covers the 59 labeled **training** soundscapes used to fit the probes
4. **Missing Bayesian priors**: The jump from ~0.87 raw Perch to 0.912 comes from site×hour priors + per-class probes + genus proxies — none of which we implemented

**The public approach runs live in 6 minutes** because `perch_v2_cpu` on Kaggle's CPU is fast, batch size is 16 files (192 windows), and the probes/priors are trivial compute. No cache step for test data is needed.

**Pivot: Port the public 0.912 approach properly — see Step 3c.**

---

### ✅ Step 3c — Perch v2 proper port (public 0.912 approach) — LB 0.912

Port of `yashanathaniel/simplerun-perch-v2embedprobe-bayesian-0-912` — dry-run output matches reference exactly:
- OOF baseline AUC: 0.487292 ✓
- Probes trained: 52/234 ✓
- Score range: -25.162 to 15.932 ✓
- Runs COMPLETE in ~6 min ✓

**Notebook**: `inference/kaggle_perch_v2_inference.ipynb`
**Kernel**: `aldisued/birdclef-2026-perch-v2-inference` (v3, COMPLETE)
**Kernel-metadata**: `inference/kernel-perch-v2-metadata.json`
**Sources**: `kdmitrie/bc26-tensorflow-2-20-0` (kernel, TF wheels) + `jaejohn/perch-meta` (dataset, training cache) + `google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1` (model)

- [x] Create notebook, port pipeline
- [x] Push & run (verified COMPLETE, dry-run matches reference)
- [x] Submit standalone — **LB 0.912** (Mar 27, tied with public top)
- [x] Blend with CNN predictions — see Step 4

---

### ✅ Step 5a — HGNetV2-B0 backbone (correct mel config) — COMPLETE, pending LB

**v23: LB 0.858 (wrong mel config — dead end)**
**v34: pending** — 4-fold v7 + 4-fold HGNetV2-Baseline blend (kernel COMPLETE, submit after midnight UTC)

Correct mel config: `n_fft=2048, hop=625` (5s@32kHz = 160,000/625 = 256 time frames → natural 256×256, no resize). `n_mels=256, fmin=20, slaney norm`.

Two HGNetV2 runs completed:
- `soundscape-v8-hgnetv2`: `--hgnetv2` only → `BirdModel` (plain linear head, logits). Val losses 0.0329–0.0335. Uploaded as `birdclef2026-soundscape-v8-hgnetv2`.
- `soundscape-v8-hgnetv2-b`: `--hgnetv2 --baseline` → `BirdModelBaseline` (GEMFreqPool + AttentionSEDHead, same arch as v7). Val losses 0.0341–0.0348. Uploaded as `birdclef2026-soundscape-v8-hgnetv2-b`. **v34 uses this.**

Notebook auto-detects model type from checkpoint key schema (`head.cls_conv.bias` = BirdModelBaseline; `backbone.head.fc.bias` = BirdModel+sigmoid wrapper).

---

### ❌ Step 5b — Circular TTA (kernel v24) — DEAD END (timeout)

**Result: TIMEOUT. TTA doubles forward passes (4 models × 23 windows × 2 = 184 vs v21's 92). No headroom.**

Root cause: public 0.892 uses ~2 checkpoints total (1-fold blend). We use 4. At 4 folds + 50% overlap, we're already at the budget ceiling without TTA.

---

### ❌ Step 5c — OpenVINO conversion — DEAD END (no internet install)

**Result: `ModuleNotFoundError: No module named 'openvino'`.**

`openvino` pip install fails on Kaggle (`enable_internet: false`). No viable offline wheel bundling path. Abandoned.

---

### ❌ Step 5c.2 — ONNX Runtime (bundled wheels) — DEAD END (ONNX slower than PyTorch on Kaggle CPU)

**Result: COMPLETE (no timeout) but no public score = timed out in scoring environment.**

ONNX Runtime was installed offline via bundled cp312 Linux wheels (dataset `aldisued/onnxruntime-121-cp312-linux`). Model converted correctly (max diff < 1e-7). But even 4 folds no TTA (same pass count as v21/LB 0.882) timed out. ONNX Runtime is slower than PyTorch on Kaggle's CPU hardware for this model, or the install+load overhead is too large. Both approaches abandoned.

---

### ❌ Step 5e — 2 best folds + circular TTA (kernel v31) — DEAD END (LB 0.857)

**Result: LB 0.857 — regression from 0.882.**

Root cause: 4-fold ensemble diversity is worth ~0.025 LB. TTA gain is smaller than the cost of dropping 2 folds. Circular TTA is only viable if all 4 folds fit within budget — requires 2× speedup (OpenVINO was the plan, but is blocked by no-internet).

---

### 🎯 Step 5c — OpenVINO conversion (unlock TTA + 4 folds within budget)

**Expected: +0.008–0.012 LB (TTA) | No retraining | Medium implementation effort**

OpenVINO gives 2–3× CPU speedup via async inference queue (4 requests in flight). With speedup, 4 folds + 50% overlap + TTA all fit in 90 min. The public HGNetV2 notebook already uses this approach.

**Workflow** (no GPU needed — pure CPU offline conversion):
1. Convert 4 soundscape-v7 `.pt` checkpoints → OpenVINO IR (`.xml` + `.bin`) using `openvino.convert_model`
2. Upload IR files as new Kaggle dataset `birdclef2026-soundscape-v7-openvino`
3. Rewrite inference notebook to use `ov.Core` + `AsyncInferQueue` (4 concurrent requests)
4. Enable `tta=True`, push kernel, submit

**Conversion script**: `competitions/birdclef-2026/convert_to_openvino.py`

Input shape: `(B, 3, 224, 313)` — dynamic batch, fixed spatial. Dynamic batch set via `ov_model.reshape`.

**Inference rewrite key snippet**:
```python
core = ov.Core()
core.set_property("CPU", {"PERFORMANCE_HINT": "THROUGHPUT", "INFERENCE_NUM_THREADS": "4", "NUM_STREAMS": "2"})
compiled = core.compile_model(ov_model, "CPU")
queue = AsyncInferQueue(compiled, jobs=4)  # 4 concurrent requests
```

- [ ] Convert + verify (max diff PyTorch vs OV < 1e-4)
- [ ] Upload to Kaggle dataset
- [ ] Rewrite inference notebook with AsyncInferQueue
- [ ] Push kernel, submit with TTA=True

---

### ✅ Step 5d — Alternative post-processing swap — TESTED, NEUTRAL

**Result: LB 0.881 (v32) — essentially neutral vs 0.882 (v21).**

Sharpened smooth + temperature T=1.1 does not improve over class-conditional pooling + persistence penalty. Current post-processing is fine.

---

### ❌ Step 5e — 2-fold + TTA — DEAD END (LB 0.857)

Fold diversity worth ~0.025 more than TTA. Don't drop folds.

---

### ✅ Step 5f — HGNetV2-B0 with correct config — COMPLETE, pending LB

See Step 5a above. v34 kernel ready to submit.

---

### 🎯 Step 5g — Per-class calibration on 66 labeled soundscapes (next after v34 LB)

**Expected: +0.005–0.020 LB | No retraining | High priority**

Learn per-class temperature T_c on OOF predictions from the 66 labeled soundscapes, maximizing per-class AP. Apply as `sigmoid(logit / T_c)` in inference. Particularly valuable for the 28 zero-shot species and 76 non-Aves classes.

Implementation options:
- Grid-search T_c ∈ [0.5, 2.0] per class using `eval_oof.py` soundscape predictions
- Apply as a 234-element vector baked into the inference notebook

**When to run**: after v34 LB result confirms HGNetV2 blend helps. If blend helps, calibrate the 8-model ensemble. If neutral, calibrate soundscape-v7 alone.

---

### 🎯 Step 4 — Perch v2 + CNN ensemble blend

**Expected: +0.005–0.020 LB on top of 0.912 | No retraining required**

**Approach**: Run both models in the SAME notebook. `kernel_sources` approach is a dead end — it loads dry-run (all-zero) outputs, not scoring-env outputs.

**Timing**: Perch ~7 min + CNN 4-fold no-overlap ~44 min = ~51 min total, fits 90-min budget.

**Blend formula**: `final = (1 - CNN_WEIGHT) × perch_probs + CNN_WEIGHT × cnn_probs`

- [x] Build blend v2 notebook (`inference/kaggle_blend_v2_inference.ipynb`)
  - Perch v2 full pipeline → `perch_submission`
  - CNN soundscape-v7 (4-fold, no overlap = 12 windows/file)
  - Blend: 0.80 × Perch + 0.20 × CNN
- [x] Push kernel `aldisued/birdclef-2026-perch-cnn-blend-v2-inference` v1
- [x] Submit Mar 29 — **awaiting score**
- [ ] If score > 0.912: iterate on CNN_WEIGHT (try 0.30, 0.40)
- [ ] If score ≤ 0.912: debug CNN inference output (check cnn_rows populated, probs non-zero)

**Dead end documented**: `kernel_sources` loads OUTPUT FILES from last manual kernel run (dry-run → all zeros). Any blend using kernel_sources is `α × perch + (1-α) × 0 = α × perch` — ranking-preserving → same cmAP.

**Local α calibration not reliable**: raw Perch logits (from jaejohn/perch-meta) give cmAP 0.35 vs CNN 0.95 on labeled soundscapes. This is incomparable to the final Perch pipeline (LB 0.912). Use LB submissions to calibrate α iteratively.

---

### 🎯 Step 5 — CNN improvements (parallel / fallback)

Public 0.892 CNN notebooks use several things we haven't tried:

---

### ❌ Step 5h — Convert Perch to TFLite — DEAD END (confirmed Mar 30)

**Expected: unlocks 4-fold CNN blend | No retraining | Medium effort**

BirdCLEF 2025 paper (arxiv 2507.08236): TF SavedModel 17s/file → TFLite 1.4s/file (**10× speedup**).
For 739 test files: 17 min (TFLite) vs 200+ min (SavedModel). Would bring total blend time to ~27 min with 4-fold CNN.

**Current bottleneck**: `perch_v2_cpu` is a TF SavedModel. TFLite requires conversion + custom ops check.

#### TFLite attempt 1 — no speedup (Mar 29, v3 COMPLETE)

| Step | Result |
|---|---|
| TFLite conversion | ✅ **SUCCESS** (28.5s, **407.3 MB** flat buffer) |
| SavedModel: 20 files | 8.80s/file → **108.4 min** for 739 files |
| TFLite: 20 files | 9.49s/file → **116.9 min** for 739 files |
| Speedup | **0.9× (actually SLOWER)** |
| Max logit diff | 0.000280 ✅ |

Root cause: conversion used only `TFLITE_BUILTINS + SELECT_TF_OPS` with **no quantization**. TF op kernels still run inside TFLite wrapper — same compute path. XNNPACK cannot accelerate SELECT_TF_OPS kernels.

#### TFLite attempt 2 — with `Optimize.DEFAULT` quantization (Mar 30, PENDING)

**Key finding from research (Mar 30)**: DS@GT BirdCLEF 2025 code (`compile.py`) used exactly:
```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]   # ← this was missing from attempt 1
converter.target_spec.supported_ops = [TFLITE_BUILTINS, SELECT_TF_OPS]
```
`Optimize.DEFAULT` applies float16/int8 weight quantization — this is what drives their 10× speedup, not the TFLite format itself.

**Important caveat**: DS@GT used Perch v1 (Keras model, `from_keras_model()`). We must use `from_saved_model()` for Perch v2. Whether `Optimize.DEFAULT` gives the same benefit with `SELECT_TF_OPS` on v2 is unknown.

**Test notebook**: `inference/kaggle_perch_tflite_test.ipynb` (Cell 3 updated to add `Optimize.DEFAULT`)
**Kernel**: `aldisued/birdclef-2026-perch-tflite-conversion-test` v2 (PENDING)

Expected outcomes:
- If quantization works: model size shrinks (<200MB), per-file time drops to ≤3s/file → blend viable
- If quantization fails / same speed: TFLite dead end confirmed for Perch v2

**Parallel benchmark findings (Mar 30)**:
Running Perch and CNN in parallel Python threads works (zero overhead, CNN fully hidden).
BUT: TF at half cores = 12.87s/file → 127 min extrap. Doesn't fit.
Parallel approach is only viable if Perch first gets faster via quantized TFLite.

#### TFLite attempt 2 results (Mar 30, v2 COMPLETE) — DEAD END CONFIRMED

| | Attempt 1 | Attempt 2 (+Optimize.DEFAULT) |
|---|---|---|
| Model size | 407 MB | 376 MB (−8%) |
| Speedup | 0.9× | 0.9× |
| Max logit diff | 0.000280 ✅ | **2.499 ❌** |

`Optimize.DEFAULT` barely shrunk the model — float16 quantization did not apply to SELECT_TF_OPS kernels (they run through TF runtime regardless). The 8% that did get quantized introduced large numerical errors (max diff 2.5 logits). **No path to TFLite speedup for Perch v2 exists via this route.**

**TFLite is definitively a dead end for Perch v2.** The model requires SELECT_TF_OPS which bypasses all TFLite optimization. Only a JAX-source conversion (chirp/export_utils.py) could produce a native TFLite model — but that requires the JAX checkpoint, not the Kaggle SavedModel.

- [x] Convert SavedModel → TFLite attempt 1 (no quantization — 0.9× speedup)
- [x] Research: DS@GT used `Optimize.DEFAULT` — but for Perch v1 Keras model, not v2 SavedModel
- [x] Convert SavedModel → TFLite attempt 2 (+Optimize.DEFAULT — 0.9× speedup, large numerical error)
- [x] **DEAD END CONFIRMED** — no TFLite speedup path for Perch v2 SavedModel

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

---

### 🎯 Step 10 — Perch → CNN knowledge distillation (Perch soft labels for CNN training)

**Expected: close the 0.882→0.912 gap | Requires precompute + retraining | High priority**

**Core idea**: Use Perch v2 (LB 0.912, better teacher) to generate soft labels on all 739 training soundscapes. Mix those into CNN training. At inference: pure CNN, no Perch needed.

**Why this is different from soundscape-v9 (which failed)**:
- v9 teacher: CNN (0.882) — weaker than student, noisy labels
- This teacher: Perch (0.912) — stronger than student, much better signal
- v9 training: separate pretraining stage → catastrophic forgetting
- This: mixed into regular training loop (no pretraining stage)

**Loss design**:
- XC clips: `BCE(pred, hard_label)` as before
- Soundscape windows (all 739 × 12): `KL(pred, perch_soft / T_perch)` only
- `T_perch ≈ 1.5` to soften Perch's spiky predictions
- `p_soundscape` = fraction of soundscape batches in each step (single tunable param)

**Tuning plan**:
1. Grid `p_soundscape ∈ {0.05, 0.15, 0.30}` × 10-epoch runs → pick best sc_cmap(held)
2. Full 30-epoch run at best ratio

**Precompute script**: `training/precompute_perch_soundscapes.py`
- Outputs: `perch_soundscape_cache/perch_sc_scores.npy` (N×234), `perch_sc_embeddings.npy` (N×1536), `perch_sc_meta.parquet`
- Runtime: ~2.3h on cluster CPU (739 files × 11s/file)

**Cluster setup** (run once):
```bash
# Download Perch model
kaggle models instances versions download \
  google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1 \
  -p ~/perch_v2_cpu --untar

# Install TF (use existing wheel if available, else pip)
pip install tensorflow-cpu==2.20.0

# Run precompute (background, log to file)
KEGO_PATH_DATA=/home/kristian/projects/kego/data \
PERCH_MODEL_DIR=~/perch_v2_cpu \
nohup uv run python competitions/birdclef-2026/training/precompute_perch_soundscapes.py \
  > /tmp/perch_precompute.log 2>&1 &
echo $! > /tmp/perch_precompute.pid
```

- [x] Download Perch model to cluster (`~/perch_v2_cpu/`)
- [x] Precompute done — `perch_pseudo_labels_soft.npz` already exists with 127,896 windows (10,658 files × 12), soft probs float32, 158/234 species covered
- [x] Add `--p-soundscape` + `--perch-temp` to train_cnn.py; `train_epoch_mixed` interleaves Perch KD batches in Stage-2 loop (Apr 1)
- [ ] Grid search p_soundscape (3 × 10-epoch fold-0 runs) — p05 + p15 running on GPU 0/1 (Apr 1)
- [ ] Full 30-epoch run at best p_soundscape
- [ ] Submit and compare vs LB 0.912

**Grid search results** (fold 0, 10 epochs, sc_cmap(held), ALL 127,896 windows):
| p_soundscape | sc_cmap@10ep | Notes |
|---|---|---|
| 0.05 | **0.519** | best epoch 9 |
| 0.15 | **0.501** | best epoch 9 |
| 0.30 | **0.504** | best epoch 10 |

**Root cause of low sc_cmap**: 97.5% of Perch windows are silent background (max_prob < 0.05). KD batches from these windows teach "predict zero for everything" → global suppression. Even at p=0.05, 95% of batch steps are XC, but the Perch suppression signal counteracts learning. p05-full at 30 epochs stuck at 0.52 with no upward trend (killed at epoch 14).

**Signal distribution**:
- max_prob ≥ 0.05: 3,146 windows (2.5%)
- max_prob ≥ 0.10: 1,307 windows (1.0%)
- max_prob ≥ 0.30: 370 windows (0.3%)

**Round 2: signal-filtered KD** (min_prob=0.1 → 1,307 windows, Apr 1):
| p_soundscape | sc_cmap@best | Notes |
|---|---|---|
| 0.15 | **0.538** (ep20) | killed at ep21 — no upward trend |
| 0.30 | **0.475** (ep14) | killed at ep19 — worse |

**DEAD END CONFIRMED** (Apr 1): Both all-windows (0.52) and signal-filtered (0.54) KD runs far below v7 baseline (0.976) with no convergence toward it. Root cause: Perch soft labels cover only 158/234 species and the soundscape distribution is fundamentally different from XC clip distribution. KD signal doesn't help the model generalize from XC → soundscape inference.

→ **Step 10 and Step 11 are both dead ends. Moving on.**

---

### 🎯 Step 11 — Perch embedding supervision for CNN (auxiliary distillation head)

**Expected: complementary to Step 10 | Requires precompute (same data) + retraining**

**Core idea**: Add an auxiliary head to CNN that predicts Perch's 1536-dim embeddings during training. Forces CNN internal representations to be "Perch-like" without needing Perch at inference time. Uses the same precomputed cache as Step 10.

**Architecture**:
```
mel → CNN encoder → feature map
                  ├→ GEMPool → AttSED → class probs   (main head, BCE)
                  └→ GAP → Linear(1536) → emb pred     (aux head, MSE vs Perch emb)
```
Loss: `BCE(pred, label) + λ_emb × MSE(emb_pred, perch_emb)`

Aux head is dropped at inference. The constraint forces the encoder to learn Perch's feature space, especially for non-Aves species where Perch excels.

**λ_emb tuning**: start at 0.1 (aux head shouldn't dominate). Tune via sc_cmap(held).

**Can combine with Step 10**: use both soft labels AND embedding supervision simultaneously.

- [ ] Precompute done (shared with Step 10)
- [ ] Add aux embedding head to BirdModelBaseline in train_cnn.py
- [ ] Add `--perch-emb-supervision λ` flag
- [ ] Train and evaluate

---

### ❌ Step 12 — EfficientNet-B3 backbone (v7 config) — DEAD END

**Result: sc_cmap(held) = 0.568 (fold 0) — far below v7 B0 baseline 0.976. Killed after fold 0.**

Same config as soundscape-v7 but larger backbone: `tf_efficientnet_b3.ns_jft_in1k` (12M params vs B0's 4M). Previous B3 run (LB 0.776) used wrong mel config — hypothesis was HTK+soundscape labels would fix it.

**Root cause**: B3 (12M params) overfits more aggressively to XC clips than B0 (4M params), losing soundscape generalization. WarmRestarts T_0=5 causes sc_cmap oscillation. Fold 0 early-stopped at epoch 23, best sc_cmap = **0.568** (vs v7 = 0.976).

| Fold | Best sc_cmap | Epoch |
|------|-------------|-------|
| fold 0 | 0.568 | 13 (early stop at 23) |
| fold 2 | ~0.543 | 14 (killed at 18) |

Conclusion: Larger backbone ≠ better soundscape generalization. B0 NoisyStudent features transfer better to bird audio. Do not retry B3/B4.

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
- **Current datasets**: `aldisued/birdclef2026-soundscape-v7` + `aldisued/birdclef2026-soundscape-v8-hgnetv2-b` (v34 blend)
- **CRITICAL**: `enable_gpu: false` — competition GPU limit is 0 min; GPU requests cause silent failure
- Push: `kaggle kernels push -p competitions/birdclef-2026/inference/` (note: new path after repo reorganization)
- Submit: `kaggle competitions submit -c birdclef-2026 -k aldisued/birdclef-2026-baseline-inference -v <int> -f submission.csv -m "..."`

---

## Hardware & Training Convention

Training on **2× RTX 3090** at `kristian@omarchyd` (Tailscale).

**Standard: 4 folds (`--n_folds 4`)**. Fits 2 GPUs in exactly 2 clean rounds (2+2).

Launch pattern (use subshell per fold to ensure `cd` applies; use `>>` to not overwrite logs):
```bash
ssh kristian@192.168.178.32 "nohup bash -c 'cd /home/kristian/projects/kego && CUDA_VISIBLE_DEVICES=0 KEGO_PATH_DATA=/home/kristian/projects/kego/data ~/.local/bin/uv run python competitions/birdclef-2026/training/train_cnn.py --baseline --soundscape-labels --htk --warm-restarts --gain-aug --fold 0 --n_folds 4 --tag <tag> >> /tmp/<tag>_fold0.log 2>&1' &"
ssh kristian@192.168.178.32 "nohup bash -c 'cd /home/kristian/projects/kego && CUDA_VISIBLE_DEVICES=1 KEGO_PATH_DATA=/home/kristian/projects/kego/data ~/.local/bin/uv run python competitions/birdclef-2026/training/train_cnn.py --baseline --soundscape-labels --htk --warm-restarts --gain-aug --fold 1 --n_folds 4 --tag <tag> >> /tmp/<tag>_fold1.log 2>&1' &"
```

**Note**: script is now at `training/train_cnn.py` (renamed from `train.py`, reorganized Mar 26). Repo structure: `training/`, `data/`, `inference/`, `eval/`, `eda/`, `research/`.

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
| `soundscape-v7` | same as v5 + `--mixup-alpha 1.0` (default) | 0–3 | **0.882** (v21) | val losses 0.0329–0.0341 (mean 0.0335); +0.006 vs v5 from class-cond pooling + persistence penalty |
| `soundscape-v8-hgnetv2` | `--hgnetv2` (BirdModel, simple head), n_fft=2048, hop=625, 256-mel, fmin=20 | 0–3 | pending (v33) | val losses 0.0329–0.0335; blended with v7 in v33 kernel |
| `soundscape-v8-hgnetv2-b` | `--hgnetv2 --baseline` (BirdModelBaseline), same mel config | 0–3 | pending (v35) | val losses 0.0341–0.0348; blended with v7 in v35 kernel (preferred) |

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
- **Circular TTA**: 2× inference → timeout with 4 folds. Safe with 2 folds (same 4 total passes as v21). See v31.
- **OpenVINO**: `openvino` pip fails on Kaggle (no internet). Dead end.
- **ONNX Runtime (bundled wheels)**: Installs offline fine, max diff < 1e-7. But ONNX is *slower* than PyTorch on Kaggle CPU — 4 folds no TTA still times out. Dead end.
- **`rglob("*")` in sanity check cell**: scanned 35K+ train audio files → hidden time cost. Use `iterdir()` only.

---

## Perch Pseudo-Labeling — Findings (Mar 21)

Perch on **unlabeled soundscapes** does NOT help (mean prob = 0.000107, 99% windows near-zero). The 0.862 baseline gap was from `train_soundscapes_labels.csv`, not Perch.

**Perch as frozen feature extractor** (Track A above) is different — it uses Perch's own logits/embeddings directly, not as pseudo-labels for CNN training.
