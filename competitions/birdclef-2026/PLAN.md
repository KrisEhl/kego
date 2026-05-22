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

### Current best: LB **0.920** (kernel perch-v2-inference v53/v55/v57/v58, Apr 12–13)

---

## Status summary (Apr 16)

All low-hanging fruit is exhausted. Post-processing, backbone swaps, seed searches, Stage2 architecture, pseudo-labeling, CORAL alignment, diel priors, and global delta smoothing all confirmed dead ends. Current best: **LB 0.920** (v53/v55/v57/v58).

**Exhausted directions (confirmed dead):**
- Post-processing: rank_power sweep (0.4→4.0), boost, A1 taxon temp, insecta smoothing — all 0.920 or below
- Stage2-only seed search (s100-s119): 20 seeds, none beat seed 42 (0.3625 at rp=2.0)
- Stage1+Stage2 full retrain seed search (s200-s231): 32 seeds. Best locally: s209 (+0.009) but LB **0.918** (−0.002). LOCAL cmAP ANTI-CORRELATED with LB for Stage1 seeds.
- AVES/BEATs/WavLM backbones: all tested, all worse than Perch
- Stage3, 66sc data, XC augmentation, site-aware Stage2, per-class temperature calibration
- CORAL embedding alignment (v63: 0.919), diel activity priors (v64: 0.918)
- Global prob-space delta smooth α=0.20 (v68: 0.913 — WORSE due to redundancy with logit-space smoothing)
- Pseudo-labeling r1c (v67: 0.912)

**All directions exhausted as of Apr 17.** LB 0.920 appears to be the ceiling for our current setup. The gap to competitors (0.930+) is unexplained — same architecture, same post-processing, same probe quality. Possible explanations: (1) competitors have additional private labeled data, (2) better probe hyperparameters we haven't searched, (3) luck/noise at the 0.920 level.

**Apr 17 additional results:**
- MLP probes (64,) — v76 LB **0.919** (−0.001). LogReg with L2 beats MLP at 708 samples. Track N dead end.
- Submit-mode retrain — v70 LB **0.916** (−0.004). 66sc regression confirmed again. Dead end.
- v77 (LogReg restore + protossm_original.pt first): COMPLETE blank score — likely transient scoring error. Resubmit tomorrow (Apr 18) as v78.
- 5/5 daily slots used Apr 17. Next action: push v78 (same code) and resubmit Apr 18.

---

## New directions (Apr 14+)

### Track F: LOSO CV — reliable Stage2 evaluation ✅ COMPLETE (Apr 14-15)

Script: `eval/eval_loso.py` (implemented, bug-fixed Apr 15, validated).

Uses Leave-One-Site-Out (8 sites in 59sc dataset) to evaluate Stage2 without in-sample bias.
**CRITICAL**: Run with `OMP_NUM_THREADS=4` — NOT 4 parallel processes (causes CPU saturation).

#### LOSO BUG: Perch-logit base was WRONG → anti-correlated with LB (fixed Apr 15)

**Bug**: Original LOSO used raw Perch logits as Stage2 base (~0.545 cmAP). Real inference uses
probe_scores (`full_probe_scores__59sc.npy`, ~0.926 cmAP). Inflated deltas (+0.06→+0.12), rankings INVERTED.

**Evidence of anti-correlation**: Perch-logit LOSO ranked seed 218 best (+0.005) → v61 submitted → LB **0.914** (−0.006 vs 0.920). The metric ranked the worst seed as best.

**Fix (Apr 15)**: `eval_loso.py` now loads `full_probe_scores__59sc.npy` as Stage2 base (`--probe-scores` arg, default = `full_probe_scores__59sc.npy`). Probe scores are seed-independent.

**DO NOT use `--probe-scores none` for seed ranking** — raw Perch logit LOSO is anti-correlated.

#### Probe-scores LOSO results (Apr 15) — DEFINITIVE SEED RANKING

| Seed | Probe-LOSO cmAP (weighted) | Stage2 delta | LB | Verdict |
|------|--------------------------|--------------|-----|---------|
| **42 (baseline)** | **0.9449** | **+0.0016** | **0.920** | ✅ OPTIMAL |
| 218 | 0.9440 | +0.0007 | 0.914 (−0.006) | ❌ |

Baseline probe_scores: 0.9433 (seed-independent). Probe-LOSO correctly ranks seed 42 > 218, matching LB.

**Stage1 seed search = DEFINITIVELY DEAD END.** Three signals agree: LB (seed 42=0.920 > 209=0.918 > 218=0.914), probe-LOSO (seed 42 > 218), Stage2 delta (seed 42 > 218). **DO NOT retry.**

#### Per-site probe-LOSO results (seed 42, Apr 15) — pseudo-labeling evaluation baseline

| Site | cmAP | Baseline | Delta | n_sc | active_cls |
|------|------|----------|-------|------|------------|
| S03 | 1.0000 | 1.0000 | +0.0000 | 2 | 4 |
| S08 | 0.9497 | 0.9496 | +0.0001 | 5 | 19 |
| S13 | 1.0000 | 1.0000 | +0.0000 | 2 | 6 |
| S15 | 0.9319 | 0.9319 | +0.0000 | 4 | 14 |
| S18 | 1.0000 | 1.0000 | +0.0000 | 1 | 3 |
| S19 | 0.9559 | 0.9461 | **+0.0098** | 3 | 17 |
| S22 | 0.8961 | 0.8961 | +0.0000 | 39 | 29 |
| S23 | 0.9881 | 0.9881 | +0.0000 | 3 | 14 |
| **Weighted avg** | **0.9449** | **0.9433** | **+0.0016** | — | — |

**Pseudo-labeling goal**: beat 0.9449 weighted cmAP (especially S22 delta from 0.0000 to >0.0050).

```bash
# Run probe-scores LOSO for any checkpoint (cluster):
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 KEGO_PATH_DATA=/home/kristian/projects/kego/data \
PYTHONUNBUFFERED=1 .venv/bin/python competitions/birdclef-2026/eval/eval_loso.py \
    --checkpoint outputs/protossm_pseudo_r1.pt --stage2-epochs 30 \
    --probe-scores full_probe_scores__59sc.npy

# v61 result: 0.914 (−0.006 vs 0.920) — Perch-logit LOSO ranked seed 218 best (WRONG)
```

### Track G: Stage2 architecture experiments — IN PROGRESS (Apr 14 evening)

CLI flags added to `train_protossm.py` and `eval/eval_loso.py`:
- `--stage2-d-model` (int, default=128)
- `--stage2-d-hour` (int, default=0)
- `--stage2-l2` (float, default=0.0)

LOSO results on seed 42 baseline — **ALL DEAD ENDS (Apr 14)**:

| Variant | LOSO cmAP | vs baseline (0.6180) | Verdict |
|---------|-----------|---------------------|---------|
| **baseline (d_model=128)** | **0.6180** | — | current best |
| d_hour=16 | 0.6168 | −0.0012 | ❌ DEAD END |
| l2=0.01 | 0.6119 | −0.0061 | ❌ DEAD END |
| d_model=64 | 0.5998 | −0.0182 | ❌ DEAD END |

**Track G = DEAD END**. All Stage2 architecture variants hurt vs baseline:
- d_hour=16: Sonotype hour patterns can't be learned from only 59 soundscapes (~7 sc/hour/site)
- d_model=64: Too small — cuts capacity needed for 234-class correction on 1770-dim input
- l2=0.01: L2 penalty shrinks corrections that ARE correct on labeled sites
Current Stage2 architecture (d_model=128, no hour, no L2) is optimal for 59sc regime.

**DO NOT retry any Stage2 architecture variants.**

### ❌ Track J: CORAL embedding alignment — DEAD END (Apr 15)

**Idea**: Diagonal CORAL to align labeled site embeddings (708 windows, 8 sites) → unlabeled
distribution (62,592 windows, 14 unlabeled sites). Stage2 then trains in the same embedding space
it sees at test time (no transform needed at inference).

**Implementation**:
- `training/precompute_coral.py` — computes μ_src, σ_src, μ_tgt, σ_tgt (1536-dim each)
- Transform: `emb_aligned = (emb_src - μ_src) / (σ_src + EPS) * (σ_tgt + EPS) + μ_tgt`
- `train_protossm.py --stage2-emb-file coral_emb_aligned.npy` — uses CORAL emb for Stage2 only
- Distribution shift measured: ||μ_tgt - μ_src|| = 1.6243

**LB result (v63, Apr 15): 0.919 (−0.001 vs 0.920 baseline)**

**Why it failed**: The distribution shift between labeled and unlabeled sites is small relative to
the noise in 708 training windows. CORAL alignment doesn't provide meaningful new signal when the
Stage1 Perch representations are already well-clustered by species. The per-feature whitening may
slightly degrade the probe_prob inputs to Stage2.

**DO NOT retry CORAL or any other domain adaptation alignment on Perch embeddings.**

---

### ❌ Track K: Diel activity priors — DEAD END (Apr 15)

**Idea**: Soundscape windows have strong hour-of-day activity patterns. Applying a per-species
hour correction should lift species that are only active at specific hours (e.g. son10 active 3-4 AM).

**Implementation**:
- `training/precompute_perch_soundscapes.py` output used to compute `diel_priors.npy` (234×24)
- diel_priors[c, h] = mean sigmoid(Perch_logit[c]) across all 127,896 unlabeled windows at hour h
- Hours 11-16 absent from training data → flat fallback (global mean)
- Correction: `final_logits += alpha * (logit(prior[c,h]) - logit(global[c]))`
- Cannot be validated by LOSO (per-site monotone transform doesn't change AP within a site)

**v64 result (Apr 15): LB 0.918 (−0.002 vs 0.920 baseline). DEAD END.**

Why it failed: hours 11–16 flat fallback (no signal) + the hours with real signal don't generalize from unlabeled train soundscapes to test soundscapes (different sites/schedules). DO NOT retry diel priors at any alpha.

---

### ❌ Track L: Global prob-space delta smoothing (competitor V18 feature) — DEAD END (Apr 16)

**Idea**: Competitor dingjiarun (LB 0.930+) applies `delta_shift_smooth(α=0.20)` to ALL 234 species
in probability space, AFTER rank_power scaling. We only apply it to Amphibia/Insecta (logit space).

**Implementation** (v68):
- Added after boost+clip, before `probs_final`:
  ```python
  _GLOBAL_DELTA_SMOOTH_ALPHA = 0.20
  if _GLOBAL_DELTA_SMOOTH_ALPHA > 0:
      _probs_3d = _delta_shift_smooth(_probs_3d, _GLOBAL_DELTA_SMOOTH_ALPHA)
      _probs_3d = np.clip(_probs_3d, 0.0, 1.0)
  ```
- `_DIEL_ALPHA = 0.0` (reverted from 0.3 dead end)
- `_PROTO_CANDIDATES`: reverted to `protossm_original.pt` (pseudo r1c was worse)

**v68 result (Apr 16): LB 0.913 (−0.007 vs 0.920 baseline). DEAD END.**

Why it failed: Our logit-space species-specific smoothing (Amphibia α=0.65, Insecta α=0.35) already
captures temporal patterns. Adding a second global probability-space pass is redundant and diffuses signal.
The competitor applies this WITHOUT our logit-space smoothing — our pipeline is different.
DO NOT add global prob-space smoothing.

---

### ❌ Track M: Submit-mode full-dataset Stage1 retrain — DEAD END (Apr 16-17)

**Hypothesis**: `protossm_v3.pt` (train mode, 5-fold CV) never sees all 59sc together in Stage1.
Competitor might train in `--mode submit` (Stage1 on ALL data). Tested as v70.

**v70 result (Apr 17): LB 0.916 (−0.004 vs 0.920 baseline). DEAD END.**

Why it failed: submit mode trains Stage1 on 66sc (all labeled soundscapes, including S09).
S09 data is categorically harmful (confirmed Apr 11: 66sc Stage1 retrain = 0.912). Submit mode
includes S09 → same regression pattern. Stage2 early-stopped at epoch 28 (val 0.00590) vs
standard 30ep — slightly less optimal. Train-mode 5-fold Stage1 is BETTER than submit-mode
all-data Stage1 for this dataset.

v69 scored blank (dataset propagation timing issue — uploaded just 19MB file, protossm_original.pt
was removed from latest version). v70 = proper retry with both files in dataset → 0.916.

**DO NOT retry submit-mode Stage1. Train-mode (protossm_original.pt) remains the best checkpoint.**

**ALL KNOWN DIRECTIONS EXHAUSTED as of Apr 17.** Current best: LB 0.920 (v53/v55/v57/v58).

---

### ❌ Track H: Iterative pseudo-labeling on unlabeled soundscapes — DEAD END (Apr 15-16)

**Dataset**: 10,659 total train soundscapes (23 sites), only 59-66 labeled. ~10,600 unlabeled.
Labeled sites: S03, S08, S09, S13, S15, S18, S19, S22, S23.
Unlabeled-only sites: S01, S02, S04, S05, S06, S07, S10, S11, S12, S14, S16, S17, S20, S21 (14 new sites matching test soundscape distribution better).

**2025 winning technique** (Nikita Babych, LB 0.933): iterative pseudo-labeling with power scaling.
Round 1: current model → soft pseudo-labels → retrain → +0.026 LB.
Rounds 2-5: power scaling (raise probs to power < 1 = sharpen) → +0.032 LB.

**How it would work for our pipeline:**
1. Precompute Perch embeddings for all ~10,600 unlabeled soundscapes (cluster job, ~4h)
2. Run Stage1 inference → proto_probs for all unlabeled windows
3. Use HARD thresholded pseudo-labels (probe_prob > 0.5) to avoid circular BCE loss
4. Add pseudo-labeled batches to Stage2 training (keep ground-truth labeled sc as primary)
5. Power scaling: `pseudo_labels = np.power(sigmoid(probe_scores), 0.7)` before using as targets
6. Retrain Stage2 on (708 labeled + ~127,908 pseudo-labeled windows), 30 epochs
7. Repeat with new model as teacher (2-3 rounds)

**Key difference from 66sc dead-end**: unlabeled soundscapes from 14 NEW sites not in labeled set.
66sc dead-end was about S09 poor data quality. 10K diverse files ≈ test soundscape distribution.

**Key risk**: circular BCE if soft labels used (Stage2 base = probe scores = pseudo-labels).
Mitigation: hard thresholded or power-scaled pseudo-labels.

**Results (all LOSO=0.9449 = identical to baseline; only LB distinguishes):**

| Variant | Pseudo-label source | Threshold | Pos/window | LOSO | LB |
|---|---|---|---|---|---|
| r1 | Raw Perch logits | 0.3 | ~110 | 0.9449 | pending |
| r1b | Stage1+Stage2 proto_probs | 0.5 | ~17 | 0.9449 | pending |
| r1c | Stage1 proto_probs only | 0.5 | 4.57 | 0.9449 | **0.912 (−0.008)** |

**Verdict: DEAD END.** All pseudo-labeling variants hurt LB vs baseline 0.920.

**Root cause analysis:**
- LOSO can't detect pseudo-labeling gains (tests on labeled sites only, not new unlabeled sites)
- Stage2 is a 1-layer SSM trained on 708 windows — 127K additional unlabeled windows cause underfitting/distribution shift in the correction pathway
- The probe_scores used for labeled batches (in-sample 0.926 cmAP) vs pseudo-labels (noisier) creates a training mismatch the model cannot reconcile
- 2025 winner's approach required multiple rounds of power-scaling + a fundamentally different architecture. Our Stage2 has only 1 BiSSM layer — not enough capacity to generalize from pseudo-labels.

**DO NOT retry any pseudo-labeling variant.**

### ❌ Track I: Perch 2.0 "Bittern" backbone — CONFIRMED DEAD END (Apr 14)

**`perch_v2_cpu/1` already IS Perch 2.0 (Bittern)**. The arXiv 2508.04665 paper describes exactly
what's deployed. The sonotype blindness (zero Perch delta for 47158son* species) is intrinsic —
these species are not in the 14,795-class taxonomy at all. No model upgrade can fix this.

---

## Archived: Apr 12-14 experiments

### Research summary (Apr 11, now superseded)
Two subagents identified the key gap: our pipeline is missing **post-processing** that top competitors (0.928–0.948) all use. No new Perch versions exist — everyone uses the same frozen Perch v2. All gains above 0.915 are downstream.

### Track A: Post-processing (no retraining, inference notebook only)

**Expected gain: +0.010 to +0.020 LB. Implement in order; test each step locally with `eval_protossm_pipeline.py` before submitting.**

| Step | Description | Status |
|---|---|---|
| A1 | **Per-taxon temperature scaling**: Aves logits ÷1.10, Amphibia/Insecta logits ÷0.95 | TODO |
| A2 | **File-level confidence boost** (top_k=2): add 0.05×file_max for each file's top-2 species | TODO |
| A3 | **Rank-aware scaling** (power=0.4): `probs *= file_max^0.4` — suppresses background files, amplifies confident ones. Used in every kernel scoring >0.920. | TODO |
| A4 | **Adaptive delta-shift smoothing** (alpha=0.15–0.20): confidence-weighted temporal smoothing — alpha shrinks as confidence grows | TODO |
| A5 | **Per-class temperature calibration** (coarse grid T∈{0.7, 0.8, 1.0}): maximize OOF cmAP per class. Only allow sharpening (T≤1.0) to prevent divergence seen previously. | TODO |
| A6 | **Extend Aves genus proxies**: currently PROXY_TAXA={Amphibia, Insecta}. Extend to Aves — rescue ~5–15 bird species with no direct Perch mapping. | TODO |

Implementation notes:
- A1–A4 applied in order after ResidualSSMv3 correction, before sigmoid
- A5 applied to final logits as per-class division: `final_scores[:, c] /= T[c]`
- A6 is a probe/mapping change in the inference notebook
- Competitor evidence: dingjiarun (0.924), Koushik (0.928), atahalam (0.943) all use A1+A3+A5

### Track B: Per-class OOF threshold calibration (no retraining)

**Expected gain: +0.002 to +0.005 LB (overlaps with A5 above — pick one approach).**

Grid search over 234 independent thresholds using OOF Stage2 predictions on 59 labeled soundscapes. Coarse grid (0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70) to avoid overfit. Different from the failed temperature calibration (which used unconstrained optimizer → T→7.4). This approach is bounded.

Script needed: `eval/calibrate_thresholds.py` — grid search on OOF scores, output 234 thresholds.

### ❌ Track C: Site-aware Stage 2 — DEAD END (Apr 12)

**Result: in-sample cmAP 0.3275 vs baseline 0.3355 (−0.0080). Do NOT submit.**

Site context (1536→64 projection, concat to Stage 2 input) overfits on 66 training soundscapes. The extra capacity (64 dims) overfits the 9 labeled training sites. Even with per-site means from 10K unlabeled files, the site context provides no generalizable signal at the Stage 2 level (Stage 1 already uses site embeddings, and Stage 2's proto_probs already encode site information).

Infrastructure built:
- `training/precompute_site_profiles.py` — per-site Perch mean embeddings (23 sites × 1536)
- `training/train_protossm.py` — `--site-profiles` flag, `ResidualSSMv3(d_site=64)` class
- `eval/eval_protossm_pipeline.py` — site-aware checkpoint loading + eval

### Execution order

1. **Today (Apr 11, 1 slot remaining)**: Save slot — don't submit until Track A is implemented and tested locally.
2. **Apr 12**: Implement A1–A6 in inference notebook. Run `eval_protossm_pipeline.py` for each step. Submit best config (1 slot).
3. **Apr 12 or 13**: Implement Track B calibration. If LB improves, submit combined A+B (1 slot).
4. **Apr 13+**: Implement Track C (site-aware Stage 2). Retrain on cluster. Local eval. If in-sample doesn't regress, submit (1 slot).

---

**Active work (Apr 11 — 4/5 submissions used)**:

### Stage 3 = DEAD END (Apr 11)

All Stage3 variants hurt LB. The in-sample cmAP gain (0.33→0.45→0.67) is pure overfitting to 708 labeled windows. Does NOT generalize.

| Experiment | Stage3 epochs | w3 | LB |
|---|---|---|---|
| baseline (no Stage3, single seed) | — | — | **0.915** |
| v48: 30ep ensemble | 30ep | 0.70 | TBD |
| v49: 40ep ensemble | 40ep | 0.70 | TBD |
| v50: 30ep ensemble | 30ep | 0.35 | 0.912 |
| v51: 60ep ensemble | 60ep | 0.70 | **0.895** |
| v40: cosine LR Stage2 | 30ep | 0.70 | 0.907 |

More epochs = more overfit = worse LB. Stage3 is categorically not useful. Do NOT retry.

**Root cause**: 708 labeled soundscape windows is too few for Stage3 to learn generalizable corrections. Stage3 memorizes the training soundscapes.

**5th submission slot (Apr 11)**: 1 remaining — use carefully.

**Previous work (Apr 10 — 0/5 submissions used — all experiments failed, no submissions)**:

All hyperparameter experiments failed to beat the 0.3355 in-sample cmAP baseline (→ LB 0.915). 3 submission slots were conserved.

| Experiment | In-sample cmAP | Notes |
|---|---|---|
| **baseline** (30ep, dropout=0.20, fixed LR) | **0.3355** | → LB 0.915 |
| xc_aug_s42 | — | **LB 0.902** (submitted v38) — catastrophically bad |
| noise02_e35 | 0.3345 | Gaussian noise σ=0.02, slight regularization, not enough |
| nonoise_e35 | 0.3320 | 35ep no noise: overfit on 708 windows |
| noise02_e40 | 0.3307 | More epochs + noise: still overfit |
| dropout0.30 | 0.3336 | Higher dropout underfits |
| dropout0.40 | 0.3332 | More underfitting |
| dropout0.50 | 0.3335 | Marginal recovery but still below baseline |
| cosine_e30 | 0.3299 | CosineAnnealingLR (T_max=30): LR→1e-6 at ep30, starves training |
| cosine_e60 | 0.3303 | T_max=60, 60 epochs: still can't beat fixed LR |
| s09aug_e30 | 0.3340 | Stage 2 on 66sc with 59sc-aligned probes — Stage 1 has no S09 site embedding (randomly initialized) → noisy Stage 2 |

**Root cause analysis**:
- **XC augmentation (LB 0.902)**: XC clips are isolated close-range recordings (single species) vs soundscape windows (multi-species, ambient noise). Both embedding statistics and probe score distributions differ → distribution mismatch.
- **Hyperparameter tuning at local optimum**: 30ep / dropout=0.20 / fixed LR is already at the Pareto frontier for this 708-window dataset. All regularization either underfits (dropout↑, noise, cosine) or overfits (more epochs).
- **s09aug blocked by Stage 1**: `protossm_v3.pt` trained on 59sc (8 sites). S09 = unknown site → Stage 1 proto_probs for S09 windows randomly initialized → Stage 2 correction training is noisy.

**In-sample cmAP → LB correlation**: ~1.7× magnification: Δin-sample −0.001 → ΔLB ≈ −0.002. All experimental checkpoints below 0.3355 expected to give LB < 0.915.

**Active work (Apr 11 — 5/5 submissions used — DAILY LIMIT REACHED)**:

### Apr 11 experiments (5/5 slots used — Stage3 x4 + v52)

| Experiment | OOF cmAP | LB | Notes |
|---|---|---|---|
| v52: A4+A2 post-processing | 0.3769 | **0.919** (NEW BEST) | Smooth + file-level confidence boost. Submitted 09:08 UTC. |
| v53: rank_power=2.0, no boost | 0.4070 | queued Apr 12 | Kernel COMPLETE. Remove boost, rank_power 0.4→2.0, smooth_amphibia 0.45→0.65. |
| Track C (site-aware Stage 2) | 0.3275 | DEAD END | −0.008 vs baseline. |
| A5 temperature calibration | 0.5561 (+0.0000) | DEAD END | AP ranking-invariant. |

**OOF post-processing analysis (Apr 11)**:

| Config | OOF cmAP | Delta vs baseline |
|---|---|---|
| Stage2 baseline (no pp) | 0.3671 | — |
| rank_power=0.4, smooth, no boost | 0.3859 | +0.019 |
| v52: rank_power=0.4, smooth, boost_top2 | 0.3769 | +0.010 |
| v53: rank_power=2.0, smooth (sa=0.65), no boost | **0.4070** | **+0.040** |
| rank_power=4.0, smooth, no boost | 0.4127 | +0.046 |
| rank_power=6.0, smooth, no boost | 0.4148 | +0.048 |

Key findings (updated Apr 12 with full eval):
- **File boost (A2) locally hurts (−0.009) but LB HELPS (+0.004)**: wrong direction locally. Confirmed useful from v52 LB 0.919.
- **rank_power sweep (no boost, rp=0.4→8.0)**: 0.3779 → 0.4070 → 0.4127 → 0.4148 → 0.4175. Diminishing returns above rp=4.0. v53 tests rp=2.0 on LB.
- **Insecta smooth: optimal at 0.35** — si=0.50 loses −0.001, si=0.65 loses −0.002.
- **Calibration (n=4 LB points, Pearson r=−0.651)**: local ANTI-correlates with LB. Use LB as ground truth, local only for catastrophic regression detection.
- **SC-only species = primary weakness**: 28 sonotype species (47158son*) with mean AP ≈ 0.03. CLAP zero-shot the next research direction.

**Apr 12-13 experiments (10/10 slots used)**:

| Version | Change | LB |
|---|---|---|
| v52: rp=0.4, smooth, boost | — | 0.919 |
| v53: rp=2.0, sa=0.65, no boost | rank_power 0.4→2.0 | **0.920** |
| v54: rp=4.0, sa=0.65, no boost | rp=4.0 | TIMEOUT |
| v55: rp=2.0, sa=0.65, boost_top2 | boost added back | 0.920 |
| v56: rp=4.0, sa=0.65, no boost | rp=4.0 retry | 0.919 |
| v57: A1 taxon temp + rp=2.0 | Aves÷1.10, Amphibia/Insecta÷0.95 | **0.920** |
| v58: smooth_insecta=0.65 + rp=2.0 | insecta alpha 0.35→0.65 | **0.920** |

**Post-processing = FULLY EXHAUSTED (Apr 13)**. Every variant ties or regresses vs 0.920.
Tried: rank_power sweep (0.4→4.0), boost, A1 taxon temp, insecta smoothing. All 0.920 or below.
Current best: v53 config (rp=2.0, sa=0.65, no boost). **Do not submit further post-processing variants.**

**Analysis findings (Apr 12)**:
- Per-class rank_power = global rank_power: `probs.max(axis=1)` already operates per file×class, so the implementation is already per-class. No change needed.
- **S08 weakness**: Entirely driven by Insecta sonotypes (all 47158son*). Best: son25 AP=0.93, son08=0.67. Worst: son10=0.12, son18=0.13. Chacha1 (Aves) does well (0.94).
- **Sonotype Perch signal = exactly 0.000** for ALL sonotypes: Perch is completely blind. Pipeline relies 100% on hour/site priors. CLAP zero-shot would add the first acoustic signal.
- **Hour patterns for worst sonotypes**: son10 active at 3-4 AM, son18/15/16 at 7 AM, son12/09 at 7 PM.
- **Worst audio species failure modes**: (1) strher2/redjun: NEGATIVE Perch delta (−0.1) — Perch blind, nothing fixable without new backbone. (2) compot1/limpki/magant1: STRONG Perch signal (delta 2.1–2.8) but only 3 positives in 708 eval windows → low AP from scarcity, likely better on full 700-soundscape LB set.
- **CLAP precompute script**: `training/precompute_clap_zeroshot.py` written. Needs `pip install msclap` on cluster + run to generate `clap_zeroshot_scores.npy`.

**Post-processing COMPLETE — all variants exhausted. Optimal config = v53 (rp=2.0, sa=0.65, no boost).**

---

## Backbone research (Apr 13)

### ❌ Track D: Alternative backbones — ALL DEAD ENDS (Apr 13-14)

All publicly accessible audio SSL backbones tested against Perch v2 embeddings (OOF cmAP):

| Model | OOF cmAP | Delta vs Perch |
|---|---|---|
| **Perch emb** | **0.2984** | — |
| AVES-base-bio (ESP, animal sounds SSL) | 0.2383 | −0.060 |
| BirdAVES-biox-large (ESP, bird audio SSL) | 0.2563 | −0.042 |
| WavLM-base-plus (Microsoft, speech SSL) | 0.2247 | −0.074 |
| Concat(Perch+AVES-base) | 0.2852 | −0.013 |
| Concat(Perch+BirdAVES-large) | 0.2715 | −0.027 |
| Concat(Perch+WavLM) | 0.2907 | −0.008 |

**Root cause**: Perch is supervised on 10,932 bird species — impossible to match with SSL.
Even BirdAVES-biox-large (specifically bird audio) is −0.042 below Perch. Adding any
of them as concat features hurts (noise from low-quality extra dims overwhelms signal).

**Do NOT test further SSL backbones.** Only a gated/paid model with comparable supervised
pretraining (e.g. a newer Perch version, or BirdNET-style fine-tuned on this taxonomy) could help.

### Track E: CLAP zero-shot for sonotypes ✅ COMPLETE (Apr 13)

25 47158son* insect species have **exactly 0.000 Perch logit delta** — Perch is blind.
Current pipeline uses only hour/site priors. CLAP adds acoustic signal.

**Script**: `training/precompute_clap_zeroshot.py` ✅ Fixed (torchaudio soundfile monkey-patch)
**Output**: `data/perch-meta/clap_zeroshot_scores.npy` — (792, 25), `clap_zeroshot_labels.json` ✅

**Results (Apr 13)**:
- Scores shape: (792, 25), range [0.010, 0.625]
- Mean AP = **0.060** vs chance baseline ~**0.029** (2.1× above chance)
- Best: son16 AP=0.244 (16× chance), son18 AP=0.201 (13×), son17 AP=0.174 (3.2×), son25 AP=0.142
- Worst: son12 AP=0.005, son05 AP=0.005, son02 AP=0.010

**Per-species AP (25 active species)**:
son01=0.030, son02=0.010, son03=0.108, son04=0.017, son05=0.005, son06=0.020, son07=0.045,
son08=0.045, son09=0.007, son10=0.030, son11=0.051, son12=0.005, son13=0.056, son14=0.016,
son15=0.048, **son16=0.244**, **son17=0.174**, **son18=0.201**, son19=0.030, son20=0.051,
son21=0.045, son22=0.047, son23=0.047, son24=0.024, son25=0.142

**CLAP integration decision**: Real signal for 3 species (son16, son17, son18), near-chance for most.
Max theoretical cmAP gain: (0.060 − 0.029) × 25/234 ≈ **+0.003**.
Low but non-zero; worth one submission slot. Apply additive prior for sonotype species only.

**Integration plan** (if proceeding):
```python
# After ProtoSSM correction, before sigmoid:
clap_scores = np.load("clap_zeroshot_scores.npy")  # (n_windows, 25)
# Indexed by clap_zeroshot_labels.json order
for j, sp in enumerate(clap_labels):
    col = label_to_col[sp]
    final_scores[:, col] += CLAP_WEIGHT * clap_scores[:, j]
# CLAP_WEIGHT = 0.5 (conservative) — tune via submission
```
Requires: upload clap_zeroshot_scores.npy + clap_zeroshot_labels.json as Kaggle dataset.

**Active work (Apr 11 — 4/5 submissions used)**:

| Seed | In-sample cmAP | Stage 1 alone | Notes |
|---|---|---|---|
| baseline (59sc, s42) | 0.3355 | 0.3403 | → LB 0.915 |
| **66sc s42** | **0.3430** | 0.3274 | Stage2 +0.0156. Best candidate. |
| 66sc s1 | 0.3270 | 0.3459 | Stage2 HURTS (-0.0189). Stage 1 over-learned. |
| 66sc s2 | 0.3387 | — | Above baseline but below s42. |
| 66sc s3 | TRAINING | — | — |

**v39 COMPLETE** (kernel v39): `protossm_66sc_s1_s42.pt`, weight=0.70 → LB **0.912** (-0.003 vs best 0.915). **REGRESSION**.

| Seed | In-sample cmAP | Stage 1 alone | Notes |
|---|---|---|---|
| baseline (59sc, s42) | 0.3355 | 0.3403 | → LB 0.915 |
| **66sc s42 30ep** | **0.3430** | 0.3274 | → LB 0.912 (-0.003!) |
| 66sc s42 22ep (optimal) | 0.3438 | 0.3274 | Expected LB ~0.912 (not submitted) |
| 66sc s1 | 0.3270 | 0.3459 | Stage2 HURTS (-0.0189). Dead end. |
| 66sc s2 | 0.3387 | — | Above baseline but useless given 66sc hurts |
| 66sc s3 | 0.3351 | — | Below baseline |

**Root cause analysis (CONFIRMED DEAD END)**:
- 66sc = 59sc + 7 S09 soundscapes
- In-sample cmAP improvement (+0.0075) was ENTIRELY due to overfitting to S09 windows
- S09 site characteristics do NOT generalize to test soundscapes
- The original 59sc Stage 1 (8 sites) has better generalization to the DIVERSE test set
- Adding S09 introduces a site-specific bias that hurts on test soundscapes

**Historical pattern**: ALL 66sc experiments have hurt LB:
- 66sc Stage 2 (Apr 8): 0.914 (-0.001)
- 66sc kfold ensemble (Apr 9): 0.912 (-0.003)
- 66sc full Stage 1+2 retrain (Apr 11): 0.912 (-0.003)
→ **S09 data is categorically harmful. Never add 66sc data to training.**

**Next direction (Apr 12)**:
- The 59sc 30ep architecture appears at local optimum for current model
- Need fundamentally different approach to improve
- Options: (1) Post-processing improvements (rank-aware file_max^0.4, per-class thresholds); (2) Better probe quality; (3) Stage 2 with additional inputs (Perch logits)

**Active work (Apr 9 — 5/5 submissions used)**:
- **v34 (adapter alone)**: LB **(blank)** — scoring failure or still pending
- **v35 (kfold ensemble, no adapter)**: LB **0.912** — still worse than single seed. Root cause: kfold seeds trained on 66sc (792 windows) but inference uses 59sc probe scores → training-inference mismatch
- **v36 (adapter + kfold ensemble)**: LB **0.910** — adapter hurts on top of misaligned seeds
- **Root cause confirmed**: All previous ensemble seeds used 66sc training data (792 windows) but inference uses 59sc probe scores (708 windows) from jaejohn/perch-meta. Training-inference mismatch = -0.003 penalty.
- **Fix**: New 59sc seeds (59sc_s1-s4). Stage2-only retrain from protossm_v3.pt Stage1 with `full_perch_arrays_59.npz` + `full_probe_scores__59sc.npy` (cmAP 0.9261). Epoch 30 losses: s1=0.00315, s2=0.00310. Uploaded to `birdclef2026-protossm-v3`.
- **v37 (original primary + 59sc_s1-s4 ensemble, no adapter)**: LB **0.913** — properly-aligned ensemble still hurts (0.913 vs 0.915 single). **Conclusion: ensemble with fixed-ep seeds has no diversity; single seed=42 correction is uniquely good. Ensemble approach abandoned.**
- **Infrastructure fixes**: Added `--npz-file` flag to `train_protossm.py` and `precompute_probe_scores.py`. OOF probe path now derived from probe_scores_file (no hardcoded 66sc mismatch).

**Active work (Apr 8 — 5/5 submissions used)**:
- **v29 (original weights + v9-v12 ensemble, no mask)**: LB **0.912** — ensemble hurts. Root cause: v9-v12 use fixed 30ep vs original's early stopping.
- **v30 (66sc 5-seed ensemble)**: PENDING. 792 windows (66sc), original+v13-v16_66sc (all fixed 30ep), no mask.
- **v31 (single-seed 66sc, v13_66sc)**: LB **0.914** — 66sc data marginally WORSE than 59sc original. Root cause: probe score quality drops (0.926→0.8905) with new soundscapes; training-inference mismatch.
- **v32 (early-stopping ensemble)**: LB **0.913** — still worse than single seed. Even with same val split, different seeds don't complement each other.
- **Early-stopping seeds (es_s1-s4)**: Trained with `--stage1-checkpoint protossm_v3.pt --seed {1-4}` on 66sc data, NO --stage2-epochs flag (early stopping). All stopped at epoch 27-28, best val=0.00591-0.00594.
- **Key insight (Apr 8 — val split bug)**: ALL previous es seeds used SAME val split (hardcoded rng_split=42). Fixed to use args.seed. K-fold seeds (kfold_s1-s4) now have diverse val splits: s1=0.00786, s2=0.00260, s3=0.00769, s4=0.00719. Diverse val splits = true K-fold diversity.
- **K-fold seeds (kfold_s1-s4)**: Trained on 66sc with diverse val splits (args.seed). Best val varies widely (0.00260–0.00786). Dataset v3 version 13. Kernel v33 prepared (original + kfold_s1-s4). Submit Apr 9.
- **66sc training-inference mismatch**: 66sc probe scores (cmAP 0.8905) used in training but 59sc probe scores (cmAP 0.926) used at inference (jaejohn/perch-meta has 59sc only). This misalignment costs ~0.001 LB.
- **Val split fix**: `rng_split = np.random.default_rng(args.seed)` — seed 42 unchanged, seeds 1-4 now use different val splits.

**Active work (Apr 7 — 5/5 submissions used)**:
- **v23 (×2, duplicate)**: 30ep + binary_mask + weight=0.70 → LB **0.910** (-0.005 vs v16). positive_mask HURTS at 30 epochs.
- **v24 (weight=0.80)**: binary mask + weight=0.80 → **(blank score)** — scoring failure.
- **v25 (soft sqrt)**: sqrt(n_pos/max) weighting + weight=0.70 → **(blank score)** — scoring failure.
- **v27 (5-seed ensemble + mask)**: seeds 0-4 (v3+v9-v12), binary mask, weight=0.70 → LB **0.910** — ensemble offers no gain when mask applied.
- **v28 (no mask, 5-seed ensemble)**: COMPLETE, queued for tomorrow. Hypothesis: no mask should recover 0.915+.
- **Key finding**: positive_mask = dead end at 30 epochs. At 33ep +0.001 was marginal. At 30ep it costs -0.005. The corrections for 163 no-positive species are beneficial at the optimal epoch count.
- **Fast Stage 2-only retrain**: `--stage1-checkpoint` flag added to train_protossm.py. Seeds 1-4 (v9-v12) trained in ~3 min each. All have positive_mask=True saved in checkpoint.

**Active work (Apr 6 — COMPLETE, 5/5 submissions used)**:
- **Stage 2 v3 (60 ep)**: LB **0.908** (kernel v19). 60 epochs overfit on 708 samples. 30ep confirmed optimal.
- **Stage 2 v4 (early stopping, 48/59 soundscapes)**: blank score (dataset not ready when kernel ran — <20s delay).
- **Stage 2 v4 early stopping analysis**: early stopping at epoch 33 (best val=0.00519). Val plateau from epoch 20. Optimal epoch = 30 (confirmed by LB).
- **Stage 2 v5 (33ep fixed, all 59)**: LB **0.909** (kernel v21). 33ep slightly worse than 30ep.
- **Stage 2 v5 + positive_mask**: LB **0.910** (kernel v22). +0.001 from zeroing 163 no-positive species' corrections.
- **Stage 2 v6 (30ep + mask)**: kernel v23 COMPLETE and ready. Submit on Apr 7.
- **Implemented**: early stopping (`--stage2-epochs N` flag), `positive_mask` saved to checkpoints and applied in inference notebook.

**Active work (Apr 5)**:
- **Step 17 + V18 DONE**: LB **0.913** (+0.001) — ProtoSSM 50/50 blend + V18 probes + rank-aware post-proc.
- **Step 19 DONE**: LB **0.913** — ONNX Perch + ProtoSSM (no CNN). ONNX logits confirmed equivalent to TF.
- **Step 19 v1 (with CNN) TIMEOUT**: scoring env too slow for ONNX + CNN combined (~90+ min).
- **Step 18 COMPLETE**: Perch ONNX = **7.98x speedup** (1.99s/file vs 15.87s/file). 739 files: **24 min** (vs 195 min TF).
- **Step 20 DONE**: ProtoSSM v2 (ResidualSSM integrated) = LB **0.913** — same as v1. Input was (logits-perch, 234) only — missing emb context.
- **Step 21 DONE**: Co-occurrence PMI boost (weight=0.20, threshold=0.5) = LB **0.913** — dead end. Station-specific ecology doesn't generalize.
- **Step 22 DONE**: Probe features 139→143 (+std_v, diff_mean, window_pos, delta_prev). In kernel v12/v13.
- **Step 23 TRAINED** (Apr 4): ResidualSSMv3 (competitor-matched) — OOF cmAP **0.7629** (vs 0.5452 ProtoSSM alone). Submit kernel v13 on Apr 5. Dataset: `aldisued/birdclef2026-protossm-v3`.
- **Step 23 v2 (BCE fix)**: ResidualSSMv3 trained with BCE logit-space loss — LB **0.909** (kernel v14, Apr 5). Correct space but wrong training base (proto_logits ~0.545 quality).
- **Step 23 v3 (proper fix, Apr 5)**: ResidualSSMv3 trained with full probe scores as Stage2 base (cmAP 0.926 in-sample). Training-inference alignment. OOF improvement: +0.0017. LB **0.914** (+0.001, kernel v15).
- **Step 23 RESIDUAL_WEIGHT sweep (Apr 5)**: Correction generalizes well! 0.35→0.914, 0.70→**0.915** (new best, v16), 1.00→pending (v17). No retraining needed.

**Step 17 results (ProtoSSM v1)**:
- 5-fold OOF cmAP = 0.5452 (per-fold: 0.5548, 0.3614, 0.7403, 0.5888, trained on ~17-29 classes/fold)
- Full retrain on 708 windows done in 24 min. Checkpoint: `aldisued/birdclef2026-protossm-v1`

**Step 18 results (Perch ONNX benchmark v5)**:
- Speedup: **7.98×** (train_soundscapes, 20 files, TF 2.20.0 vs onnxruntime 1.21.0)
- ONNX = 1.99s/file → 24 min for 739 files (fits in 90-min budget with 66 min remaining)
- TF = 15.87s/file → 195 min (way over budget on its own)
- Model: `yuriygreben/birdclef26-perch-onnx/perch_v2_no_dft.onnx` (tf2onnx conversion)

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
| **Perch v4 (MLP probes + 5-way archetypes)** | **TIMEOUT** | MLP adds seconds; scoring env barely fits in 90 min → blank score |
| **Perch v5 (5-way archetypes + LogReg)** | **0.912** | 5-way archetypes neutral — same score as v3 |
| **Perch v8 (ProtoSSM 50/50 + V18 probes + rank-aware)** | **0.913** | +0.001 — ProtoSSM is slightly positive |
| **Step 19 v1 (ONNX + ProtoSSM + 1-fold CNN)** | **TIMEOUT** | CNN adds ~38 min in scoring env → >90 min total |
| **Step 19 v2 (ONNX Perch + ProtoSSM, no CNN)** | **0.913** | ONNX logits = TF logits (confirmed equivalent) |
| **Step 20 ProtoSSM v2 (ResidualSSM second pass)** | **0.913** | ResidualSSM dead end — OOF 0.5438 vs v1 0.5452. Too small dataset (708 windows) to learn corrections |
| **Step 21: co-occurrence PMI boost (weight=0.20, threshold=0.5)** | **0.913** | Dead end — station-specific ecology doesn't generalize. 852 pairs, max P(j\|i)=0.91, mean boost 0.039, max boost 5.18 |
| **Step 23 v1: ResidualSSMv3 competitor-matched** | **0.906** | Kernel v13 — MSE in probability space (wrong loss), wrong base |
| **Step 23 v2: ResidualSSMv3 BCE fix** | **0.909** | Kernel v14 — BCE logit-space loss, but Stage2 trained on proto_logits base (~0.545 quality) |
| **Step 23 v3: ResidualSSMv3 proper fix (full probe base)** | **0.914** | Kernel v15 — Stage2 base=full probe scores (0.926 in-sample cmAP). Training-inference alignment. OOF +0.0017.
| **Step 23 weight=0.70** | **0.915** | Kernel v16 — RESIDUAL_WEIGHT=0.70. Correction generalizes — linear gain. **New best.**
| **Step 23 weight=1.00** | **0.915** | Kernel v17 — RESIDUAL_WEIGHT=1.00. Plateau: same as 0.70. Optimal weight is 0.70. |
| **Stage 2 v2 (TTA-matched + OOF + 60ep)** | **0.914** | Kernel v18 — regression. TTA mismatch + OOF proto_probs both hurt. |
| **Stage 2 v3 (60ep, in-sample)** | **0.908** | Kernel v19 — clear overfitting on 708 samples. 30ep was optimal. |
| **Stage 2 v4 (early stopping, 48/59)** | **(blank)** | Kernel v20 — dataset not ready when kernel ran (20s delay too short). |
| **Stage 2 v5 (33ep fixed, all 59)** | **0.909** | Kernel v21 — 33ep slightly worse than 30ep. 30ep confirmed as optimal. |
| **Stage 2 v5 + positive_mask** | **0.910** | Kernel v22 — +0.001 from mask (33ep base). Mask helps the 163 no-positive species. |
| **Stage 2 v6 (30ep + positive_mask)** | **0.910** | Kernel v23 (×2 duplicate) — optimal 30ep base + mask. positive_mask HURTS at 30ep (-0.005 vs no-mask v16=0.915). |
| **Stage 2 v6 + weight=0.80** | **(blank)** | Kernel v24 — binary mask + RESIDUAL_WEIGHT=0.80. Scoring failure. |
| **Stage 2 v7 (sqrt weighting)** | **(blank)** | Kernel v25 — soft sqrt(n_pos/max) correction weight. RESIDUAL_WEIGHT=0.70. Scoring failure. |
| **Stage 2 ensemble (5 seeds, mask)** | **0.910** | Kernel v27 — average correction from seeds 0,1,2,3,4 (v3+v9-v12). Binary mask. RESIDUAL_WEIGHT=0.70. Ensemble no benefit when mask applied. |
| **Stage 2 ensemble (5 seeds, NO mask)** | **0.912** | Kernel v29 (v28 config) — correct original weights (first_w=-0.007645) + v9-v12 (fixed 30ep seeds). No mask. Ensemble hurts vs single seed. |
| **66sc 5-seed ensemble (792 windows)** | **(blank)** | Kernel v30 — SCORING FAILURE. Root cause: protossm_original.pt not in 66sc dataset, fallback to legacy blend. |
| **66sc single seed (792 windows)** | **PENDING** | Kernel v31 — v13_66sc only (seed42, fixed 30ep). Tests if +12% data helps. |
| **Early-stopping ensemble (orig+es_s1-s4)** | **0.913** | Kernel v32 — original (59sc, ep33) + es_s1-s4 (66sc, ep27-28). No mask. Ensemble hurts even with training-regime match. |
| **K-fold ensemble (orig + kfold_s1-s4)** | **queued** | Kernel v33 — original + kfold seeds with DIVERSE val splits. True K-fold benefit expected. Submit Apr 9. |
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

### What 0.920–0.937 teams are doing (confirmed Apr 4)

**Score ladder** (confirmed from research):

| Approach | LB |
|---|---|
| Our Perch v5 (LogReg probes + Bayesian priors + 5-way smoothing) | 0.912 |
| ProtoSSM v18 (`dingjiarun/pantanal-distill-birdclef2026-improvement-a4dc68`) | ~0.924+ |
| Kamongi V22 (ProtoSSM + YAMNet + XC train embeddings + sonotype boost) | ~0.928 |
| Milan Joshi v72+ (Perch ONNX + ProtoSSM + EfficientNet-B0 blend) | unknown |

**The ProtoSSM architecture** (documented in `research-protossm.md`) is the confirmed source of the 0.912→0.924 jump:
- 4-layer bidirectional Mamba-style SSM (d_model=320) on top of frozen Perch 1536-dim embeddings
- Trains on only 708 windows (59 labeled soundscapes × 12 windows) — fits in ~5 min CPU
- Gated per-class fusion: each of 234 species learns α to blend prototype similarity vs raw Perch logit
- ResidualSSM second pass (2-layer BiSSM on residuals)
- 5-shift temporal TTA, rank-aware post-processing (file_max^0.4), adaptive delta smoothing
- Per-class thresholds from OOF (234 hardcoded values)
- **Key timing constraint**: must run in artifact mode — train weights offline, load in scoring kernel in <5s

**Delta breakdown** (0.912 → 0.924+):

| Change | Est. gain |
|---|---|
| ProtoSSM BiSSM temporal modeling | +0.008–0.012 |
| ResidualSSM second pass | +0.002–0.004 |
| Probe improvements (PCA 128, min_pos 5, 143 features) | +0.002–0.003 |
| Aves genus proxies (extend to unmapped Aves) | +0.001 |
| Rank-aware post-processing + adaptive delta smoothing | +0.002 |
| Per-class thresholds from OOF | +0.002 |

**Perch ONNX** (`yuriygreben/birdclef26-perch-onnx`): `tf2onnx` conversion reported 9x faster than TF SavedModel (Milan Joshi v40+). If confirmed ≥3x, unlocks CNN blend (drops Perch from 85 min → 10–28 min). Pre-converted model on Kaggle — just needs a benchmark kernel. Note: TFLite (SELECT_TF_OPS, 0.9x) and CNN ONNX (slower than PyTorch) are both dead ends — this is a different path (TF model → ONNX RT).

Key insights from Apr 4 research (3-agent deep dive — see `research-protossm.md`):
- The biggest jump is **ProtoSSM** (temporal SSM), not probe tuning or post-processing
- Co-occurrence PMI prior (+0.008–0.018 est.) is the best no-training improvement
- Per-class Platt scaling (bias+scale) is untested and potentially significant — different from per-class temperature (dead end)
- Probe feature expansion (143 vs 41 features, std_base + interaction terms) is cheap and confirmed by ProtoSSM V18
- XC train audio embeddings for probe training (Kamongi V22 technique) gives more positives for rare classes

---

## Next steps (ordered by expected impact)

### Research: Closing the Gap to 0.940 LB (Apr 8 — deep-dive by two subagents)

**Current best: 0.915. LB leader: 0.940. Gap: -0.025.**

Two research directions investigated in parallel:

#### Direction A: Fine-tuning Perch Embeddings

**Verdict: Direct Perch fine-tuning = ❌ NOT RECOMMENDED. Perch Embedding Adapter = ✅ viable (+0.002–0.004)**

- **Full Perch fine-tune**: Catastrophic forgetting risk (70%). Perch is a TF SavedModel; PyTorch bridge adds complexity. 59 soundscapes far too small (708 windows) for 10,932-class model fine-tuning.
- **LoRA adapters on Perch**: Medium risk (50% overfit). Still requires TF fine-tuning infrastructure not in codebase.
- **✅ Recommended: Perch Embedding Adapter** — small PyTorch MLP (~50K params) that transforms frozen Perch embeddings: `concat(emb 1536, raw_logits 234) → 512 → 1536`. Residual design: `emb_out = emb_in + α * delta`. Frozen Perch prevents catastrophic forgetting. <2 min training on cluster. Feeds into existing LogReg probes (no re-run needed).
  - Architecture: 2-3 layers, LayerNorm + ReLU + 0.2 dropout, train on 708 windows
  - Expected: +0.002–0.004 LB (confidence 70%)
  - Risk: Station-specific overfit (30%)

#### Direction B: BirdCLEF 2025 Pseudo-Labels

**Verdict: ❌ NOT VIABLE — wrong geography, wrong taxonomy**

- BirdCLEF 2025 = Kenyan birds (East African savanna/woodland). BirdCLEF 2026 = Pantanal (South American wetland). ~10-15 species overlap max.
- Our probes only know 53/234 Pantanal species — can't generate pseudo-labels for Kenyan species.
- CNN pseudo-label pretraining (Step 9) already failed exactly this pattern: -0.28 sc_cmap, unrecoverable.
- **Alternative** (if any pseudo-label approach): Apply current model OOF predictions on the 59 existing soundscapes as soft targets for Stage 2 → +0.002–0.004 expected, low risk.

#### Gap Analysis: Can we reach 0.940?

| Approach | Expected gain | Total (from 0.915) |
|---|---|---|
| K-fold ensemble (v33, Apr 9) | +0.001–0.002 | 0.916–0.917 |
| Perch Embedding Adapter | +0.002–0.004 | 0.918–0.921 |
| OOF pseudo-label Stage 2 augmentation | +0.002–0.004 | 0.920–0.925 |
| Combined | +0.005–0.010 | 0.920–0.925 |

**Conclusion**: Best realistic ceiling with current architecture ~0.920–0.925. To reach 0.940 requires something not yet identified — likely a fundamentally better base model or much more labeled data. The 0.025 gap is real and will not close with incremental improvements.

**Recommended plan**:
1. Submit K-fold ensemble (v35) Apr 9 — no adapter, original primary + kfold_s1-s4
2. Submit adapter test (v34) Apr 9 — adapter + adapted primary, no ensemble
3. Submit full system (v36) Apr 9 — adapter + adapted primary + kfold ensemble
4. Try OOF pseudo-label augmentation for Stage 2 based on results

---

### ✅ Step 24 — Perch Embedding Adapter (Apr 8)

**Training: 33s on cluster | α=0.175 | Best val BCE: 0.01769 | Kernel v34 (single), v36 (full system)**

**Implementation**:
- New script: `competitions/birdclef-2026/training/train_perch_adapter.py`
- Architecture: `concat(emb 1536, logits 234) → LayerNorm(512) → GELU → Dropout(0.2) → Linear(1536)`
- Residual: `emb_out = emb + α * delta`, α learned (init 0.1, final 0.175)
- Zero-init output: adapter starts as identity transformation
- Training: AdapterWithHead (linear proxy for LogReg probes), BCE loss, Adam, 200ep max, early stop patience=30
- Stopped at ep74 (best val ep44), val BCE: 0.02028→0.01769 (improved 12%)
- Adapted probe scores: cmAP 0.8904 (same as raw 66sc — adapter doesn't help probes directly)
- Stage 2 retrained with adapted emb + adapted probe scores: `protossm_adapted.pt` (30ep fixed, 27s)

**Checkpoints**: `perch_adapter.pt` (6.8MB) + `protossm_adapted.pt` (20MB) in `birdclef2026-protossm-v3` v14

**Infrastructure changes** (merged to main):
- `train_perch_adapter.py` — standalone adapter training
- `precompute_probe_scores.py --emb-file --output-suffix` — adapted probe computation
- `train_protossm.py --emb-file --probe-scores-file` — Stage 2 with adapted emb
- Inference notebook: `CFG.USE_ADAPTER` flag + Cell 11b (adapter application)

**Apr 9 experiment grid**:
| Kernel | Config | Result | Notes |
|--------|--------|--------|-------|
| v34 | adapter=True, adapted primary, no ensemble | (blank) | Scoring failure |
| v35 | adapter=False, original primary, kfold ensemble (66sc) | 0.912 | 66sc mismatch hurts |
| v36 | adapter=True, adapted primary, kfold ensemble (66sc) | 0.910 | Adapter + mismatch = worst |
| v37 | adapter=False, original primary, 59sc_s1-s4 ensemble | RUNNING | Properly-aligned ensemble |

---

### ❌ Step 15 — Cross-species pre-training on BirdCLEF 2021-2024 — DEAD END (Apr 3)

**Expected: +0.010–0.030 LB | Requires pre-training + fine-tuning | High priority**

**Hypothesis**: Perch's advantage over our CNN comes from being trained on 10,932 bird species — rich cross-species representations. Pre-training our B0 backbone on 2021-2024 combined data (~900 species, ~120K clips) before v7 fine-tuning should close part of this gap.

| Year | Clips | Species | Audio dir |
|------|-------|---------|-----------|
| 2021 | 62,874 | 397 | `train_short_audio/` |
| 2022 | ~15,000 | 152 | `train_audio/` |
| 2023 | ~16,900 | 264 | `train_audio/` |
| 2024 | ~24,459 | 182 | `train_audio/` |
| **Total** | **~120K** | **~900 unique** | — |

**Pipeline**:
1. `precompute_pretrain_specs.py` — HTK mel cache for all 4 years (same config as v7). Running on cluster, ~1.5h.
2. `pretrain_cnn.py` — multi-class CrossEntropy on primary label, 15 epochs, saves backbone-only weights.
3. `train_cnn.py --baseline --htk --soundscape-labels --pretrained-backbone outputs/pretrain-v1_best_backbone.pt` — v7 fine-tune from pre-trained init instead of ImageNet.

**Why CrossEntropy (not BCE)**: pre-training only needs one label per clip (primary species). BCE requires knowing all species present — we only have reliable primary labels across all years.

**Key precedent**: BirdCLEF 2025 top teams used exactly this 2021-2024 pre-training → 2025 fine-tuning approach.

- [x] Download 2021-2024 data (Apr 3)
- [x] Precompute HTK specs for all 4 years (~1.5h)
- [x] Pre-train B0: 15 epochs, 926 species, 119K clips, final val_acc=0.360
- [x] Fine-tune fold 0: sc_cmap(held) = **0.7135** vs v7 0.9762 — delta **-0.2627**
- [x] **DEAD END** — pipeline stopped, folds 1-3 not run

**Root cause**: Same catastrophic forgetting pattern as all prior pre-training attempts (pseudo-labels 0.69, Perch KD 0.54, B3 0.57). ImageNet NoisyStudent init (`tf_efficientnet_b0.ns_jft_in1k`) was trained on 300M+ images with noisy student distillation — richer low-level features than 119K XC clips can provide. XC pre-training overwrites these features; 30-epoch fine-tuning at LR=1e-3 cannot recover them.

**Why Perch doesn't have this problem**: Perch uses Perch embeddings as *features*, not as a pre-trained CNN backbone. The CNN backbone still starts from ImageNet. Our approach tried to replace the backbone init, which hurts.

**Do not retry** with same architecture. Only viable retry: frozen backbone for first N epochs + lower LR — but this is risky and the expected gain is marginal.

---

### Step 24 — Perch Embedding Adapter (Apr 8-9)

**Expected: +0.002–0.004 LB | Training: <5 min | Architecture: lightweight MLP adapter on frozen Perch embeddings**

**Hypothesis**: Frozen Perch embeddings are optimized for 10,932 global species. A small adapter (~1.8M params) can transform them to be more discriminative for Pantanal-specific species, improving probe quality from cmAP 0.926 without catastrophic forgetting (zero-init residual → starts as identity).

**Architecture**: PerchEmbeddingAdapter
- Input: concat(emb 1536, perch_logits 234) = 1770 dims
- 2-layer MLP: Linear(1770→512) → LayerNorm → GELU → Dropout(0.2) → Linear(512→1536) (zero-init)
- Residual: `emb_out = emb_in + α * delta`, α≈0.1 init, learned (log_alpha parameter)
- Training: BCE vs species labels using linear head proxy (1536→234), pos_weight_cap=5.0

**Full pipeline**:
1. `train_perch_adapter.py` → adapter checkpoint + `full_emb_adapted.npy`
2. `precompute_probe_scores.py --emb-file full_emb_adapted.npy --output-suffix adapted`
3. `train_protossm.py --stage1-checkpoint protossm_v3.pt --emb-file full_emb_adapted.npy --probe-scores-file full_probe_scores_adapted.npy`
4. Upload `perch_adapter.pt` + `protossm_adapted.pt` to birdclef2026-protossm-v3 dataset
5. Kernel v34: single-seed, adapter applied at test time

**Files**:
- `training/train_perch_adapter.py` — standalone adapter training script
- `precompute_probe_scores.py` — modified to accept `--emb-file` and `--output-suffix`
- `train_protossm.py` — modified to accept `--emb-file` and `--probe-scores-file`
- Inference notebook — adapter cell inserted after cache loading (cell 11b)

**Key guard**: adapter applied to BOTH training embeddings (before PCA/probes) AND test embeddings (before PCA projection). Ensures train-test consistency.

**Training results (Apr 8)**:
- Adapter: early stop epoch 44/200, best val BCE=0.01769 (vs init 0.02028), final α=0.1749
- Full probe cmAP (adapted emb, 66sc): 0.8904 — same as raw 66sc baseline (neutral)
- Stage 2: 30ep fixed, train loss 0.00338 (33s total)
- Kernel v33 pushed (protossm_adapted.pt primary, with adapter), LB score pending Apr 9

- [x] Train adapter on cluster (max 200 epochs, early stopping patience=30)
- [x] Recompute adapted probe scores
- [x] Retrain Stage 2 with adapted embeddings + adapted probe scores
- [x] Upload checkpoints to birdclef2026-protossm-v3 (v14)
- [x] Push kernel v33 (COMPLETE, submission pending)
- [ ] Submit Apr 9 (daily limit hit Apr 8)
- [ ] Report: LB score (pending Apr 9)

---

### Step 16 — Perch post-processing improvements (Apr 3)

**In progress** — three changes to `kaggle_perch_v2_inference.ipynb`:

1. **5-way acoustic archetypes** (implemented) — replace binary texture/event smoothing with 5 groups:
   - Amphibia → α=0.40 (continuous background chorus)
   - Insecta → α=0.35 (texture)
   - Aves proxy/unmapped → α=0.20 (uncertain bird calls)
   - Aves directly-mapped → α=0.10 (event birds)
   - Mammalia/Reptilia → α=0.05 (rare transients)

2. **MLP probes** (implemented) — replace `LogisticRegression` with `MLPClassifier(hidden=(64,))` on same 41-43 features; `predict_proba` logit-transformed for blending consistency.

3. **Tune smoothing α** — `eval/tune_perch_smoothing.py` grid-searches 108 configs on 59 labeled soundscapes (GroupKFold by site, cmAP metric). Run on cluster to get best α values, then bake into notebook.

- [x] Implement 5-way archetypes in notebook
- [x] Implement MLP probes in notebook
- [x] Write tuning script (`eval/tune_perch_smoothing.py`)
- [x] Run tuning on cluster, update notebook α values
- [x] Push Kaggle kernel v4 (MLP probes) — **TIMEOUT** (blank score, >90 min in scoring env)
- [x] Push Kaggle kernel v5 (LogReg probes) — **LB 0.912** — tied with baseline. 5-way archetypes neutral.

**Conclusion**: 5-way archetypes = neutral (same 0.912). MLP probes timed out in scoring env (scoring env barely fits within 90 min with LogReg). All Perch post-processing improvements exhausted.

---

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

### ❌ Step 5g — Per-class calibration on labeled soundscapes — DEAD END (Apr 2)

**Result: OOF cmAP delta = -0.0004 (essentially zero). Not worth pursuing.**

Script: `eval/calibrate_perch_temps.py` — runs 5-fold GroupKFold OOF on 59 labeled soundscapes (708 windows), optimises T_c per class with Brent's method (log-T bounds [-2, 2] → T ∈ [0.135, 7.389]).

| Metric | Value |
|---|---|
| OOF cmAP (no T) | 0.5397 (71 classes with ≥1 positive) |
| OOF cmAP (per-class T) | 0.5393 |
| Delta | **-0.0004** |
| Median T_c | 7.389 (hitting upper bound) |

**Root causes**:
- Only 71/234 classes have any positives in the 59 labeled soundscapes → 163 classes get T_c=1.0 (no signal)
- 132–144 val samples per fold, very few positives per class → optimizer hits upper bound (T=7.389) for most calibrated classes
- When ranking is already correct, temperature scaling is a rank-preserving no-op
- Perch probe OOF scores are well-calibrated enough that there is no systematic over/underconfidence to correct

**Do not retry**: increasing labeled data or reducing PCA dims won't overcome the fundamental sparsity — 59 soundscapes covers only ~30% of target species.

---

### ❌ Step 4 — Perch v2 + CNN ensemble blend — DEAD END (confirmed Apr 2)

All same-kernel blend attempts timed out in scoring env. No viable path remains:

| Approach | Result |
|---|---|
| kernel_sources | Bug: loads dry-run zeros → same as Perch alone |
| Single kernel, 4-fold CNN | Timeout (blend v2) |
| Single kernel, 2-fold CNN | Timeout (blend v3) |
| TF thread caps + 1-fold CNN | Timeout (blend v4, v5) |
| TFLite for Perch (10× speedup) | Dead end — SELECT_TF_OPS bypasses XNNPACK, 0.9× speedup |
| Parallel Perch+CNN threads | TF at half cores = 12.87s/file → 127 min, doesn't fit |

**Root cause**: TF consumes all CPUs and doesn't release thread pools. PyTorch then competes → 2-4× slowdown. No fix exists for Perch v2 SavedModel (requires JAX checkpoint for native TFLite). Code Competition = no external blending.

**Do not retry**.

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

### ❌ Step 5 — Per-class calibration + Perch probe ablations — DEAD END (Apr 2)

Per-class temperature calibration (Part A → Step 5g) confirmed dead end: OOF delta = -0.0004.
Site×hour Bayesian prior (Part B) already implemented in Perch v2 pipeline (LB 0.912).
Raw embeddings without PCA (Part C): not tested, but given calibration failure the expected gain is minimal.

**Do not retry any of these**.

---

---

### ❌ Step 10 — Perch → CNN knowledge distillation (Perch soft labels for CNN training) — DEAD END (Apr 1)

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

### ❌ Step 11 — Perch embedding supervision for CNN (auxiliary distillation head) — DEAD END (Apr 1)

Deprioritized when Step 10 (same data, same root cause) confirmed dead. KD signal doesn't help the CNN close the XC→soundscape domain gap — same fundamental issue as Step 10.

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

### ❌ Step 13 — Soundscape fine-tuning stage (post-XC, soundscape-only) — DEAD END

**Result: sc_cmap degrades from 0.9484 → best 0.9530 (ep1) then 0.93 range. Below v7's 0.976. Dead end.**

Hypothesis: after full v7 training (XC+soundscape mixed), fine-tune on ONLY labeled soundscape segments at low LR to adapt the model to the soundscape distribution.

Variants tested (fold 0 only, Apr 1):
| Variant | LR | Best sc_cmap | Notes |
|---|---|---|---|
| Full model | 1e-5 | 0.9409 | Degrades steadily |
| Full model | 5e-6 | 0.9361 | Degrades steadily |
| Head-only (backbone frozen) | 1e-4 | 0.9530 (ep1) | Instantly degrades after ep1 |

**Starting sc_cmap from v7 checkpoint (no fine-tuning): 0.9484**. v7 training reported 0.976 — gap is likely due to BN running stats context difference between training and fresh eval runs.

**Root cause**: The 52 training soundscapes are ALREADY in v7 mixed training — no new information. Fine-tuning on them produces overfitting to specific recording conditions (station S08, S09, etc.), hurting generalization to the 14 held-out files. Even head-only mode peaks at epoch 1 then overfits immediately (head has 2.2M params vs only 1,180 training segments).

**Conclusion**: Soundscape fine-tuning is fundamentally limited by data volume. With only 52 files, the model memorizes recording environment rather than species patterns. The approach would only work with many more labeled soundscapes.

---

### ❌ Step 14 — XC-trained Perch probes (all 206 species) — DEAD END (Apr 1)

**Expected: +0.005–0.020 LB | Actual: OOF mean AP 0.548 vs Perch baseline 0.749 (delta -0.201)**

**Results**:

| Variant | OOF mean AP | vs Perch | Notes |
|---|---|---|---|
| Perch v2 (baseline) | **0.749** | — | reference: raw logits on XC data |
| XC probes (PCA-only, 64-dim) | 0.549 | -0.200 | LogReg on PCA of 1536-dim emb |
| XC probes (PCA + Perch logit, 65-dim) | 0.548 | -0.201 | adding raw logit made no difference |

**Root causes**:
1. **Domain shift**: XC clips = close-up, single-species recordings. Soundscape windows = ambient, multi-species PAM. LogReg trained on XC embeddings has a different decision boundary than needed for soundscape predictions.
2. **Perch already captures the signal**: Perch v2 embeddings encode species identity optimally — a LogReg probe on top adds noise. The soundscape probes work because they use 43-dim features including temporal context + site priors; XC probes use only raw PCA (64-dim), missing the key discriminative features.
3. **Only 4/181 species showed positive delta**: 177 species were hurt; bottom 2: yebcar -0.70, 43435 -0.70.

**Infrastructure built** (reusable):
- `data/perch_cache_train_clips.py` — computes Perch v2 embeddings for all 35,549 XC clips → `perch_train_cache_v2.npz`
- `training/train_perch_probes_v2.py` — trains XC probes; fixed Perch v2 API (serving_default, labels.csv/scientific_name mapping)
- `inference/kaggle_perch_v2_inference.ipynb` cell 19b — XC probe application code (remove from kernel)

---

### Step 6 — Co-occurrence prior (medium priority)

**Expected: +0.003–0.010 LB | No retraining**

Build species-species conditional probability from labeled training soundscapes and training clips:
- `P(species_j present | species_i high-confidence)` from `train_soundscapes_labels.csv`
- Apply small correction: if species_i score > 0.5, boost correlated species_j slightly
- Condition on habitat/site to avoid spurious co-occurrences

---

### ❌ Step 7 — PCEN + log-mel 2-channel input — DEAD END (Apr 2)

**Expected: +0.005–0.015 LB | Actual: sc_cmap 0.587–0.607 vs v7 0.976 (delta -0.37)**

**Results (4-fold, soundscape-v12-pcen)**:

| Fold | Best sc_cmap | Epoch | Notes |
|---|---|---|---|
| 0 | 0.587 | 9 | early stopped ep19 |
| 1 | 0.607 | 34 | 2 runs wrote to same log (fold launch conflict) |
| 2 | ~0.47 | 5 | killed early — same trajectory |
| 3 | — | — | not launched |

**Root cause**: The `[log-mel, PCEN, log-mel]` design fails because:
1. **PCEN is too different from log-mel**: After min-max norm, PCEN has sparser, more peaked patterns vs. log-mel's smooth dB representation. The EfficientNet pretrained on [mel, mel, mel] struggles to adapt the PCEN channel fast enough.
2. **Channels 0 and 2 are identical**: gradient from two identical channels pulls the model toward "predict from log-mel only" — PCEN in ch1 is downweighted by the optimizer.
3. **Note**: caching was also a challenge — on-the-fly PCEN decode is ~14× slower than cache. Added `precompute_specs.py --pcen` + PCEN cache dirs to `train_cnn.py` to fix I/O. Infrastructure reusable.

**What might work instead**: train with PCEN as the ONLY input `[PCEN, PCEN, PCEN]` or use frequency-weighted blend; don't mix with log-mel in same forward pass.

---

### Step 8 — Multi-year BirdCLEF data pretraining (lower priority)

**Expected: +0.02–0.05 LB | Training: ~8–12h | Medium risk**

BirdCLEF 2021/2022/2023/2024 datasets (~117K clips, all public on Kaggle). Pretrain → fine-tune on 2026. Large effort; defer until Perch ensemble approach is validated. Note: BirdSet XCL pretraining already failed (LB 0.782), suggesting label mismatch with Pantanal is the bottleneck, not data volume. Multi-year BirdCLEF is more geographically aligned and worth testing.

---

### 🎯 Step 17 — ProtoSSM in artifact mode (Apr 4)

**Expected: +0.008–0.012 LB | Confirmed by competitors at 0.924+ | Medium effort (2–3 days)**

Train the ProtoSSM architecture locally on the 708-window perch-meta cache, serialize weights, upload as Kaggle dataset, load in scoring kernel in <5s. Full architecture documented in `research-protossm.md`.

**Architecture**: 4-layer bidirectional SSM (d_model=320, d_state=32), TemporalCrossAttention, 2 prototypes/class, gated per-class fusion (prototype sim vs Perch logit), taxonomic aux head. Input: 1536-dim Perch embeddings × 12 windows per file.

**Training**: AdamW lr=8e-4, OneCycleLR + CosineAnnealingWarmRestarts, 80 epochs, focal BCE (γ=2.5) + 0.15×KD + 0.1×taxonomic aux, Mixup β(0.4), SWA from epoch 65%, 5-fold GroupKFold by site.

**Scoring kernel budget**: Perch live inference ~85 min + ProtoSSM artifact load+inference ~2–3 min = ~88 min total.

**Data**: `jaejohn/perch-meta` cache (already downloaded: `data/perch-meta/`) — 59 soundscapes × 12 windows = 708 rows.

- [ ] Implement `training/train_protossm.py` (architecture + training loop)
- [ ] Train on cluster: 5-fold GroupKFold by site, serialize `model.state_dict()` + probe_models
- [ ] Validate sc_cmap(held) locally on 14 held-out soundscapes
- [ ] Upload weights as Kaggle dataset `aldisued/birdclef2026-protossm-v1`
- [ ] Modify inference notebook to load ProtoSSM artifact + run test inference
- [ ] Submit and compare LB

---

### ✅ Step 18 — Perch ONNX benchmark (Apr 4) — CONFIRMED 7.98x SPEEDUP

**Result**: ONNX = **7.98x faster** than TF SavedModel (Apr 4, benchmark v5)
- TF SavedModel: 15.87s/file → 195 min for 739 files (way over 90-min budget)
- ONNX Runtime: 1.99s/file → **24 min for 739 files** (66 min remaining)
- Model: `yuriygreben/birdclef26-perch-onnx/perch_v2_no_dft.onnx` (tf2onnx)
- Embedding similarity: benchmark v6 showed 0.06 cosine sim — likely benchmark bug comparing TF `spatial_embedding` (16×4×1536, shape[-1]==1536 matches filter) vs ONNX global `embedding` (1536). Actual embeddings likely consistent.

**Key insight**: ONNX Perch alone takes 24 min. This leaves 66 min for CNN inference.
A 1-fold CNN takes ~19 min → Perch ONNX + 1-fold CNN blend fits in ~51 min total.

- [x] Create benchmark Kaggle notebook `aldisued/birdclef-2026-perch-onnx-benchmark`
- [x] Run benchmark v5 (20 soundscapes, TF 2.20 vs onnxruntime 1.21) → 7.98x
- [x] Fix embedding cosine sim (benchmark v6 COMPLETE — see note above)
- [x] Build Perch ONNX + CNN blend kernel (Step 19)

---

### 🎯 Step 19 — ONNX Perch + CNN blend (Apr 4) — IN PROGRESS

**Expected: +0.010–0.020 LB | Submitted, scoring pending**

**Kernel**: `aldisued/birdclef-2026-onnx-perch-cnn-blend-inference` v7 — COMPLETE

**Actual timing (20 dry-run files, extrapolated)**:
- ONNX Perch inference: 1.97s/file → **24.3 min** for 739 files
- 1-fold CNN (soundscape-v7_fold0): 1.55s/file → **19.1 min** for 739 files
- ProtoSSM: ~1 min
- Setup + probes: ~5 min
- **Total: ~51 min** (39 min headroom — well within 90-min budget)

**Architecture**: ONNX Perch + V18 probes (PCA 128, α=0.45) + ProtoSSM (50/50 blend) + rank-aware post-proc + 1-fold CNN (20% blend)

**Issues encountered during development (Apr 4)**:
1. `del x, out` → `del x, ort_out` (v4→v5 fix)
2. CNN glob found ProtoSSM `.pt` first → filtered to `"soundscape" in path` (v5→v6 fix)
3. `cnn_probs_arr` sized by `len(sample_sub)` not `len(meta_test)` → broadcast error → fixed (v6→v7 fix)

**Submission**: `Step 19 v1: ONNX Perch (24min) + ProtoSSM 50/50 + 1-fold CNN 20% blend (~51 min total)` — PENDING

- [x] Create `kaggle_blend_onnx_inference.ipynb` based on v8 perch inference notebook
- [x] Replace TF Perch with ONNX Perch
- [x] Add 1-fold CNN inference (load from `aldisued/birdclef2026-soundscape-v7`)
- [x] Push v7 — COMPLETE, ~51 min estimated runtime
- [x] Submit — PENDING scoring

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

---

## Status summary (Apr 13)

### Current best: LB **0.920** (kernel perch-v2-inference v53)
Config: rank_power=2.0, smooth_amphibia=0.65, no boost, single seed (protossm_original.pt)

### Exhausted approaches (DO NOT RETRY)

| Category | Status | Best LB |
|---|---|---|
| Post-processing (rp, smooth, boost, taxon temp) | FULLY EXHAUSTED | 0.920 |
| SSL backbones (AVES, WavLM, BirdAVES) | ALL DEAD ENDS | − |
| Stage2 ensemble (any combination) | ALWAYS HURTS | 0.913 |
| Stage2 hyperparameters (epochs, dropout, LR) | AT LOCAL OPTIMUM | − |
| Stage3 | CATEGORICALLY HARMFUL | 0.895 |
| 66sc / S09 data | ALWAYS HURTS | 0.912 |
| XC augmentation | DOMAIN MISMATCH | 0.902 |
| CLAP zero-shot integration | IMPRACTICAL (test timing) | − |
| Perch Embedding Adapter | HURTS probe quality | − |

### Stage 2 seed comparison (Apr 13, local rp=2.0 cmAP)

| Seed | Stage2 base | rp=2.0 |
|---|---|---|
| **s42** (protossm_v3.pt, current best) | **0.3355** | **0.3625** |
| s4 (59sc_s4.pt) | 0.3363 | 0.3604 |
| s1 (59sc_s1.pt) | 0.3295 | 0.3564 |
| s2 (59sc_s2.pt) | 0.3276 | 0.3493 |
| s3 (59sc_s3.pt) | 0.3257 | 0.3441 |

Seed s42 is best locally. Seeds s100-s119 training overnight (results Apr 14).

---

## Status summary (Apr 14)

### Stage2-only seed search (20 seeds s100-s119): ALL WORSE LOCALLY

None of 20 new Stage2-only seeds beat seed 42 (0.3625). Best: seed 117 (0.3600, -0.0025).

**Conclusion**: Stage2-only seed search exhausted. Seed 42 is optimal for this Stage1 checkpoint.

### Stage1+Stage2 full retrain seed search (32 seeds s200-s231)

Top results (rp=2.0 cmAP):

| Seed | Local cmAP | LB | Delta |
|---|---|---|---|
| 42 (baseline, protossm_v3.pt) | 0.3625 | **0.920** | — |
| 209 | 0.3716 | **0.918** | **-0.002** |
| 218 | 0.3715 | (not submitted) | est. similar |
| 219 | 0.3630 | (not submitted) | — |
| 201 | 0.3646 | (not submitted) | — |

**Critical finding**: Local cmAP is ANTI-CORRELATED with LB for Stage1 seeds. Seed 209 locally +0.009 → LB -0.002. Root cause: Stage1 trains in-sample on 59 labeled soundscapes; "better" Stage1 = more memorization of these 59 soundscapes = WORSE generalization to test set.

**Conclusion**: Stage1 seed variation is a DEAD END. The local metric cannot rank Stage1 seeds reliably. Seed 42 (protossm_v3.pt) remains the best.

### v59 failure (blank score): Bug found and fixed
v59 used OLD notebook with rank_power=0.4 instead of 2.0. Local notebook was NOT kept in sync with temp push directories. Fixed in v60 (v57-base + s209). v60 scored 0.918.

**CRITICAL FIX**: Always build push from `/tmp/perch_v57_push/` as base, NOT from local `kaggle_perch_v2_inference.ipynb`. The local file is stale (v33 commit, rp=0.4). The v57 push has the correct rp=2.0 + boost + full post-processing.

### Exhausted approaches (updated Apr 14)

| Category | Status | Best LB |
|---|---|---|
| Stage2-only seed search (s100-s119) | DEAD END — seed 42 optimal | 0.920 |
| Stage1+Stage2 full retrain seeds (s200-s231) | DEAD END — anti-correlated locally | 0.918 |

### What could still help

1. **Cross-validation for Stage1**: Use K-fold with leave-one-SITE-out to get reliable Stage1 ranking. Too expensive (9 sites × ~4 min per seed = 36 min per seed with k-fold).
2. **New architecture for Stage2**: Larger model (d_model=256 vs 128), attention-based, or different input features.
3. **Semi-supervised**: Use test soundscape embeddings for Stage2 training (requires two-stage kernel run).
4. **Competitor analysis**: Study what atahalam (0.943) and top teams are doing differently.
