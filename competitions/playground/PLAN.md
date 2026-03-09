# Plan: Next Steps — Maximize LB Score

## Final Result (Competition Closed 2026-02-28)

**Private LB: 0.95528, rank 377 / 4,371 (top 8.6%)**. Public LB was 0.95388 (rank 492). Jumped +115 spots on private evaluation.

## Status (2026-02-27)

Current best: **0.95388 LB** (submit-v12, 134 learners, Ridge stacking, retrain-full-v2 + orig-stats).
Leaderboard rank ~490 / 3,593 (top 13.6%). Bronze cutoff: ~0.95395 (~+0.00007 needed).

**Latest attempts (2026-02-27)**:
- **catboost-tune-v1** (100 Optuna trials): Completely flat landscape after trial ~31. Best OOF 0.9533. Tuned params (depth=5, lr=0.02385, Bernoulli, subsample=0.778, l2_leaf_reg=20.26): identical to defaults in practice.
- **lgbm-tune-v2 with max_bin** (100 trials, ablation-pruned, 5-fold): Best OOF 0.9537 (trial #98, lr=0.0382, num_leaves=37, max_bin=386). Retrain-full too slow (~12h/task on CPU, no GPU support in pip LightGBM). Abandoned.
- **catboost_tuned retrain-full** (10 seeds, ablation-pruned, 5+10f): OOF 0.9552 = identical to default catboost. Not selected by ensemble.
- **submit-v11** (retrain-full-v2 + catboost_tuned): Ridge 0.9556, LB **0.95378** — slightly below retrain-full-v2 alone (0.95380). catboost_tuned adds zero diversity.

**Key pattern**: Tuned GBDTs consistently fail to add ensemble diversity. The same 8 models keep winning (xgboost/raw, catboost/raw, ft_transformer/raw, lightgbm/ablation-pruned). HP optimization within a model family produces correlated predictions that Ridge already handles optimally. The gap to bronze requires either better features or completely different model architectures.

**Previous attempt (tuned-retrain-v1 combined)**: No improvement. Added 60 tuned GBDTs (lgbm_tuned + xgboost_tuned + catboost × 5 seeds on `all`+`ablation-pruned`) to retrain-full-v2. Holdout AUC 0.9557, LB 0.9538 — identical to retrain-full-v2 alone. Hill climbing collapsed to uniform weights.

### Leaderboard Context (as of 2026-02-25)

| Position | Score | Gap |
|---|---|---|
| #1 (Pirhosseinlou) | 0.95414 | +0.00034 |
| #2 (Tshithihi) | 0.95410 | +0.00030 |
| #3 (Chris Deotte) | 0.95410 | +0.00030 |
| #4–20 (~17 teams) | 0.95408 | +0.00028 |
| Bronze (~top 10%, ~rank 360) | ~0.95395 | ~+0.00007 |
| **Us (~rank ~460)** | **0.95388** | — |

**Leaderboard has moved since last check:** the cluster that was at 0.95406 has now risen to 0.95408, and new entries at 0.95410 have appeared. The bronze cutoff has shifted up — what was +0.00008 away is now estimated at +0.00015. The gap is wider than previously thought.

The cluster at 0.95408 comes from the public "S6E2 Heart Disease Top1 Multi Seed" notebook. Analysis shows **the gap is primarily in HP quality and GPU model tuning**, not architecture — they score 0.95408 with just 3 GBDTs + 5 seeds, while we have 104 learners.

### Key Features We're Missing (from public notebook)

| Feature | Formula | Category |
|---|---|---|
| `rate_pressure_product` | Max HR × BP / 1000 | Cardiology |
| `cardiac_reserve` | Max HR / (220 - Age), clipped [0.5, 1.2] | Cardiology |
| `st_ratio` | ST depression / (Max HR + 1) | Exercise physiology |
| `metabolic_syndrome` | hypertension + high_chol + FBS | Composite |
| `hypertension` | BP > 140 (binary) | Binary threshold |
| `high_chol` | Cholesterol > 200 (binary) | Binary threshold |
| `very_high_chol` | Cholesterol > 240 (binary) | Binary threshold |
| `risk_age` | Age > 55 (binary) | Binary threshold |
| `severe_vessels` | Vessels >= 2 (binary) | Binary threshold |
| `age_x_vessels` | Age × Vessels | Interaction |
| `rpp_x_st` | RPP × ST depression | Interaction |
| `chol_x_bp` | Cholesterol × BP / 10000 | Interaction |
| Frequency encoding | Value counts for all 8 categoricals | Encoding |
| Target encoding | OOF TE for all 8 categoricals (we only do 4) | Encoding |

## Plan (local Mac, no cluster)

### Step 1: Add missing features to training script ✅

Added 13 features from the public notebook to `_engineer_features()`:
- Binary thresholds: hypertension, high_chol, very_high_chol, risk_age, severe_vessels
- Cardiology: rate_pressure_product, cardiac_reserve (clipped), st_ratio
- Composites: metabolic_syndrome, score_proxy
- Interactions: age_x_vessels, rpp_x_st, chol_x_bp

Also: expanded TE_FEATURES from 4 → all 8 categoricals, added frequency encoding to `make_te_preprocess()`.

Total features: 53 (was 40).

### Step 2: Local validation of new features ✅

Local 5-fold CV (LightGBM, single model):
- Ablation-pruned baseline: 0.95526
- Ablation-pruned + notebook features: 0.95515 (-0.00011)
- Ablation-pruned + TE(8) + freq(8): 0.95525 (neutral)
- All features + TE(8) + freq(8): 0.95509 (-0.00016)

**Result: neutral for single LightGBM** — expected since GBDTs learn binary splits natively. The benefit should show in non-GBDT models (LogReg, neural) and ensemble diversity. Real test is full ensemble on LB.

### Step 3: Investigate CDC BRFSS dataset (400K real rows) ✅ — Not usable

The BRFSS dataset (445K rows, 40 cols) is **self-reported phone survey data** — smoking, sleep, general health, etc. Almost no overlap with our clinical features (ECG, stress test, cholesterol labs, fluoroscopy). Only `Sex` matches directly. 9/13 features have no equivalent. Dead end.

### Step 4: Re-ensemble existing predictions with different meta-learners ✅ — No improvement

Loaded 104 learners from retrain-full-v2, tested meta-learners on OOF predictions:
- **Ridge (original, alpha=10): 0.95568** — already optimal
- Ridge (broader alphas): 0.95567 (same, alpha converges to ~10-17)
- LogReg (C=0.001): 0.95536 (-0.00032)
- Rank averaging: 0.95517 (-0.00051)
- Ridge on ranks: 0.95491 (-0.00077)

**Result: Ridge stacking is already the best meta-learner. No gain from switching.**

### Step 5: Add KNN model locally ✅ — No improvement

Trained 6 subsampled KNN variants (k=5, 10, 20, 50, 100, 200) with 5-fold CV, 50K training rows per fold:
- Individual OOF AUC: 0.922 (k=5) to 0.947 (k=200)
- Ensemble impact: **zero** — adding any KNN variant to 104 learners gives exactly 0.95568 (±0.000001)
- KNN weights near zero in Ridge ensemble

**Result: KNN adds no ensemble diversity. Not useful.**

### Step 6: Retrain-full CPU models with new features ✅ — No improvement

Retrained 5 CPU models (lightgbm × 4 variants + logistic_regression) on feature set `all` (53 features incl. 13 new) and `ablation-pruned`, 5+10 folds, 3 seeds. 60 tasks, ~1h 20m on Mac.

Individual OOF AUCs:
- lightgbm/all: 0.9552 vs lightgbm/ablation-pruned: 0.9553 (neutral for LightGBM)
- logistic_regression/all: 0.9531 vs logistic_regression/ablation-pruned: 0.9528 (+0.0003 for LogReg)
- cpu-retrain-v1 alone: Ridge OOF AUC 0.9554

Combined with retrain-full-v2 (124 learners total): **Ridge OOF AUC 0.95568 — zero improvement**.
New `/all/` learners have small weights that cancel out; highly correlated with existing GPU model predictions.

**Result: CPU-only new features don't move the needle. Need GPU models (XGBoost, CatBoost) with new features on cluster.**

### Step 8: Local Optuna HP tuning for CPU models ✅

Ran 100-trial Optuna studies locally (50K sample, 5-fold, `all` features) for LightGBM, LogReg, and XGBoost (CPU mode). Results:

| Model | Best OOF AUC | Notes |
|---|---|---|
| lightgbm | 0.9532 | Tuned params: num_leaves=16, max_depth=12, lr=0.0206, subsample=0.563, colsample=0.466, reg_alpha=0.328, path_smooth=73.7 |
| xgboost (CPU) | 0.9532 | Tuned params: max_depth=3, lr=0.0129, min_child_weight=8, subsample=0.759, colsample=0.449, reg_lambda=0.031 |
| logistic_regression | 0.9526 | Flat landscape — no improvement over default (C=1.0, lbfgs) |

Added `lightgbm_tuned` and `xgboost_tuned` model variants to `get_models()`. The XGBoost tuned params use `device=cuda` for cluster use. LogReg config unchanged (converged).

### Step 9: Local multi-seed validation ✅ — LightGBM only, cluster needed for full test

Trained `lightgbm_tuned` + `lightgbm_large` + `logistic_regression` × 5 seeds locally, `all` features, 5-fold CV (standard mode):

| Model | Holdout AUC (avg 5 seeds) |
|---|---|
| lightgbm_tuned | **0.9558** |
| lightgbm_large | 0.9557 |
| logistic_regression | 0.9538 |
| Ridge ensemble (all 15 learners) | 0.9558 |

`lightgbm_tuned` is the best LightGBM variant. Ridge gives `lightgbm_tuned` 78% weight, `lightgbm_large` 27%, LogReg −5% (hurts). Multi-seed LightGBM alone ≈ existing 104-learner ensemble quality, but **single-family submission would score ~0.953 LB** (worse than 0.95380). The bottleneck is XGBoost/CatBoost diversity.

### Step 9b: 3-family CPU multi-seed test ✅

Trained `lightgbm_tuned + catboost_cpu + xgboost_tuned_cpu` × 5 seeds locally, `all` features, 5-fold CV:

| Model | Holdout AUC (avg 5 seeds) |
|---|---|
| lightgbm_tuned | 0.9558 |
| catboost_cpu | **0.9558** |
| xgboost_tuned_cpu (CPU) | 0.9553 |
| Ridge ensemble | **0.9559** |

Ridge weights: catboost_cpu=0.761, lightgbm_tuned=0.578, xgboost_tuned_cpu=**−0.339** (negative!).

Key findings:
- 3-family diversity gives +0.0001 over single-family (0.9559 vs 0.9558)
- XGBoost gets negative Ridge weight — the CPU tuned variant underperforms the GPU version. On cluster, use `xgboost_tuned` (device=cuda) instead.
- Estimated LB: ~0.9538 (gap of 0.0018-0.0020 from holdout) — potentially just above bronze cutoff (0.95388)

### Step 9c: 2-family CPU multi-seed test ✅

Trained `lightgbm_tuned + catboost_cpu` × 5 seeds, `all` features, 5+10 folds:

| Learner | Holdout AUC (avg 5 seeds) |
|---|---|
| lightgbm_tuned/all/5f | 0.9558 |
| lightgbm_tuned/all/10f | 0.9558 |
| catboost_cpu/all/5f | 0.9558 |
| catboost_cpu/all/10f | **0.9559** |
| Ridge ensemble (20 learners) | **0.9559** |

Ridge weights: lgbm/10f=0.877, cb/10f=0.644, cb/5f=0.033, lgbm/5f=**−0.553** (5f redundant given 10f).

Key findings:
- Adding more seeds and fold counts doesn't move the needle beyond 0.9559 for 2-family CPU
- The CPU-only ceiling is 0.9559 holdout → estimated LB ~0.9533 (well below current best 0.95380)
- XGBoost GPU is required to push past this ceiling — no path around it on local Mac (CUDA-only, no Metal/MPS support)
- **Conclusion: cluster is required for the next LB improvement**

---

## Next Steps (requires cluster)

### Step 10: Retrain-full tuned 3-family ensemble ✅ SUCCEEDED (raysubmit_wJc2TuKaxbZJgGqz)

Submitted directly to retrain-full (skipping holdout eval to save time). Training `lightgbm_tuned + xgboost_tuned + catboost` on 100% of data:

```bash
cd cluster && make submit-diverse \
  TAG=tuned-retrain-v1 \
  DIVERSE_MODELS="lightgbm_tuned xgboost_tuned catboost" \
  DIVERSE_FEATURES="all ablation-pruned" \
  DIVERSE_FOLDS="5 10" \
  DIVERSE_SEEDS_PER=5 \
  RETRAIN_FULL=1 \
  DESCRIPTION="3-family tuned retrain-full, 5 seeds, 5+10 folds"
```

60 learners (3 models × 2 feature sets × 2 fold counts × 5 seeds). No holdout eval — OOF AUC only.

### Step 11: Combine with retrain-full-v2 and submit ✅ COMPLETED — no improvement

Combined 114 learners (retrain-full-v2 + tuned-retrain-v1). Best: Ridge, holdout AUC **0.9557**, LB **0.9538** = 0.95380.

Results:
| Method | Holdout AUC |
|---|---|
| average | 0.9552 |
| **ridge (α=10)** | **0.9557** |
| hill_climbing | 0.9552 (uniform weights — couldn't improve on average) |
| rank_blending | 0.9552 |
| l2_preds_only | 0.9555 |

Key finding: Hill climbing assigned equal weights (1/114) to all learners — strong signal the tuned models add no useful diversity. Ridge dominant contributors remain the same retrain-full-v2 models (catboost_shallow/raw, xgboost/raw, xgboost/forward-selected). Tuned model weights: small positive or negative, net zero effect.

Note: tuned-retrain-v1 used retrain-full-direct (OOF=630K spans train+holdout), so the combine.py size mismatch fix was needed and applied.

### Step 18: orig-stats feature set (target statistics from UCI original dataset) ✅ COMPLETED

Identified from the public RealMLP notebook (LB 0.95397 solo): compute per-value target statistics from the 270-row UCI original dataset (mean/median/std/skew/count per feature × value), appended to ablation-pruned features. 86 features total vs 21.

**Holdout validation results so far** (10f, 3 seeds, `playground-s6e2-orig-stats-v1`):

| Model | ablation-pruned | orig-stats | Δ |
|---|---|---|---|
| logistic_regression | 0.9535 | **0.9543** | +0.0008 |
| xgboost | 0.9561 | **0.9562** | +0.0001 |
| catboost | 0.9560 | **0.9561** | +0.0001 |
| lightgbm | running | running | — |
| tabpfn | OOM (fixed) | OOM (fixed) | — |

LogReg gains most (+0.0008): it can't learn interactions itself, so pre-aggregated stats help directly. Tree models get consistent marginal gains since they can learn these patterns from raw data anyway.

**Jobs submitted** (2026-02-27):
- `raysubmit_vVWVgSkE5bdJ421d` — retrain-full on orig-stats, 16 models × 5+10f × 3 seeds (MLflow: `playground-s6e2-retrain-full-orig-stats-v1`)
- `raysubmit_CfuUdy2kzpsVTJYg` — full validation (holdout eval) on orig-stats, same 16 models × resume from orig-stats-v1 (MLflow: `playground-s6e2-orig-stats-full-v1`)

**OOM fix**: Added `memory=4*1024**3` Ray scheduling hint for `lightgbm_dart` and `lightgbm_large` to limit concurrent tasks on the 15.2GB worker node.

**Result (submit-v12)**: Combined `retrain-full-v2` (104 learners) + `retrain-full-orig-stats-v1` (30 learners) = 134 learners total. Best: Ridge, holdout AUC 0.9557, **LB 0.95388** (+0.00008 vs 0.95380). Several orig-stats learners got large positive Ridge weights: `catboost/orig-stats/10f` (+0.134), `catboost_shallow/orig-stats/10f` (+0.134), `xgboost/orig-stats/10f` (+0.173), `lightgbm_large/orig-stats/10f` (+0.113). Still 0.00007 below bronze cutoff (~0.95395).

---

## Remaining Options (gap: ~0.00007 to bronze)

These are the options not yet confirmed as dead ends, roughly ordered by expected impact.

### Option A: LOO encoding for Thallium + Slope of ST → new `orig-stats-loo` feature set

From `research_features.py` greedy selection: `Thallium_loo` (+0.00008, rank 2) and `Slope of ST_loo` (+0.00002, rank 6) survived drop-one validation. **Not yet in any retrain-full pipeline** — only tested locally with LightGBM.

These are LOO-encoded features that provide signal orthogonal to the orig-stats target statistics. Combining both into a single feature set `orig-stats-loo` (ablation-pruned + UCI target statistics + LOO encoding for Thallium/Slope_of_ST) and running retrain-full would produce a third diverse feature set for ensemble. **Low effort** — just add two LOO columns to `_engineer_features()` under a new feature set name.

### Option B: Research features retrain-full (cluster)

`playground-s6e2-research-v1` (13 runs, job stopped early) was never completed. The research features showed +0.00053 local AUC with LightGBM (5-fold, `ablation-pruned` baseline → `research`). This was never submitted to LB due to job interruption. Adding retrain-full predictions on `research` features (XGBoost, CatBoost, LightGBM) would provide a fourth diverse feature set.

**Caveat**: Research features didn't produce LB gain in the one partial test we did. But we never had a clean complete retrain-full run on research features to actually combine.

### Option C: Adversarial validation ✅ DONE — no shift

LightGBM 5-fold CV on train-vs-test labels: **AUC = 0.501 ± 0.002**. Classifier cannot distinguish train from test. CTGAN generation was perfectly consistent. No reweighting opportunity.

### Option D: L2 meta-model with `std(OOF)` confidence meta-feature ✅ DONE — no improvement

Implemented `l2_confidence` in `kego/ensemble/combine.py` — appends `std(oof_matrix, axis=1)` as an extra column for the L2 LightGBM meta-model. Tested on `playground-s6e2-full` + `playground-s6e2-diverse-v1` (65 holdout-evaluated learners):

| Method | Holdout AUC |
|--------|-------------|
| ridge | 0.9562 |
| l2_preds_only | 0.9562 |
| **l2_confidence** | **0.9562** |
| l2_raw/ablation-pruned/forward-selected | 0.9562 |

All L2 variants tie Ridge at 0.9562. Confidence feature adds nothing — the std signal is fully captured by the linear combination already. Dead end.

### Option E: GPU models on `all` feature set (cluster)

XGBoost and CatBoost on `all` (53 features: ablation-pruned + notebook clinical features + TE8 + freq8) were **never tested in GPU retrain-full mode**. CPU models (Step 6) showed zero improvement, but CatBoost/XGBoost are the dominant ensemble contributors — their representation of the `all` features might differ. Low probability of improvement given Step 6 result; cluster job cost is moderate (~45 min).

---

## Already Tried / Won't Help

(Formerly "Ideas To Try" — confirmed dead ends from experiments + research)

- **More meta-learner experiments**: Ridge at alpha=10 is confirmed optimal. LogReg, rank averaging, Ridge on ranks all worse. L2 LightGBM (5 variants: preds_only, +raw, +ablation-pruned, +forward-selected, +std(OOF) confidence) all tie Ridge at 0.9562. No non-linear signal to exploit. No more to try.
- **Polynomial features**: Trees discover interactions natively. Ablation confirmed interaction features are harmful for GBDTs. Only useful for LogReg/neural.
- **Repeated k-fold beyond 10**: Research shows diminishing returns. 10f is the practical ceiling.
- **Snapshot ensembling during GBDT training**: Only adds value when not already using multi-seed. With 5 seeds × 2 fold counts, this adds no new diversity.
- **Calibration (isotonic/Platt)**: AUC is rank-invariant to monotonic transforms. Already applied opportunistically in pipeline.
- **More seeds (>5)**: From 1→3 seeds: +0.00001 LB. Diminishing returns well before 5.
- **TabPFN**: Near-zero ensemble weight at 630K rows. Designed for small datasets.
- **SVM, KNN**: Near-zero or zero ensemble weight. Too slow and/or too weak individually.
- **Tuned GBDTs (lgbm_tuned + xgboost_tuned + catboost) on `all`+`ablation-pruned` features, retrain-full**: 60 learners, 5 seeds, 5+10 folds. Combined with retrain-full-v2 (114 total): LB 0.9538, no improvement. Hill climbing went uniform. The `all` feature set and tuned HPs don't add diversity beyond the existing ensemble.
- **catboost_tuned Optuna (catboost-tune-v1, 100 trials)**: Flat landscape, best OOF 0.9533. Tuned params (depth=5, lr=0.02385, Bernoulli bootstrap, high regularization) give OOF 0.9552 in retrain-full — identical to default catboost. submit-v11 (retrain-full-v2 + catboost_tuned): LB 0.95378 < 0.95380. Confirmed dead end.
- **lgbm_tuned retrain-full**: Too slow (~12h/task on CPU). LightGBM pip wheel lacks CUDA support (requires `-DUSE_CUDA=1` recompile). Not practical without GPU-compiled LightGBM.
- **Pseudo-labeling (hard + soft)**: Both definitively failed. Soft labels collapse model to 0.929 round 1 → 0.70 round 2.
- **CatBoost Optuna HP tuning (catboost-tune-v1, 100 trials)**: Flat landscape after trial ~31, best OOF 0.9533. Tuned params give retrain-full OOF 0.9552 = identical to default catboost. submit-v11 (retrain-full-v2 + catboost_tuned): LB 0.95378 < 0.95380. HP tuning within CatBoost adds zero diversity.
- **HP config diversity (top Optuna trials as additional learners)**: LightGBM and XGBoost tuned variants already combined with retrain-full-v2 (Step 11) — zero improvement. CatBoost Optuna confirmed flat. No reason to extract additional Optuna trials.
- **GPU models on `all` feature set**: CPU models (LightGBM × 4, LogReg) on `all` features showed zero improvement when combined with retrain-full-v2 (Step 6). GPU models unlikely to differ.

---

## Previous Results

| Submission | AUC | Public LB | Models | Method |
|---|---|---|---|---|
| submit-v9 | 0.9558 holdout | 0.95372 | 8 greedy-selected x 3 seeds | Ridge |
| submit-v10 | 0.9562 holdout | 0.95372 | 93 learners from 4 experiments | Ridge |
| **retrain-full-v2** | **0.9557 OOF** | **0.95380** | **104 learners, full data** | **Ridge** |
| submit-v11 | 0.9556 OOF | 0.95378 | 114 learners (retrain-full-v2 + catboost_tuned) | Ridge |
| **submit-v12** | **0.9557 OOF** | **0.95388** | **134 learners (retrain-full-v2 + retrain-full-orig-stats-v1)** | **Ridge** |

### Key Findings (from previous steps)

- Retrain-full (train on 100% data) gives +0.00008 LB
- SVM: near-zero ensemble weight, not useful
- Research features (6 clinical): +0.00053 local AUC but no LB gain
- Neural models on research features: resnet broken, ft_transformer degraded
- More models != better LB for holdout-evaluated ensembles
- 104 learners slightly edges 8 curated for retrain-full (0.95380 vs 0.95378)
