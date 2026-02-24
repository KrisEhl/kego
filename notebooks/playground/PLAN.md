# Plan: Next Steps — Maximize LB Score

## Status (2026-02-23)

Current best: **0.95380 LB** (retrain-full-v2, 104 learners, Ridge stacking).
Leaderboard rank ~490 / 3,593 (top 13.6%). Bronze cutoff ~0.95388 (+0.00008 needed).

### Leaderboard Context

| Position | Score | Gap |
|---|---|---|
| #1 (Pirhosseinlou) | 0.95414 | +0.00034 |
| #2-4 (Deotte et al.) | 0.95407 | +0.00027 |
| #5-75 (71 teams, public notebook cluster) | 0.95406 | +0.00026 |
| Bronze (~top 10%) | ~0.95388 | +0.00008 |
| **Us** | **0.95380** | — |

The 71-team cluster at 0.95406 comes from the public "S6E2 Heart Disease Top1 Multi Seed" notebook. Analysis shows **the gap is primarily in features**, not models — they score 0.95406 with just 3 GBDTs + 5 seeds, while we have 104 learners.

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

### Step 7 (when cluster available): Retrain-full GPU models with new features

Retrain XGBoost and CatBoost variants on feature set `all` (53 features) with retrain-full mode. These are the dominant ensemble contributors (top weights are all xgboost/catboost). Their improved feature representations should actually shift the ensemble.

### Step 8: Local Optuna HP tuning for CPU models ✅

Ran 100-trial Optuna studies locally (50K sample, 5-fold, `all` features) for LightGBM, LogReg, and XGBoost (CPU mode). Results:

| Model | Best OOF AUC | Notes |
|---|---|---|
| lightgbm | 0.9532 | Tuned params: num_leaves=16, max_depth=12, lr=0.0206, subsample=0.563, colsample=0.466, reg_alpha=0.328, path_smooth=73.7 |
| xgboost (CPU) | 0.9532 | Tuned params: max_depth=3, lr=0.0129, min_child_weight=8, subsample=0.759, colsample=0.449, reg_lambda=0.031 |
| logistic_regression | 0.9526 | Flat landscape — no improvement over default (C=1.0, lbfgs) |

Added `lightgbm_tuned` and `xgboost_tuned` model variants to `get_models()`. The XGBoost tuned params use `device=cuda` for cluster use. LogReg config unchanged (converged).

### Step 9 (when cluster available): More seeds + CatBoost Optuna HP tuning

Increase seed pool to 5-10. Run Optuna tuning (100+ trials) for CatBoost on the cluster with GPU.

---

## Previous Results

| Submission | AUC | Public LB | Models | Method |
|---|---|---|---|---|
| submit-v9 | 0.9558 holdout | 0.95372 | 8 greedy-selected x 3 seeds | Ridge |
| submit-v10 | 0.9562 holdout | 0.95372 | 93 learners from 4 experiments | Ridge |
| **retrain-full-v2** | **0.9557 OOF** | **0.95380** | **104 learners, full data** | **Ridge** |
| submit-v11 | 0.9556 OOF | 0.95378 | 8 curated, full data | Ridge |

### Key Findings (from previous steps)

- Retrain-full (train on 100% data) gives +0.00008 LB
- SVM: near-zero ensemble weight, not useful
- Research features (6 clinical): +0.00053 local AUC but no LB gain
- Neural models on research features: resnet broken, ft_transformer degraded
- More models != better LB for holdout-evaluated ensembles
- 104 learners slightly edges 8 curated for retrain-full (0.95380 vs 0.95378)
