# Playground Series S6E2 — Predicting Heart Disease

Binary classification: predict Heart Disease (Presence/Absence) from 13 clinical features.

## Competition

- **Kaggle**: https://www.kaggle.com/competitions/playground-series-s6e2
- **Metric**: AUC
- **Train**: 630k rows, **Test**: 270k rows

## Original Data

The competition's synthetic data is generated from the original UCI heart disease dataset:

- **Source**: https://www.kaggle.com/datasets/neurocipher/heartdisease/data?select=Heart_Disease_Prediction.csv
- **Rows**: 270
- **License**: Apache 2.0

The original data is combined with the synthetic training data during training to provide additional real-world signal.

## Scripts

- `train_s6e2_baseline.py` — 19-model ensemble with multi-seed averaging and Ridge stacking (runs on Ray cluster)
- `compare_stacking.py` — Stacking comparison: simple average vs Ridge vs LightGBM meta-models
- `analyze_ensemble.py` — Greedy forward selection and leave-one-out analysis of ensemble members
- `select_features.py` — Fine-grained feature selection: per-feature ablation + forward selection with multi-seed averaging
- `test_features_local.py` — Local CPU feature engineering comparison (LightGBM + LogReg)
- `submit_s6e2.sh` — Submit predictions via Kaggle CLI
- `explore_s6e2.py` — EDA and data exploration

## Current Best

- **Leaderboard (public)**: 0.95360
- **Holdout AUC (ensemble)**: 0.9562
- **Method**: Ridge stacking over 8 models x 3 seeds

## Local Validation vs Leaderboard

| Run | Ensemble Method | Holdout AUC | LB (public) | Gap |
|-----|----------------|-------------|-------------|-----|
| 1 | Ridge stacking, 3 models | 0.9561 | 0.95354 | -0.0026 |
| 2 | Ridge stacking, 8 models | 0.9562 | 0.95359 | -0.0026 |
| 5 | Ridge stacking, 8 models x 3 seeds | 0.9562 | 0.95360 | -0.0026 |
| submit-v1 | Ridge stacking, all models, 10 folds | — | 0.95341 | — |

The holdout AUC consistently overestimates the leaderboard score by ~0.0026. This gap is stable across runs, so holdout improvements should translate 1:1 to LB improvements.

### Individual Model Holdout AUC (ablation-pruned, 21 features, 3 seeds, 10 folds)

| Model | Avg Holdout AUC | Type |
|-------|-----------------|------|
| catboost | 0.9560 | GPU GBDT |
| catboost_shallow | 0.9560 | GPU GBDT |
| lightgbm | 0.9560 | CPU GBDT |
| xgboost_reg | 0.9560 | GPU GBDT |
| xgboost_shallow | 0.9560 | GPU GBDT |
| lightgbm_small | 0.9559 | CPU GBDT |
| xgboost | 0.9559 | GPU GBDT |
| lightgbm_large | 0.9558 | CPU GBDT |
| catboost_deep | 0.9558 | GPU GBDT |
| xgboost_dart | 0.9558 | GPU GBDT |
| xgboost_deep | 0.9555 | GPU GBDT |
| realmlp | 0.9547 | GPU neural |
| ft_transformer | 0.9545 | GPU neural |
| lightgbm_dart | 0.9540 | CPU GBDT |
| resnet | 0.9539 | GPU neural |
| logistic_regression | 0.9529 | CPU linear |
| random_forest | 0.9528 | CPU tree |
| extra_trees | 0.9513 | CPU tree |

## Experiment Log

| # | Experiment | Holdout AUC | LB Score | LB Delta | Notes |
|---|---|---|---|---|---|
| 1 | 3-model ensemble (XGB + LGB + CB) | 0.9561 | 0.95354 | — | Baseline with Ridge stacking |
| 2 | Expand to 8 models (+ LR, RF, ET, XGB-reg, LGB-dart) | 0.9562 | 0.95359 | +0.00005 | More model diversity helped |
| 3 | StandardScaler for LogisticRegression | 0.9562 | 0.95359 | +0.00000 | Fixed convergence warnings but Ridge already compensated for scale |
| 4 | Pseudo-labeling (high-confidence test preds as extra training data) | 0.9562 | 0.95359 | +0.00000 | 136k confident samples added, no improvement. Reverted |
| 5 | Multi-seed ensembling (3 seeds x 8 models) | 0.9562 | 0.95360 | +0.00001 | Tiny variance reduction |
| 6 | 19 models on Ray cluster (+ neural + GBDT variants), 35 features | — | — | — | Completed, individual runs in MLflow |
| submit-v1 | Curated ensemble, 14 models x 3 seeds, 10 folds | — | 0.95341 | -0.00019 | 40 runs; missing ft_transformer/realmlp (not finished), catboost only 1 seed |
| 7 | Feature selection: ablation-pruned (21 features), 18 models x 3 seeds x 10 folds | — | — | — | All models complete. Top GBDTs at 0.9560, neural models 0.9539-0.9547 |

### Local Feature Validation (5-fold CV on full train, CPU, single LightGBM/LogReg)

| Setup | LightGBM CV AUC | LogReg CV AUC |
|-------|-----------------|---------------|
| Raw features (no engineering) | **0.95532** | 0.95047 |
| New FE (Thallium interactions + risk scores) | 0.95522 | **0.95113** (+0.00066) |
| Old FE (age/bp/chol interactions) | 0.95514 | 0.95048 |

Trees discover interactions natively — FE doesn't help them. New features improve LogReg (+0.00066) which helps ensemble diversity.

## Stacking Comparison (15 models, 10 folds, 3 seeds)

Compared simple averaging vs learned meta-models on holdout AUC. Script: `compare_stacking.py`.

| Method | Holdout AUC | Delta vs Average |
|---|---|---|
| Simple Average | 0.95545 | baseline |
| Ridge Regression (alpha=1.0) | 0.95605 | +0.00060 |
| LightGBM (preds only) | 0.95594 | +0.00049 |
| LightGBM (preds + features) | 0.95595 | +0.00050 |

**Verdict**: Gap < 0.001 — stacking is not worth the added complexity over simple averaging. Ridge weights reveal catboost (+0.61) and xgboost_reg (+0.40) dominate; original features add negligible signal beyond what base models capture.

## Ensemble Prediction

After training all models (multi-seed averaged), the final submission is produced by the best of four ensemble methods, selected automatically by holdout AUC:

1. **Simple Average** — equal-weight mean of all model predictions
2. **Ridge Stacking** — `RidgeCV` (alphas 0.01–100) trained on OOF predictions, learns per-model weights
3. **Hill Climbing** — greedy weight optimization on OOF AUC (step size 0.01, up to 100 iterations)
4. **Rank Blending** — converts each model's predictions to percentile ranks, then averages. Robust to different calibration scales across models.

The method with the highest holdout AUC wins. The winning predictions are then optionally post-processed with **isotonic calibration** (fit on OOF, applied to holdout/test) — only used if it improves holdout AUC.

## Encoding Strategy

Each model family uses a different combination of categorical encoding, numerical preprocessing, and numerical embeddings. Target encoding and bin computation happen **per CV fold** to prevent data leakage.

### Categorical Features

8 categorical columns: Sex, Chest pain type, FBS over 120, EKG results, Exercise angina, Slope of ST, Number of vessels fluro, Thallium.

| Model family | Encoding method |
|---|---|
| XGBoost | pandas `category` dtype (`enable_categorical=True`) |
| CatBoost | native `cat_features` parameter |
| LightGBM | native `categorical_feature` in fit kwargs |
| Random Forest / Extra Trees | ordinal integers (no special handling) |
| LogisticRegression | target encoding (`drop_original=True`) — replaces categoricals with per-fold mean target |
| ResNet / ResNet-PLE | target encoding (`drop_original=True`) — can't handle categoricals natively |
| FT-Transformer / FT-Transformer-PLE | label encoding (integer mapping) → learned categorical embeddings, plus target encoding for TE features |
| RealMLP | pandas `category` dtype (native handling), plus target encoding for TE features |

**Target encoding** is applied to 4 high-cardinality categoricals (Thallium, Chest pain type, Slope of ST, EKG results) via `make_te_preprocess`. It creates `{col}_te` columns with the mean target value per category, computed from fold training data only.

### Numerical Preprocessing

| Model family | Preprocessing |
|---|---|
| Tree models (XGB, LGB, CB, RF, ET) | None (trees are invariant to monotonic transforms) |
| LogisticRegression | `StandardScaler` |
| ResNet, FT-Transformer, RealMLP | `QuantileTransformer(output_distribution="normal")` |

### Numerical Embeddings (Neural Models)

Neural models transform each scalar feature into a higher-dimensional embedding before feeding it to the network backbone. Two strategies:

| Embedding | Models | How it works | Key params |
|---|---|---|---|
| **PeriodicEmbeddings** | `resnet`, `ft_transformer` | Learned sinusoidal basis functions. Each feature is mapped to `d_embedding` dimensions via `n_frequencies` learned sine/cosine pairs. | `n_frequencies=48`, `d_embedding=24` (ResNet) or `d_block=96` (FT-Transformer) |
| **PiecewiseLinearEmbeddings (PLE)** | `resnet_ple`, `ft_transformer_ple` | Target-aware binning via decision trees, then piecewise linear interpolation within bins. Bin boundaries are learned per fold from training data + labels. | `n_bins=48`, `version="B"`, `tree_kwargs={"min_samples_leaf": 64}` |
| Internal (RealMLP) | `realmlp` | RealMLP's built-in embedding scheme (not configurable) | — |

**PLE vs Periodic**: PLE uses `compute_bins(X, y, regression=False)` to find bin edges supervised by the target, so bins concentrate where the feature-target relationship changes most. Periodic embeddings are unsupervised — they learn frequency representations during backpropagation. PLE variants use identical hyperparameters to their periodic counterparts for clean A/B comparison.

**Gaussian noise** (`std=0.01`) is added to numerical features during training only (both ResNet and FT-Transformer) as regularization for synthetic data.

## Ensemble Member Analysis

Script: `analyze_ensemble.py` — greedy forward selection and leave-one-out analysis on the 18-model ensemble from `playground-s6e2-full` (ablation-pruned features, 3 seeds, 10 folds).

### Greedy Forward Selection

Starting from empty, adds models one at a time picking the highest holdout AUC gain:

| Step | Model Added | Ensemble AUC | Delta | Spearman r |
|------|-------------|-------------|-------|------------|
| 1 | xgboost_reg | 0.95603 | — | — |
| 2 | catboost_shallow | 0.95606 | +0.00003 | 0.999 |
| — | *16 models rejected* | — | -0.00002 to -0.00065 | 0.983–1.000 |

Only 2 of 18 models survive. The greedy 2-model ensemble (0.95606) beats the full 18-model average (0.95558).

### Leave-One-Out (full 18-model ensemble, AUC 0.95558)

| Model | AUC without | Delta | Spearman r | Verdict |
|-------|------------|-------|------------|---------|
| extra_trees | 0.95566 | -0.00008 | 0.990 | HARMFUL |
| random_forest | 0.95565 | -0.00007 | 0.987 | HARMFUL |
| resnet | 0.95562 | -0.00004 | 0.996 | neutral |
| ... 13 more neutral ... | | | | neutral |
| xgboost_dart | 0.95553 | +0.00004 | 0.997 | helpful-ish |
| xgboost_reg | 0.95553 | +0.00004 | 0.998 | helpful-ish |

### Key Finding

All 18 models are extremely correlated (Spearman r = 0.983–1.000), meaning the ensemble gets almost no diversity benefit. Adding models beyond the top 2 dilutes the best predictions rather than improving them. The extra_trees and random_forest models actively hurt — they are the least accurate and most different, but different in the wrong direction.

**Implication:** To improve the ensemble, new models need to be genuinely diverse (different feature sets, different preprocessing, different problem formulations), not just more hyperparameter variants of the same GBDT recipe.

## What Worked

- **Model diversity**: Adding sklearn models (LR, RF, ET) and GBDT variants (XGB-reg, LGB-dart) alongside the original XGB/LGB/CB improved the ensemble
- **Original UCI data**: Combining 270 real samples with 630k synthetic rows helped slightly
- **Ridge stacking**: Consistently outperforms simple averaging and hill climbing
- **Multi-seed averaging**: Small but real improvement from training each model with 3 different StratifiedKFold seeds

## What Didn't Work

- **Stacking meta-models**: Ridge, LightGBM (preds-only), and LightGBM (preds+features) all gain < 0.001 AUC over simple averaging — not worth the complexity
- **StandardScaler for LR**: Ridge stacking already handles different prediction scales, so normalizing LR inputs had no effect on the ensemble
- **Pseudo-labeling**: Even with 136k confident test predictions (prob > 0.95 or < 0.05), the second training round didn't improve. Likely because the synthetic data is already large enough and pseudo-labels don't add new signal
- **Neural models on CPU**: RealMLP, FTTransformer, ResNet were too slow to train on CPU (hours per model). Now running on GPU cluster.
- **Old feature engineering for trees**: age/bp/chol interactions are redundant — LightGBM scores 0.95532 raw vs 0.95514 with old FE. Trees discover these natively.
- **Cholesterol imputation**: Synthetic data has 0 rows with Cholesterol=0. Only the 270 original rows had missing values — negligible impact.
- **BP, FBS over 120, Cholesterol**: Near-zero predictive value (BP r=0.005, FBS MI=0.002). FBS has negative permutation importance.

## Key Observations

- The top GBDT models (XGB, LGB, CB variants) all plateau at ~0.9560 holdout AUC individually with ablation-pruned features
- Neural models (RealMLP: 0.9547, FT-Transformer: 0.9545, ResNet: 0.9539) are weaker individually but contribute ensemble diversity
- Weaker models (LR: 0.9529, RF: 0.9528, ET: 0.9513) still contribute to the ensemble through diversity
- Hill climbing tends to put nearly all weight on the strongest models, making it equivalent to just using those models
- Ridge stacking is more robust, finding useful signal even in weaker models
- Ablation-pruned (21 features) slightly outperforms all-35 features for LightGBM (0.95122 vs 0.95039 on local validation)
- Improvements are in the 5th decimal place — this competition has a very tight leaderboard

## Feature Importance

1. **Thallium** (r=0.606, MI=0.194) — dominant predictor, biggest gap in old FE
2. **Chest pain type** (r=0.461, MI=0.147)
3. **Exercise angina** (r=0.442)
4. **Max HR** (r=0.441, MI=0.127)
5. **Number of vessels fluro** (r=0.439)
6. **ST depression** / **Slope of ST**
7. **Age**, **Sex** — moderate
8. **BP**, **FBS over 120**, **Cholesterol** — near-zero value

## Feature Selection (Fine-Grained)

Script: `select_features.py` — per-feature ablation + forward selection, multi-seed averaged (seeds 42/123/777, 56K train, 14K holdout).

### Method

1. **Permutation importance** ranks features by how much shuffling them hurts AUC
2. **Drop-one-at-a-time ablation** trains LightGBM on all-but-one feature (3 seeds), measures AUC delta vs baseline — identifies features that actively hurt
3. **Forward selection** adds features one at a time in permutation importance order (3 seeds), finds the optimal count where adding more stops helping

### Ablation Results (multi-seed baseline AUC: 0.95039)

Features sorted by impact of removal (positive delta = removing it **improves** AUC):

| Feature | AUC without | Delta | Verdict |
|---|---|---|---|
| maxhr_per_age | 0.95070 | +0.00031 | **HARMFUL** |
| Max HR_dev_sex | 0.95051 | +0.00012 | **HARMFUL** |
| thallium_x_chestpain | 0.95048 | +0.00008 | **HARMFUL** |
| thallium_abnormal | 0.95045 | +0.00006 | **HARMFUL** |
| thallium_x_stdep | 0.95045 | +0.00005 | **HARMFUL** |
| heart_load | 0.95042 | +0.00003 | **HARMFUL** |
| vessels_x_thallium | 0.95042 | +0.00003 | **HARMFUL** |
| Exercise angina | 0.95041 | +0.00002 | **HARMFUL** |
| age_x_maxhr | 0.95041 | +0.00001 | harmful-ish |
| hr_reserve_pct | 0.95041 | +0.00001 | harmful-ish |
| thallium_x_sex | 0.95040 | +0.00001 | harmful-ish |
| Thallium | 0.95040 | +0.00001 | harmful-ish |
| chestpain_x_angina | 0.95040 | +0.00001 | harmful-ish |
| BP | 0.95039 | +0.00000 | neutral |
| Number of vessels fluro | 0.95038 | -0.00001 | helpful |
| age_x_stdep | 0.95038 | -0.00001 | helpful |
| thallium_x_slope | 0.95038 | -0.00001 | helpful |
| Cholesterol_dev_sex | 0.95037 | -0.00002 | helpful |
| FBS over 120 | 0.95037 | -0.00002 | helpful |
| BP_dev_sex | 0.95037 | -0.00002 | helpful |
| signal_conflict | 0.95037 | -0.00002 | helpful |
| abnormal_count | 0.95036 | -0.00003 | helpful |
| risk_score | 0.95036 | -0.00003 | helpful |
| ST depression_dev_sex | 0.95035 | -0.00004 | helpful |
| angina_x_stdep | 0.95035 | -0.00004 | helpful |
| chestpain_x_slope | 0.95034 | -0.00005 | helpful |
| Cholesterol | 0.95033 | -0.00006 | helpful |
| Sex | 0.95033 | -0.00006 | helpful |
| ST depression | 0.95033 | -0.00006 | helpful |
| Max HR | 0.95033 | -0.00006 | helpful |
| Slope of ST | 0.95033 | -0.00006 | helpful |
| top4_sum | 0.95029 | -0.00010 | helpful |
| Chest pain type | 0.95026 | -0.00013 | helpful |
| Age | 0.95024 | -0.00015 | helpful |
| EKG results | 0.94940 | -0.00099 | **critical** |

### Forward Selection (optimal feature count)

Adding features in permutation importance order, AUC peaks at **16 features** (0.95068):

| N | Added feature | AUC | Delta |
|---|---|---|---|
| 1 | abnormal_count | 0.91278 | — |
| 2 | top4_sum | 0.93152 | +0.01875 |
| 3 | Max HR | 0.94496 | +0.01344 |
| 4 | Chest pain type | 0.94804 | +0.00307 |
| 5 | maxhr_per_age | 0.94905 | +0.00102 |
| 6 | EKG results | 0.95019 | +0.00113 |
| 7-10 | thallium_x_chestpain, Sex, risk_score, ST depression | 0.95042 | +0.00023 |
| 11-16 | vessels_fluro, chestpain_x_slope, Age, BP, Cholesterol, chestpain_x_angina | **0.95068** | +0.00026 |
| 17-35 | remaining 19 features | 0.95041 | -0.00027 |

The first 6 features get 99.5% of the way there. Features 17+ actively degrade AUC.

### Feature Set Comparison (multi-seed, LightGBM + LogReg)

| Feature set | LightGBM AUC | LogReg AUC |
|---|---|---|
| All features (35) | 0.95039 | 0.94725 |
| Raw only (13) | 0.95097 | 0.83882 |
| Forward-selected (16) | 0.95068 | 0.94701 |
| **Ablation-pruned (21)** | **0.95122** | 0.94668 |

### Why Features Help or Hurt

**Most valuable features (removing them hurts significantly):**
- **EKG results** (-0.00099): Strongest individual contribution. Carries unique signal about cardiac electrical activity that no other feature captures.
- **Age** (-0.00015): Independent risk factor, interacts with everything.
- **Chest pain type** (-0.00013): Core diagnostic feature, 4 categories with clear risk gradient.
- **top4_sum** (-0.00010): Composite of Thallium + Chest pain + Vessels + Angina. Gives trees a pre-computed "overall risk" split point that's hard to reconstruct from individual features.

**Harmful features (removing them improves AUC):**
- **maxhr_per_age** (+0.00031, worst offender): Despite ranking #5 in permutation importance, it's redundant — Max HR and Age are both present, and trees can learn their ratio implicitly. The explicit ratio adds noise through correlated splits.
- **Max HR_dev_sex** (+0.00012): Deviation from sex-group mean is just a linear transform of Max HR + Sex, both already present. Adds collinearity without new signal.
- **thallium_x_chestpain** (+0.00008): Both Thallium and Chest pain type are categorical-ish — their product creates a noisy pseudo-continuous feature that trees handle worse than splitting on the originals separately.
- **thallium_abnormal** (+0.00006): Binary threshold of Thallium >= 6. Strictly less informative than the original Thallium ordinal values.
- **Thallium** (+0.00001): Surprising — the raw Thallium column is mildly harmful because `abnormal_count` and `top4_sum` already encode its signal more effectively in composite form. Trees don't need the raw column when the composites are present.
- **Exercise angina** (+0.00002): Similarly absorbed into `abnormal_count` and `top4_sum`.
- **heart_load** / **vessels_x_thallium** / **age_x_maxhr**: Interaction features that trees discover natively. Adding them explicitly just creates correlated split candidates that dilute feature sampling (`colsample_bytree=0.9`).

**Key insight:** Permutation importance and ablation disagree on several features (e.g., `maxhr_per_age` ranks #5 by importance but is the #1 most harmful). This is because permutation importance measures how much *shuffling* a feature hurts predictions on a model that was *trained with it*, while ablation measures whether the model is *better off without it entirely*. A feature can be important to a model that learned to use it, yet harmful compared to a model that never saw it.

### Recommendations

- **For trees (LightGBM/XGBoost/CatBoost):** Use ablation-pruned (21 features) — it scored highest at 0.95122. Alternatively, raw-only (13) at 0.95097 is simpler and nearly as good.
- **For NNs (ResNet/FTTransformer/RealMLP):** Use forward-selected (16 features) or raw-only. NNs benefit more from engineered composites (abnormal_count, top4_sum) since they struggle with discrete interactions. LogReg AUC (proxy for NNs) favors keeping engineered features.
- **For ensemble diversity:** Train tree models on ablation-pruned and NNs on forward-selected to maximize prediction decorrelation.
