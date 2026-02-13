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

The holdout AUC consistently overestimates the leaderboard score by ~0.0026. This gap is stable across runs, so holdout improvements should translate 1:1 to LB improvements.

### Individual Model Holdout AUC (Run 6, in progress)

| Model | Holdout AUC | Type |
|-------|-------------|------|
| xgboost_reg | 0.9559 | GPU GBDT |
| xgboost_shallow | 0.9559 | GPU GBDT |
| lightgbm_small | 0.9559 | CPU GBDT |
| xgboost | 0.9558 | GPU GBDT |
| xgboost_deep | 0.9554 | GPU GBDT |
| logistic_regression | 0.9514 | CPU linear |
| extra_trees | 0.9509 | CPU tree |
| random_forest | 0.9504 | CPU tree |

*24/57 tasks done. CatBoost, remaining LightGBM, neural models pending.*

## Experiment Log

| # | Experiment | Holdout AUC | LB Score | LB Delta | Notes |
|---|---|---|---|---|---|
| 1 | 3-model ensemble (XGB + LGB + CB) | 0.9561 | 0.95354 | — | Baseline with Ridge stacking |
| 2 | Expand to 8 models (+ LR, RF, ET, XGB-reg, LGB-dart) | 0.9562 | 0.95359 | +0.00005 | More model diversity helped |
| 3 | StandardScaler for LogisticRegression | 0.9562 | 0.95359 | +0.00000 | Fixed convergence warnings but Ridge already compensated for scale |
| 4 | Pseudo-labeling (high-confidence test preds as extra training data) | 0.9562 | 0.95359 | +0.00000 | 136k confident samples added, no improvement. Reverted |
| 5 | Multi-seed ensembling (3 seeds x 8 models) | 0.9562 | 0.95360 | +0.00001 | Tiny variance reduction |
| 6 | 19 models on Ray cluster (+ neural + GBDT variants) | TBD | TBD | TBD | Running: `raysubmit_4B4eqYJpfAyje4Xz` |
| 7 | Categorical handling + new Thallium FE + risk scores | — | — | — | Queued |

### Local Feature Validation (5-fold CV on full train, CPU, single LightGBM/LogReg)

| Setup | LightGBM CV AUC | LogReg CV AUC |
|-------|-----------------|---------------|
| Raw features (no engineering) | **0.95532** | 0.95047 |
| New FE (Thallium interactions + risk scores) | 0.95522 | **0.95113** (+0.00066) |
| Old FE (age/bp/chol interactions) | 0.95514 | 0.95048 |

Trees discover interactions natively — FE doesn't help them. New features improve LogReg (+0.00066) which helps ensemble diversity.

## What Worked

- **Model diversity**: Adding sklearn models (LR, RF, ET) and GBDT variants (XGB-reg, LGB-dart) alongside the original XGB/LGB/CB improved the ensemble
- **Original UCI data**: Combining 270 real samples with 630k synthetic rows helped slightly
- **Ridge stacking**: Consistently outperforms simple averaging and hill climbing
- **Multi-seed averaging**: Small but real improvement from training each model with 3 different KFold seeds

## What Didn't Work

- **StandardScaler for LR**: Ridge stacking already handles different prediction scales, so normalizing LR inputs had no effect on the ensemble
- **Pseudo-labeling**: Even with 136k confident test predictions (prob > 0.95 or < 0.05), the second training round didn't improve. Likely because the synthetic data is already large enough and pseudo-labels don't add new signal
- **Neural models on CPU**: RealMLP, FTTransformer, ResNet were too slow to train on CPU (hours per model). Now running on GPU cluster.
- **Old feature engineering for trees**: age/bp/chol interactions are redundant — LightGBM scores 0.95532 raw vs 0.95514 with old FE. Trees discover these natively.
- **Cholesterol imputation**: Synthetic data has 0 rows with Cholesterol=0. Only the 270 original rows had missing values — negligible impact.
- **BP, FBS over 120, Cholesterol**: Near-zero predictive value (BP r=0.005, FBS MI=0.002). FBS has negative permutation importance.

## Key Observations

- The top GBDT models (XGB, XGB-reg, LGB, CB) all plateau at ~0.9559 holdout AUC individually
- Weaker models (LR: 0.9514, RF: 0.9504, ET: 0.9509) still contribute to the ensemble through diversity
- Hill climbing tends to put nearly all weight on the strongest models, making it equivalent to just using those models
- Ridge stacking is more robust, finding useful signal even in weaker models
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
