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

- `train_s6e2_baseline.py` — 8-model ensemble with multi-seed averaging and Ridge stacking
- `submit_s6e2.sh` — Submit predictions via Kaggle CLI
- `explore_s6e2.py` — EDA and data exploration

## Current Best

- **Leaderboard (public)**: 0.95360
- **Holdout AUC**: 0.9562
- **Method**: Ridge stacking over 8 models x 3 seeds

## Experiment Log

| # | Experiment | Holdout AUC | LB Score | Delta | Notes |
|---|---|---|---|---|---|
| 1 | 3-model ensemble (XGB + LGB + CB) | 0.9561 | 0.95354 | — | Baseline with Ridge stacking |
| 2 | Expand to 8 models (+ LR, RF, ET, XGB-reg, LGB-dart) | 0.9562 | 0.95359 | +0.00005 | More model diversity helped |
| 3 | StandardScaler for LogisticRegression | 0.9562 | 0.95359 | 0 | Fixed convergence warnings but Ridge already compensated for scale |
| 4 | Pseudo-labeling (high-confidence test preds as extra training data) | 0.9562 | 0.95359 | 0 | 136k confident samples added, no improvement in R2. Reverted |
| 5 | Multi-seed ensembling (3 seeds x 8 models) | 0.9562 | 0.95360 | +0.00001 | Tiny variance reduction |

## What Worked

- **Model diversity**: Adding sklearn models (LR, RF, ET) and GBDT variants (XGB-reg, LGB-dart) alongside the original XGB/LGB/CB improved the ensemble
- **Original UCI data**: Combining 270 real samples with 630k synthetic rows helped slightly
- **Ridge stacking**: Consistently outperforms simple averaging and hill climbing
- **Multi-seed averaging**: Small but real improvement from training each model with 3 different KFold seeds

## What Didn't Work

- **StandardScaler for LR**: Ridge stacking already handles different prediction scales, so normalizing LR inputs had no effect on the ensemble
- **Pseudo-labeling**: Even with 136k confident test predictions (prob > 0.95 or < 0.05), the second training round didn't improve. Likely because the synthetic data is already large enough and pseudo-labels don't add new signal
- **Neural models on CPU**: RealMLP, FTTransformer, ResNet were too slow to train on CPU (hours per model). Commented out — would need GPU access

## Key Observations

- The top GBDT models (XGB, XGB-reg, LGB, CB) all plateau at ~0.9561 holdout AUC individually
- Weaker models (LR: 0.9514, RF: 0.9516, ET: 0.9524) still contribute to the ensemble through diversity
- Hill climbing tends to put nearly all weight on the strongest models, making it equivalent to just using those models
- Ridge stacking is more robust, finding useful signal even in weaker models
- Improvements are in the 5th decimal place — this competition has a very tight leaderboard
