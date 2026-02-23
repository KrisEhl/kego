# Plan: Next Steps — Maximize LB Score

## Status (2026-02-23)

Steps 1-7 completed. **Retrain-full achieved new best: 0.95380 LB** (+0.00008 over submit-v9).

### Results Summary

| Submission | AUC | Public LB | Models | Method |
|---|---|---|---|---|
| submit-v9 (previous best) | 0.9558 holdout | 0.95372 | 8 greedy-selected x 3 seeds | Ridge |
| submit-v10 (93 learners) | 0.9562 holdout | 0.95372 | 93 learners from 4 experiments | Ridge |
| **retrain-full-v2 (104 learners)** | **0.9557 OOF** | **0.95380** | **104 learners trained on full data** | **Ridge** |
| submit-v11 (8 curated) | 0.9556 OOF | 0.95378 | 8 curated learners trained on full data | Ridge |

### Key Findings

- **Retrain-full works**: Training on 100% of labeled data (train+holdout) gives +0.00008 LB improvement. This is the only change that has moved the LB needle since submit-v9.
- **More models still helps slightly for retrain-full**: 104 learners (0.95380) edges out 8 curated (0.95378).
- **SVM**: near-zero ensemble weight across all methods. Not useful for this dataset.
- **Research features**: small positive Ridge weights for `catboost/research/10f` (0.125) and `xgboost_dart/research/10f` (0.081), but didn't move the LB needle.
- **Neural models on research features**: resnet completely broken (AUC 0.72-0.86), ft_transformer degraded (0.91), only realmlp performed well (0.9547).
- **Ridge top weights (retrain-full-v2)**: `xgboost/raw/10f` (0.42), `xgboost/forward-selected/10f` (0.26), `catboost_shallow/raw/10f` (0.26), `catboost_shallow/raw/5f` (0.22).
- More models != better LB for holdout-evaluated ensembles. But retrain-full benefits slightly from more learners.

### Experiments Available

| Experiment | Runs | Models | Feature Sets | Folds |
|---|---|---|---|---|
| `playground-s6e2-full` | 146 | 20 models | all, raw, ablation-pruned, forward-selected | 5, 10 |
| `playground-s6e2-diverse-v1` | 84 | 5 core models | raw, ablation-pruned, forward-selected | 5, 10 |
| `playground-s6e2-research-v1` | ~60 | 22 models (most done) | research | 10 |
| `playground-s6e2-svm-v1` | 24 | SVM | raw, ablation-pruned, forward-selected, research | 5, 10 |
| `retrain-full-v2` | 360 | 19 models | raw, ablation-pruned, forward-selected | 10 |

### Fixes Applied During Execution

- Fixed `submit-neural` Makefile target (was missing `FEATURES` passthrough)
- Had to `git pull` on worker node (was missing `SubsampledSVC` class)
- Ran jobs sequentially to avoid OOM on worker (15GB RAM)

## Completed Steps

### 1. Pull latest code on head node — DONE
### 2. Debug smoke test with SVM — DONE (passed)
### 3. Train SVM across all feature sets — DONE (24/24 runs)

Best SVM result: `svm/research/10f` at AUC 0.9368. Research features slightly better than other feature sets for SVM, 10 folds > 5 folds.

### 4. Complete research features run — DONE (46/54, remaining 8 neural tasks skipped)

All GBDTs completed on research features. Neural models: realmlp done (0.9547), resnet broken (0.72-0.86), ft_transformer degraded (0.91). Remaining ft_transformer_ple and resnet_ple tasks were skipped.

### 5. Analyze ensemble across ALL experiments — DONE

93 learners loaded. Ridge stacking: 0.9562 holdout AUC.

### 6. Submit to Kaggle — DONE

Public LB: **0.95372** (no improvement over submit-v9).

### 7. Submit retrain-full-v2 — DONE (new best!)

Fixed code to allow `--retrain-full` with `--from-experiment` and `--from-ensemble` (needed to set correct data dimensions: holdout=test, 270K rows, OOF AUC for method selection).

- **retrain-full-v2 (all 104 learners)**: Ridge stacking, OOF AUC 0.9557, **Public LB: 0.95380** (+0.00008)
- **submit-v11 (8 curated learners)**: xgboost/raw/10f, xgboost/ablation-pruned/10f, catboost/raw/10f, lightgbm/ablation-pruned/10f, ft_transformer/raw/10f, xgboost/raw/5f, catboost/raw/5f, ft_transformer/raw/5f. Ridge OOF AUC 0.9556, **Public LB: 0.95378**

### Fixes Applied

- Fixed `submit-neural` Makefile target (was missing `FEATURES` passthrough)
- Had to `git pull` on worker node (was missing `SubsampledSVC` class)
- Ran jobs sequentially to avoid OOM on worker (15GB RAM)
- Allowed `--retrain-full` with `--from-experiment`/`--from-ensemble` for retrain-full ensembles

## Possible Next Steps

- **Blend submissions**: rank-average retrain-full-v2 (0.95380) with submit-v9 (0.95372) — different training data and weights might complement each other
- **Train new models on full data**: retrain-full-v2 only has raw/ablation-pruned/forward-selected. Could add research features or new model variants
- **Hyperparameter tuning**: Optuna tuning for top models (xgboost, catboost), then retrain-full with tuned params
- **Post-processing**: isotonic calibration, probability clipping
- **Feature engineering**: generate more candidates beyond the 6 research features
