# Plan: Next Steps — Maximize LB Score

## Status (2026-02-22)

Steps 1-6 completed. **No LB improvement** — 93-learner Ridge ensemble scored **0.95372 LB** (same as submit-v9 with 8 models). Step 7 (retrain-full) skipped since no improvement.

### Results Summary

| Submission | Holdout AUC | Public LB | Models | Method |
|---|---|---|---|---|
| submit-v9 (previous best) | ~0.9558 | **0.95372** | 8 greedy-selected x 3 seeds | Ridge |
| submit-v10 (93 learners) | 0.9562 | **0.95372** | 93 learners from 4 experiments | Ridge |

### Key Findings

- **SVM**: near-zero ensemble weight across all methods. Not useful for this dataset.
- **Research features**: small positive Ridge weights for `catboost/research/10f` (0.125) and `xgboost_dart/research/10f` (0.081), but didn't move the LB needle.
- **Neural models on research features**: resnet completely broken (AUC 0.72-0.86), ft_transformer degraded (0.91), only realmlp performed well (0.9547).
- **Hill climbing** heavily concentrated: 66% on `xgboost/raw/10f`, 12% `lightgbm/forward-selected/10f`, 11% `lightgbm/raw/10f`.
- **Ridge top weights**: `xgboost/raw/10f` (0.42), `xgboost/ablation-pruned/10f` (0.23), `catboost/raw/10f` (0.22), `lightgbm_large/all/10f` (0.17).
- More models != better LB. The marginal holdout improvement (+0.0004) didn't generalize.

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

### 7. Retrain-full — SKIPPED (no LB improvement)

## Possible Next Steps

- **Try different ensemble strategies**: greedy forward selection with fewer models, different alpha values for Ridge
- **Feature engineering**: look for new features beyond the 6 research features
- **Hyperparameter tuning**: Optuna tuning for top models on best feature sets
- **Post-processing**: calibration, rank averaging with submit-v9
- **Blend with submit-v9**: average the two submission files (same LB but different internal weights might help)
