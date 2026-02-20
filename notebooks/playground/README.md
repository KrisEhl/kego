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
- `analyze_disagreement.py` — Model disagreement matrix and best-run analysis using OOF predictions from MLflow
- `benchmark_models.py` — Standalone neural model benchmarking (no Ray): timing, profiling, MLflow logging
- `select_features.py` — Fine-grained feature selection: per-feature ablation + forward selection with multi-seed averaging
- `test_features_local.py` — Local CPU feature engineering comparison (LightGBM + LogReg)
- `submit_s6e2.sh` — Submit predictions via Kaggle CLI
- `explore_s6e2.py` — EDA and data exploration

## CLI Reference — `train_s6e2_baseline.py`

### Training Modes

| Flag | Folds | Seeds | Models | Purpose |
|------|-------|-------|--------|---------|
| *(none)* | 10 | 3 | all 19 | Full training run |
| `--fast` | 5 | 1 | core GBDTs | Quick iteration (~3-5 min) |
| `--fast-full` | 10 | 3 | core GBDTs | GBDT-only with full CV (~15-20 min) |
| `--neural` | 5 | 1 | resnet, ft_transformer, realmlp | Neural models only |
| `--debug` | 2 | 1 | all | 2000 rows, sanity check |

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--tag TAG` | str | `""` | Custom MLflow experiment name suffix (e.g. `--tag gbdt-v2`) |
| `--features F [F ...]` | choice | `ablation-pruned` | Feature set(s): `all`, `raw`, `ablation-pruned`, `forward-selected` |
| `--folds N [N ...]` | int | mode-dependent | CV fold counts (e.g. `--folds 5 10`) |
| `--models M [M ...]` | str | all | Only train these models (e.g. `--models catboost realmlp`) |
| `--seed-pool S [S ...]` | int | `42 123 777` | Seed pool for reproducibility |
| `--seeds-per-learner N` | int | all seeds | Rotate N seeds per learner from the pool |
| `--resume EXPERIMENT` | str | — | Skip tasks completed in a previous MLflow experiment |
| `--retrain-full` | flag | — | Retrain on train+holdout combined (no holdout eval) |
| `--description TEXT` | str | `""` | Free-text description logged to MLflow |

### Tuning Mode

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--tune M [M ...]` | str | — | Run Optuna HP tuning for these models |
| `--tune-trials N` | int | `50` | Optuna trials per model |
| `--tune-sample N` | int | all rows | Subsample training data for faster tuning |

### Ensemble-from-MLflow Mode (no training)

| Argument | Type | Description |
|----------|------|-------------|
| `--from-experiment E [E ...]` | str | Load predictions from MLflow experiment(s) |
| `--from-ensemble NAME` | str | Load predictions from a curated ensemble (tagged runs) |
| `--submit` | flag | Submit to Kaggle and log leaderboard score to MLflow |

### Examples

```bash
# Full training on Ray cluster
cd cluster && make submit-full TAG=full-v2

# Fast iteration with specific models and features
cd cluster && make submit-fast TAG=test DESCRIPTION="testing catboost tuned params"

# Tune CatBoost and LightGBM (50 trials each)
cd cluster && make submit-tune TUNE_MODELS="catboost lightgbm" TUNE_TRIALS=50

# Re-ensemble from previous experiments (no training)
cd cluster && make submit-ensemble EXPERIMENTS="full-v1 diverse-v1"

# Submit best ensemble to Kaggle
cd cluster && make submit-kaggle ENSEMBLE=submit-v9

# Local usage (without Ray cluster)
uv run python notebooks/playground/train_s6e2_baseline.py --fast --tag local-test
```

## Cluster Commands (`cluster/Makefile`)

All commands run from the `cluster/` directory.

### Training Jobs

| Target | Description | Key Variables |
|--------|-------------|---------------|
| `make submit-fast` | 5 folds, 1 seed, core GBDTs (~3-5 min) | `TAG=`, `RESUME=`, `DESCRIPTION=` |
| `make submit-fast-full` | 10 folds, 3 seeds, core GBDTs (~15-20 min) | `TAG=`, `FEATURES=`, `RESUME=`, `DESCRIPTION=` |
| `make submit-full` | All 19 models, 10 folds, 3 seeds | `TAG=`, `FEATURES=`, `RESUME=`, `DESCRIPTION=` |
| `make submit-neural` | Neural models only, 5 folds, 1 seed | `TAG=`, `RESUME=`, `DESCRIPTION=` |
| `make submit-debug` | Debug: 2K rows, fast mode | `TAG=` |
| `make submit-tune` | Optuna HP tuning | `TUNE_MODELS=`, `TUNE_TRIALS=`, `TUNE_SAMPLE=`, `FEATURES=`, `FOLDS=`, `TAG=` |
| `make submit-diverse` | Diverse models/features/seeds | `DIVERSE_MODELS=`, `DIVERSE_FEATURES=`, `DIVERSE_FOLDS=`, `DIVERSE_SEED_POOL=`, `DIVERSE_SEEDS_PER=`, `TAG=`, `RESUME=`, `DESCRIPTION=` |

### Ensemble & Submission

| Target | Description | Key Variables |
|--------|-------------|---------------|
| `make submit-ensemble` | Re-ensemble from MLflow runs (no training) | `ENSEMBLE=` or `EXPERIMENTS=` |
| `make submit-kaggle` | Generate submission + submit to Kaggle | `ENSEMBLE=` or `EXPERIMENTS=` |
| `make log-score` | Log leaderboard score to latest ensemble run | `SCORE=` (required) |

### Ensemble Curation

| Target | Description | Key Variables |
|--------|-------------|---------------|
| `make promote` | Tag specific runs into a named ensemble | `ENSEMBLE=`, `RUN_ID=` (both required) |
| `make auto-promote` | Auto-select best runs per model from experiments | `ENSEMBLE=`, `EXPERIMENT=` or `ALL=1`, `FOLDS=`, `MODELS=`, `FEATURES=` |
| `make list-ensemble` | List all runs in an ensemble | `ENSEMBLE=` |
| `make clear-ensemble` | Remove all runs from an ensemble | `ENSEMBLE=` |
| `make search-runs` | Search MLflow runs matching filters | `EXPERIMENT=` or `ALL=1`, `FOLDS=`, `MODELS=`, `SEEDS=`, `FEATURES=` |

### Cluster Management

| Target | Description |
|--------|-------------|
| `make start-head` | Start Ray head node (run on head machine) |
| `make start-worker` | Connect this node as a Ray worker |
| `make start-worker-light` | Connect as worker with `light_gpu` resource only |
| `make restart-worker` | Stop and reconnect worker |
| `make stop` | Stop Ray on this node |
| `make status` | List all Ray jobs and their status |
| `make logs` | Show parsed progress of the running job |
| `make logs-raw` | Show last N raw log lines (`N=20` default) |

### MLflow

| Target | Description |
|--------|-------------|
| `make mlflow-start` | Start MLflow tracking server |
| `make mlflow-stop` | Stop MLflow server |

## Resuming Failed Runs

When a training run has partial failures (GPU OOM, joblib crashes, worker disconnects), `--resume` avoids re-running the entire experiment. It queries MLflow for completed tasks and only re-trains what's missing.

```bash
# Resume a failed full run — skip completed tasks, retrain only failed/missing
cd cluster && make submit-full RESUME=playground-s6e2-full TAG=full

# Resume with a description for tracking
cd cluster && make submit-full RESUME=playground-s6e2-full TAG=full DESCRIPTION="retry after OOM fix"

# Local (non-cluster) usage
uv run python notebooks/playground/train_s6e2_baseline.py --resume playground-s6e2-full --tag full
```

**How it works:**

1. Each training task (model + seed + folds + features + hyperparams) is hashed into a 12-char **config fingerprint** stored in MLflow
2. `--resume EXPERIMENT` queries the named MLflow experiment for completed runs and collects their fingerprints
3. Before submitting each Ray task, the driver computes the fingerprint and skips it if already completed
4. Predictions from completed runs are loaded from MLflow artifacts and merged with newly trained results
5. The final ensemble combines both preloaded and freshly trained learners

**Config change detection:** If you modify ANY parameter (hyperparams, features, folds), the fingerprint changes automatically — stale results are never reused. Only exact config matches are skipped.

**Partial learners:** If a learner has some seeds completed but not all (e.g. 2 of 3 seeds finished), none of its seeds are preloaded — all seeds are retrained. This avoids complexity of un-averaging partial results.

## Current Best

- **Leaderboard (public)**: 0.95372
- **Holdout AUC (ensemble)**: 0.9563
- **Method**: Ridge stacking over 8 greedy-selected models x 3 seeds (submit-v9)

## Local Validation vs Leaderboard

| Run | Ensemble Method | Holdout AUC | LB (public) | Gap |
|-----|----------------|-------------|-------------|-----|
| 1 | Ridge stacking, 3 models | 0.9561 | 0.95354 | -0.0026 |
| 2 | Ridge stacking, 8 models | 0.9562 | 0.95359 | -0.0026 |
| 5 | Ridge stacking, 8 models x 3 seeds | 0.9562 | 0.95360 | -0.0026 |
| submit-v1 | Ridge stacking, all models, 10 folds | — | 0.95341 | — |
| submit-v9 | Ridge stacking, 8 greedy-selected models x 3 seeds | 0.9563 | 0.95372 | -0.0026 |
| submit-v2* | Ridge stacking, 65 learners (19 models + TabPFN), 223 runs | 0.9562 | 0.95372 | -0.0026 |

*\*New submit-v2 = rebuilt ensemble from `full` + `diverse-v1` experiments (223 runs, 19 model types including TabPFN, 3 feature sets, 5/10 folds). Same LB as submit-v9 despite 8x more learners.*

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
| tabpfn | 0.9537 | GPU neural |
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
| submit-v9 | Greedy-selected 8 of 28 learners, Ridge stacking, 3 seeds | 0.9563 | 0.95372 | +0.00012 | Best 8 from multi-strategy greedy forward selection. XGB, FT-Transformer, LGB, CatBoost, LogReg across raw/ablation-pruned/forward-selected features |
| submit-v2* | Ridge stacking, 65 learners from 223 runs (19 models + TabPFN) | 0.9562 | 0.95372 | +0.00000 | Massive ensemble from `full` + `diverse-v1`. TabPFN adds negligible value. Same LB as lean submit-v9 |

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

After training all models (multi-seed averaged), the final submission is produced by the best of eight ensemble methods, selected automatically by holdout AUC:

1. **Simple Average** — equal-weight mean of all model predictions
2. **Ridge Stacking** — `RidgeCV` (alphas 0.01–100) trained on OOF predictions, learns per-model weights
3. **Hill Climbing** — greedy weight optimization on OOF AUC (step size 0.01, 10 iterations)
4. **Rank Blending** — converts each model's predictions to percentile ranks, then averages. Robust to different calibration scales across models.
5. **L2 LightGBM (preds_only)** — 5-fold StratifiedKFold LightGBM meta-model on L1 OOF predictions
6. **L2 LightGBM (raw)** — L2 meta-model on L1 predictions + 13 raw features
7. **L2 LightGBM (ablation-pruned)** — L2 meta-model on L1 predictions + 21 ablation-pruned features
8. **L2 LightGBM (forward-selected)** — L2 meta-model on L1 predictions + 16 forward-selected features

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

Script: `analyze_ensemble.py` — multi-strategy greedy forward selection and leave-one-out analysis on the `submit-candidates` ensemble (28 learners: 5 model types x 3 feature sets x 2 fold counts, 3 seeds each, 84 runs total).

At each greedy step, the script evaluates 4 blending strategies per candidate model: simple mean, rank blending, Ridge stacking (fit on OOF, evaluate on holdout), and hill climbing weight optimization. It picks the (candidate, strategy) pair with the highest holdout AUC.

### Greedy Forward Selection (multi-strategy)

| Step | Model Added | Strategy | Ensemble AUC | Delta | Spearman r |
|------|-------------|----------|-------------|-------|------------|
| 1 | xgboost/raw/10f | mean | 0.95624 | — | — |
| 2 | ft_transformer/forward-selected/10f | ridge | 0.95626 | +0.00002 | 0.995 |
| 3 | lightgbm/ablation-pruned/10f | ridge | 0.95627 | +0.00001 | 0.998 |
| 4 | xgboost/raw/5f | ridge | 0.95627 | +0.00000 | 0.999 |
| 5 | catboost/raw/5f | ridge | 0.95627 | +0.00000 | 0.999 |
| 6 | logistic_regression/forward-selected/10f | ridge | 0.95627 | +0.00000 | 0.989 |
| 7 | ft_transformer/ablation-pruned/10f | ridge | 0.95627 | +0.00000 | 0.997 |
| 8 | ft_transformer/raw/5f | ridge | 0.95627 | +0.00000 | 0.998 |
| — | *20 models rejected* | ridge | — | -0.00000 to -0.00001 | 0.992–0.999 |

8 of 28 models selected. Ridge stacking is the winning strategy from step 2 onward.

### Leave-One-Out (full 28-model ensemble, ridge strategy, AUC 0.95626)

| Model | AUC without | Delta | Spearman r | Verdict |
|-------|------------|-------|------------|---------|
| xgboost/raw/10f | 0.95625 | +0.00002 | 0.998 | neutral |
| logistic_regression/raw/10f | 0.95625 | +0.00001 | 0.995 | neutral |
| logistic_regression/raw/5f | 0.95625 | +0.00001 | 0.995 | neutral |
| xgboost/raw/5f | 0.95625 | +0.00001 | 0.998 | neutral |
| ... 24 more ... | 0.95626 | +/-0.00000 | 0.992–0.999 | neutral |

All 28 models are neutral under Ridge stacking — no model is harmful, none is indispensable. Ridge assigns appropriate weights to compensate for weak or redundant models.

### Key Findings

**Multi-strategy vs mean-only:** The previous mean-only greedy selection on 18 models selected just 2 (AUC 0.95606). Multi-strategy with Ridge selects 8 from 28 candidates (AUC 0.95627), a +0.00021 improvement. Ridge stacking can assign low weights to weaker models, so it benefits from including more diverse inputs.

**Model diversity matters for Ridge:** The selected 8 include 3 model types (XGBoost, FT-Transformer, LightGBM, LogisticRegression, CatBoost) across different feature sets (raw, forward-selected, ablation-pruned) and fold counts (5f, 10f). Ridge benefits from this diversity even when individual models are r=0.999 correlated.

**Leave-one-out is flat under Ridge:** With Ridge stacking over 28 models, no single model's removal changes AUC by more than 0.00002. This is because Ridge can redistribute weight. Under simple averaging the same ensemble shows more sensitivity to individual models.

## What Worked

- **Model diversity**: Adding sklearn models (LR, RF, ET) and GBDT variants (XGB-reg, LGB-dart) alongside the original XGB/LGB/CB improved the ensemble
- **Original UCI data**: Combining 270 real samples with 630k synthetic rows helped slightly
- **Ridge stacking**: Consistently outperforms simple averaging and hill climbing
- **Multi-seed averaging**: Small but real improvement from training each model with 3 different StratifiedKFold seeds

## What Didn't Work

- **Stacking meta-models**: Ridge, LightGBM (preds-only), and LightGBM (preds+features) all gain < 0.001 AUC over simple averaging. L2 LightGBM stacking (5-fold CV, 4 feature set variants) ties Ridge at 0.9562 — no non-linear signal to exploit
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
- TabPFN (prior-fitted network) underperforms GBDTs despite being a strong baseline on small datasets — likely because the 630K training set is well beyond TabPFN's sweet spot
- Improvements are in the 5th decimal place — this competition has a very tight leaderboard
- **Ensemble size doesn't matter**: 8 learners (submit-v9) ties 65 learners (submit-v2) at 0.95372 LB. Ridge stacking extracts the same signal from a lean or massive ensemble

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

## Ideas To Try

Researched from Playground Series winner writeups and top solutions. Ranked by expected impact.

| # | Idea | Expected Gain | Effort | Notes |
|---|------|---------------|--------|-------|
| 1 | **More feature engineering** | Small-Medium | Medium | Groupby stats (mean/std of numerics per categorical), frequency/count encoding, log transforms on BP/Cholesterol, binned continuous features. Generate hundreds of candidates, let selection prune. |
| 2 | **Retrain on full data** for final submission | Small | Low | After selecting hyperparameters via CV, retrain on train+holdout for submission. Common in top solutions. |
| 3 | **Adversarial validation** | Diagnostic | Low | Train classifier to distinguish train vs test. May reveal distribution shift issues. |
| 4 | **Revisit pseudo-labeling** | Small | Medium | Previous attempt (136k hard labels) failed. Try: soft labels, higher confidence threshold (0.99), per-fold pseudo-labels, ensemble-generated labels. |
| 5 | **KNN / SVM** for diversity | Small | Low | Different inductive bias from trees/NNs, adds stacking diversity. |
| 6 | **FT-Transformer HP tuning** | Small | In progress | Optuna tuning with 5 trials (patience=5, max_epochs=50). Currently running. |

### Already tried / won't help

- **More seeds (>3)**: Tested — only +0.00001 LB going from 1→3 seeds, diminishing returns
- **Pseudo-labeling (hard labels)**: Tried 136k confident predictions, no improvement
- **StandardScaler for LogReg**: No effect, Ridge already compensates
- **TabPFN**: Added to ensemble (6 learners across 3 feature sets x 2 fold counts, 18 runs). Ridge assigns near-zero/negative weights. Greedy forward selection picks it at step 11 of 65 with <0.0001 AUC gain. LB unchanged at 0.95372. Not helpful for this dataset (630K rows — TabPFN is designed for smaller datasets)
- **Massive ensemble (65 learners, 19 model types)**: submit-v2 with 223 runs from `full` + `diverse-v1` scores identically to submit-v9 (8 learners). More models ≠ better LB when Ridge stacking is already optimal
- **2-level stacking (L2 LightGBM)**: Tested 4 variants — preds-only, +raw features, +ablation-pruned, +forward-selected. All tie Ridge at 0.9562 holdout AUC. The 65-model L1 predictions are already well-captured by linear combination; LightGBM meta-model can't find non-linear interactions to exploit on this dataset
