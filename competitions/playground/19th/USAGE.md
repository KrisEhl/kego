# 19th Benchmark - Usage Guide

## Overview

Benchmarks models trained by [19th](https://19th.com) against classical ML (LightGBM, logistic
regression, Optuna-tuned XGBoost) by training size, on two Kaggle Playground datasets:

| Dataset | Competition | Task | Rows | Sizes benchmarked |
|---|---|---|---:|---|
| `heart` | Playground S6E2, heart disease | binary, 44% positive | 630,000 | 100 - 20,000 |
| `fraud` | Playground S6E11, card fraud | binary, 1.7% positive | 699,960 | 1,000 - 599,960 (all four 19th modes up to 200,000) |

`heart` also compares Rimay quantum features (see `../quantum/`). Findings are in
[RESULTS.md](RESULTS.md); open `dashboard_heart.html` or `dashboard_fraud.html` in a browser for
the charts.

19th runs are driven through its MCP server from Claude Code. The scripts here do everything
that touches files. Every command takes `--dataset heart` or `--dataset fraud`.

| Script | What it does |
|--------|--------------|
| `benchmark_services.py` | Dataset configs; builds samples and holdout, runs classical baselines, scores predictions, bootstraps 19th vs classical |
| `nineteenth_workflow.py` | Prepares upload CSVs and the `prep_run` declaration, uploads, verifies a model bundle, predicts the holdout or the competition test set |
| `build_dashboard.py` | Builds `dashboard_<dataset>.html` (self-contained, opens offline) from `dashboard_template.html` |

## Setup

```bash
cd competitions/playground/19th
uv sync
```

Needs the competition data under `data/playground/<competition>/` (or `KEGO_PATH_DATA`), e.g.
`uv run kaggle competitions download playground-series-s6e11 -p data/playground` from the repo
root after accepting the rules on Kaggle. `heart` also needs the Rimay run in
`../quantum/quantum_results/` (its 700 training and 300 test rows anchor the samples).

Connect 19th once, at user scope:

```bash
claude mcp add --transport http --scope user 19th https://app.19th.com/mcp
```

Then run `/mcp` in Claude Code, select `19th` and sign in. No API key is needed.

## Workflow

### 1. Build samples and classical baselines

```bash
uv run python benchmark_services.py --dataset fraud prepare    # nested samples + holdout
uv run python benchmark_services.py --dataset fraud baseline   # LightGBM + logistic regression
uv run python benchmark_services.py --dataset fraud baseline --sizes 200000   # only new sizes, keep the rest
uv run python benchmark_services.py --dataset fraud baseline --models xgboost_tuned   # only one model
uv run python benchmark_services.py --dataset heart score-rimay   # heart only
```

`fraud` baselines take about 15 minutes (LightGBM on 480k rows, 3 seeds); `heart` about 1 minute.
Tuned XGBoost adds up to 15 minutes of Optuna search per size (60 trials or the time limit,
whichever comes first). The search uses `kego.pipeline.tune.run_study` with the `HPSpace` list
`XGBOOST_SEARCH_SPACE`; each trial trains on 80% of the sample and is scored on the other 20%,
never on the holdout. Best parameters are cached in `results/xgboost_tuned_parameters.json`;
delete a size's entry to retune it.

### 2. Prepare the 19th uploads

```bash
uv run python nineteenth_workflow.py --dataset fraud prepare
```

Writes `benchmark/<dataset>/samples/nineteenth/train_<size>.csv` (no `id` column) and
`manifest.json` with each file's byte count, SHA-256, idempotency key and the `prediction`
declaration (binary task, `event_probability`, positive label `1`, the raw feature columns).

### 3. Train on 19th (MCP, from Claude Code)

For each size:

1. `prep_run` `action: "create"` with the manifest's `file` fields, `prediction` and `idempotency_key`.
2. Upload to the returned signed URL:
   ```bash
   uv run python nineteenth_workflow.py --dataset fraud upload --size 5000 --url '<upload.url from prep_run>'
   ```
3. `prep_run` `action: "finalize"` with the `prep_id`.
4. `submit_run` with the `prep_id`, a run name like `Card fraud 5000 rows high`, a
   `training_mode` (`instant`, `standard`, `high` or `ultra`) and an idempotency key like
   `fraud-benchmark-run-5000-high-v1`. One preparation can be trained in every mode.
5. `get_results` with the `run_id` until `state` is `succeeded`. Standard runs take 6-10
   seconds up to 20,000 rows, about 1 minute at 100,000 and about 5 minutes at 600,000. High
   takes about 2x standard; ultra about 10x at small sizes and over 30 minutes at 200,000.

### 4. Download, verify and predict

Download a model's bundle from the `get_results` link (Fast, Turbo or Warpspeed), then:

```bash
uv run python nineteenth_workflow.py --dataset fraud predict --size 5000 --mode high --variant fast \
  --bundle model_bundle.zip --expected-sha256 <models[fast].download.sha256 from get_results>
```

`--mode` defaults to `standard` and `--variant` to `fast`.

For a Kaggle submission from a model, `predict-test` takes the same arguments and writes
`benchmark/<dataset>/submissions/submission_19th_<size>_<mode>_<variant>.csv` from the competition's
`test.csv`. Submitting it to Kaggle is a separate, manual step.

The script refuses a bundle whose checksum differs from what `get_results` reported, or whose
prediction code differs from the reviewed version (`REVIEWED_BUNDLE_CODE_SHA256`). When 19th
ships a new runtime, read the bundle's `predict.py` and `runtime/` before adding its hash.

### 5. Score

```bash
uv run python benchmark_services.py --dataset fraud score --mode high --size 5000
uv run python benchmark_services.py --dataset fraud compare --mode high   # paired bootstrap, every size
uv run python benchmark_services.py --dataset fraud summary
uv run python build_dashboard.py --dataset fraud
```

A Kaggle submission from tuned XGBoost refit on every labeled row (holdout included) comes from
`uv run python benchmark_services.py --dataset fraud xgboost-submission`, which writes
`benchmark/fraud/submissions/submission_xgboost_tuned_all.csv`.

Scores are stored as service `19th_<mode>` (Fast) or `19th_<mode>_<variant>`. `compare` caches the
seed-averaged classical predictions under `benchmark/<dataset>/cache/`, so only the first run
per size refits LightGBM and logistic regression.

## Output Files

```
benchmark/<dataset>/
├── samples/                     # gitignored, rebuilt by `prepare`
│   ├── train_<size>.csv         # nested training samples (id, features, target 0/1)
│   ├── holdout_features.csv     # holdout rows, no labels
│   ├── holdout_labels.csv       # labels (+ in_rimay_holdout flag for heart)
│   └── nineteenth/              # upload CSVs + manifest.json
├── models/                      # gitignored, downloaded 19th bundles (train_<size>_<mode>_<variant>)
├── cache/                       # gitignored, seed-averaged classical holdout predictions
├── submissions/                 # gitignored, Kaggle submission files
├── predictions/
│   └── pred_<size>_<mode>_<variant>.csv  # 19th holdout probabilities (tracked for heart only)
└── results/
    ├── classical_scores.csv     # LightGBM + logistic regression by size
    ├── service_scores.csv       # 19th by mode/variant (and Rimay for heart) by size
    ├── comparison_vs_classical.csv  # bootstrap lead of each 19th mode over best classical
    └── leaderboard.csv          # Kaggle public scores of submitted models (fraud; shown on the dashboard)
dashboard_<dataset>.html         # results dashboard, built by build_dashboard.py
```

## Design Choices

- **Nested samples**: each smaller sample is contained in the larger ones. For `heart`,
  `train_700.csv` is exactly Rimay's training set; for `fraud`, positives and negatives are
  taken in fixed proportion so every sample keeps the 1.67% fraud rate.
- **Holdout**: `heart` uses Rimay's 300 test rows plus 700 more; `fraud` a stratified
  100,000 rows (1,670 frauds), so differences of about 0.003 AUC are resolvable.
- **No test-set early stopping**: LightGBM early-stops on 20% of the training rows. The quantum
  project's workflow early-stops on the test set, which inflates its scores.
- **Raw features only**: every method gets the same columns, with booleans as 0/1 and no
  feature engineering.
