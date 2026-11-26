# 19th vs Classical ML - Results

Benchmarks run on 2026-11-26. Metric: ROC AUC on a fixed holdout never used for training or
early stopping. Two datasets:

- **Card fraud** (Playground S6E11): harder, imbalanced, up to 600,000 rows. Charts:
  [dashboard_fraud.html](dashboard_fraud.html).
- **Heart disease** (Playground S6E2): easy, also compared with Rimay quantum features. Charts:
  [dashboard_heart.html](dashboard_heart.html).

## Across both datasets

- 19th's advantage is a **small-to-medium data effect**. Against the best classical model,
  including an Optuna-tuned XGBoost, it is clearly ahead from 100 to 700 rows on heart and from
  5,000 to 20,000 rows on fraud.
- **With enough data a tuned XGBoost catches up and passes it.** On fraud, 19th levels off near
  0.943 AUC from 100,000 rows; tuned XGBoost ties it at 100,000-200,000 and is clearly ahead on
  the full training set (0.9471 vs 0.9430). On heart it ties 19th from 2,000 rows.
- **19th's training mode barely matters.** Instant, standard, high and ultra land within about
  0.002 AUC of each other at every size tested; ultra takes up to 10x longer for no clear gain.
- Leads are small in absolute terms: at most +0.007 AUC on heart and +0.010 on fraud.

# Card fraud (Playground S6E11)

Data: `train.csv`, 699,960 rows, 24 raw features (19 numeric/boolean, 5 categorical), 1.67%
fraud. Holdout: a stratified 100,000 rows (1,670 frauds). Training samples: stratified and
nested, 1,000 to 599,960 rows (all remaining rows). 19th was run in standard mode at every size
and in instant, high and ultra up to 200,000 rows.

## Summary

- **19th is clearly ahead at 5,000 rows (+0.0095) and 20,000 rows (+0.0051)**, both well outside
  the bootstrap noise.
- **At 1,000 rows it ties logistic regression** (+0.0010, within noise). With only 17 frauds in
  the sample, LightGBM collapses to 0.790; 19th and logistic regression both reach about 0.895.
- **Tuned XGBoost is the strongest classical model from 20,000 rows.** It cuts 19th's 20,000-row
  lead from +0.0051 (over logistic regression) to +0.0033, ties 19th at 100,000 and 200,000 rows
  (-0.0004 and -0.0007, within noise) and is clearly ahead on the full 599,960 rows (-0.0041).
  Untuned LightGBM alone was clearly ahead only at 599,960 rows (-0.0029).
- **19th hit a capacity limit on the full data.** It warned that it used fewer labeled rows than
  supplied.
- **The capacity limit is not what holds 19th back.** At 200,000 rows 19th trained on every row
  with no warning and scored 0.9432, the same as at 100,000 (0.9429) and 599,960 (0.9430). 19th
  levels off near 0.943; LightGBM and tuned XGBoost are still improving at 600,000 rows.
- **Higher training modes barely help.** High scores 0.0000 to +0.0003 above standard at every
  size. Ultra ranges from -0.0015 (1,000 rows) to +0.0006 (100,000 and 200,000 rows), which turns
  a tie with tuned XGBoost into a slightly better tie (+0.0002 and -0.0001). Instant is up to
  0.0013 lower from 5,000 rows on but highest at 1,000; at 200,000 rows it is the only mode
  clearly behind tuned XGBoost (-0.0013).
- **Smaller model files cost almost nothing here.** Across all 21 runs, Turbo scores within
  0.0002 of Fast and Warpspeed within 0.0004. Warpspeed files are 5-26x smaller and its 1,000-row
  batch predictions 6-27x faster (standard vs ultra Fast models).
- **The leaderboard is higher.** Public leaderboard leaders score about 0.971, presumably with
  feature engineering and ensembles that no method here used. All numbers below use raw features.

## 19th vs best classical model (100,000-row holdout)

| Training rows | 19th | Best classical | Lead | 95% interval | P(19th better) |
|---:|---:|---|---:|---|---:|
| 1,000 | 0.8958 | Logistic regression 0.8948 | +0.0010 | -0.0038 to +0.0060 | 0.66 |
| 5,000 | 0.9358 | Logistic regression 0.9263 | +0.0095 | +0.0070 to +0.0118 | 1.00 |
| 20,000 | 0.9408 | XGBoost (tuned) 0.9375 | +0.0033 | +0.0017 to +0.0048 | 1.00 |
| 100,000 | 0.9429 | XGBoost (tuned) 0.9433 | -0.0004 | -0.0015 to +0.0008 | 0.25 |
| 200,000 | 0.9432 | XGBoost (tuned) 0.9439 | -0.0007 | -0.0018 to +0.0003 | 0.09 |
| 599,960 | 0.9430 | XGBoost (tuned) 0.9471 | -0.0041 | -0.0054 to -0.0028 | 0.00 |

Paired bootstrap with 1,000 resamples. The classical side is the seed-averaged (3 seeds)
prediction of the better model, which scores a little higher than the mean of single-seed
scores below. Source: `benchmark/fraud/results/comparison_vs_classical.csv` (service `19th_standard`).

## Learning curves (100,000-row holdout)

| Training rows | 19th (standard) | LightGBM | Logistic regression | XGBoost (tuned) |
|---:|---:|---:|---:|---:|
| 1,000 | **0.8958** | 0.7900 | 0.8948 | 0.7230 |
| 5,000 | **0.9358** | 0.8984 | 0.9263 | 0.8974 |
| 20,000 | **0.9408** | 0.9269 | 0.9357 | 0.9358 |
| 100,000 | **0.9429** | 0.9400 | 0.9401 | 0.9428 |
| 200,000 | **0.9432** | 0.9413 | 0.9401 | **0.9432** |
| 599,960 | 0.9430 | 0.9451 | 0.9406 | **0.9468** |

Classical scores are the mean of 3 seeds. Tuned XGBoost searched 60 trials per size (8 seconds
at 1,000 rows, 15 minutes at 599,960). At 1,000 rows its 200-row tuning split holds about 3
frauds, so the search is close to random and the score is poor. Sources: `benchmark/fraud/results/classical_scores.csv`,
`benchmark/fraud/results/service_scores.csv`.

## 19th training modes (Fast model, 100,000-row holdout)

| Training rows | Instant | Standard | High | Ultra | Best classical |
|---:|---:|---:|---:|---:|---|
| 1,000 | **0.8977** | 0.8958 | 0.8958 | 0.8943 | Logistic regression 0.8948 |
| 5,000 | 0.9345 | 0.9358 | 0.9361 | **0.9362** | Logistic regression 0.9263 |
| 20,000 | 0.9404 | 0.9408 | **0.9409** | **0.9409** | XGBoost (tuned) 0.9375 |
| 100,000 | 0.9425 | 0.9429 | 0.9432 | **0.9435** | XGBoost (tuned) 0.9433 |
| 200,000 | 0.9426 | 0.9432 | 0.9434 | **0.9438** | XGBoost (tuned) 0.9439 |

Best classical is the seed-averaged prediction used in the bootstrap. Every mode's lead over it
is outside noise at 5,000 and 20,000 rows and inside noise at 1,000 and 100,000 rows. At
200,000 rows instant is clearly behind (-0.0013) and the other modes are within noise.
Per-mode bootstrap rows: `benchmark/fraud/results/comparison_vs_classical.csv`.

Each run also returns two faster variants of its model, Turbo and Warpspeed. Holdout AUC
(Fast / Turbo / Warpspeed):

| Training rows | Instant | Standard | High | Ultra |
|---:|---|---|---|---|
| 1,000 | 0.8977 / 0.8977 / 0.8977 | 0.8958 / 0.8958 / 0.8959 | 0.8958 / 0.8956 / 0.8956 | 0.8943 / 0.8942 / 0.8942 |
| 5,000 | 0.9345 / 0.9346 / 0.9344 | 0.9358 / 0.9357 / 0.9356 | 0.9361 / 0.9361 / 0.9360 | 0.9362 / 0.9361 / 0.9359 |
| 20,000 | 0.9404 / 0.9404 / 0.9403 | 0.9408 / 0.9408 / 0.9407 | 0.9409 / 0.9409 / 0.9408 | 0.9409 / 0.9409 / 0.9408 |
| 100,000 | 0.9425 / 0.9426 / 0.9426 | 0.9429 / 0.9429 / 0.9428 | 0.9432 / 0.9431 / 0.9428 | 0.9435 / 0.9435 / 0.9432 |
| 200,000 | 0.9426 / 0.9426 / 0.9426 | 0.9432 / 0.9431 / 0.9431 | 0.9434 / 0.9434 / 0.9431 | 0.9438 / 0.9438 / 0.9434 |
| 599,960 | — | 0.9430 / 0.9430 / 0.9428 | — | — |

| Variant | Model file (standard, 200,000 rows) | 1,000-row batch | Model file (ultra) | 1,000-row batch |
|---|---:|---:|---:|---:|
| Fast | 1.7 MB | 2.0 ms | 8.6 MB | 8.8 ms |
| Turbo | 1.0 MB | 1.1 ms | 4.4 MB | 4.5 ms |
| Warpspeed | 0.33 MB | 0.32 ms | 0.33 MB | 0.32 ms |

| Mode | Fast model file | Fast 1,000-row batch | Run time (5,000 rows) | Run time (200,000 rows) |
|---|---:|---:|---:|---:|
| Instant | 0.4 MB | 0.5 ms | 5 s | 2 min |
| Standard | 1.1-1.7 MB | 1.3-2.0 ms | 10 s | 3.5 min |
| High | 1.1-1.7 MB | 1.3-2.0 ms | 11 s | 7 min |
| Ultra | 8.5 MB | 8.8 ms | 2 min | 32 min |

Sizes and prediction times are 19th's reference CPU estimates; run times are from submission to
completion, with all runs queued at once, so they include some queueing. High and ultra runs
carried a generic warning ("Additional training limitation recorded"); instant and standard did not.

## 19th run details (fraud)

| Item | Value |
|---|---|
| Training mode | `standard` |
| Inputs | 19 numeric (booleans sent as 0/1) and 5 categorical, all read as declared; no exclusions |
| Run time | 6-10 seconds up to 20,000 rows, about 1 minute at 100,000, about 5 minutes at 599,960 |
| Warnings | Standard at 599,960 rows: "Training used fewer labeled examples because of capacity limits; some supplied labels were not used." and "Training used a random 8000 of the 399960 unlabeled rows." (every supplied row was labeled) |
| Submission | `benchmark/fraud/submissions/submission_19th_599960_standard_fast.csv`, submitted 2026-11-26: **public leaderboard 0.94503** (holdout 0.9430) |
| Comparison submission | Tuned XGBoost refit on all 699,960 rows (parameters tuned at 599,960, 3-seed average; `benchmark_services.py --dataset fraud xgboost-submission`): **public leaderboard 0.94785** (holdout 0.9471 when trained on 599,960 rows) |

# Heart disease (Playground S6E2)

Data: `train.csv`, 630,000 rows, 13 raw features, binary target.

## Summary

- **19th is clearly ahead of the best classical model up to 700 rows**, by about +0.005 AUC.
  These leads are outside the bootstrap noise.
- **A tuned XGBoost erases the lead from 2,000 rows.** With Optuna-tuned XGBoost added, the best
  classical model at 2,000 rows ties 19th (-0.0001); at 5,000 and 20,000 rows 19th is ahead by
  +0.0010 and +0.0003, within noise. Against untuned LightGBM and logistic regression alone,
  19th had been clearly ahead at 2,000 rows (+0.0024).
- **Tuning hurts XGBoost on tiny samples.** At 100-250 rows the 20% tuning split holds only 20-50
  rows, so the search overfits it: tuned XGBoost scores 0.934-0.946, below untuned LightGBM.
- **In data terms**, 19th trained on 100-250 rows scores about what logistic regression reaches
  with 2,000.
- **Rimay quantum features do not help.** At 700 rows they score 0.921 alone and lower LightGBM
  from 0.950 to 0.940 when added to the raw features. 19th scores 0.958 on the same rows.
- **The dataset is easy.** Logistic regression already reaches 0.954 with 100 rows, so every
  method sits within about 0.02 AUC of the others.

## 19th vs best classical model (1,000-row holdout)

| Training rows | 19th | Best classical | Lead | 95% interval | P(19th better) |
|---:|---:|---|---:|---|---:|
| 100 | 0.9597 | Logistic regression 0.9543 | +0.0053 | +0.0006 to +0.0106 | 0.99 |
| 250 | 0.9628 | Logistic regression 0.9581 | +0.0046 | +0.0013 to +0.0083 | 1.00 |
| 500 | 0.9644 | XGBoost (tuned) 0.9591 | +0.0053 | +0.0015 to +0.0094 | 1.00 |
| 700 | 0.9639 | Logistic regression 0.9592 | +0.0046 | +0.0018 to +0.0077 | 1.00 |
| 2,000 | 0.9650 | XGBoost (tuned) 0.9651 | -0.0001 | -0.0021 to +0.0020 | 0.48 |
| 5,000 | 0.9664 | XGBoost (tuned) 0.9654 | +0.0010 | -0.0009 to +0.0029 | 0.85 |
| 20,000 | 0.9666 | XGBoost (tuned) 0.9663 | +0.0003 | -0.0013 to +0.0018 | 0.64 |

Lead and interval come from a paired bootstrap (2,000 resamples of the holdout). The classical
side is the seed-averaged prediction of whichever of LightGBM, logistic regression and tuned
XGBoost scores higher on the full holdout, which slightly favors it. Source:
`benchmark/heart/results/comparison_vs_classical.csv`.

## Learning curves (1,000-row holdout)

| Training rows | 19th (standard) | LightGBM | Logistic regression | XGBoost (tuned) |
|---:|---:|---:|---:|---:|
| 100 | **0.9597** | 0.9443 | 0.9543 | 0.9335 |
| 250 | **0.9628** | 0.9515 | 0.9581 | 0.9455 |
| 500 | **0.9644** | 0.9521 | 0.9573 | 0.9545 |
| 700 | **0.9639** | 0.9565 | 0.9592 | 0.9550 |
| 2,000 | **0.9650** | 0.9606 | 0.9626 | 0.9612 |
| 5,000 | **0.9664** | 0.9635 | 0.9635 | 0.9650 |
| 20,000 | **0.9666** | 0.9656 | 0.9636 | 0.9660 |

Classical scores are the mean of 5 seeds; the bootstrap compares against the average of the seeds'
predictions, which scores higher (tuned XGBoost: 0.9651 at 2,000 rows vs a 0.9612 seed mean). Logistic regression plateaus near 0.964 from 2,000
rows; LightGBM keeps improving and closes the gap. Sources:
`benchmark/heart/results/classical_scores.csv`, `benchmark/heart/results/service_scores.csv`.

## Rimay quantum features (700 training rows, Rimay's 300 test rows)

| Method | AUC |
|---|---:|
| 19th (standard) | **0.9575** |
| Logistic regression, raw features | 0.9562 |
| LightGBM, raw features | 0.9501 |
| XGBoost (tuned), raw features | 0.9477 |
| LightGBM, raw + Rimay quantum features | 0.9402 |
| LightGBM, Rimay quantum features only | 0.9207 |

Rimay (Kipu Quantum, via PlanQK; execution `b9c72ba0`, 500 shots, 1 run) returned 25 quantum
features per row for exactly these 700 + 300 rows, so only the 300-row score exists for it.
LightGBM scores are the mean of 5 seeds.

### Correction to the earlier Rimay evaluation

`quantum/quantum_rimay_workflow.py --evaluate-only` reported quantum features alone at 0.575
AUC. That number is wrong: the script rebuilds its training split, which no longer matches
the 700 rows sent to Rimay on 2026-03-30 (only 52% of labels agree), so it paired Rimay's
features for one set of patients with another set's labels. Scored on the rows Rimay actually
received (`quantum/quantum_results/`), quantum features alone reach 0.921. The script also
early-stops on the test set, which inflates all of its scores. The example numbers in
`quantum/USAGE.md` are illustrative, not measured.

## 19th run details

| Item | Value |
|---|---|
| Training mode | `standard` (default; `instant`, `high` and `ultra` also offered) |
| Model scored | Fast (Turbo and Warpspeed also returned; 19th reports them within -0.03 to +0.01 quality points) |
| Prediction | Local, from the downloaded `npz` bundle (oracle_numpy runtime, numpy only) |
| Run time | 6-10 seconds per run, including validation |
| Runs | One per size, named `Heart disease probability <size> rows` in the 19th account |
| Inputs | The 13 raw columns, all read as numeric; no exclusions or warnings |

## Caveats

- **One 19th run per size.** Run-to-run variation of 19th is not measured.
- **Only `standard` mode.** `high` or `ultra` may do better, most likely at larger sizes.
- **Untuned baselines.** LightGBM and logistic regression use fixed settings; CatBoost and
  tuned gradient boosting were not tried.
- **Noisy small holdout.** On the 300-row Rimay holdout, differences under about 0.01 AUC are
  within noise.
- **Easy, synthetic data.** Playground data is generated from a real dataset; with AUC near
  0.96 at 100 rows, this benchmark says little about harder problems.

## Reproduce

See [USAGE.md](USAGE.md). Classical baselines and scoring rerun locally; 19th runs need the
19th MCP server.
