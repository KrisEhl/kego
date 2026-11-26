# Feedback for 19th

From a benchmark of 19th against classical ML on two Kaggle Playground datasets (November 2026),
run through the 19th MCP server. Full numbers: [RESULTS.md](RESULTS.md); overview:
[dashboard_summary.html](dashboard_summary.html).

## Summary

19th is strong with little data and easy to use, but it stops improving around 100,000 rows,
the training level barely changes results, and the capacity limit is undocumented. A tuned
XGBoost matches it from 100,000 rows and beats it on the full card-fraud data, on our holdout
(0.9471 vs 0.9430) and on the Kaggle public leaderboard (0.94785 vs 0.94503).

## What worked well

- **Small-data accuracy.** 19th beat LightGBM, logistic regression and an Optuna-tuned XGBoost
  by +0.005 AUC at 100-700 rows (heart disease) and +0.003 to +0.010 at 5,000-20,000 rows (card
  fraud, 1.7% positives). Classical models needed about 5x the rows to match it.
- **Workflow.** prep_run / submit_run / get_results with idempotency keys, signed uploads, SHA-256
  checks and model bundles that run locally on numpy alone were reliable across about 30 runs.
  The bundle's prediction code was byte-identical across all runs, which made it easy to review once.
- **Model variants.** Turbo and Warpspeed scored within 0.0002 and 0.0004 AUC of Fast on our
  holdout, with Warpspeed files 5-26x smaller and predictions 6-27x faster. This is a great
  default to advertise.

## Issues and requests

1. **Document the training-row cap.** The docs say "No row limit. Very large tables are sampled
   during training". In practice a 599,960-row table (all labeled) trained on about 200,000
   labeled rows: the rest were treated as unlabeled, and only 8,000 of those were used. A 200,000-row
   upload trained without warnings. Please state the cap per training level, and say how the
   labeled subset is chosen. For imbalanced data it matters whether the sample is stratified.
2. **Make label coverage consistent with what training used.** For that run, `label_coverage`
   reported 599,960 labeled and 0 unlabeled rows, while the warning said "a random 8000 of the
   399960 unlabeled rows" were used. Reporting the labeled and unlabeled rows actually used for
   training would remove the confusion.
3. **Scaling with data.** Card-fraud AUC was 0.9429 at 100,000 rows, 0.9432 at 200,000 (no cap
   hit) and 0.9430 at 599,960, while LightGBM and tuned XGBoost kept improving. Any way to use
   more rows, even at a higher level, would close the large-data gap.
4. **Training levels give little.** Instant, standard, high and ultra were within about 0.002
   AUC of each other at every size (1,000-200,000 rows). Ultra took 32 minutes at 200,000 rows
   (standard: 3.5) for +0.0006 AUC, with 5x larger models. Guidance on when a higher level is
   expected to pay off would help users avoid long runs.
5. **Make the high/ultra warning actionable.** Every high and ultra run carried "Additional
   training limitation recorded; review the data and model suitability before relying on
   predictions." It does not say what the limitation is or what to change. Instant and standard
   runs on the same data had no warning.
6. **Report absolute validation metrics.** Results give each variant's quality only relative to
   Fast (`quality.difference_points`). An absolute validation score (AUC for probabilities)
   would let users compare runs without their own holdout. The relative estimates also looked
   noisy. For example, Warpspeed was reported at -0.0325 at ultra/200,000 rows, but measured
   -0.0004 on our 100,000-row holdout. Stating the validation size behind the estimate would
   help users judge it.
7. **Run-to-run variation.** There is no seed or repeat option, so we could not separate level
   differences from run noise. A seed parameter, or reported variation across internal
   resamples, would make comparisons more trustworthy.
8. **Lighter MCP responses.** Each get_results call returns about 8,000 tokens, most of it the
   same presentation guide and three signed URLs, and the links expire after an hour. Scoring
   all variants of 21 runs needed one full call per run. A compact mode, field selection, or a
   batch call for download links across runs would cut agent cost a lot.
9. **Docs consistency.** The Training API docs say `training_mode` is required with "no
   default", while get_capabilities reports `default_mode: standard`.

## Setup we used

- Datasets: Playground S6E2 (heart disease, 13 features) and S6E11 (card fraud, 24 features
  including 5 categorical, 1.7% positives); raw columns only, no feature engineering.
- Nested training samples (100 to 599,960 rows) and a fixed holdout (1,000 / 100,000 rows) that
  no model trained, tuned or early-stopped on; paired bootstrap for 19th vs the best classical
  model.
- Event-probability task, positive label 1, all columns declared as inputs; Fast models run
  locally from the npz bundle.
