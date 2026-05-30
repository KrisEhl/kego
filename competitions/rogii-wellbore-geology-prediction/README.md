# Rogii — Wellbore Geology Prediction

**Task**: Predict TVT (True Vertical Thickness) in the post-PS zone of ~200 horizontal wells from GR logs and a typewell reference.
**Metric**: RMSE (ft) on post-PS rows. Lower is better.
**Deadline**: 5 August 2026

Metric verified 2026-05-30 from Kaggle competition assets: `AI_wellbore_geology_prediction_task_en.pptx`
states prediction quality is RMSE over `dTVT = manualTVT - predictedTVT` for each predicted point.

## Problem summary

Each horizontal well is drilled vertically then curves to horizontal. At the **Prediction Start (PS)** point, exact TVT measurements stop. The model must predict TVT for the remaining 4–6k ft of horizontal drilling using:

- `GR` — Gamma Ray log along the horizontal well (noisy, ~30% NaN after PS)
- `TVT_input` — exact TVT up to PS (anchor)
- Typewell `TVT, GR` — vertical reference well nearby (clean, 0.5 ft resolution)
- `X, Y, Z, MD` — borehole geometry

**TVT is a geological depth coordinate**, not physical depth. It measures "at this drill location, what depth in the typewell has the same rock?" After PS the well is nearly horizontal so TVT barely moves (±15 ft std), but predicting those small deviations precisely is the challenge.

## Baselines

| Method | Post-PS RMSE |
|---|---|
| Constant (predict TVT_at_PS) | 15.9 ft |
| Nearest-neighbour deviation (5-NN blend) | 13.8 ft |
| Current LightGBM (OOF all rows) | ~18 ft* |
| Public LB best (DWT-based) | 9.25 ft |

\* OOF measured across all rows including easy before-PS region. Post-PS only not yet measured.

**Key finding**: spatial proximity between wells does **not** help — TVT deviations are locally driven and don't correlate even between wells 400 ft apart. The GR signal is the only path to beating the constant baseline.

## Data

- **Train**: 773 horizontal wells + typewells, full TVT available for all rows
- **Test**: ~200 hidden wells (placeholder: 3 wells locally)
- **Coverage**: ~34 × 25 mile field
- **Submission**: `{well_id}_{row_idx}, tvt` for post-PS rows only

## Plan

### [x] Track 1 — Fix CV metric (immediate)
Evaluate OOF RMSE on **post-PS rows only**. Done — `post_ps_rmse` now logged as primary metric and configured as `primary_metric` in `kego.toml`.

### [~] Track 2 — Feature improvements (fast)
Add to `train_rogii.py`:
- [x] `tvt_anchor` — TVT at PS (constant per well)
- [x] `delta_md_from_ps` — distance from PS anchor in MD
- [x] Predict deviation `TVT - tvt_anchor` instead of absolute TVT
- [x] GR pattern matching (4D KNN) against typewell + pre-PS self-reference

### [x] Track 3 — GR↔typewell sliding window correlation (main signal)
DEAD END: individual signals (GR, Z, dip) all fail. The post-PS TVT range (±20 ft) is too narrow for typewell GR matching. Pre-PS dip has zero correlation (r=-0.05) with post-PS dip.

### [~] Track 3b — Sequence model (dilated TCN, `train_seq.py`)
Causal 1D dilated TCN. Features: GR_norm, Z_delta, dX, dY, delta_MD, is_post_ps, tvt_dev_known.
Full sequences, dynamic padding per batch, loss only on post-PS rows.

**Diagnosis**: model barely beats constant. Root cause — model uses `tvt_dev_known` as passthrough
for pre-PS rows without learning from GR. To predict post-PS it just outputs ~0 (the PS anchor value).

**Next**: input masking — randomly zero `tvt_dev_known` for 50% of pre-PS rows during training,
forcing the model to learn GR→TVT from the pre-PS data. Also increase capacity (d_model 128, 8 layers).

### [~] Track 4 — Physics-informed tabular features (from public solutions analysis)
Public solutions get 8.9–9.4 ft using tabular GBM + physics features, NOT neural nets.
Neural alone = 15.5 ft (same as us). Key signals:

**Priority 1 — Multi-scale Pearson NCC** (r=0.9993 with true TVT)
For each row, compute Pearson correlation between h_gr[i-hw:i+hw] and tw_gr[j-hw:j+hw]
at candidate typewell positions j. Half-widths hw ∈ {8, 15, 25}. Softmax blend.
This is NORMALISED cross-correlation — scale-invariant, handles lateral GR variation.

**Priority 2 — Formation spatial KNN with bias calibration**
TVT ≈ -Z + formation_depth(X,Y) + bias_well
KNN from training wells' formation surfaces; bias_well calibrated from anchor zone.
Current implementation note: direct `EGFDU` features are excluded from model inputs because
`EGFDU` exists in train horizontals but not test horizontals; any formation KNN must be
fit fold-aware from training wells only. A first fold-aware sampled surface is implemented
behind `--formation-knn`, but debug CV was much worse, so it is off by default.

**Priority 3 — Beam search (Viterbi HMM)**
Forward-Viterbi on typewell GR emissions, 4 variants (loose/medium/tight sigma pairs).

**Priority 4 — Particle filter**
500 particles tracking TVT through typewell GR likelihood.

**Priority 5 — Savitzky-Golay smoothing post-processing**

### Track 5 — Ensemble
Stack all signal families with NNLS blend (LightGBM + CatBoost + XGBoost × seeds).

## Running

```bash
# Local (full CV)
uv run kego run competitions/rogii-wellbore-geology-prediction/train_rogii.py

# Cluster (5-fold fan-out)
uv run kego run competitions/rogii-wellbore-geology-prediction/train_rogii.py \
    --target cluster --folds 0,1,2,3,4

# Debug smoke test
uv run kego run competitions/rogii-wellbore-geology-prediction/train_rogii.py --debug

# Experimental fold-aware formation KNN
uv run kego run competitions/rogii-wellbore-geology-prediction/train_rogii.py --formation-knn
```

## Dead ends

| Approach | Result | Why it failed |
|---|---|---|
| 5-NN spatial deviation | 13.8 ft (worse) | TVT deviations don't correlate spatially even at 400 ft |
| GR cross-correlation (full typewell) | 138 ft | Searches wrong TVT range without constraint |
| GR cross-correlation (constrained ±150 ft) | 71 ft | Horizontal GR and typewell GR waveforms don't match (corr=0.41, lateral heterogeneity) |
| GR calibrated typewell NN | 324 ft | Post-PS TVT range is ±20 ft — GR flat in typewell across that range, inversion impossible |
| EGFDU spatial interpolation from training wells | 30-65 ft | Different typewells have 644 ft std in egfdu_tw — values not comparable across wells |
| EGFDU XY-dip extrapolation from pre-PS | 27-45 ft | Pre-PS dip (curved section) doesn't match post-PS horizontal dip |
| Post-PS dip interpolation from training wells | 31 ft mean, catastrophic outliers | Dip varies ±0.18 ft/ft across field — interpolation fails for far wells |
| corr(pre-PS dip, post-PS dip) | r = -0.05 | Pre-PS dip has ZERO predictive power for post-PS dip |
| dTVT vs dGR correlation | r ≈ 0.0 | GR changes don't predict TVT changes at 1-ft scale |
| dTVT vs dZ correlation | r = -0.20 | Z only explains 4% of TVT variance (+0.4% improvement over constant) |
| Tabular GBM on GR matching features | ~15.96 ft | GBM can't exploit sequential GR structure — stuck at constant baseline |
| tvt_anchor + delta_md_from_ps features | 16.95 ft post-PS | Model predicting absolute TVT — drifts from anchor |
| Tabular GBM on GR matching features | ~15.96 ft | GBM can't exploit sequential GR structure — stuck at constant baseline |
| Fold-aware sampled formation KNN | 71.79 ft debug CV | Spatial `TVT + Z` surface does not generalise on the 20-well smoke sample; gated behind `--formation-knn` |
| Beam search / Viterbi HMM (sigma 10–50, step 0.5–2 ft) | 32–48 ft on 25 wells | Greedy/hard-window decode follows GR noise; sigma has no effect with hard max-pool transition; worse than constant on all wells |
| NCC as direct predictor (±10/20/40/80/150 ft search) | 13.4 ft best (±10), worse on 17/25 wells | **KEY: GR does not disambiguate TVT within the ±15 ft reservoir band.** Public "r=0.9993" is a global-range artifact (pre-PS TVT spans 100s of ft on the curve). NCC kept as a weak GBM feature only, not a predictor. |
| Savitzky-Golay smoothing of OOF preds (win 11–51) | −0.02 ft | Error is slow drift, not high-frequency noise — nothing to smooth |

**Strategic conclusion (2026-05-30)**: Individual GR/geometric signals (NCC, beam, formation, dip) are ALL weak at the post-PS scale — the well steers within a thin reservoir where GR can't pin absolute TVT. Constant (~15.9 ft) is a very strong baseline. Public top solutions reach 8.9 ft via a large multi-feature 3-model ensemble (XGB+CatBoost+HGB, NNLS blend) where many individually-weak signals collectively reduce variance — NOT via any single strong feature. Path forward: replicate the ensemble + breadth of simple features, not chase one magic signal.

## Results log

| Run | Model | Post-PS RMSE | Notes |
|---|---|---|---|
| — | constant baseline | 15.9 ft | predict TVT_at_PS |
| — | 5-NN deviation | 13.8 ft | spatial proximity only |
| v1 | LightGBM 5-fold | TBD (all-rows only) | baseline, submission format fixed |
| v2 | + tvt_anchor + delta_md_from_ps | 16.95 ft | worse than constant — model predicting absolute TVT not anchored |
| v3 | + deviation target (TVT - tvt_anchor) | 15.99 ft | matches constant baseline — correct framing |
| v4 | + typewell pattern NN + prePS self-ref | 15.96 ft | marginal gain — tabular GBM can't exploit GR sequence structure |
| seq-v1 | TCN 5-fold truncated (500+2000) | 17.33 ft | truncation drops context; model uses tvt_dev_known passthrough |
| seq-v2 | TCN 5-fold full sequences | 15.86 ft | fold 1: 14.79 ft — below constant |
| seq-v3 | TCN 3-fold full sequences | 15.86 ft | fold 1: 14.79 ft — same; model still near constant baseline |
| seq-v4 | + input masking 50% + d128/8L | **15.54 ft** | consistently below constant — masking forces GR learning |
| seq-v5 | mask=0.9 + denoising aux loss | 16.42 ft | too aggressive — loses pre-PS TVT context |
| seq-v6 | mask=1.0 + denoising aux loss | 15.65 ft | full blind — fold 1 best (14.58) but fold 0/2 hurt |
| v5-invalid | + NCC + direct EGFDU formation features | 1.14 ft child-fold mean | invalid CV — `EGFDU` is train-only for horizontal wells and unavailable in test |
| debug | NCC default, no formation KNN | 30.99 ft | 20-well smoke only; confirms no train-only `EGFDU` inputs |
| debug | + `--formation-knn` fold-aware sampled surface | 71.79 ft | 20-well smoke only; worse, keep disabled |
| v6-ncc | NCC (hw=8,15,25) + existing features, 4-fold | 15.94 ft | NCC alone doesn't help — typewell GR flat in ±15 ft post-PS zone |
| v7-anchor-slope | + anchor stats + slope + GR-residual, ALL-rows training | ~16.4 ft (fold 3; killed) | **Reference divergence found**: trained on all rows like v2-v6. Public XGB Starter trains post-PS-only → 15.01 ft. Killed early once root cause identified. |
| v8-postPS | INVALID — cluster ran stale code | 16.37 ft (=v7) | cluster auto-pull failed (stuck 5 commits behind at 80e50e3); post-PS filter never applied. Killed + resubmitted as v8b. |
| v8b-postPS | LightGBM, v7 features, **post-PS-only training** | **15.58 ft** | folds 15.10/15.63/16.05/15.54. Filter verified in log (3.78M/5.09M rows). First result below constant via correct training setup. |
| v9-xgb | XGBoost, post-PS-only, slimmed df | **15.54 ft** | folds 15.43/15.20/15.99/15.52. Marginally beats LightGBM. **Slimming fix: 4–5 min/fold (was ~45 min)** — memory swap eliminated. |
| v10-morefeats | XGBoost + 16 more XGB-Starter features (~52 total) | 15.54 ft | **No change** — simple-feature lever EXHAUSTED. Diagnostic: `last_known_gr` useful (corr 0.26) but overlaps existing signal; `gr_diff_1/10`,`row_frac` are noise (corr ~0); `baseline_tvt_recent` BROKEN (std 679k ft — recent-50 slope extrapolation explodes, model ignores it). |
| v12-dip-lags | XGBoost + physics dip + GR lags/leads (67 feat) | **15.37 ft** | folds 15.52/14.95/15.46/15.55. BREADTH CONFIRMED — weak estimators help in aggregate (−0.17 vs v9/v10). Validates investing in full estimator suite. |
| v13-seqfeats | XGBoost + **full ported estimator suite** (198 feat: 7 beams + 2 particle filters + multi-scale NCC + FormationPlaneKNN + DenseANCCImputer + dip) | **fold 0 = 10.59 ft** | **BREAKTHROUGH — plateau broken.** Same fold 0 was 15.52 (v12) → **−4.93 ft**. Reproduces public R3→R6 jump (reference OOF 10.01). Cluster default ran `--fold 0` only; full 4-fold OOF pending (`--all-folds`). Build 740s (beams+PFs per well). Sanity: 10.59 ≈ reference 10.01 (consistent, not too-good); imputers use same leave-one-well-out as reference. **CV-optimism caveat**: imputers fit on all 773 train wells *before* the fold split → in fold *k* a val well's surface includes other val-fold wells (absent at test). Leave-one-out blocks self-leakage but not cross-val-well; expect OOF slightly optimistic vs LB. Mitigant: reference LB 9.41 < its OOF 10.01, so global-fit optimism is not severe there. **Treat LB as ground truth — not a confirmed win until submitted.** Gap to reference (10.59 vs 10.01) ≈ single-xgb vs their tree+blend ensemble (R7–R11 ladder). |
| v13b-allfolds | (killed mid-run) | fold 0 = 10.31 ft | Second independent fold-0 confirmation — brackets 10.3–10.6 with v13 (CPU-xgb thread nondeterminism); consistent w/ ref 10.01. **Perf**: 4 sequential CPU xgb fits using ~3/22 cores + both 3090s idle; one-time pandas `.replace()` (3.78M×198) was single-threaded. Killed at fold 1 → switched to **v13c** (GPU + vectorised prep). |
| v13c-gpu | XGBoost **GPU** (`tree_method=hist, device=cuda`) + vectorised `inf→nan` (np), full 198-feat suite, `--all-folds` 1 job | TBD | Same pipeline as v13/v13b on a 3090; `np.isfinite` mask replaces single-threaded pandas `.replace()`. Auto-detects cuda w/ CPU fallback. Expect OOF ≈ 10.3–10.6 (GPU/CPU hist differ slightly). Establishes fast path for R7–R11 re-runs. |
