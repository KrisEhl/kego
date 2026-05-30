# Rogii — Wellbore Geology Prediction

**Task**: Predict TVT (True Vertical Thickness) in the post-PS zone of ~200 horizontal wells from GR logs and a typewell reference.
**Metric**: RMSE (ft) on post-PS rows. Lower is better.
**Deadline**: 30 July 2026

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
Evaluate OOF RMSE on **post-PS rows only**. Done — `post_ps_rmse` now logged as primary metric.

### [~] Track 2 — Feature improvements (fast)
Add to `train_rogii.py`:
- [x] `tvt_anchor` — TVT at PS (constant per well)
- [x] `delta_md_from_ps` — distance from PS anchor in MD
- [x] Predict deviation `TVT - tvt_anchor` instead of absolute TVT
- [x] GR pattern matching (4D KNN) against typewell + pre-PS self-reference

### [x] Track 3 — GR↔typewell sliding window correlation (main signal)
DEAD END: individual signals (GR, Z, dip) all fail. The post-PS TVT range (±20 ft) is too narrow for typewell GR matching. Pre-PS dip has zero correlation (r=-0.05) with post-PS dip.

### [~] Track 3b — Sequence model (end-to-end trajectory learning)
For each post-PS point, slide a window of horizontal GR along the typewell GR curve and find the TVT offset with the best cross-correlation. This is domain-correct: it's how geologists "correlate" wells manually. The DWT-based LB 9.25 notebook is doing this with wavelet features.

Implementation:
1. For each post-PS row, extract GR window `[i-W, i+W]` (W=25–50 ft)
2. Slide along typewell GR, compute cross-correlation at each TVT offset
3. Return the TVT of the best-matching typewell position
4. This replaces point-wise `typewell_tvt_nn` with sequence-aware matching

### Track 4 — Sequence model on cluster (2× RTX 3090)
Encode the pre-PS (GR, TVT) sequence → decode post-PS TVT autoregressively using GR as input signal. Train on 773 wells via `kego run --target cluster --folds 0,1,2,3,4`.

Candidates: 1D conv, small Transformer, or TCN (Temporal Convolutional Network).

### Track 5 — Ensemble
Stack Tracks 2–4 with Ridge regression on OOF predictions.

## Running

```bash
# Local (full CV)
uv run kego run competitions/rogii-wellbore-geology-prediction/train_rogii.py

# Cluster (5-fold fan-out)
uv run kego run competitions/rogii-wellbore-geology-prediction/train_rogii.py \
    --target cluster --folds 0,1,2,3,4

# Debug smoke test
uv run kego run competitions/rogii-wellbore-geology-prediction/train_rogii.py --debug
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

## Results log

| Run | Model | Post-PS RMSE | Notes |
|---|---|---|---|
| — | constant baseline | 15.9 ft | predict TVT_at_PS |
| — | 5-NN deviation | 13.8 ft | spatial proximity only |
| v1 | LightGBM 5-fold | TBD (all-rows only) | baseline, submission format fixed |
| v2 | + tvt_anchor + delta_md_from_ps | 16.95 ft | worse than constant — model predicting absolute TVT not anchored |
| v3 | + deviation target (TVT - tvt_anchor) | 15.99 ft | matches constant baseline — correct framing |
| v4 | + typewell pattern NN + prePS self-ref | 15.96 ft | marginal gain — tabular GBM can't exploit GR sequence structure |
