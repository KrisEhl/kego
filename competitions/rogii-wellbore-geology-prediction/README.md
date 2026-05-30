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
- [~] Predict deviation `TVT - tvt_anchor` instead of absolute TVT
- [ ] GR sliding-window correlation score against typewell at each TVT candidate

### Track 3 — GR↔typewell sliding window correlation (main signal)
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
| 5-NN spatial deviation | 13.8 ft (worse than constant) | TVT deviations don't correlate spatially even at 400 ft |
| tvt_anchor + delta_md_from_ps features | 16.95 ft post-PS | LightGBM doesn't exploit the anchor — constant prediction (just predicting tvt_anchor) would score 15.9 ft |

## Results log

| Run | Model | Post-PS RMSE | Notes |
|---|---|---|---|
| — | constant baseline | 15.9 ft | predict TVT_at_PS |
| — | 5-NN deviation | 13.8 ft | spatial proximity only |
| v1 | LightGBM 5-fold | TBD (all-rows only) | baseline, submission format fixed |
| v2 | + tvt_anchor + delta_md_from_ps | 16.95 ft | worse than constant (15.9) — model not yet exploiting anchor correctly |
