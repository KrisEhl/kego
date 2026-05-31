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
| Public LB best (drift-NCC ensemble, R11) | **8.905 ft** |

\* OOF measured across all rows including easy before-PS region. Post-PS only not yet measured.

**Key finding**: spatial proximity between wells does **not** help — TVT deviations are locally driven and don't correlate even between wells 400 ft apart. The GR signal is the only path to beating the constant baseline.

## Reference ladder → path to 8.905 (from `drift-targeting-ncc` notebook, analyzed 2026-05-30)

| Round | Change | OOF | LB |
|---|---|---|---|
| R6 | 163 feat + Optuna + NNLS blend (LGB+XGB+CB) | 10.01 | 9.410 |
| R7 | CatBoost depth5, NNLS XGB 55.9%+CB 44.1% | 10.05 | 9.398 |
| R10 | +21 "v4 features" (estimator divergence, short-window slopes, DWT GR) | 9.91 | — |
| R11 | + HistGradientBoosting (max_iter=5000, `early_stopping=False`), NNLS XGB+CB+HGB | **9.85** | **8.905** |

- **⚠️ CORRECTED (audit 2026-05-31) — we ported the WRONG/weaker reference.** Our `rogii_features.py` is a faithful port of the **`wellbore-geology-prediction-lightgbm` / `aeroridge`** solution (OOF **~10.6**, identical fold splits) — which IS our LB-10.538 source. The **`drift-targeting-ncc` (8.905)** is a SEPARATE, richer solution we did **NOT** port. My earlier claim "gap = ensemble + Optuna, NOT missing features" is **FALSE**: their *single*-model OOF is already ~10.00 (vs our ~10.55) — the gap precedes ensembling and is **missing features + HP tuning + post-processing**.
- **Missing feature families** the 8.905 solution has and we lack: single-scale GR xcorr (`GR_xcorr_*`), trajectory **kinematics** (incl/azi/dls/build_rate/cos·sin incl), **apparent-dip / plane-dip physics** (`b_dip_*`, `plane_dip_*`), **RowKNN** row-level ANCC, anchor-NCC template, **DWT-GR** (`pywt`). (DTW = correctly low-value: ref dead-ends it.)
- **Missing POST-PROCESSING (the signature 8.905 lever, GPU-free, never tried)**: drift attenuation `d *= alpha·(1−exp(−md_since/τ))` + optional PF-blend `d = d·(1−w)+pf·w` + per-well **Savitzky-Golay** smooth — **Optuna-tuned (alpha/τ/w) on OOF**. Operates on final predictions → structural, above the ~0.05 CV-noise floor. (Our earlier "SG alone on OOF −0.02" dead-end did NOT test the attenuation it's combined with.)
- **HGB must use `early_stopping=False`** — internal `validation_fraction` leaks per-well GR; GroupKFold OOF is the stopping criterion.
- **8.905 is single-sourced** (the drift-ncc author's own writeup table); corroborated public LB ceiling across ranked notebooks is **~9.5** (AeroRidge 9.571, ROGII-v10 9.537). Treat 8.905 as plausible-but-unverified.
- **CORRECTED next-step ladder**: **(A) drift-attenuation + SG post-processing, Optuna on cached OOF (GPU-free, highest EV, above noise) → (B) wire LightGBM (ref's workhorse: num_leaves=255, ~6000 trees, 3 seeds) + raise XGB/CB capacity (depth8, 6000 trees) + DWT-GR features → (C) Optuna-tune single models → (D) multi-seed NNLS/Ridge stack LAST.** v22 single-seed ensemble is NOT the headline lever.

## Data

- **Train**: 773 horizontal wells + typewells, full TVT available for all rows
- **Test**: ~200 hidden wells (placeholder: 3 wells locally)
- **Coverage**: ~34 × 25 mile field
- **Submission**: `{well_id}_{row_idx}, tvt` for post-PS rows only
- **⚠️ CODE COMPETITION** (confirmed 2026-05-30): direct CSV upload via `kaggle competitions submit` returns **400 / "No submissions found"** — LB scores require a re-run Kaggle **script kernel** (`pattern=script`, `enable_gpu=false`). Use `kego push <id>` (models→dataset) + `kego submit <id>` (push inference kernel). Test data IS fully provided locally (~200 wells, 14,151 post-PS rows; our `outputs/submission_seq_feats.csv` matches `sample_submission.csv` id-set exactly), but the LB still needs the kernel path. **Inference kernel BUILT + validated (2026-05-30)**: `inference_kernel.py` = `rogii_features.py` (verbatim) + `_inference_main.py`, assembled by `build_kernel.sh` (single-file kernel; Kaggle script kernels can't import aux `.py`). Train-in-kernel on CPU (`device=cpu`), GroupKFold(4) 4 models averaged = exact v13c setup; reindexes test to train feature cols (handles hidden-set column drift). Debug smoke (`ROGII_KERNEL_DEBUG=1`, 12 wells) → 14,151 valid rows, 0 bad. `kernel-metadata.json`: `competition_sources=[slug]`, `enable_gpu=false`, single code_file. **Pushed directly via `kaggle kernels push`** (NOT `kego submit` — that force-appends a `*_fold*.pt` checkpoint dataset this self-contained kernel doesn't need). LB submit via `kaggle competitions submit -k <kernel> -v <ver>`. kego.toml repointed `inference_notebook→inference_kernel.py`. **Local/public test = 3 visible wells (14,151 rows); hidden test ~200 wells re-run by Kaggle.** LB anchor pending kernel run.

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
| Trajectory kinematics + apparent-dip physics (v25, 16 feat, ported from 8.905 ref) | +0.015 ± 0.042 (3-seed, neutral) | No gain on single XGB — incl/dip/dls/build_rate signal is **redundant** with the existing beam/PF/NCC drift extrapolations. Geometric features are tapped out for our suite. Kept in code (`--no-kinematics` to drop). |
| XGB capacity bump depth8/6000 trees (v26, ref-style) | +0.091 (3-seed, all worse) | Naive capacity increase **overfits** — depth8 worse than depth7 on all 3 paired seeds (+0.076/+0.129/+0.069). Data wants MORE regularization, not more depth. depth7/3000 (LB-10.538 config) is optimal. The ref's depth8 was paired with Optuna-tuned reg; bumping depth alone hurts. |

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
| v13c-gpu | XGBoost **GPU** (`tree_method=hist, device=cuda`) + vectorised `inf→nan` (np), full 198-feat suite, `--all-folds` 1 job | **10.62 ft (4-fold OOF)** | **PLATEAU BROKEN — full OOF confirmed.** Folds: 10.21/10.61/10.12/11.48 (fold 3 worst, spread 1.4 ft, no leakage-low fold). **15.37→10.62 (−4.75 ft).** GPU @97% util, 17m total (vs ~50m CPU) — both fixes confirmed. 14,151 test preds saved → `outputs/submission_seq_feats.csv`. Sanity: ≈ ref OOF 10.01 (gap = single-xgb vs their tree+blend); not bit-identical to v13b → fresh code. **→ submitted to Kaggle as the inference kernel: LB 10.538 (OOF≈LB within 0.1).** |
| v14-catboost | CatBoost **GPU** (`task_type=GPU`, depth7, lr0.03, 3000it, od_wait80), same 198-feat suite, `--all-folds` | 10.81 ft | Folds 10.63/10.67/10.51/11.42 (same fold-3-hardest pattern as xgb → consistent). **XGBoost (10.62) remains best single model** (catboost +0.19). Family comparison done. CatBoost close enough to keep as ensemble diversity for later (priority 4). |
| v15-divergence | XGBoost GPU + v4 Phase-1 estimator-divergence features (212 cols) | 10.48 ft | ⚠️ **NOT A WIN** (corrected): the −0.14 vs v13c is INSIDE the 0.29 single-run noise band; 5-seed A/B later proved divergence NEUTRAL (Δ0.01). Folds 10.03/10.28/10.35/11.21. Left in code (neutral, `--no-divergence` to drop). |
| v16-slopes | + v4 slope features (217 cols) on top of divergence | 10.76 ft (DEAD END) | Folds 10.08/10.96/10.50/11.46 — **every fold worse than v15** (+0.28 OOF). Per-well scalar slopes degrade GroupKFold generalization (model keys off per-well constants that don't transfer to held-out wells). **Reverted.** v15 (divergence, 10.48) remains best. |
| v17-catboost-div | CatBoost GPU on v15's **212 divergence features** | 10.57 ft | Divergence helped catboost too (v14 198-feat=10.81 → 10.57). Strong diverse ensemble member (xgb 10.48 / cb 10.57). |
| v18-ensemble | **NNLS blend XGB+CB** on 212 divergence features (NNLS w: xgb 0.40 / cb 0.67) | 10.44 ft | Blend < both members (xgb 10.63 / cb 10.53 *this run*). **⚠️ INSTABILITY FOUND**: v18's xgb OOF=10.63 but v15's (same config/seed/features) =10.48 — **~0.15 OOF swing run-to-run** (fold 1: 10.28 vs 10.86). GPU xgb + early-stopping is not reproducible. **Small CV deltas (divergence +0.14, ensemble +0.04) are within this noise band — NOT cleanly confirmed.** LB is the only trustworthy signal. |
| v19-seed7 | v18 ensemble seed 7 | KILLED | Cluster powered off (10h maintenance) before finish. Noise measured locally instead (v20). |
| v20-local-xgb | xgb seed 42, 212 divergence feat, **CPU/local** (offline sqlite) | 10.77 ft | Build 390s (Mac CPU faster than cluster) + cached. 3rd same-config data point. |
| v22-ens3-gpu | full xgb+cb+hgb NNLS, single-seed GPU | KILLED | Single-seed (not submittable per audit) + HGB ran 1-CPU (default `--cpu 1`) → ~28min/fold, GPU 0%. Killed; ensemble is priority-4 (post-proc first). Lesson: use `--cpu 8` for HGB ensembles. |
| v23-postproc | **drift-attenuation + PF-blend + savgol**, tuned on 212-feat 4-fold xgb OOF (GPU-free) | 10.727 ft (CPU OOF, −0.041) | ⚠️ **NARRATIVE CORRECTED (audit 2026-05-31)**: the gain is **entirely the PF-blend**, NOT the attenuation. Decomposing `oof_v23.npz`: blend(w=0.1) alone = +0.0376, τ=39 attenuation alone = **+0.0005 (inert)** — only 0.78% of rows have `md_since<39` (median 2458 ft → factor≈1), and savgol(win=31) was a +0.003 grid-**boundary** fit (gain monotone-rising in window → noise-fit). A tune-on-half/eval-on-disjoint-half test showed the full tuned pipeline transfers at mean +0.029 (min **−0.080**, 1/12 negative) while the blend **alone** transfers at +0.055 (min −0.006). → keep blend only. Superseded by v24. |
| v24-blend-kernel | **PF-blend only (w=0.10)** on the actual 195-feat model (no divergence), re-tuned via `tune_pp_blend.py` | **LB 10.105** (CPU OOF 10.795) | 🎯 **NEW LB BEST — 10.538 → 10.105 (−0.433!).** Kernel v2, COMPLETE. The ONLY change vs the 10.538 anchor is the PF-blend (verified by git diff `9464a76..f7f6bb9`: estimator features identical, divergence dropped in `_xy`). `d = drift·0.9 + pf_ancc_delta·0.10`. ⚠️ **Magnitude vastly exceeds OOF**: −0.433 LB vs −0.054 OOF (8×). Almost certainly amplified by the tiny **4-well public test** (high variance — the blend strongly corrects XGB over-drift on those wells); **private-LB gain may be smaller**. Sound + leak-free (pf_ancc_delta is an input; kernel audited). Params `outputs/pp_blend_198.json`. |
| v25-kinematics | **+ trajectory kinematics + apparent-dip physics** (16 feat), ported from the 8.905 reference | **NEUTRAL — mean Δ +0.015 ± 0.042 (3 paired seeds, within noise)** | Biggest documented missing lever, but **no gain** on single XGB. Paired (kin−base): s1 −0.066 / s2 +0.039 / s3 +0.072 (base mean 10.743, kin mean 10.758). Despite `tvt_dip_late_d` corr 0.34 w/ target, the dip/kinematics signal is **redundant** with the existing beam/PF/NCC extrapolations → marginal contribution ≈ 0. Build validated (incl median 90.6° in horizontal zone ✓, 0% NaN, leak-free: geometry + anchor TVT_input only, plane-dip leave-one-well-out). Kept in code (neutral), `--no-kinematics` to drop; kernel stays 195-feat. **Implication**: geometric/physics features are tapped out for our suite — the gap to the ref's single-model ~10.0 likely lives in DWT-GR (untried, fundamentally different signal repr) or the ensemble+tuning, NOT more geometry. |
| v26-depth8 | XGB **depth8/6000 trees** (ref-style capacity), 195-feat, vs depth7/3000 baseline | **WORSE — mean Δ +0.091 (3 paired seeds, all worse)** | Paired (d8−d7): s1 +0.076 / s2 +0.129 / s3 +0.069 (d7 mean 10.743, d8 mean 10.834). Naive capacity bump **overfits** — consistent regression across all seeds (above noise). **Data wants more regularization, not more depth.** depth7/3000 remains optimal. HP lever now bracketed: depth8 worse (this), depth5/mcw20 marginal (prior probe) → depth7 near-optimal. Next: LightGBM (ref workhorse, untried family) or multi-seed ensemble. |
| v27-regularize | XGB **depth6** and **mcw20** (more regularization), 195-feat, vs depth7/mcw5 | **depth6 −0.047 (3 seeds, all better)**; mcw20 −0.019 | depth6 paired Δ: s1 −0.043 / s2 −0.058 / s3 −0.041 (consistent, above noise) → **regularization helps** (confirms depth8-overfit read; data wants depth6). mcw20 milder (−0.019). **NOT submitted** — audit FAIL: sub-LB-noise + justified by the LB-inverted OOF metric + only 3-seed (methodology needs ≥5). Kept depth7 kernel. |
| v29-consensus-blend | Blend toward **median(pf_ancc_delta, beam_med_d, tvt_dense_d) w=0.125** vs pf_ancc-alone w=0.10 (the 10.105 blend) | OOF +0.0372 (best blend; +0.005 vs pf-alone) | **SUBMITTED as a more-robust private-LB candidate** (3rd anchor, kernel v3, PENDING). Audit: correctness PASS, strategy lean-PASS. **Justified by the disjoint-half transfer test** (not the +0.005 OOF): median3 transfers robustly (mean +0.026, **0/12 splits negative**) while pf-alone **overfits its w-optimum** (mean +0.017, 4/12 negative). MECHANISM (audit-corrected): beam/dense are individually weak — the median acts as a **robust clipper of pf's extreme corrections**, NOT a consensus of 3 good estimators. On the proven LB lever (blend, OOF-directional). Leak-free (all 3 are input drift estimates). 10.105 (pf-alone) kept banked as 2nd candidate; will NOT de-select median3 on a noisy public delta (it's a private-LB/generalization bet). |
| v31-ensemble-XC | **NNLS ensemble XGB+CatBoost** (both GPU on cluster, 195-feat, no-div/no-kin), 3 seeds | **OOF 10.603 (3-seed mean, std 0.004) — −0.150 vs single XGB** | **CONFIRMED structural win** (priority-4 lever pays off). Per-seed ensemble: 10.598/10.606/10.605 (rock-solid). vs single XGB 10.753, single CatBoost 10.710. So: **CatBoost > XGB by ~0.04 (single)**, and **NNLS blend > best-single by ~0.11**, consistent across all 3 seeds → far above the ~0.05–0.15 noise band. NNLS weights ~XGB 0.48–0.52 / CB 0.55–0.60. Leak-free (NNLS fit on OOF, applied to test); structural (variance reduction) → should transfer to private LB. HGB dropped (28min/fold CPU bottleneck, marginal). (Threading: 3-concurrent OOMs the 62GB node @ ~18GB/job → run ≤2 concurrent.) **Ensemble kernel built (ca1a134); audit (3 judges): correctness PASS, statistical CONCERNS flagged the ensemble+blend stack as unvalidated/double-count.** |
| v32-ens-blend-validate | Re-tune consensus blend ON the ensemble OOF + disjoint-half transfer (`tune_ensemble_blend.py`) | ensemble OOF 10.560; **blend on ensemble = NEGATIVE transfer (8/12 splits, mean −0.046)** | **Audit CONFIRMED**: the consensus blend does NOT transfer on the ensemble base (8/12 negative, vs 0/12 on single-XGB) — the ensemble already incorporates pf/beam/dense (inputs) + NNLS sum 1.063 de-shrinks while the blend shrinks (opposing). **DECISION: ship ENSEMBLE-ONLY** (drop the blend) as the 4th anchor — the clean validated structural lever. The blend stays on the single-XGB kernels (10.105 / v3) where it DOES transfer. Ensemble OOF saved → `oof_ensemble.npz`. |

## 🎯 LB BEST (2026-05-31): **public LB 10.105** — v24 kernel (195 feat single xgb + PF-blend w=0.10, train-in-kernel CPU)

**2 anchors now (both same kernel features, the only diff is the blend):**
| Kernel | Change | CPU OOF | public LB |
|---|---|---|---|
| v13c (anchor #1) | 195-feat single xgb, no pp | 10.62 (GPU) | 10.538 |
| v24 (anchor #2, **BEST**) | + PF-blend w=0.10 | 10.795 (CPU) | **10.105** |

**🔑 PIVOTAL FINDING — OOF is a POOR proxy for LB at this scale; post-processing toward the PF drift estimate is the DOMINANT LB lever.** The blend moved LB **−0.433** but OOF only −0.054 (8×). And the CV↔LB relationship is *inverted/uncorrelated*: v24 has WORSE OOF (10.795 > 10.62) yet BETTER LB (10.105 < 10.538). Root cause: the public test is only **4 wells / 14,151 rows** → high variance; the blend strongly corrects XGB over-drift on those specific wells. **Implications**: (1) trust LB, not OOF, for ranking pp/blend changes; (2) the small-feature/HP CV experiments (kinematics, depth8, regularization) barely matter for LB — the blend dominates; (3) **private-LB gain may be smaller** than the public −0.433 (don't over-trust 4 wells); (4) **risk: tuning the blend weight on the 4-well public LB overfits it** — w=0.10 is OOF-principled, push higher only cautiously. Reference best LB = 8.905, R6 = 9.41. Submission = code-comp kernel (`-f submission.csv -k <kernel> -v <ver>`, ~80min scoring re-run).

### Strategy after the 10.105 audit (2026-05-31)
**depth6 + v28 ensemble were NOT submitted — audit FAIL (do not churn sub-noise submissions).** A focused audit panel (2 judges) on a proposed depth6+blend submission returned FAIL on strategy: (1) depth6's −0.047 is measured on the **OOF metric this README proves is inverted/untrustworthy for LB**; (2) it's only **3-seed** — our own CV-noise methodology requires ≥5 (the README already records a near-identical 3-seed reg tweak that *flipped* at 5 seeds); (3) the effect (~0.05) is **far below** the 4-well public-LB sampling noise (~4.5 ft RMSE std on a random 4-well subset) **and** the kernel re-run noise (±0.15–0.3 ft) → a submission yields **zero decision-quality info**. Same logic kills the v28 ensemble (XGB-d6+HGB seed1 = 10.7073, only −0.021 vs depth6-single → also sub-noise; HGB also costs ~28min/seed).
**The blend is the ONLY lever with demonstrated LB traction (−0.433), and crucially the blend's OOF was *directionally* predictive (helped OOF AND LB) — unlike depth6/features where OOF is inverted.** So the disciplined next move is to **improve the proven lever (the blend), OOF-validated**: test blending the XGB drift toward a robust **consensus of independent drift estimators** (median of pf_ancc_delta, beam_med_d, tvt_dense_d, form_mean_d, sc_ens_d) vs pf_ancc-alone, + a w-sweep, on the cached OOF. A consensus blend is *structurally* more robust (less reliant on one estimator that may be lucky on 4 public wells) → better for the PRIVATE LB. Kernel stays at the banked 10.105 config (depth7, 195-feat, pf_ancc blend w=0.10) until a principled blend improvement is OOF-validated.

## ⚠️ CV NOISE FINDING (2026-05-30) — fine-grained deltas are NOT trustworthy

Same xgb config (seed 42, 212 divergence feat), **three runs: 10.48 (v15 GPU) / 10.63 (v18 GPU) / 10.77 (v20 CPU)** — a **0.29 OOF spread** (GPU early-stopping nondeterminism + CPU/GPU backend differences). Noise concentrated in folds 1 & 3 (10.28–10.86, 11.21–11.91).

**Consequences — all recent fine deltas are within noise, NOT confirmed:**
- divergence "+0.14" (v13c 10.62 sits INSIDE the divergence range 10.48–10.77) → **not a real win**
- slopes "+0.28 worse" → vs the divergence *mean* (~10.6) it's ~+0.15 → within noise, not clearly a dead end
- ensemble "+0.04" → noise

**Only the coarse result is real**: ported estimator suite ~10.6 vs 15.4 constant-ish plateau (>4 ft, far above noise). **Methodology fix**: use **multi-seed averaging** (mean over ≥5 seeds) for any feature/model A/B; single-run deltas < ~0.3 are meaningless. LB is the only single-shot trustworthy signal.

**5-seed divergence A/B VERDICT (offline CPU, 2026-05-31)** — divergence is **NEUTRAL**:
- div mean **10.804** (std 0.043) vs nodiv mean **10.814** (std 0.074) → **Δ 0.01, far below std**.
- ⚠️ At 3 seeds nodiv "led" by 0.04; seeds 4–5 FLIPPED it. **3 seeds was still noise** — need ≥5.
- So divergence/slopes/ensemble gains were ALL noise. **Honest best = single-xgb 198-feat, LB 10.538.** Divergence kept in code (neutral, `--no-divergence` to drop). Next real lever must be >noise: full ensemble (XGB+CB+HGB NNLS) + Optuna (ref's path to 9.85/8.905) — GPU-bound; test HGB diversity locally meanwhile.
- Note: CPU OOF (~10.81) runs ~0.25 above GPU (~10.55); A/B valid (both CPU).

**HGB ensemble member (offline CPU, 2026-05-31)** — promising, awaiting multi-seed:
- xgb+HGB NNLS, **seed 1**: xgb 10.79 (w 0.74) + HGB 10.93 (w **0.29**) → blend **10.75**. Blend beats best member; **NNLS gives HGB a real 0.29 weight → HGB is genuinely diverse** (different algo). But ~0.04 < noise std → need multi-seed mean (3-seed loop running, ~37min/run; HGB ~10min/fold CPU).
- **3-seed paired VERDICT (CPU)**: xgb+HGB NNLS blend beats xgb-alone on **all 3 seeds** (paired Δ −0.040/−0.024/−0.069, **mean −0.044**); HGB weight stable ~0.30. **Real (small) gain** — paired comparison controls seed noise, consistent (unlike divergence A/B). HGB is a useful diverse member.
- **xgb regularization probe (depth5/mcw20), 5-seed paired vs depth7**: mean Δ **−0.031** (4/5 wins, but seed 4 flipped +0.072; ~1 SEM from zero) → **MARGINAL/inconclusive**. 3-seed had said −0.041 — overstated (noise again). NOT adopting; keep depth7 (LB-10.538). Optuna will tune depth on GPU. (Correction: ref's depth-5 was CatBoost R7, not xgb — no xgb precedent.)
- **Lesson reinforced**: single tweaks here are ~0.03–0.05, below/at the ~0.05–0.07 noise even at 5 seeds. The reference's gains come from the FULL STACK compounding, not any one tweak. **Local CPU has hit diminishing returns** — every remaining single-lever probe is noise-edge.
- **HGB downgraded (audit J7)**: +0.044 is 3-seed paired, t=3.36 but **95% CI [−0.101, +0.012] includes zero** — *promising, NOT confirmed*. Same magnitude as the divergence "+0.04" that flipped at 5 seeds; needs ≥5-seed confirmation before anchoring the stack.
- **v22 caveat**: it is **single-seed** (seed 42) — its OOF + NNLS weights are un-denoised; a single-seed ensemble OOF is NOT safely comparable to the LB-10.538 baseline (run-to-run swing ~0.29). **Do not submit on v22 alone**; multi-seed required.
- **Kernel provenance (audit J8 — RESOLVED 2026-05-31)**: the LB-10.538 kernel was **198-feat (commit `9464a76`, pre-divergence)**. The v24 kernel now **drops div_* in `_xy`** (→ ~195 feat, matching the anchor's no-divergence stance; minor 195-vs-198 drift from intervening feature commits is the only residual confound) so the PF-blend is the sole headline change. `inference_kernel.py` is generated by `build_kernel.sh` from the committed `rogii_features.py` + `_inference_main.py` → reconstructible from the recorded git_sha. Caveat unchanged: kernel re-runs are CPU + `xgboost` unpinned → expect ±0.15-0.3 ft re-run variance.
- **REVISED priority (audit J6/J9)**: post-processing (drift-attenuation + SG, Optuna on cached OOF, GPU-free) is the highest-EV next step and is BEFORE the ensemble in the priority order — see Reference ladder "CORRECTED next-step ladder". GPU ensemble (v22) may run on idle GPUs but is priority-4 and must be multi-seed before any submit.
