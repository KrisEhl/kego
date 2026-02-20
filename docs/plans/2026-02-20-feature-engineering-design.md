# Design: Domain-Driven Feature Engineering for S6E2

**Date**: 2026-02-20
**Goal**: Improve leaderboard AUC (currently 0.95372) by generating ~125 domain-informed feature candidates and systematically evaluating them with a clean-slate forward selection approach.

## Problem Statement

Current feature engineering is ad-hoc: 22 features were created from intuition, and ablation showed 14 of them are harmful to trees. The ablation-pruned set (21 features) only marginally beats raw (13 features) for LightGBM (0.95122 vs 0.95097). We need a systematic, domain-informed approach that questions every feature — including the 13 raw ones.

## Approach

### Clean-Slate Evaluation

No feature is guaranteed a spot. The script generates ~125 candidates (13 raw + ~112 engineered/encoded) and runs forward selection from zero — every feature must earn its place.

### Three Evaluation Configurations

1. **Native categoricals + candidates** — LightGBM native categorical handling on, all candidates available
2. **No native categoricals** — Categoricals only appear through numeric encodings (TE, GLMM, WoE, etc.)
3. **Full clean slate** — Forward selection starts empty, adds one feature at a time from the entire pool

### Evaluation Methodology

- **Data**: Train (56K) + holdout (14K) stratified split, original UCI data included
- **Model**: LightGBM 5-fold CV, 3-seed averaging (seeds 42/123/777)
- **Metrics**: AUC on holdout, averaged across seeds
- **Phases**:
  1. Generate all ~125 candidates
  2. Permutation importance ranking on full candidate set
  3. Drop-one-at-a-time ablation (identify harmful features)
  4. Forward selection from empty (greedy add, best-first)
  5. Compare: new optimal set vs ablation-pruned (21) vs raw (13)

## Deliverables

1. **`notebooks/playground/research_features.py`** — standalone evaluation script
2. **`notebooks/playground/FEATURES.md`** — feature catalog with formulas, rationale, and references

## Script Architecture

```
research_features.py
├── _engineer_static_features(df)           # ~55 features computed once
├── _engineer_fold_features(X_tr, y_tr, X_val)  # ~57 features per fold
├── _run_lightgbm_cv(X, y, features, seeds)     # 5-fold CV, 3 seeds
├── evaluate_all_features(seeds)            # ablation + forward selection
├── compare_configurations()                # native cats vs no cats vs clean slate
└── print_report()                          # ranked tables
```

## Feature Candidates (~125 total)

### Raw Features (13)

Age, Sex, Chest pain type, BP, Cholesterol, FBS over 120, EKG results, Max HR, Exercise angina, ST depression, Slope of ST, Number of vessels fluro, Thallium

### Static Features (~55)

#### Clinical Scores (4)

| Feature | Formula | Source |
|---------|---------|--------|
| `framingham_partial` | Sex-specific log-linear: men `3.06*ln(Age) + 1.12*ln(Chol) + 1.93*ln(BP) + 0.57*FBS`; women `2.33*ln(Age) + 1.21*ln(Chol) + 2.76*ln(BP) + 0.69*FBS` | Framingham 2008 |
| `heart_score_partial` | Age points (0/<45, 1/45-64, 2/65+) + EKG points + risk factor points (FBS + HTN) | HEART Score (Six et al. 2008) |
| `duke_treadmill_approx` | `est_exercise_min - 5*ST_dep - 4*angina_index` where exercise time estimated from Max HR via Bruce protocol | Duke Treadmill Score (Mark et al. 1991) |
| `timi_partial` | `(Age>=65) + FBS + (BP>140) + (ST_dep>0)` | TIMI Risk Score (Antman et al. 2000) |

#### Exercise Physiology (10)

| Feature | Formula |
|---------|---------|
| `chronotropic_incompetence` | `(Max HR < 0.80 * (220 - Age)).astype(int)` |
| `chronotropic_response_index` | `(Max HR - (60+0.2*BP)) / (220-Age - (60+0.2*BP))` |
| `hr_reserve_pct_tanaka` | `Max HR / (208 - 0.7*Age)` |
| `hr_reserve_absolute` | `(220 - Age) - Max HR` |
| `st_hr_index` | `ST_dep * 1000 / Max HR` |
| `st_hr_hysteresis` | `ST_dep / (Max HR - (60+0.2*BP))` |
| `rate_pressure_product` | `Max HR * BP` |
| `supply_demand_mismatch` | `(Max HR * BP / 10000) * ST_dep * (1 + Exercise angina)` |
| `estimated_mets` | `0.05 * Max HR - 1.0` |
| `poor_exercise_capacity` | `(estimated_mets < 5).astype(int)` |

#### Clinical Categories (6)

| Feature | Bins |
|---------|------|
| `age_risk_category` | 0/<45, 1/45-54, 2/55-64, 3/65+ |
| `age_sex_risk` | Men >=45, Women >=55 (Framingham thresholds) |
| `bp_category` | AHA: 0/Normal, 1/Elevated, 2/HTN1, 3/HTN2 |
| `cholesterol_category` | ATP III: 0/Desirable, 1/Borderline, 2/High |
| `hr_achievement_category` | 0/<60%, 1/<80%, 2/80-85%, 3/>85% of predicted |
| `st_depression_category` | 0/None, 1/Mild, 2/Moderate, 3/Severe |

#### Domain Interactions (10)

| Feature | Formula |
|---------|---------|
| `diabetes_hypertension` | `FBS * (BP > 140)` |
| `multivessel_ischemia` | `(vessels >= 2) * (ST_dep + Exercise angina)` |
| `anatomic_severity` | `vessels * (Thallium >= 6)` |
| `exercise_test_positive` | `(ST_dep >= 1) + (Slope >= 2) + Exercise angina` |
| `age_sex_interaction` | `Age * Sex` |
| `triple_threat` | `(ChestPain==4) * (Thallium>=6) * (vessels>=1)` |
| `cholesterol_age_risk` | `Cholesterol * (Age > 50)` |
| `cardiac_efficiency` | `Max HR / BP` |
| `rest_exercise_concordance` | `(EKG>=1) * ((ST_dep>0) + Exercise angina)` |
| `ekg_with_hypertension` | `(EKG>=1) * (BP>140)` |

#### Composites (5)

| Feature | Formula |
|---------|---------|
| `ischemic_burden` | `ST_dep * slope_weight + 2*angina + 3*(Thallium>=6)` |
| `risk_factor_count` | `FBS + (BP>140) + (Chol>240) + (Age>55) + Sex` |
| `thallium_severity` | Ordinal severity: normal=0, fixed=2, reversible=3, plus angina and ST flags |
| `modified_duke` | Duke with refined angina index (limiting vs non-limiting) |
| `o2_supply_demand` | `(BP*MaxHR/10000) * (1 - supply_proxy)` |

#### Existing Engineered (22, re-tested)

All features from `_engineer_features()`: `thallium_x_*`, `chestpain_x_*`, `vessels_x_thallium`, `top4_sum`, `abnormal_count`, `risk_score`, `maxhr_per_age`, `hr_reserve_pct`, `age_x_*`, `heart_load`, `*_dev_sex`, `signal_conflict`, `thallium_abnormal`

#### Competition Tricks (~8)

| Feature | Method |
|---------|--------|
| `{cat}_freq` (x4) | Frequency encoding for Thallium, Chest pain, EKG, Slope |
| `age_squared`, `cholesterol_squared`, `st_depression_squared` | Polynomial terms |
| `risk_logodds` | Log-odds transform of risk_score |

### Fold-Aware Features (~57)

These are computed per CV fold to prevent target leakage.

#### Advanced Encodings (~30)

For each of 5 categoricals (Thallium, Chest pain type, Slope of ST, EKG results, Number of vessels fluro):
- Target encoding (existing, re-tested)
- GLMM encoding (`category_encoders.GLMMEncoder`)
- James-Stein encoding (`category_encoders.JamesSteinEncoder`)
- Leave-one-out encoding (`category_encoders.LeaveOneOutEncoder`)
- WoE via optimal binning (`optbinning.OptimalBinning`)
- Bayesian M-estimate encoding

For 4 categorical pairs: target-encoded pair interactions
- (Thallium, Chest pain), (Thallium, Slope), (Chest pain, Exercise angina), (EKG, Slope)

#### Residual Features (3)

- `maxhr_residual`: residual of `Max HR ~ Age`
- `chol_residual`: residual of `Cholesterol ~ Age + Sex`
- `bp_residual`: residual of `BP ~ Age + Sex`

#### Dimensionality Reduction (5)

- PCA first 3 components on standardized continuous features
- Supervised UMAP 2 coordinates

#### Anomaly/Distance Scores (5)

- Mahalanobis distance from positive/negative class centroids (2 + ratio)
- Isolation Forest anomaly score
- LOF score (subsampled if needed for speed)

#### KNN Features (4)

- Mean distance to 10 nearest positive cases
- Mean distance to 10 nearest negative cases
- Distance ratio (pos/neg)
- Neighborhood target rate (20-NN)

#### Meta-Model OOF (5)

- Logistic regression OOF probability
- Naive Bayes OOF probability
- KNN (K=50) OOF probability
- Model disagreement (|LR - NB|)
- Decision tree leaf ID (target-encoded)

#### Splines & Transforms (~5)

- Spline basis functions for Age (4 cols) and Max HR (4 cols)
- Yeo-Johnson transforms for ST depression, Cholesterol, BP

## Dependencies

Additional packages needed beyond current environment:
- `category_encoders` — GLMM, James-Stein, LOO encoders
- `optbinning` — supervised optimal binning with WoE
- `umap-learn` — supervised UMAP coordinates

## Output

The script prints:
1. **Baseline AUCs**: raw (13 features), current ablation-pruned (21), current all (35)
2. **Per-feature permutation importance** ranking on the full ~125 feature set
3. **Ablation table**: drop-one-at-a-time delta for all features
4. **Forward selection curve**: greedy add from empty, showing AUC at each step
5. **Three configuration comparison**: native cats vs no cats vs clean slate
6. **Recommended feature set**: the optimal set from forward selection with feature list

## Success Criteria

- New feature set achieves higher holdout AUC than current ablation-pruned (0.95122)
- At least some research features survive forward selection (domain knowledge adds value)
- Clear report enables informed decisions about which features to promote to the full pipeline
