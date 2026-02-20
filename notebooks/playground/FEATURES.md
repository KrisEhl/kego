# Feature Engineering Reference — S6E2 Heart Disease

Catalog of all feature candidates for the Playground Series S6E2 binary classification task. Each feature includes its formula, clinical or statistical rationale, and source references.

## Raw Features (13)

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numeric | Patient age in years |
| Sex | Binary | 1=male, 0=female |
| Chest pain type | Categorical (1-4) | 1=typical angina, 2=atypical angina, 3=non-anginal, 4=asymptomatic |
| BP | Numeric | Resting systolic blood pressure (mmHg) |
| Cholesterol | Numeric | Serum total cholesterol (mg/dL) |
| FBS over 120 | Binary | Fasting blood sugar > 120 mg/dL (proxy for diabetes) |
| EKG results | Categorical (0-2) | 0=normal, 1=ST-T wave abnormality, 2=probable/definite LVH |
| Max HR | Numeric | Maximum heart rate achieved during exercise test |
| Exercise angina | Binary | Exercise-induced angina (1=yes, 0=no) |
| ST depression | Numeric | ST depression induced by exercise relative to rest (mm) |
| Slope of ST | Categorical (1-3) | 1=upsloping, 2=flat, 3=downsloping |
| Number of vessels fluro | Ordinal (0-3) | Number of major vessels colored by fluoroscopy |
| Thallium | Categorical (3/6/7) | 3=normal, 6=fixed defect, 7=reversible defect |

## Clinical Scoring Systems

### Framingham Partial Risk Score

Sex-specific log-linear risk score using available features. Missing HDL cholesterol and smoking status from the full Framingham model.

```python
framingham_partial = np.where(
    Sex == 1,  # male
    3.06*ln(Age) + 1.12*ln(Cholesterol) + 1.93*ln(BP) + 0.57*FBS,
    2.33*ln(Age) + 1.21*ln(Cholesterol) + 2.76*ln(BP) + 0.69*FBS
)
```

**Rationale**: The Framingham Risk Score is the most widely used cardiovascular risk prediction tool, developed from 30+ years of prospective cohort data. Even with missing predictors, the sex-specific coefficients encode validated relative risk weights that linear/neural models cannot learn from raw features alone.

**References**:
- D'Agostino RB Sr, et al. "General cardiovascular risk profile for use in primary care: the Framingham Heart Study." *Circulation*. 2008;117(6):743-753. [doi:10.1161/CIRCULATIONAHA.107.699579](https://doi.org/10.1161/CIRCULATIONAHA.107.699579)
- Wilson PWF, et al. "Prediction of coronary heart disease using risk factor categories." *Circulation*. 1998;97(18):1837-1847. [doi:10.1161/01.CIR.97.18.1837](https://doi.org/10.1161/01.CIR.97.18.1837)

### HEART Score (Partial)

Emergency department chest pain triage score. We can compute 3 of 5 components (History and Troponin unavailable).

```python
age_pts = 0 if Age<45 else 1 if Age<65 else 2
ekg_pts = 0 if EKG==0 else 1 if EKG==1 else 2
risk_pts = min(FBS + (BP>140), 2)
heart_score_partial = age_pts + ekg_pts + risk_pts
```

**Rationale**: Validated across >12,000 patients. Score 0-3 = 1.7% MACE rate, 4-6 = 16.6%, 7-10 = 50.1%.

**References**:
- Six AJ, et al. "Chest pain in the emergency room: value of the HEART score." *Netherlands Heart Journal*. 2008;16(6):191-196. [doi:10.1007/BF03086144](https://doi.org/10.1007/BF03086144)
- Backus BE, et al. "A prospective validation of the HEART score for chest pain patients at the emergency department." *International Journal of Cardiology*. 2013;168(3):2153-2158. [doi:10.1016/j.ijcard.2013.01.255](https://doi.org/10.1016/j.ijcard.2013.01.255)

### Duke Treadmill Score (Approximated)

The Duke Treadmill Score (DTS) is the most validated exercise test prognostic score. Since exercise duration is unavailable, we estimate it from Max HR using Bruce protocol relationships.

```python
est_exercise_min = (Max HR - 80) / 8  # rough Bruce protocol HR-to-time
angina_index = 2 * Exercise angina  # 0=none, 2=exercise-limiting
duke_treadmill_approx = est_exercise_min - 5*ST depression - 4*angina_index
```

Modified variant uses refined angina grading:
```python
angina_index = 0 if no angina else 1 if non-limiting else 2 if limiting
# limiting = Exercise angina + Chest pain type 4 (asymptomatic/severe)
```

**Rationale**: DTS ranges from -25 (highest risk, 5-year mortality 35%) to +15 (lowest risk, 5-year mortality 0.25%). Validated in >30,000 patients across multiple centers.

**References**:
- Mark DB, et al. "Prognostic value of a treadmill exercise score in outpatients with suspected coronary artery disease." *New England Journal of Medicine*. 1991;325(12):849-853. [doi:10.1056/NEJM199109193251204](https://doi.org/10.1056/NEJM199109193251204)
- Shaw LJ, et al. "Prognostic Value of the Duke Treadmill Score." *American Heart Journal*. 2005;150(5):1074. [doi:10.1016/j.ahj.2005.08.023](https://doi.org/10.1016/j.ahj.2005.08.023)

### TIMI Risk Score (Partial)

The TIMI (Thrombolysis In Myocardial Infarction) score for UA/NSTEMI. We can compute 4 of 7 components.

```python
timi_partial = (Age>=65) + FBS + (BP>140) + (ST depression>0)
```

**References**:
- Antman EM, et al. "The TIMI risk score for unstable angina/non-ST elevation MI." *JAMA*. 2000;284(7):835-842. [doi:10.1001/jama.284.7.835](https://doi.org/10.1001/jama.284.7.835)

## Exercise Physiology Features

### Chronotropic Incompetence

Failure to reach 80% of age-predicted maximum heart rate is an independent predictor of cardiovascular mortality and major adverse cardiac events.

```python
chronotropic_incompetence = (Max HR < 0.80 * (220 - Age)).astype(int)
```

**References**:
- Brubaker PH, Kitzman DW. "Chronotropic incompetence: causes, consequences, and management." *Circulation*. 2011;123(9):1010-1020. [doi:10.1161/CIRCULATIONAHA.110.940577](https://doi.org/10.1161/CIRCULATIONAHA.110.940577)
- Lauer MS, et al. "Impaired chronotropic response to exercise stress testing as a predictor of mortality." *JAMA*. 1999;281(6):524-529. [doi:10.1001/jama.281.6.524](https://doi.org/10.1001/jama.281.6.524)

### Chronotropic Response Index (CRI)

Refined version using estimated resting heart rate from blood pressure.

```python
resting_hr_est = 60 + 0.2 * BP  # population estimate
cri = (Max HR - resting_hr_est) / ((220 - Age) - resting_hr_est)
```

**Rationale**: CRI < 0.80 indicates chronotropic incompetence. More accurate than simple HR/predicted ratio because it accounts for resting heart rate and the actual heart rate reserve used.

**References**:
- Wilkoff BL, Miller RE. "Exercise testing for chronotropic assessment." *Cardiology Clinics*. 1992;10(4):705-717. [PMID: 1423740](https://pubmed.ncbi.nlm.nih.gov/1423740/)

### Heart Rate Reserve (Tanaka Formula)

The Tanaka formula (208 - 0.7 * Age) is more accurate than 220 - Age, especially for ages >40.

```python
hr_reserve_pct_tanaka = Max HR / (208 - 0.7 * Age)
hr_reserve_absolute = (220 - Age) - Max HR
```

**References**:
- Tanaka H, et al. "Age-predicted maximal heart rate revisited." *Journal of the American College of Cardiology*. 2001;37(1):153-156. [doi:10.1016/S0735-1097(00)01054-8](https://doi.org/10.1016/S0735-1097(00)01054-8)

### ST/HR Index

Normalizes ST depression by heart rate achieved. Improved exercise test sensitivity from 57% to 91% in the original study.

```python
st_hr_index = (ST depression * 1000) / Max HR  # convert mm to µV
```

**References**:
- Kligfield P, et al. "Heart rate adjustment of ST segment depression for improved detection of coronary artery disease." *Circulation*. 1989;79(2):245-255. [doi:10.1161/01.CIR.79.2.245](https://doi.org/10.1161/01.CIR.79.2.245)
- Okin PM, et al. "Heart rate adjustment of ST-segment depression and performance of the exercise electrocardiogram: a critical evaluation." *Journal of the American College of Cardiology*. 1995;25(7):1726-1735. [doi:10.1016/0735-1097(95)00085-I](https://doi.org/10.1016/0735-1097(95)00085-I)

### ST/HR Hysteresis Index

Normalizes ST depression by the actual heart rate increase achieved (not absolute HR).

```python
st_hr_hysteresis = ST depression / (Max HR - (60 + 0.2*BP))
```

**References**:
- Lehtinen R, et al. "Accurate detection of coronary artery disease by integrated analysis of the ST-segment depression/heart rate patterns during the exercise and recovery phases of the exercise electrocardiography test." *American Journal of Cardiology*. 1996;78(9):1002-1006. [doi:10.1016/S0002-9149(96)00525-1](https://doi.org/10.1016/S0002-9149(96)00525-1)

### Rate-Pressure Product (RPP)

The gold-standard non-invasive estimate of myocardial oxygen consumption.

```python
rate_pressure_product = Max HR * BP
rpp_normalized = (rate_pressure_product - 10000) / 30000
```

**Rationale**: Normal range at peak exercise: 25,000-40,000. Values <25,000 at peak exercise suggest inadequate effort or severe disease. RPP captures a multiplicative physiological relationship that trees need many splits to approximate.

**References**:
- Gobel FL, et al. "The rate-pressure product as an index of myocardial oxygen consumption during exercise in patients with angina pectoris." *Circulation*. 1978;57(3):549-556. [doi:10.1161/01.CIR.57.3.549](https://doi.org/10.1161/01.CIR.57.3.549)

### Supply-Demand Mismatch

Combines oxygen demand proxy (RPP) with ischemia markers.

```python
supply_demand_mismatch = (Max HR * BP / 10000) * ST depression * (1 + Exercise angina)
```

**Rationale**: High demand (high RPP) combined with evidence of ischemia (ST depression + angina) = supply-demand mismatch, the fundamental mechanism of coronary artery disease symptoms.

### Estimated METs

Metabolic equivalents estimated from heart rate. <5 METs = poor exercise capacity (high mortality risk).

```python
estimated_mets = 0.05 * Max HR - 1.0
poor_exercise_capacity = (estimated_mets < 5).astype(int)
```

**References**:
- Saito M, et al. "Estimating the metabolic equivalent of one MET from resting heart rate." *International Journal of Environmental Research and Public Health*. 2018;15(11):2449. [doi:10.3390/ijerph15112449](https://doi.org/10.3390/ijerph15112449)
- Myers J, et al. "Exercise capacity and mortality among men referred for exercise testing." *New England Journal of Medicine*. 2002;346(11):793-801. [doi:10.1056/NEJMoa011858](https://doi.org/10.1056/NEJMoa011858)

### Ischemic Burden Score

Weighted ischemia composite that accounts for ST depression slope type — mirrors how cardiologists actually read exercise tests.

```python
slope_weight = 2 if Slope==2 else 1 if Slope==1 else 0  # down > flat > up
ischemic_burden = ST depression * slope_weight + 2*Exercise angina + 3*(Thallium>=6)
```

**Rationale**: Horizontal/downsloping ST depression is more specific for ischemia than upsloping. Weighting by slope type is standard clinical practice.

**References**:
- Froelicher VF, Myers J. *Exercise and the Heart*. 5th ed. Saunders; 2006.
- Ashley EA, et al. "An evidence-based review of the resting electrocardiogram as a screening technique for heart disease." *Progress in Cardiovascular Diseases*. 2001;44(1):55-67. [doi:10.1053/pcad.2001.24683](https://doi.org/10.1053/pcad.2001.24683)

## Clinical Category Features

### Blood Pressure Categories (AHA/ACC 2017)

```python
bp_category = 0 if BP<120 else 1 if BP<130 else 2 if BP<140 else 3
# 0=Normal, 1=Elevated, 2=HTN Stage 1, 3=HTN Stage 2
```

**References**:
- Whelton PK, et al. "2017 ACC/AHA Guideline for the Prevention, Detection, Evaluation, and Management of High Blood Pressure in Adults." *Hypertension*. 2018;71(6):e13-e115. [doi:10.1161/HYP.0000000000000065](https://doi.org/10.1161/HYP.0000000000000065)

### Cholesterol Categories (ATP III)

```python
cholesterol_category = 0 if Cholesterol<200 else 1 if Cholesterol<240 else 2
# 0=Desirable, 1=Borderline High, 2=High
```

**References**:
- National Cholesterol Education Program. "Third Report of the NCEP Expert Panel on Detection, Evaluation, and Treatment of High Blood Cholesterol in Adults (ATP III)." *NIH Publication No. 02-5215*. 2002. [NHLBI](https://www.nhlbi.nih.gov/files/docs/guidelines/atp3xsum.pdf)

### Age Risk Categories

```python
age_risk_category = 0 if Age<45 else 1 if Age<55 else 2 if Age<65 else 3
age_sex_risk = (Age >= 45 if Sex==1 else Age >= 55)  # Framingham thresholds
```

**References**:
- Jousilahti P, et al. "Sex, age, cardiovascular risk factors, and coronary heart disease." *Circulation*. 1999;99(9):1165-1172. [doi:10.1161/01.CIR.99.9.1165](https://doi.org/10.1161/01.CIR.99.9.1165)

### Heart Rate Achievement and ST Depression Categories

```python
hr_achievement = 0 if pct<0.60 else 1 if pct<0.80 else 2 if pct<0.85 else 3
st_depression_category = 0 if ST==0 else 1 if ST<=1 else 2 if ST<=2 else 3
```

## Domain Interaction Features

### Diabetes + Hypertension Comorbidity

```python
diabetes_hypertension = FBS * (BP > 140)
```

**Rationale**: Diabetes + hypertension doubles cardiovascular risk beyond either alone — recognized as a "deadly duo" in cardiology.

**References**:
- Stamler J, et al. "Diabetes, other risk factors, and 12-yr cardiovascular mortality for men screened in the MRFIT." *Diabetes Care*. 2993;16(2):434-444. [doi:10.2337/diacare.16.2.434](https://doi.org/10.2337/diacare.16.2.434)

### Multi-Vessel Disease + Ischemia

```python
multivessel_ischemia = (vessels >= 2) * (ST depression + Exercise angina)
```

**References**:
- Hachamovitch R, et al. "Comparison of the short-term survival benefit associated with revascularization compared with medical therapy in patients with no prior coronary artery disease undergoing stress myocardial perfusion single photon emission computed tomography." *Circulation*. 2003;107(23):2900-2907. [doi:10.1161/01.CIR.0000072790.23090.41](https://doi.org/10.1161/01.CIR.0000072790.23090.41)

### Exercise Test Positive (Clinical Definition)

```python
exercise_test_positive = (ST depression >= 1) + (Slope >= 2) + Exercise angina
```

**Rationale**: A "positive" exercise stress test in clinical terms requires >= 1mm horizontal/downsloping ST depression. This codifies the cardiologist's interpretation.

**References**:
- Fletcher GF, et al. "Exercise standards for testing and training: a scientific statement from the AHA." *Circulation*. 2013;128(8):873-934. [doi:10.1161/CIR.0b013e31829b5b44](https://doi.org/10.1161/CIR.0b013e31829b5b44)

### Triple Threat

```python
triple_threat = (Chest pain type == 4) * (Thallium >= 6) * (vessels >= 1)
```

**Rationale**: When anatomy (vessels), perfusion (thallium), and symptoms all indicate disease, probability is near-certain. Captures a 3-way interaction that trees need 3+ splits to find.

### Thallium Defect Severity Score

```python
thal_severity = (0 if Thal==3 else 2 if Thal==6 else 3 if Thal==7 else 1)
              + (1 if Exercise angina else 0) + (1 if ST depression > 2 else 0)
```

**Rationale**: Reversible defects (7) indicate active ischemia (treatable); fixed defects (6) indicate infarction. Weighting reversible > fixed matches clinical significance.

**References**:
- Berman DS, et al. "Roles of nuclear cardiology, cardiac computed tomography, and cardiac magnetic resonance: assessment of patients with suspected coronary artery disease." *Journal of Nuclear Medicine*. 2006;47(1):74-82. [PMID: 16391190](https://pubmed.ncbi.nlm.nih.gov/16391190/)

### Other Interactions

```python
cardiac_efficiency = Max HR / BP                    # cardiac workload efficiency
age_sex_interaction = Age * Sex                     # sex modifies age-risk
cholesterol_age_risk = Cholesterol * (Age > 50)     # cumulative plaque burden
rest_exercise_concordance = (EKG>=1) * ((ST>0) + angina)  # concordant findings
ekg_with_hypertension = (EKG>=1) * (BP>140)         # LVH + HTN combo
anatomic_severity = vessels * (Thallium>=6)          # anatomy + perfusion
risk_factor_count = FBS + (BP>140) + (Chol>240) + (Age>55) + Sex
```

## Advanced Encoding Features (Per-Fold)

All encoding features below must be computed per CV fold (fit on fold train, transform fold val) to prevent target leakage.

### GLMM Encoding

Mixed-effects logistic regression with categorical features as random effects. Automatic shrinkage regularizes rare categories toward the grand mean.

**Apply to**: Thallium, Chest pain type, Slope of ST, EKG results, Number of vessels fluro

**References**:
- `category_encoders` library. [Documentation](https://contrib.scikit-learn.org/category_encoders/glmm.html)
- McCulloch CE, Searle SR. *Generalized, Linear, and Mixed Models*. 2nd ed. Wiley; 2008.

### James-Stein Encoding

Shrinks category means toward the grand mean using the James-Stein estimator: `encoded = grand_mean + B * (category_mean - grand_mean)`. Dominates MLE for 3+ categories (Stein's paradox).

**References**:
- James W, Stein C. "Estimation with quadratic loss." *Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability*. 1961;1:361-379.
- `category_encoders.JamesSteinEncoder`. [Documentation](https://contrib.scikit-learn.org/category_encoders/jamesstein.html)

### Leave-One-Out Encoding

For each sample, encode as the mean target of all *other* samples with the same category.

**References**:
- `category_encoders.LeaveOneOutEncoder`. [Documentation](https://contrib.scikit-learn.org/category_encoders/leaveoneout.html)

### Weight of Evidence (Optimal Binning)

Uses mathematical programming to find optimal bin boundaries that maximize Information Value with respect to the target, with monotonicity constraints.

**Apply to**: Continuous features (Age, BP, Cholesterol, Max HR, ST depression) and ordinal categoricals

**References**:
- Navas-Palencia G. "Optimal binning: mathematical programming formulation." [OptBinning documentation](https://gnpalencia.org/optbinning/tutorials/tutorial_binary.html)
- Siddiqi N. *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Wiley; 2005.

### Target-Encoded Categorical Pair Interactions

Concatenate pairs of categoricals, then target-encode the combined key. Captures specific target rates for each combination.

**Pairs**: (Thallium, Chest pain type), (Thallium, Slope of ST), (Chest pain type, Exercise angina), (EKG results, Slope of ST)

## Statistical / Geometric Features (Per-Fold)

### Mahalanobis Distance from Class Centroids

```python
mahal_pos = sqrt((x - mu_pos)^T @ Sigma_pos_inv @ (x - mu_pos))
mahal_neg = sqrt((x - mu_neg)^T @ Sigma_neg_inv @ (x - mu_neg))
mahal_ratio = mahal_neg / (mahal_pos + epsilon)
```

**Rationale**: Accounts for feature correlations and scales. Captures class-conditional geometric structure.

**References**:
- Mahalanobis PC. "On the generalized distance in statistics." *Proceedings of the National Institute of Sciences of India*. 1936;2:49-55.
- De Maesschalck R, et al. "The Mahalanobis distance." *Chemometrics and Intelligent Laboratory Systems*. 2000;50(1):1-18. [doi:10.1016/S0169-7439(99)00047-7](https://doi.org/10.1016/S0169-7439(99)00047-7)

### Isolation Forest Anomaly Score

```python
anomaly_score = IsolationForest(n_estimators=100).decision_function(X)
```

**References**:
- Liu FT, et al. "Isolation forest." *IEEE International Conference on Data Mining*. 2008:413-422. [doi:10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17)

### PCA Components

First 3 principal components on standardized continuous features.

### Supervised UMAP Coordinates

First 2 coordinates from UMAP with label supervision.

**References**:
- McInnes L, et al. "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." *arXiv:1802.03426*. 2018. [arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)

### KNN-Based Features

```python
dist_k_pos = mean distance to 10 nearest positive cases
dist_k_neg = mean distance to 10 nearest negative cases
dist_ratio = dist_k_pos / (dist_k_neg + eps)
knn_target_rate = mean target of 20 nearest neighbors
```

**References**:
- Fix E, Hodges JL. "Discriminatory analysis: nonparametric discrimination, consistency properties." *USAF School of Aviation Medicine*. 1951. Technical Report 4, Project 21-49-004.

## Meta-Model Features (Per-Fold)

### OOF Predictions as Features

```python
lr_oof_prob = LogisticRegression OOF predicted probability
nb_oof_prob = GaussianNB OOF predicted probability
knn_oof_prob = KNeighborsClassifier(n_neighbors=50) OOF predicted probability
model_disagreement = abs(lr_oof_prob - nb_oof_prob)
dt_leaf_te = target-encoded DecisionTree(max_depth=5) leaf IDs
```

**Rationale**: Each model captures different inductive biases. LR provides a linear view, NB assumes independence, KNN captures local density. Their disagreement highlights samples near decision boundaries.

**References**:
- He X, et al. "Practical lessons from predicting clicks on ads at Facebook." *ADKDD*. 2014:1-9. [doi:10.1145/2648584.2648589](https://doi.org/10.1145/2648584.2648589) (decision tree leaf IDs as features)
- Wolpert DH. "Stacked generalization." *Neural Networks*. 1992;5(2):241-259. [doi:10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080(05)80023-1) (stacking/OOF methodology)

## Transform Features

### Residuals from Linear Models (Per-Fold)

```python
maxhr_residual = Max HR - LinearRegression().fit(Age).predict(Age)
chol_residual = Cholesterol - LinearRegression().fit([Age, Sex]).predict([Age, Sex])
bp_residual = BP - LinearRegression().fit([Age, Sex]).predict([Age, Sex])
```

**Rationale**: Removes expected age/sex variation, leaving only the clinically meaningful deviation. More principled than the existing `*_dev_sex` features because it adjusts for multiple confounders simultaneously.

### Spline Basis Functions (Per-Fold)

```python
SplineTransformer(n_knots=5, degree=3).fit_transform(Age)   # 7 columns
SplineTransformer(n_knots=5, degree=3).fit_transform(Max HR)  # 7 columns
```

**Rationale**: Allows logistic regression to model non-linear effects without polynomial feature explosion. Especially powerful for age-risk and HR-risk curves that are known to be non-linear.

**References**:
- Perperoglou A, et al. "A review of spline function procedures in R." *BMC Medical Research Methodology*. 2019;19:46. [doi:10.1186/s12874-019-0666-3](https://doi.org/10.1186/s12874-019-0666-3)

### Yeo-Johnson Power Transforms (Per-Fold)

Applied to skewed features: ST depression, Cholesterol, BP.

**References**:
- Yeo I, Johnson RA. "A new family of power transformations to improve normality or symmetry." *Biometrika*. 2000;87(4):954-959. [doi:10.1093/biomet/87.4.954](https://doi.org/10.1093/biomet/87.4.954)

### Polynomial Features

```python
age_squared = Age ** 2
cholesterol_squared = Cholesterol ** 2
st_depression_squared = ST depression ** 2
```

**Rationale**: Risk increases non-linearly. Age-squared captures accelerating risk post-midlife. Only useful for linear models.

### Other Transforms

```python
risk_logodds = log(risk_prob / (1 - risk_prob))  # log-odds of risk_score
{col}_freq = frequency encoding for 4 categoricals
{col}_dev_agegroup = deviation from age-group mean (10-year bins)
quantile_dev_{col} = abs(percentile_rank - 0.5) * 2  # extremeness measure
```

## Competition Technique References

- Prokhorenkova L, et al. "CatBoost: unbiased boosting with categorical features." *NeurIPS*. 2018. [arXiv:1706.09516](https://arxiv.org/abs/1706.09516) (ordered target statistics)
- Pargent F, et al. "Regularized target encoding outperforms traditional methods in supervised machine learning with high cardinality features." *Computational Statistics*. 2022;37:2671-2692. [doi:10.1007/s00180-022-01207-6](https://doi.org/10.1007/s00180-022-01207-6)
- Hancock JT, Khoshgoftaar TM. "Survey on categorical data for neural networks." *Journal of Big Data*. 2020;7:28. [doi:10.1186/s40537-020-00305-w](https://doi.org/10.1186/s40537-020-00305-w)
- NVIDIA. "Kaggle Grandmasters Playbook: 7 Battle-Tested Modeling Techniques." [Developer Blog](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- Neptune.ai. "Tabular Data Binary Classification: Tips and Tricks from 5 Kaggle Competitions." [Blog](https://neptune.ai/blog/tabular-data-binary-classification-tips-and-tricks-from-5-kaggle-competitions)

## General Medical References

- Gibbons RJ, et al. "ACC/AHA 2002 guideline update for exercise testing." *Circulation*. 2002;106(14):1883-1892. [doi:10.1161/01.CIR.0000034670.06526.15](https://doi.org/10.1161/01.CIR.0000034670.06526.15)
- Froelicher VF, Myers J. *Exercise and the Heart*. 5th ed. Saunders/Elsevier; 2006.
- Detrano R, et al. "International application of a new probability algorithm for the diagnosis of coronary artery disease." *American Journal of Cardiology*. 1989;64(5):304-310. [doi:10.1016/0002-9149(89)90524-9](https://doi.org/10.1016/0002-9149(89)90524-9) (original UCI Heart Disease dataset)
