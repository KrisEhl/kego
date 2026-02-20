# Feature Engineering Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create `research_features.py` — a standalone script that generates ~125 domain-informed feature candidates and evaluates them with clean-slate forward selection using LightGBM.

**Architecture:** Single script with two feature generation phases (static + per-fold), followed by a 4-step evaluation pipeline (permutation importance → ablation → forward selection → comparison). Follows the exact patterns of `select_features.py` and `test_features_local.py`.

**Tech Stack:** Python 3.10+, LightGBM, scikit-learn, category_encoders, optbinning, umap-learn, pandas, numpy

**Design doc:** `docs/plans/2026-02-20-feature-engineering-design.md`
**Feature reference:** `notebooks/playground/FEATURES.md`

---

## Context for the Implementer

- This project is a Kaggle competition (Playground S6E2, Heart Disease binary classification, AUC metric)
- The script runs locally on Mac (no Ray cluster, no GPU) — CPU only
- Follow the patterns in `notebooks/playground/select_features.py` exactly — same data loading, same split, same LightGBM params, same multi-seed evaluation
- The key data files are in `data/playground/playground-series-s6e2/` (train.csv, Heart_Disease_Prediction.csv)
- Pre-commit hooks enforce: black, isort, flake8, mypy, autoflake — code must pass all of these
- Use `uv run python ...` to execute (never bare `python`)
- The FEATURES.md reference doc has formulas for every feature

## Existing Files to Reference

- `notebooks/playground/select_features.py` — PRIMARY TEMPLATE: data loading, split, LightGBM eval, ablation, forward selection patterns
- `notebooks/playground/test_features_local.py` — target encoding per fold pattern, feature engineering functions
- `notebooks/playground/FEATURES.md` — formulas for all ~125 features
- `notebooks/playground/train_s6e2_baseline.py` — full model configs (lines 639-698 for `_engineer_features()`)

---

### Task 1: Script scaffold with data loading and static features

**Files:**
- Create: `notebooks/playground/research_features.py`

**Step 1: Create the script with imports, constants, data loading, and all static feature generation**

The script should:
1. Have a module docstring explaining what it does and how to run it
2. Import the same libraries as `select_features.py` plus `category_encoders` and `optbinning`
3. Define the same constants: `DATA_DIR`, `TARGET`, `RAW_FEATURES`, `CAT_FEATURES`
4. Copy `_impute_cholesterol()` from `select_features.py` (identical pattern — can't import from baseline due to heavy deps)
5. Copy the existing `_engineer_features()` as `_engineer_existing_features()` — these 22 features are candidates too
6. Create `_engineer_research_features(df)` that generates all ~55 static research features from FEATURES.md:
   - Clinical scores: `framingham_partial`, `heart_score_partial`, `duke_treadmill_approx`, `timi_partial`, `modified_duke`
   - Exercise physiology: `chronotropic_incompetence`, `chronotropic_response_index`, `hr_reserve_pct_tanaka`, `hr_reserve_absolute`, `st_hr_index`, `st_hr_hysteresis`, `rate_pressure_product`, `rpp_normalized`, `supply_demand_mismatch`, `estimated_mets`, `poor_exercise_capacity`
   - Clinical categories: `age_risk_category`, `age_sex_risk`, `bp_category`, `cholesterol_category`, `hr_achievement_category`, `st_depression_category`
   - Domain interactions: `diabetes_hypertension`, `multivessel_ischemia`, `anatomic_severity`, `exercise_test_positive`, `age_sex_interaction`, `triple_threat`, `cholesterol_age_risk`, `cardiac_efficiency`, `rest_exercise_concordance`, `ekg_with_hypertension`
   - Composites: `ischemic_burden`, `risk_factor_count`, `thallium_severity`, `o2_supply_demand`
   - Competition tricks: `{cat}_freq` for 4 categoricals, `age_squared`, `cholesterol_squared`, `st_depression_squared`, `risk_logodds`
7. Create `_generate_all_static(df)` that calls both `_engineer_existing_features(df)` and `_engineer_research_features(df)`, returning the combined df
8. Add a `main()` with argparse (args: `--seeds`, `--train-sample`, `--holdout-sample`) and data loading that mirrors `select_features.py` lines 245-310 exactly (load train.csv + original, concat, sample, split, impute, generate static features)

**Important formula details** (from FEATURES.md):

```python
# framingham_partial (sex-specific log-linear)
log_age = np.log(df["Age"].clip(lower=20))
log_chol = np.log(df["Cholesterol"].clip(lower=100))
log_bp = np.log(df["BP"].clip(lower=80))
df["framingham_partial"] = np.where(
    df["Sex"] == 1,
    3.06*log_age + 1.12*log_chol + 1.93*log_bp + 0.57*df["FBS over 120"],
    2.33*log_age + 1.21*log_chol + 2.76*log_bp + 0.69*df["FBS over 120"],
)

# heart_score_partial
age_pts = np.where(df["Age"] < 45, 0, np.where(df["Age"] < 65, 1, 2))
ekg_pts = np.where(df["EKG results"] == 0, 0, np.where(df["EKG results"] == 1, 1, 2))
risk_pts = np.minimum(df["FBS over 120"] + (df["BP"] > 140).astype(int), 2)
df["heart_score_partial"] = age_pts + ekg_pts + risk_pts

# duke_treadmill_approx
est_exercise_min = ((df["Max HR"] - 80) / 8).clip(0, 21)
df["duke_treadmill_approx"] = est_exercise_min - 5*df["ST depression"] - 4*df["Exercise angina"]*2

# modified_duke (refined angina grading)
angina_index = np.where(
    df["Exercise angina"] == 0, 0,
    np.where(df["Chest pain type"] == 4, 2, 1)
)
est_time = ((df["Max HR"] - 60) / 20).clip(0, 21)
df["modified_duke"] = est_time - 5*df["ST depression"] - 4*angina_index

# timi_partial
df["timi_partial"] = (
    (df["Age"] >= 65).astype(int)
    + df["FBS over 120"]
    + (df["BP"] > 140).astype(int)
    + (df["ST depression"] > 0).astype(int)
)

# chronotropic_incompetence
df["chronotropic_incompetence"] = (df["Max HR"] < 0.80 * (220 - df["Age"])).astype(int)

# chronotropic_response_index (with estimated resting HR)
resting_hr = 60 + 0.2 * df["BP"]
predicted_max = 220 - df["Age"]
df["chronotropic_response_index"] = (
    (df["Max HR"] - resting_hr) / (predicted_max - resting_hr).clip(lower=1)
)

# hr_reserve_pct_tanaka (Tanaka formula: 208 - 0.7*Age)
df["hr_reserve_pct_tanaka"] = df["Max HR"] / (208 - 0.7 * df["Age"])

# hr_reserve_absolute
df["hr_reserve_absolute"] = (220 - df["Age"]) - df["Max HR"]

# st_hr_index (ST/HR normalization)
df["st_hr_index"] = (df["ST depression"] * 1000) / df["Max HR"].clip(lower=60)

# st_hr_hysteresis
df["st_hr_hysteresis"] = df["ST depression"] / (df["Max HR"] - resting_hr).clip(lower=1)

# rate_pressure_product
df["rate_pressure_product"] = df["Max HR"] * df["BP"]
df["rpp_normalized"] = (df["rate_pressure_product"] - 10000) / 30000

# supply_demand_mismatch
df["supply_demand_mismatch"] = (
    df["Max HR"] * df["BP"] / 10000
    * df["ST depression"]
    * (1 + df["Exercise angina"])
)

# estimated_mets and poor_exercise_capacity
df["estimated_mets"] = 0.05 * df["Max HR"] - 1.0
df["poor_exercise_capacity"] = (df["estimated_mets"] < 5).astype(int)

# ischemic_burden (slope-weighted)
slope_weight = np.where(df["Slope of ST"] == 2, 2, np.where(df["Slope of ST"] == 1, 1, 0))
df["ischemic_burden"] = (
    df["ST depression"] * slope_weight
    + 2 * df["Exercise angina"]
    + 3 * (df["Thallium"] >= 6).astype(int)
)

# Clinical categories
df["age_risk_category"] = pd.cut(df["Age"], bins=[0, 44, 54, 64, 200], labels=[0, 1, 2, 3]).astype(int)
df["age_sex_risk"] = np.where(df["Sex"] == 1, (df["Age"] >= 45).astype(int), (df["Age"] >= 55).astype(int))
df["bp_category"] = pd.cut(df["BP"], bins=[0, 119, 129, 139, 500], labels=[0, 1, 2, 3]).astype(int)
df["cholesterol_category"] = pd.cut(df["Cholesterol"].clip(lower=1), bins=[0, 199, 239, 1000], labels=[0, 1, 2]).astype(int)
pct = df["Max HR"] / (220 - df["Age"])
df["hr_achievement_category"] = pd.cut(pct, bins=[0, 0.60, 0.80, 0.85, 5.0], labels=[0, 1, 2, 3]).astype(int)
df["st_depression_category"] = pd.cut(df["ST depression"], bins=[-0.1, 0, 1, 2, 100], labels=[0, 1, 2, 3]).astype(int)

# Domain interactions
df["diabetes_hypertension"] = df["FBS over 120"] * (df["BP"] > 140).astype(int)
df["multivessel_ischemia"] = (df["Number of vessels fluro"] >= 2).astype(int) * (df["ST depression"] + df["Exercise angina"])
df["anatomic_severity"] = df["Number of vessels fluro"] * (df["Thallium"] >= 6).astype(int)
df["exercise_test_positive"] = (df["ST depression"] >= 1).astype(int) + (df["Slope of ST"] >= 2).astype(int) + df["Exercise angina"]
df["age_sex_interaction"] = df["Age"] * df["Sex"]
df["triple_threat"] = ((df["Chest pain type"] == 4).astype(int) * (df["Thallium"] >= 6).astype(int) * (df["Number of vessels fluro"] >= 1).astype(int))
df["cholesterol_age_risk"] = df["Cholesterol"] * (df["Age"] > 50).astype(int)
df["cardiac_efficiency"] = df["Max HR"] / df["BP"].clip(lower=80)
df["rest_exercise_concordance"] = (df["EKG results"] >= 1).astype(int) * ((df["ST depression"] > 0).astype(int) + df["Exercise angina"])
df["ekg_with_hypertension"] = (df["EKG results"] >= 1).astype(int) * (df["BP"] > 140).astype(int)

# Composites
df["risk_factor_count"] = (
    df["FBS over 120"]
    + (df["BP"] > 140).astype(int)
    + (df["Cholesterol"] > 240).astype(int)
    + (df["Age"] > 55).astype(int)
    + df["Sex"]
)
df["thallium_severity"] = (
    np.where(df["Thallium"] == 3, 0, np.where(df["Thallium"] == 6, 2, np.where(df["Thallium"] == 7, 3, 1)))
    + df["Exercise angina"]
    + (df["ST depression"] > 2).astype(int)
)
supply_proxy = ((df["Thallium"] == 3).astype(int) * (4 - df["Number of vessels fluro"]) / 4)
df["o2_supply_demand"] = (df["Max HR"] * df["BP"] / 10000) * (1 - supply_proxy)

# Competition tricks
for col in ["Chest pain type", "EKG results", "Slope of ST", "Thallium"]:
    freq = df[col].value_counts(normalize=True)
    df[f"{col}_freq"] = df[col].map(freq)
df["age_squared"] = df["Age"] ** 2
df["cholesterol_squared"] = df["Cholesterol"] ** 2
df["st_depression_squared"] = df["ST depression"] ** 2
risk_prob = (df["risk_score"] + 0.5) / (df["risk_score"].max() + 1) if "risk_score" in df.columns else 0.5
df["risk_logodds"] = np.log(risk_prob / (1 - risk_prob))
```

**Step 2: Verify the script loads data and generates features without errors**

Run: `uv run python notebooks/playground/research_features.py --train-sample 5000 --holdout-sample 2000`
Expected: Prints feature counts and exits (no evaluation yet). Should show ~80+ total features (13 raw + 22 existing + ~55 research).

**Step 3: Commit**

```bash
git add notebooks/playground/research_features.py
git commit -m "feat: add research_features.py scaffold with static feature generation"
```

---

### Task 2: Per-fold feature generation

**Files:**
- Modify: `notebooks/playground/research_features.py`

**Step 1: Add `_engineer_fold_features(X_train, y_train, X_val)` function**

This function receives fold-level train/val DataFrames and returns augmented X_train, X_val with per-fold features. It must fit all encoders/models on X_train only and transform both.

Features to implement (see FEATURES.md for details):

```python
def _engineer_fold_features(X_train, y_train, X_val):
    """Generate fold-aware features (fit on train, transform both)."""
    X_train = X_train.copy()
    X_val = X_val.copy()

    ENCODE_CATS = ["Thallium", "Chest pain type", "Slope of ST",
                   "EKG results", "Number of vessels fluro"]
    CONTINUOUS = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]

    # --- 1. Target encoding (existing pattern from test_features_local.py) ---
    for col in ENCODE_CATS:
        if col not in X_train.columns:
            continue
        means = y_train.groupby(X_train[col]).mean()
        global_mean = y_train.mean()
        X_train[f"{col}_te"] = X_train[col].map(means).fillna(global_mean)
        X_val[f"{col}_te"] = X_val[col].map(means).fillna(global_mean)

    # --- 2. GLMM encoding ---
    try:
        from category_encoders import GLMMEncoder
        enc = GLMMEncoder(cols=ENCODE_CATS)
        enc.fit(X_train[ENCODE_CATS], y_train)
        glmm_tr = enc.transform(X_train[ENCODE_CATS])
        glmm_val = enc.transform(X_val[ENCODE_CATS])
        for col in ENCODE_CATS:
            X_train[f"{col}_glmm"] = glmm_tr[col].values
            X_val[f"{col}_glmm"] = glmm_val[col].values
    except ImportError:
        pass

    # --- 3. James-Stein encoding ---
    try:
        from category_encoders import JamesSteinEncoder
        enc = JamesSteinEncoder(cols=ENCODE_CATS)
        enc.fit(X_train[ENCODE_CATS], y_train)
        js_tr = enc.transform(X_train[ENCODE_CATS])
        js_val = enc.transform(X_val[ENCODE_CATS])
        for col in ENCODE_CATS:
            X_train[f"{col}_js"] = js_tr[col].values
            X_val[f"{col}_js"] = js_val[col].values
    except ImportError:
        pass

    # --- 4. Leave-one-out encoding ---
    try:
        from category_encoders import LeaveOneOutEncoder
        enc = LeaveOneOutEncoder(cols=ENCODE_CATS)
        enc.fit(X_train[ENCODE_CATS], y_train)
        loo_tr = enc.transform(X_train[ENCODE_CATS])
        loo_val = enc.transform(X_val[ENCODE_CATS])
        for col in ENCODE_CATS:
            X_train[f"{col}_loo"] = loo_tr[col].values
            X_val[f"{col}_loo"] = loo_val[col].values
    except ImportError:
        pass

    # --- 5. WoE via optimal binning (continuous features) ---
    try:
        from optbinning import OptimalBinning
        for col in CONTINUOUS:
            optb = OptimalBinning(name=col, dtype="numerical", solver="cp")
            optb.fit(X_train[col].values, y_train.values)
            X_train[f"{col}_woe"] = optb.transform(X_train[col].values, metric="woe")
            X_val[f"{col}_woe"] = optb.transform(X_val[col].values, metric="woe")
    except ImportError:
        pass

    # --- 6. TE pair interactions ---
    TE_PAIRS = [
        ("Thallium", "Chest pain type"),
        ("Thallium", "Slope of ST"),
        ("Chest pain type", "Exercise angina"),
        ("EKG results", "Slope of ST"),
    ]
    for c1, c2 in TE_PAIRS:
        combo_tr = X_train[c1].astype(str) + "_" + X_train[c2].astype(str)
        combo_val = X_val[c1].astype(str) + "_" + X_val[c2].astype(str)
        means = y_train.groupby(combo_tr).mean()
        global_mean = y_train.mean()
        name = f"{c1}_{c2}_te"
        X_train[name] = combo_tr.map(means).fillna(global_mean)
        X_val[name] = combo_val.map(means).fillna(global_mean)

    # --- 7. Residual features ---
    from sklearn.linear_model import LinearRegression
    # MaxHR residual ~ Age
    lr = LinearRegression().fit(X_train[["Age"]], X_train["Max HR"])
    X_train["maxhr_residual"] = X_train["Max HR"] - lr.predict(X_train[["Age"]])
    X_val["maxhr_residual"] = X_val["Max HR"] - lr.predict(X_val[["Age"]])
    # Cholesterol residual ~ Age + Sex
    lr = LinearRegression().fit(X_train[["Age", "Sex"]], X_train["Cholesterol"])
    X_train["chol_residual"] = X_train["Cholesterol"] - lr.predict(X_train[["Age", "Sex"]])
    X_val["chol_residual"] = X_val["Cholesterol"] - lr.predict(X_val[["Age", "Sex"]])
    # BP residual ~ Age + Sex
    lr = LinearRegression().fit(X_train[["Age", "Sex"]], X_train["BP"])
    X_train["bp_residual"] = X_train["BP"] - lr.predict(X_train[["Age", "Sex"]])
    X_val["bp_residual"] = X_val["BP"] - lr.predict(X_val[["Age", "Sex"]])

    # --- 8. PCA (3 components) ---
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train[CONTINUOUS])
    X_val_scaled = scaler.transform(X_val[CONTINUOUS])
    pca = PCA(n_components=3, random_state=42)
    pca_tr = pca.fit_transform(X_tr_scaled)
    pca_val = pca.transform(X_val_scaled)
    for i in range(3):
        X_train[f"pca_{i}"] = pca_tr[:, i]
        X_val[f"pca_{i}"] = pca_val[:, i]

    # --- 9. Supervised UMAP (2 components) ---
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
        umap_tr = reducer.fit_transform(X_tr_scaled, y=y_train.values)
        umap_val = reducer.transform(X_val_scaled)
        X_train["umap_0"] = umap_tr[:, 0]
        X_train["umap_1"] = umap_tr[:, 1]
        X_val["umap_0"] = umap_val[:, 0]
        X_val["umap_1"] = umap_val[:, 1]
    except ImportError:
        pass

    # --- 10. Mahalanobis distance ---
    from scipy.spatial.distance import mahalanobis
    from sklearn.covariance import EmpiricalCovariance
    for label, suffix in [(1, "pos"), (0, "neg")]:
        mask = y_train == label
        cov = EmpiricalCovariance().fit(X_tr_scaled[mask])
        vi = cov.precision_
        center = cov.location_
        dist_tr = np.array([mahalanobis(x, center, vi) for x in X_tr_scaled])
        dist_val = np.array([mahalanobis(x, center, vi) for x in X_val_scaled])
        X_train[f"mahal_{suffix}"] = dist_tr
        X_val[f"mahal_{suffix}"] = dist_val
    X_train["mahal_ratio"] = X_train["mahal_neg"] / (X_train["mahal_pos"] + 1e-8)
    X_val["mahal_ratio"] = X_val["mahal_neg"] / (X_val["mahal_pos"] + 1e-8)

    # --- 11. Isolation Forest anomaly score ---
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1)
    iso.fit(X_tr_scaled)
    X_train["isolation_score"] = iso.decision_function(X_tr_scaled)
    X_val["isolation_score"] = iso.decision_function(X_val_scaled)

    # --- 12. KNN features ---
    from sklearn.neighbors import BallTree
    tree = BallTree(X_tr_scaled)
    # Distance to 10 nearest positive/negative
    for label, suffix in [(1, "pos"), (0, "neg")]:
        mask = y_train == label
        class_tree = BallTree(X_tr_scaled[mask])
        dist_tr, _ = class_tree.query(X_tr_scaled, k=10)
        dist_val, _ = class_tree.query(X_val_scaled, k=10)
        X_train[f"knn_dist_{suffix}"] = dist_tr.mean(axis=1)
        X_val[f"knn_dist_{suffix}"] = dist_val.mean(axis=1)
    X_train["knn_dist_ratio"] = X_train["knn_dist_pos"] / (X_train["knn_dist_neg"] + 1e-8)
    X_val["knn_dist_ratio"] = X_val["knn_dist_pos"] / (X_val["knn_dist_neg"] + 1e-8)
    # Neighborhood target rate (20-NN)
    _, idx_tr = tree.query(X_tr_scaled, k=21)  # +1 because self is included
    _, idx_val = tree.query(X_val_scaled, k=20)
    y_arr = y_train.values
    X_train["knn_target_rate"] = np.array([y_arr[ii[1:]].mean() for ii in idx_tr])
    X_val["knn_target_rate"] = np.array([y_arr[ii].mean() for ii in idx_val])

    # --- 13. Meta-model OOF predictions ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    # LogReg
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_tr_scaled, y_train)
    X_train["lr_oof"] = lr_model.predict_proba(X_tr_scaled)[:, 1]
    X_val["lr_oof"] = lr_model.predict_proba(X_val_scaled)[:, 1]

    # Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_tr_scaled, y_train)
    X_train["nb_oof"] = nb_model.predict_proba(X_tr_scaled)[:, 1]
    X_val["nb_oof"] = nb_model.predict_proba(X_val_scaled)[:, 1]

    # KNN
    knn_model = KNeighborsClassifier(n_neighbors=50, n_jobs=-1)
    knn_model.fit(X_tr_scaled, y_train)
    X_train["knn_oof"] = knn_model.predict_proba(X_tr_scaled)[:, 1]
    X_val["knn_oof"] = knn_model.predict_proba(X_val_scaled)[:, 1]

    # Model disagreement
    X_train["model_disagreement"] = np.abs(X_train["lr_oof"] - X_train["nb_oof"])
    X_val["model_disagreement"] = np.abs(X_val["lr_oof"] - X_val["nb_oof"])

    # Decision tree leaf ID (target-encoded)
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_tr_scaled, y_train)
    leaf_tr = dt.apply(X_tr_scaled)
    leaf_val = dt.apply(X_val_scaled)
    leaf_means = y_train.groupby(pd.Series(leaf_tr)).mean()
    X_train["dt_leaf_te"] = pd.Series(leaf_tr).map(leaf_means).fillna(y_train.mean()).values
    X_val["dt_leaf_te"] = pd.Series(leaf_val).map(leaf_means).fillna(y_train.mean()).values

    # --- 14. Spline basis functions ---
    from sklearn.preprocessing import SplineTransformer
    for col in ["Age", "Max HR"]:
        spline = SplineTransformer(n_knots=5, degree=3)
        sp_tr = spline.fit_transform(X_train[[col]])
        sp_val = spline.transform(X_val[[col]])
        for i in range(sp_tr.shape[1]):
            X_train[f"{col}_spline_{i}"] = sp_tr[:, i]
            X_val[f"{col}_spline_{i}"] = sp_val[:, i]

    # --- 15. Yeo-Johnson transforms ---
    from sklearn.preprocessing import PowerTransformer
    pt = PowerTransformer(method="yeo-johnson")
    for col in ["ST depression", "Cholesterol", "BP"]:
        pt.fit(X_train[[col]])
        X_train[f"{col}_yj"] = pt.transform(X_train[[col]]).ravel()
        X_val[f"{col}_yj"] = pt.transform(X_val[[col]]).ravel()

    return X_train, X_val
```

**Step 2: Update the LightGBM evaluation to use fold-aware features**

Add `_eval_lgbm_with_fold_features()` that wraps the CV loop:

```python
def _eval_lgbm_with_fold_features(
    train_df, y_train, holdout_df, y_holdout, static_features, seeds,
    use_native_cats=True
):
    """Evaluate with per-fold feature generation.

    For each seed:
      1. Train LightGBM on train with all features (static + fold-generated)
      2. Evaluate on holdout
    Returns: (mean_auc, list of all feature names used)
    """
    # Since we do train/holdout (not CV), fold features are computed once:
    # fit on train, transform both train and holdout
    X_tr = train_df[static_features].copy()
    X_ho = holdout_df[static_features].copy()
    X_tr, X_ho = _engineer_fold_features(X_tr, y_train, X_ho)

    all_features = [c for c in X_tr.columns if c not in ["id", TARGET]]

    cat_feats = [c for c in CAT_FEATURES if c in all_features] if use_native_cats else []

    aucs = []
    for seed in seeds:
        params = {**LGBM_PARAMS, "random_state": seed}
        model = LGBMClassifier(**params)
        model.fit(
            X_tr[all_features], y_train,
            eval_set=[(X_ho[all_features], y_holdout)],
            categorical_feature=cat_feats,
            callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)],
        )
        preds = model.predict_proba(X_ho[all_features])[:, 1]
        aucs.append(roc_auc_score(y_holdout, preds))

    return float(np.mean(aucs)), all_features
```

**Step 3: Verify fold features generate without errors**

Run: `uv run python notebooks/playground/research_features.py --train-sample 5000 --holdout-sample 2000`
Expected: Prints total feature count (should be ~125+), no errors.

**Step 4: Commit**

```bash
git add notebooks/playground/research_features.py
git commit -m "feat: add per-fold feature generation (encodings, KNN, meta-models, splines)"
```

---

### Task 3: Evaluation pipeline — ablation and forward selection

**Files:**
- Modify: `notebooks/playground/research_features.py`

**Step 1: Add the evaluation pipeline to `main()`**

Follow the exact pattern of `select_features.py` steps 1-4 (lines 322-498), but adapted for the research features:

```python
# In main(), after data loading and static feature generation:

# Generate fold features once (fit on train, transform holdout)
static_features = [c for c in train.columns if c not in ["id", TARGET]]
X_tr = train[static_features].copy()
X_ho = holdout[static_features].copy()
X_tr, X_ho = _engineer_fold_features(X_tr, y_train, X_ho)

all_features = [c for c in X_tr.columns if c not in ["id", TARGET]]
print(f"\nTotal candidate features: {len(all_features)}")

# ── Step 1: Baselines ────────────────────────────────
print(f"\n{'='*70}")
print("STEP 1: BASELINES (multi-seed)")
print(f"{'='*70}")

# 1a. Current ablation-pruned (21 features) — reference baseline
FEATURES_ABLATION_PRUNED = [...]  # copy from train_s6e2_baseline.py
auc_ablation = _eval_lgbm_multiseed(train, y_train, holdout, y_holdout,
    FEATURES_ABLATION_PRUNED, seeds)
print(f"Current ablation-pruned ({len(FEATURES_ABLATION_PRUNED)}): {auc_ablation:.5f}")

# 1b. Raw only (13 features)
auc_raw = _eval_lgbm_multiseed(train, y_train, holdout, y_holdout,
    RAW_FEATURES, seeds)
print(f"Raw only ({len(RAW_FEATURES)}): {auc_raw:.5f}")

# 1c. All features (static + fold) with native cats
auc_all_native = _eval_lgbm_multiseed_fold(X_tr, y_train, X_ho, y_holdout,
    all_features, seeds, use_native_cats=True)
print(f"All features native cats ({len(all_features)}): {auc_all_native:.5f}")

# 1d. All features without native cats
numeric_features = [f for f in all_features if f not in CAT_FEATURES]
auc_all_no_cats = _eval_lgbm_multiseed_fold(X_tr, y_train, X_ho, y_holdout,
    numeric_features, seeds, use_native_cats=False)
print(f"All features no native cats ({len(numeric_features)}): {auc_all_no_cats:.5f}")

# ── Step 2: Permutation importance ────────────────────────
# Train one model on all features, rank by permutation importance
# (Same pattern as select_features.py lines 329-374)

# ── Step 3: Drop-one-at-a-time ablation ──────────────────
# For each feature, train on all-but-one, measure AUC delta
# (Same pattern as select_features.py lines 376-413)
# NOTE: use X_tr/X_ho (which already have fold features) and a simple
# _eval_lgbm_multiseed that takes pre-computed feature DataFrames

# ── Step 4: Forward selection (clean slate) ───────────────
# Start from EMPTY set, greedily add features in permutation importance order
# (Same pattern as select_features.py lines 415-448)
# At each step, train on just the selected features

# ── Step 5: Comparison ────────────────────────────────────
# Compare: new forward-selected, new ablation-pruned, old ablation-pruned, raw
```

For the ablation and forward selection, use a simplified eval function that works on the pre-computed X_tr/X_ho DataFrames:

```python
def _eval_features_multiseed(X_tr, y_train, X_ho, y_holdout, features, seeds,
                              use_native_cats=True):
    """Evaluate a feature subset on pre-computed DataFrames."""
    cat_feats = [c for c in CAT_FEATURES if c in features] if use_native_cats else []
    aucs = []
    for seed in seeds:
        params = {**LGBM_PARAMS, "random_state": seed}
        model = LGBMClassifier(**params)
        model.fit(
            X_tr[features], y_train,
            eval_set=[(X_ho[features], y_holdout)],
            categorical_feature=cat_feats,
            callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)],
        )
        preds = model.predict_proba(X_ho[features])[:, 1]
        aucs.append(roc_auc_score(y_holdout, preds))
    return float(np.mean(aucs))
```

**Step 2: Add the FEATURES_ABLATION_PRUNED constant**

Copy the 21-feature list from `train_s6e2_baseline.py` for the reference baseline comparison.

**Step 3: Verify the full pipeline runs on a small sample**

Run: `uv run python notebooks/playground/research_features.py --train-sample 5000 --holdout-sample 2000`
Expected: All 5 steps complete, prints ablation table, forward selection curve, comparison table. May take 2-3 minutes on small sample.

**Step 4: Commit**

```bash
git add notebooks/playground/research_features.py
git commit -m "feat: add evaluation pipeline (ablation, forward selection, comparison)"
```

---

### Task 4: Full run and report formatting

**Files:**
- Modify: `notebooks/playground/research_features.py`

**Step 1: Add final report formatting**

After the evaluation pipeline, print a summary:

```python
# ── Summary ───────────────────────────────────────────────
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

print(f"\n{'Feature set':<45} {'AUC':>10} {'Delta vs raw':>14}")
print("-" * 73)
for name, auc in results:
    delta = auc - auc_raw
    print(f"{name:<45} {auc:>10.5f} {delta:>+14.5f}")

print(f"\nNew forward-selected features ({len(new_forward)}):")
for f in new_forward:
    source = "RAW" if f in RAW_FEATURES else "EXISTING" if f in existing_features else "RESEARCH"
    print(f"  - {f} [{source}]")

print(f"\nNew ablation-pruned features ({len(new_ablation)}):")
for f in new_ablation:
    source = "RAW" if f in RAW_FEATURES else "EXISTING" if f in existing_features else "RESEARCH"
    print(f"  - {f} [{source}]")
```

**Step 2: Run the full evaluation with default sample sizes**

Run: `uv run python notebooks/playground/research_features.py`
Expected: Full run with 50K train / 20K holdout, 3 seeds. Should take ~10-15 minutes.

**Step 3: Commit**

```bash
git add notebooks/playground/research_features.py
git commit -m "feat: add final report formatting and complete research_features.py"
```

---

### Task 5: Install dependencies and verify end-to-end

**Files:**
- Possibly modify: `notebooks/playground/pyproject.toml` or root `pyproject.toml`

**Step 1: Check which dependencies are already available**

Run: `uv run python -c "import category_encoders; import optbinning; import umap; print('all available')"`

If any fail, install them. Since the script runs locally (not on Ray), add to the playground workspace member or install directly.

**Step 2: Run the full script with default settings**

Run: `uv run python notebooks/playground/research_features.py`
Expected: Complete run, all features generated, evaluation pipeline runs, report printed.

**Step 3: Verify pre-commit hooks pass**

Run: `uv run pre-commit run --files notebooks/playground/research_features.py`
Expected: All hooks pass.

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore: install deps and finalize research_features.py"
```

---

### Task 6: Update documentation

**Files:**
- Modify: `notebooks/playground/README.md`

**Step 1: Add `research_features.py` to the Scripts section**

Add after the existing script list:
```
- `research_features.py` — Domain-driven feature research: generates ~125 candidates from clinical literature and evaluates via clean-slate forward selection
```

**Step 2: Add CLI reference for the new script**

Add a brief CLI section:
```markdown
### `research_features.py` CLI

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seeds` | str | `42,123,777` | Comma-separated seeds for multi-seed averaging |
| `--train-sample` | int | `50000` | Subsample training set (0=no sampling) |
| `--holdout-sample` | int | `20000` | Subsample holdout set (0=no sampling) |
```

**Step 3: Commit**

```bash
git add notebooks/playground/README.md
git commit -m "docs: add research_features.py to README"
```
