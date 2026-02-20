"""Domain-driven feature research for S6E2 Heart Disease.

Generates ~80 static feature candidates (13 raw + 22 existing + ~45 research) plus
~45 per-fold features from clinical literature, advanced encodings,
and competition techniques. Evaluates via clean-slate forward selection with
LightGBM (CPU only, no Ray).

Usage:
    uv run python notebooks/playground/research_features.py
    uv run python notebooks/playground/research_features.py --train-sample 5000 --holdout-sample 2000
"""

import argparse
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from kego.datasets.split import split_dataset  # noqa: E402

DATA_DIR = (
    Path(os.environ.get("KEGO_PATH_DATA", project_root / "data"))
    / "playground"
    / "playground-series-s6e2"
)
TARGET = "Heart Disease"

RAW_FEATURES = [
    "Age",
    "Sex",
    "Chest pain type",
    "BP",
    "Cholesterol",
    "FBS over 120",
    "EKG results",
    "Max HR",
    "Exercise angina",
    "ST depression",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
]

CAT_FEATURES = [
    "Sex",
    "Chest pain type",
    "FBS over 120",
    "EKG results",
    "Exercise angina",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
]

CONTINUOUS_FEATURES = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]

FEATURES_ABLATION_PRUNED = [
    "Age",
    "Sex",
    "Chest pain type",
    "Cholesterol",
    "FBS over 120",
    "EKG results",
    "Max HR",
    "ST depression",
    "Slope of ST",
    "Number of vessels fluro",
    "thallium_x_slope",
    "chestpain_x_slope",
    "angina_x_stdep",
    "top4_sum",
    "abnormal_count",
    "risk_score",
    "age_x_stdep",
    "Cholesterol_dev_sex",
    "BP_dev_sex",
    "ST depression_dev_sex",
    "signal_conflict",
]

LGBM_PARAMS = {
    "n_estimators": 1500,
    "max_depth": 4,
    "num_leaves": 15,
    "learning_rate": 0.08,
    "metric": "auc",
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 3,
    "reg_alpha": 0.01,
    "reg_lambda": 0.1,
    "random_state": 123,
    "verbosity": -1,
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _impute_cholesterol(df: pd.DataFrame) -> pd.DataFrame:
    """Replace Cholesterol=0 (missing) with grouped median by Sex and Age bin."""
    df = df.copy()
    if (df["Cholesterol"] == 0).any():
        df["_age_bin"] = pd.cut(df["Age"], bins=[0, 40, 50, 60, 200])
        median_map = (
            df[df["Cholesterol"] > 0]
            .groupby(["Sex", "_age_bin"])["Cholesterol"]
            .median()
        )
        mask = df["Cholesterol"] == 0
        for idx in df[mask].index:
            key = (df.loc[idx, "Sex"], df.loc[idx, "_age_bin"])
            if key in median_map.index:
                df.loc[idx, "Cholesterol"] = median_map[key]
            else:
                df.loc[idx, "Cholesterol"] = df.loc[~mask, "Cholesterol"].median()
        df = df.drop(columns=["_age_bin"])
    return df


# ---------------------------------------------------------------------------
# Feature engineering: existing features (from select_features.py)
# ---------------------------------------------------------------------------


def _engineer_existing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven interaction and composite features (22 features)."""
    df = df.copy()

    # --- Thallium interactions (Thallium is the #1 predictor) ---
    df["thallium_x_chestpain"] = df["Thallium"] * df["Chest pain type"]
    df["thallium_x_slope"] = df["Thallium"] * df["Slope of ST"]
    df["thallium_x_sex"] = df["Thallium"] * df["Sex"]
    df["thallium_x_stdep"] = df["Thallium"] * df["ST depression"]
    df["thallium_abnormal"] = (df["Thallium"] >= 6).astype(int)

    # --- Other strong interactions ---
    df["chestpain_x_slope"] = df["Chest pain type"] * df["Slope of ST"]
    df["chestpain_x_angina"] = df["Chest pain type"] * df["Exercise angina"]
    df["vessels_x_thallium"] = df["Number of vessels fluro"] * df["Thallium"]
    df["angina_x_stdep"] = df["Exercise angina"] * df["ST depression"]

    # --- Composite risk scores ---
    df["top4_sum"] = (
        df["Thallium"]
        + df["Chest pain type"]
        + df["Number of vessels fluro"]
        + df["Exercise angina"]
    )
    df["abnormal_count"] = (
        (df["Thallium"] >= 6).astype(int)
        + (df["Number of vessels fluro"] >= 1).astype(int)
        + (df["Chest pain type"] >= 3).astype(int)
        + (df["Exercise angina"] == 1).astype(int)
        + (df["Slope of ST"] >= 2).astype(int)
        + (df["ST depression"] > 1).astype(int)
        + (df["Sex"] == 1).astype(int)
    )
    df["risk_score"] = (
        3 * (df["Thallium"] >= 6).astype(int)
        + 2 * (df["Number of vessels fluro"] >= 1).astype(int)
        + 2 * (df["Chest pain type"] >= 3).astype(int)
        + 2 * (df["Exercise angina"] == 1).astype(int)
        + (df["Slope of ST"] >= 2).astype(int)
        + (df["ST depression"] > 1).astype(int)
    )

    # --- Ratio features ---
    df["maxhr_per_age"] = df["Max HR"] / df["Age"]
    df["hr_reserve_pct"] = df["Max HR"] / (220 - df["Age"])
    df["age_x_stdep"] = df["Age"] * df["ST depression"]
    df["age_x_maxhr"] = df["Age"] * df["Max HR"]
    df["heart_load"] = df["BP"] * df["Cholesterol"] / df["Max HR"].clip(lower=1)

    # --- Grouped deviation features (individual risk vs demographic peers) ---
    for col in ["Cholesterol", "BP", "Max HR", "ST depression"]:
        grp_mean = df.groupby("Sex")[col].transform("mean")
        df[f"{col}_dev_sex"] = df[col] - grp_mean

    # --- Signal conflict: top predictors disagree on risk direction ---
    df["signal_conflict"] = (
        (df["Thallium"] >= 6) & (df["Chest pain type"] <= 3)
    ).astype(int) + ((df["Thallium"] == 3) & (df["Chest pain type"] == 4)).astype(int)

    return df


# ---------------------------------------------------------------------------
# Feature engineering: research features (~44 new features)
# ---------------------------------------------------------------------------


def _engineer_research_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate ~44 research feature candidates from clinical literature."""
    df = df.copy()

    # === Clinical Scores (5) ===
    # framingham_partial
    log_age = np.log(df["Age"].clip(lower=20))
    log_chol = np.log(df["Cholesterol"].clip(lower=100))
    log_bp = np.log(df["BP"].clip(lower=80))
    df["framingham_partial"] = np.where(
        df["Sex"] == 1,
        3.06 * log_age + 1.12 * log_chol + 1.93 * log_bp + 0.57 * df["FBS over 120"],
        2.33 * log_age + 1.21 * log_chol + 2.76 * log_bp + 0.69 * df["FBS over 120"],
    )

    # heart_score_partial
    age_pts = np.where(df["Age"] < 45, 0, np.where(df["Age"] < 65, 1, 2))
    ekg_pts = np.where(
        df["EKG results"] == 0, 0, np.where(df["EKG results"] == 1, 1, 2)
    )
    risk_pts = np.minimum(df["FBS over 120"] + (df["BP"] > 140).astype(int), 2)
    df["heart_score_partial"] = age_pts + ekg_pts + risk_pts

    # duke_treadmill_approx
    est_exercise_min = ((df["Max HR"] - 80) / 8).clip(0, 21)
    df["duke_treadmill_approx"] = (
        est_exercise_min - 5 * df["ST depression"] - 4 * df["Exercise angina"] * 2
    )

    # modified_duke
    angina_index = np.where(
        df["Exercise angina"] == 0,
        0,
        np.where(df["Chest pain type"] == 4, 2, 1),
    )
    est_time = ((df["Max HR"] - 60) / 20).clip(0, 21)
    df["modified_duke"] = est_time - 5 * df["ST depression"] - 4 * angina_index

    # timi_partial
    df["timi_partial"] = (
        (df["Age"] >= 65).astype(int)
        + df["FBS over 120"]
        + (df["BP"] > 140).astype(int)
        + (df["ST depression"] > 0).astype(int)
    )

    # === Exercise Physiology (11) ===
    resting_hr = 60 + 0.2 * df["BP"]
    predicted_max = 220 - df["Age"]

    df["chronotropic_incompetence"] = (df["Max HR"] < 0.80 * predicted_max).astype(int)
    denom = (predicted_max - resting_hr).clip(lower=1)
    df["chronotropic_response_index"] = (df["Max HR"] - resting_hr) / denom
    df["hr_reserve_pct_tanaka"] = df["Max HR"] / (208 - 0.7 * df["Age"])
    df["hr_reserve_absolute"] = predicted_max - df["Max HR"]
    df["st_hr_index"] = (df["ST depression"] * 1000) / df["Max HR"].clip(lower=60)
    hr_delta = (df["Max HR"] - resting_hr).clip(lower=1)
    df["st_hr_hysteresis"] = df["ST depression"] / hr_delta
    df["rate_pressure_product"] = df["Max HR"] * df["BP"]
    df["rpp_normalized"] = (df["rate_pressure_product"] - 10000) / 30000
    df["supply_demand_mismatch"] = (
        df["Max HR"]
        * df["BP"]
        / 10000
        * df["ST depression"]
        * (1 + df["Exercise angina"])
    )
    df["estimated_mets"] = 0.05 * df["Max HR"] - 1.0
    df["poor_exercise_capacity"] = (df["estimated_mets"] < 5).astype(int)

    # === Clinical Categories (6) ===
    df["age_risk_category"] = pd.cut(
        df["Age"], bins=[0, 44, 54, 64, 200], labels=[0, 1, 2, 3]
    ).astype(int)
    df["age_sex_risk"] = np.where(
        df["Sex"] == 1, (df["Age"] >= 45).astype(int), (df["Age"] >= 55).astype(int)
    )
    df["bp_category"] = pd.cut(
        df["BP"], bins=[0, 119, 129, 139, 500], labels=[0, 1, 2, 3]
    ).astype(int)
    df["cholesterol_category"] = pd.cut(
        df["Cholesterol"].clip(lower=1), bins=[0, 199, 239, 1000], labels=[0, 1, 2]
    ).astype(int)
    pct = df["Max HR"] / predicted_max
    df["hr_achievement_category"] = pd.cut(
        pct, bins=[0, 0.60, 0.80, 0.85, 5.0], labels=[0, 1, 2, 3]
    ).astype(int)
    df["st_depression_category"] = pd.cut(
        df["ST depression"], bins=[-0.1, 0, 1, 2, 100], labels=[0, 1, 2, 3]
    ).astype(int)

    # === Domain Interactions (10) ===
    df["diabetes_hypertension"] = df["FBS over 120"] * (df["BP"] > 140).astype(int)
    df["multivessel_ischemia"] = (df["Number of vessels fluro"] >= 2).astype(int) * (
        df["ST depression"] + df["Exercise angina"]
    )
    df["anatomic_severity"] = df["Number of vessels fluro"] * (
        df["Thallium"] >= 6
    ).astype(int)
    df["exercise_test_positive"] = (
        (df["ST depression"] >= 1).astype(int)
        + (df["Slope of ST"] >= 2).astype(int)
        + df["Exercise angina"]
    )
    df["age_sex_interaction"] = df["Age"] * df["Sex"]
    df["triple_threat"] = (
        (df["Chest pain type"] == 4).astype(int)
        * (df["Thallium"] >= 6).astype(int)
        * (df["Number of vessels fluro"] >= 1).astype(int)
    )
    df["cholesterol_age_risk"] = df["Cholesterol"] * (df["Age"] > 50).astype(int)
    df["cardiac_efficiency"] = df["Max HR"] / df["BP"].clip(lower=80)
    df["rest_exercise_concordance"] = (df["EKG results"] >= 1).astype(int) * (
        (df["ST depression"] > 0).astype(int) + df["Exercise angina"]
    )
    df["ekg_with_hypertension"] = (df["EKG results"] >= 1).astype(int) * (
        df["BP"] > 140
    ).astype(int)

    # === Composites (4) ===
    slope_weight = np.where(
        df["Slope of ST"] == 2, 2, np.where(df["Slope of ST"] == 1, 1, 0)
    )
    df["ischemic_burden"] = (
        df["ST depression"] * slope_weight
        + 2 * df["Exercise angina"]
        + 3 * (df["Thallium"] >= 6).astype(int)
    )
    df["risk_factor_count"] = (
        df["FBS over 120"]
        + (df["BP"] > 140).astype(int)
        + (df["Cholesterol"] > 240).astype(int)
        + (df["Age"] > 55).astype(int)
        + df["Sex"]
    )
    df["thallium_severity"] = (
        np.where(
            df["Thallium"] == 3,
            0,
            np.where(df["Thallium"] == 6, 2, np.where(df["Thallium"] == 7, 3, 1)),
        )
        + df["Exercise angina"]
        + (df["ST depression"] > 2).astype(int)
    )
    supply_proxy = (
        (df["Thallium"] == 3).astype(int) * (4 - df["Number of vessels fluro"]) / 4
    )
    df["o2_supply_demand"] = (df["Max HR"] * df["BP"] / 10000) * (1 - supply_proxy)

    # === Competition Tricks (~8) ===
    for col in ["Chest pain type", "EKG results", "Slope of ST", "Thallium"]:
        freq = df[col].value_counts(normalize=True)
        df[f"{col}_freq"] = df[col].map(freq)
    df["age_squared"] = df["Age"] ** 2
    df["cholesterol_squared"] = df["Cholesterol"] ** 2
    df["st_depression_squared"] = df["ST depression"] ** 2

    # risk_logodds — only compute if risk_score exists (from _engineer_existing_features)
    if "risk_score" in df.columns:
        risk_prob = (df["risk_score"] + 0.5) / (df["risk_score"].max() + 1)
        df["risk_logodds"] = np.log(risk_prob / (1 - risk_prob))

    return df


# ---------------------------------------------------------------------------
# Feature engineering: fold-aware features (~57 per fold)
# ---------------------------------------------------------------------------


def _engineer_fold_features(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate ~57 fold-aware features fit on X_train only, applied to both."""
    from sklearn.covariance import EmpiricalCovariance
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import BallTree, KNeighborsClassifier
    from sklearn.preprocessing import (
        PowerTransformer,
        SplineTransformer,
        StandardScaler,
    )
    from sklearn.tree import DecisionTreeClassifier

    X_train = X_train.copy()
    X_val = X_val.copy()
    y_train = pd.Series(y_train.values, index=X_train.index)

    ENCODE_CATS = [
        "Thallium",
        "Chest pain type",
        "Slope of ST",
        "EKG results",
        "Number of vessels fluro",
    ]
    CONTINUOUS = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]

    # === 1. Target encoding (5 features) ===
    global_mean = y_train.mean()
    for col in ENCODE_CATS:
        means = y_train.groupby(X_train[col]).mean()
        X_train[f"{col}_te"] = X_train[col].map(means).fillna(global_mean)
        X_val[f"{col}_te"] = X_val[col].map(means).fillna(global_mean)

    # === 2. GLMM encoding (5 features) ===
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

    # === 3. James-Stein encoding (5 features) ===
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

    # === 4. Leave-one-out encoding (5 features) ===
    try:
        from category_encoders import LeaveOneOutEncoder

        enc = LeaveOneOutEncoder(cols=ENCODE_CATS, sigma=0.05)
        loo_tr = enc.fit_transform(X_train[ENCODE_CATS], y_train)
        loo_val = enc.transform(X_val[ENCODE_CATS])
        for col in ENCODE_CATS:
            X_train[f"{col}_loo"] = loo_tr[col].values
            X_val[f"{col}_loo"] = loo_val[col].values
    except ImportError:
        pass

    # === 5. WoE via optimal binning (5 features) ===
    try:
        from optbinning import OptimalBinning

        for col in CONTINUOUS:
            optb = OptimalBinning(name=col, dtype="numerical", solver="cp")
            optb.fit(X_train[col].values, y_train.values)
            X_train[f"{col}_woe"] = optb.transform(X_train[col].values, metric="woe")
            X_val[f"{col}_woe"] = optb.transform(X_val[col].values, metric="woe")
    except ImportError:
        pass

    # === 6. TE pair interactions (4 features) ===
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
        name = f"{c1}_{c2}_te"
        X_train[name] = combo_tr.map(means).fillna(global_mean)
        X_val[name] = combo_val.map(means).fillna(global_mean)

    # === 7. Residual features (3 features) ===
    lr = LinearRegression().fit(X_train[["Age"]], X_train["Max HR"])
    X_train["maxhr_residual"] = X_train["Max HR"] - lr.predict(X_train[["Age"]])
    X_val["maxhr_residual"] = X_val["Max HR"] - lr.predict(X_val[["Age"]])

    lr = LinearRegression().fit(X_train[["Age", "Sex"]], X_train["Cholesterol"])
    X_train["chol_residual"] = X_train["Cholesterol"] - lr.predict(
        X_train[["Age", "Sex"]]
    )
    X_val["chol_residual"] = X_val["Cholesterol"] - lr.predict(X_val[["Age", "Sex"]])

    lr = LinearRegression().fit(X_train[["Age", "Sex"]], X_train["BP"])
    X_train["bp_residual"] = X_train["BP"] - lr.predict(X_train[["Age", "Sex"]])
    X_val["bp_residual"] = X_val["BP"] - lr.predict(X_val[["Age", "Sex"]])

    # === Standardize continuous features (shared for PCA, UMAP, etc.) ===
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train[CONTINUOUS])
    X_val_scaled = scaler.transform(X_val[CONTINUOUS])

    # === 8. PCA (3 features) ===
    pca = PCA(n_components=3, random_state=42)
    pca_tr = pca.fit_transform(X_tr_scaled)
    pca_val = pca.transform(X_val_scaled)
    for i in range(3):
        X_train[f"pca_{i}"] = pca_tr[:, i]
        X_val[f"pca_{i}"] = pca_val[:, i]

    # === 9. Supervised UMAP (2 features) ===
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

    # === 10. Mahalanobis distance (3 features) ===
    for label, suffix in [(1, "pos"), (0, "neg")]:
        mask = y_train.values == label
        cov = EmpiricalCovariance().fit(X_tr_scaled[mask])
        vi = cov.precision_
        center = cov.location_
        diff_tr = X_tr_scaled - center
        dist_tr = np.sqrt(np.sum(diff_tr @ vi * diff_tr, axis=1))
        diff_val = X_val_scaled - center
        dist_val = np.sqrt(np.sum(diff_val @ vi * diff_val, axis=1))
        X_train[f"mahal_{suffix}"] = dist_tr
        X_val[f"mahal_{suffix}"] = dist_val
    X_train["mahal_ratio"] = X_train["mahal_neg"] / (X_train["mahal_pos"] + 1e-8)
    X_val["mahal_ratio"] = X_val["mahal_neg"] / (X_val["mahal_pos"] + 1e-8)

    # === 11. Isolation Forest anomaly score (1 feature) ===
    iso = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1)
    iso.fit(X_tr_scaled)
    X_train["isolation_score"] = iso.decision_function(X_tr_scaled)
    X_val["isolation_score"] = iso.decision_function(X_val_scaled)

    # === 12. KNN features (4 features) ===
    tree = BallTree(X_tr_scaled)
    for label, suffix in [(1, "pos"), (0, "neg")]:
        mask = y_train.values == label
        class_tree = BallTree(X_tr_scaled[mask])
        dist_tr, _ = class_tree.query(X_tr_scaled, k=10)
        dist_val, _ = class_tree.query(X_val_scaled, k=10)
        X_train[f"knn_dist_{suffix}"] = dist_tr.mean(axis=1)
        X_val[f"knn_dist_{suffix}"] = dist_val.mean(axis=1)
    X_train["knn_dist_ratio"] = X_train["knn_dist_pos"] / (
        X_train["knn_dist_neg"] + 1e-8
    )
    X_val["knn_dist_ratio"] = X_val["knn_dist_pos"] / (X_val["knn_dist_neg"] + 1e-8)
    # Neighborhood target rate (20-NN)
    _, idx_tr = tree.query(X_tr_scaled, k=21)  # +1 for self
    _, idx_val = tree.query(X_val_scaled, k=20)
    y_arr = y_train.values
    X_train["knn_target_rate"] = y_arr[idx_tr[:, 1:]].mean(axis=1)
    X_val["knn_target_rate"] = y_arr[idx_val].mean(axis=1)

    # === 13. Meta-model OOF predictions (5 features) ===
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
    X_train["dt_leaf_te"] = (
        pd.Series(leaf_tr).map(leaf_means).fillna(y_train.mean()).values
    )
    X_val["dt_leaf_te"] = (
        pd.Series(leaf_val).map(leaf_means).fillna(y_train.mean()).values
    )

    # === 14. Spline basis functions (~14 features) ===
    for col in ["Age", "Max HR"]:
        spline = SplineTransformer(n_knots=5, degree=3)
        sp_tr = spline.fit_transform(X_train[[col]])
        sp_val = spline.transform(X_val[[col]])
        for i in range(sp_tr.shape[1]):
            X_train[f"{col}_spline_{i}"] = sp_tr[:, i]
            X_val[f"{col}_spline_{i}"] = sp_val[:, i]

    # === 15. Yeo-Johnson transforms (3 features) ===
    for col in ["ST depression", "Cholesterol", "BP"]:
        pt = PowerTransformer(method="yeo-johnson")
        X_train[f"{col}_yj"] = pt.fit_transform(X_train[[col]]).ravel()
        X_val[f"{col}_yj"] = pt.transform(X_val[[col]]).ravel()

    return X_train, X_val


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def _eval_features_multiseed(
    X_tr, y_train, X_ho, y_holdout, features, seeds, use_native_cats=True
):
    """Evaluate a feature subset with multi-seed LightGBM on pre-computed DataFrames."""

    cat_feats = [c for c in CAT_FEATURES if c in features] if use_native_cats else []
    aucs = []
    for seed in seeds:
        params = {**LGBM_PARAMS, "random_state": seed}
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr[features],
            y_train,
            eval_set=[(X_ho[features], y_holdout)],
            categorical_feature=cat_feats,
            callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)],
        )
        preds = model.predict_proba(X_ho[features])[:, 1]
        aucs.append(roc_auc_score(y_holdout, preds))
    return float(np.mean(aucs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Domain-driven feature research for S6E2 Heart Disease"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,777",
        help="Comma-separated seeds for multi-seed averaging (default: 42,123,777)",
    )
    parser.add_argument(
        "--train-sample",
        type=int,
        default=50000,
        help="Subsample training set to N rows (default: 50000, 0=no sampling)",
    )
    parser.add_argument(
        "--holdout-sample",
        type=int,
        default=20000,
        help="Subsample holdout set to N rows (default: 20000, 0=no sampling)",
    )
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    # --- Load & prepare data (same pipeline as select_features.py) ---
    total_sample = (args.train_sample or 0) + (args.holdout_sample or 0)

    train_full = pd.read_csv(DATA_DIR / "train.csv")
    original = pd.read_csv(DATA_DIR / "Heart_Disease_Prediction.csv")

    train_full[TARGET] = train_full[TARGET].map({"Presence": 1, "Absence": 0})
    original[TARGET] = original[TARGET].map({"Presence": 1, "Absence": 0})

    original["id"] = -1
    train_full = pd.concat([train_full, original], ignore_index=True)

    # Downsample before split to keep memory low
    if total_sample and len(train_full) > total_sample:
        train_full = train_full.sample(n=total_sample, random_state=42).reset_index(
            drop=True
        )

    train, holdout, _ = split_dataset(
        train_full,
        train_size=0.8,
        validate_size=0.2,
        stratify_column=TARGET,
    )
    del train_full  # free memory
    train = train.reset_index(drop=True)
    holdout = holdout.reset_index(drop=True)

    train = _impute_cholesterol(train)
    holdout = _impute_cholesterol(holdout)

    train = _engineer_existing_features(train)
    holdout = _engineer_existing_features(holdout)

    train = _engineer_research_features(train)
    holdout = _engineer_research_features(holdout)

    all_features = [c for c in train.columns if c not in ["id", TARGET]]
    existing_features = [c for c in all_features if c not in RAW_FEATURES]
    y_train = train[TARGET].values
    y_holdout = holdout[TARGET].values

    print(f"Total features: {len(all_features)}")
    print(f"  Raw: {len(RAW_FEATURES)}")
    print(f"  Engineered (existing + research): {len(existing_features)}")
    print(f"  Ablation-pruned baseline: {len(FEATURES_ABLATION_PRUNED)}")
    print(f"Train: {len(train)}, Holdout: {len(holdout)}")
    print(f"Seeds: {seeds}")

    # Generate fold-aware features (fit on train, transform holdout)
    static_features = [c for c in train.columns if c not in ["id", TARGET]]
    X_tr = train[static_features].copy()
    X_ho = holdout[static_features].copy()
    X_tr, X_ho = _engineer_fold_features(X_tr, pd.Series(y_train), X_ho)

    all_features = [c for c in X_tr.columns if c not in ["id", TARGET]]
    fold_features = [c for c in all_features if c not in static_features]
    print(f"  Fold-aware features: {len(fold_features)}")
    print(f"  Total after fold features: {len(all_features)}")

    # ===================================================================
    # Step 1: Baselines (multi-seed)
    # ===================================================================
    print(f"\n{'='*70}")
    print("STEP 1: BASELINES")
    print(f"{'='*70}")

    # 1a. Raw only (13 features) — use native cats
    raw_in_xtr = [f for f in RAW_FEATURES if f in X_tr.columns]
    auc_raw = _eval_features_multiseed(
        X_tr, y_train, X_ho, y_holdout, raw_in_xtr, seeds
    )
    print(f"Raw only ({len(raw_in_xtr)}): {auc_raw:.5f}")

    # 1b. Current ablation-pruned (21 features) — reference
    abl_pruned_in_xtr = [f for f in FEATURES_ABLATION_PRUNED if f in X_tr.columns]
    auc_abl_ref = _eval_features_multiseed(
        X_tr, y_train, X_ho, y_holdout, abl_pruned_in_xtr, seeds
    )
    print(f"Ablation-pruned ref ({len(abl_pruned_in_xtr)}): {auc_abl_ref:.5f}")

    # 1c. All features with native categoricals
    auc_all_native = _eval_features_multiseed(
        X_tr, y_train, X_ho, y_holdout, all_features, seeds, use_native_cats=True
    )
    print(f"All features native cats ({len(all_features)}): {auc_all_native:.5f}")

    # 1d. All features without native categoricals
    auc_all_no_cats = _eval_features_multiseed(
        X_tr, y_train, X_ho, y_holdout, all_features, seeds, use_native_cats=False
    )
    print(f"All features no native cats ({len(all_features)}): {auc_all_no_cats:.5f}")

    # ===================================================================
    # Step 2: Permutation importance
    # ===================================================================
    print(f"\n{'='*70}")
    print("STEP 2: PERMUTATION IMPORTANCE")
    print(f"{'='*70}")

    # Train one model on all features for permutation importance
    from sklearn.inspection import permutation_importance

    model_all = lgb.LGBMClassifier(**LGBM_PARAMS)
    fit_cats = [c for c in CAT_FEATURES if c in all_features]
    model_all.fit(
        X_tr[all_features],
        y_train,
        eval_set=[(X_ho[all_features], y_holdout)],
        categorical_feature=fit_cats,
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)],
    )

    result = permutation_importance(
        model_all,
        X_ho[all_features],
        y_holdout,
        n_repeats=10,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
    )

    imp_df = pd.DataFrame(
        {
            "feature": all_features,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    imp_df["significant"] = imp_df["importance_mean"] > 2 * imp_df["importance_std"]

    print(f"\n{'Feature':<40} {'Mean':>10} {'Std':>10} {'Sig':>5}")
    print("-" * 70)
    for _, row in imp_df.iterrows():
        sig = "***" if row["significant"] else ""
        sign = "-" if row["importance_mean"] < 0 else ""
        print(
            f"{row['feature']:<40} {sign}{abs(row['importance_mean']):>9.5f} "
            f"{row['importance_std']:>10.5f} {sig:>5}"
        )

    features_by_importance = imp_df["feature"].tolist()

    # ===================================================================
    # Step 3: Drop-one-at-a-time ablation (multi-seed)
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"STEP 3: ABLATION ({len(all_features)} features x {len(seeds)} seeds)")
    print(f"{'='*70}")

    baseline_ms = _eval_features_multiseed(
        X_tr, y_train, X_ho, y_holdout, all_features, seeds
    )
    print(f"\nAll-features baseline: {baseline_ms:.5f}")

    ablation_results = []
    for i, feat in enumerate(all_features):
        reduced = [f for f in all_features if f != feat]
        auc_without = _eval_features_multiseed(
            X_tr, y_train, X_ho, y_holdout, reduced, seeds
        )
        delta = auc_without - baseline_ms
        ablation_results.append((feat, auc_without, delta))
        print(
            f"  [{i+1}/{len(all_features)}] -{feat:<35} "
            f"AUC={auc_without:.5f} (delta={delta:+.5f})"
        )

    ablation_results.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'Feature':<40} {'AUC without':>12} {'Delta':>10} {'Verdict':>10}")
    print("-" * 76)
    for feat, auc_without, delta in ablation_results:
        verdict = "HARMFUL" if delta > 0 else "helpful"
        print(f"{feat:<40} {auc_without:>12.5f} {delta:>+10.5f} {verdict:>10}")

    harmful_features = [f for f, _, d in ablation_results if d > 0]
    ablation_pruned = [f for f in all_features if f not in harmful_features]
    print(f"\nHarmful ({len(harmful_features)}): {harmful_features}")
    print(f"Ablation-pruned set: {len(ablation_pruned)} features")

    # ===================================================================
    # Step 4: Forward selection (greedy, importance order)
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"STEP 4: FORWARD SELECTION ({len(all_features)} features)")
    print(f"{'='*70}")

    forward_history = []
    for i, _ in enumerate(features_by_importance, start=1):
        subset = features_by_importance[:i]
        auc_fwd = _eval_features_multiseed(
            X_tr, y_train, X_ho, y_holdout, subset, seeds
        )
        forward_history.append((i, subset[-1], auc_fwd))
        print(
            f"  [{i}/{len(features_by_importance)}] +{subset[-1]:<35} "
            f"AUC={auc_fwd:.5f}"
        )

    # Find optimal
    best_n, best_feat, best_fwd_auc = max(forward_history, key=lambda x: x[2])
    forward_selected = features_by_importance[:best_n]

    print(f"\n{'N':>3} {'Added feature':<40} {'AUC':>10} {'Delta':>10}")
    print("-" * 67)
    prev_auc = 0.0
    for n, feat, auc in forward_history:
        delta = auc - prev_auc if n > 1 else 0.0
        print(f"{n:>3} {feat:<40} {auc:>10.5f} {delta:>+10.5f}")
        prev_auc = auc

    print(f"\nOptimal: {best_n} features, AUC={best_fwd_auc:.5f}")

    # ===================================================================
    # Step 5: Comparison
    # ===================================================================
    print(f"\n{'='*70}")
    print("STEP 5: COMPARISON")
    print(f"{'='*70}")

    results = [
        (f"Raw only ({len(raw_in_xtr)})", auc_raw),
        (f"Ablation-pruned ref ({len(abl_pruned_in_xtr)})", auc_abl_ref),
        (f"All features ({len(all_features)})", auc_all_native),
        (
            f"New ablation-pruned ({len(ablation_pruned)})",
            _eval_features_multiseed(
                X_tr, y_train, X_ho, y_holdout, ablation_pruned, seeds
            ),
        ),
        (f"Forward-selected ({len(forward_selected)})", best_fwd_auc),
    ]

    print(f"\n{'Feature set':<45} {'AUC':>10} {'Delta vs raw':>14}")
    print("-" * 73)
    for name, auc in results:
        delta = auc - auc_raw
        print(f"{name:<45} {auc:>10.5f} {delta:>+14.5f}")

    # Determine feature sources
    raw_set = set(RAW_FEATURES)
    fold_set = set(fold_features)
    existing_eng_names = {
        "thallium_x_chestpain",
        "thallium_x_slope",
        "thallium_x_sex",
        "thallium_x_stdep",
        "thallium_abnormal",
        "chestpain_x_slope",
        "chestpain_x_angina",
        "vessels_x_thallium",
        "angina_x_stdep",
        "top4_sum",
        "abnormal_count",
        "risk_score",
        "maxhr_per_age",
        "hr_reserve_pct",
        "age_x_stdep",
        "age_x_maxhr",
        "heart_load",
        "Cholesterol_dev_sex",
        "BP_dev_sex",
        "Max HR_dev_sex",
        "ST depression_dev_sex",
        "signal_conflict",
    }

    print(f"\nForward-selected features ({len(forward_selected)}):")
    for f in forward_selected:
        if f in raw_set:
            source = "RAW"
        elif f in existing_eng_names:
            source = "EXISTING"
        elif f in fold_set:
            source = "FOLD"
        else:
            source = "RESEARCH"
        print(f"  - {f} [{source}]")


if __name__ == "__main__":
    main()
