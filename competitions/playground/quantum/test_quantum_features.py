"""Quantum Feature Extraction Experiment.

Compares LightGBM performance across 4 setups, all evaluated on the same holdout:
  1. Vanilla (small):  700 train rows,  ablation-pruned features
  2. Quantum (small):  700 train rows,  ablation-pruned + quantum features
  2b. Quantum-only:    700 train rows,  quantum features only
  3. Vanilla (full):   ~630K train rows, ablation-pruned features
  4. Quantum (full):   ~630K train rows, ablation-pruned + quantum features

Quantum features are generated locally via a fast numpy statevector simulator
(~1.5ms/sample), or optionally via Qiskit or the Kipu Quantum Rimay API.

Usage:
  # Default: load quantum features from previous Rimay results
  uv run python test_quantum_features.py

  # Local simulation (no Rimay API needed)
  uv run python test_quantum_features.py --local

  # Submit new Rimay job and wait for results
  uv run python test_quantum_features.py --submit

  # Submit Rimay job without waiting
  uv run python test_quantum_features.py --submit --submit-only

  # Full-data quantum features (~15 min for 630K rows)
  uv run python test_quantum_features.py --full-quantum

Environment variables (for --rimay only):
  PLANQK_ACCESS_TOKEN     - PlanQK personal access token
  PLANQK_ORGANIZATION_ID  - PlanQK organization ID
  PLANQK_CONSUMER_KEY     - Service consumer key (from application)
  PLANQK_CONSUMER_SECRET  - Service consumer secret (from application)
  PLANQK_SERVICE_ENDPOINT - Rimay service endpoint URL

Data directory (optional):
  KEGO_PATH_DATA - override default data path (default: ../../data)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from quantum_features import (
    download_quantum_features,
    generate_quantum_features_fast,
    submit_to_rimay,
    wait_for_result,
)
from sklearn.metrics import roc_auc_score

from kego.datasets.split import split_dataset
from kego.features.selection import (
    SelectionResult,
    drop_one_ablation,
    forward_selection,
    greedy_add_one_screening,
)

# ── Constants ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = (
    Path(os.environ.get("KEGO_PATH_DATA", PROJECT_ROOT / "data"))
    / "playground"
    / "playground-series-s6e2"
)
RESULTS_DIR = Path(__file__).resolve().parent / "quantum_results"
TARGET = "Heart Disease"

# 13 raw features (fits within 15-qubit Rimay limit)
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

# 21 ablation-pruned features (10 raw + 11 engineered)
FEATURES_ABLATION_PRUNED = [
    # Raw features that help (10)
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
    # Engineered features that help (11)
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

# LightGBM params for small data (700 rows) — more regularization, fewer trees
LGB_PARAMS_SMALL = {
    "n_estimators": 300,
    "max_depth": 3,
    "num_leaves": 7,
    "learning_rate": 0.1,
    "metric": "auc",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

# LightGBM params for full data (~630K rows) — production params from baseline
LGB_PARAMS_FULL = {
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
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

# Split sizes for quantum experiments
TRAIN_SAMPLE = 700
HOLDOUT_SAMPLE = 300


# ── Feature Engineering ───────────────────────────────────────────────────


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven features needed for ablation-pruned set."""
    df = df.copy()

    # Thallium interactions
    df["thallium_x_slope"] = df["Thallium"] * df["Slope of ST"]

    # Other strong interactions
    df["chestpain_x_slope"] = df["Chest pain type"] * df["Slope of ST"]
    df["angina_x_stdep"] = df["Exercise angina"] * df["ST depression"]

    # Composite risk scores
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

    # Ratio/interaction features
    df["age_x_stdep"] = df["Age"] * df["ST depression"]

    # Grouped deviation features
    for col in ["Cholesterol", "BP", "ST depression"]:
        grp_mean = df.groupby("Sex")[col].transform("mean")
        df[f"{col}_dev_sex"] = df[col] - grp_mean

    # Signal conflict
    df["signal_conflict"] = (
        (df["Thallium"] >= 6) & (df["Chest pain type"] <= 3)
    ).astype(int) + ((df["Thallium"] == 3) & (df["Chest pain type"] == 4)).astype(int)

    return df


# ── Data Loading ──────────────────────────────────────────────────────────


def load_data() -> pd.DataFrame:
    """Load competition data + original UCI data (if available)."""
    train_full = pd.read_csv(DATA_DIR / "train.csv")
    train_full[TARGET] = train_full[TARGET].map({"Presence": 1, "Absence": 0})

    original_path = DATA_DIR / "Heart_Disease_Prediction.csv"
    if original_path.exists():
        original = pd.read_csv(original_path)
        original[TARGET] = original[TARGET].map({"Presence": 1, "Absence": 0})
        original["id"] = -1
        combined = pd.concat([train_full, original], ignore_index=True)
        print(
            f"Data loaded: {len(combined)} rows "
            f"({len(train_full)} synthetic + {len(original)} original)"
        )
    else:
        combined = train_full
        print(f"Data loaded: {len(combined)} rows (synthetic only)")

    return combined


def create_splits(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create 3 splits: quantum_train (700), holdout (300), full_train (rest).

    Uses kego split_dataset for stratified splitting.
    full_train = everything except holdout.
    quantum_train = small subsample from full_train.
    """
    holdout_fraction = HOLDOUT_SAMPLE / len(df)
    quantum_fraction = TRAIN_SAMPLE / (len(df) - HOLDOUT_SAMPLE)

    # Split into full_train and holdout
    full_train, holdout, _ = split_dataset(
        df,
        train_size=1 - holdout_fraction,
        validate_size=holdout_fraction,
        test_size=0.0,
        stratify_column=TARGET,
    )

    # Subsample quantum_train from full_train
    quantum_train, _, _ = split_dataset(
        full_train,
        train_size=quantum_fraction,
        validate_size=1 - quantum_fraction,
        test_size=0.0,
        stratify_column=TARGET,
    )

    print(
        f"  quantum_train target dist: {quantum_train[TARGET].value_counts().to_dict()}"
    )
    print(f"  holdout target dist: {holdout[TARGET].value_counts().to_dict()}")

    return quantum_train, holdout, full_train


def concat_quantum_features(
    X_classical: pd.DataFrame,
    quantum_array: np.ndarray,
    quantum_feature_names: list[str],
) -> pd.DataFrame:
    """Concatenate classical DataFrame with quantum feature array."""
    return pd.concat(
        [
            X_classical.reset_index(drop=True),
            pd.DataFrame(quantum_array, columns=quantum_feature_names),
        ],
        axis=1,
    )


# ── Evaluation ────────────────────────────────────────────────────────────


def train_evaluate_lgb(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_holdout: np.ndarray | pd.DataFrame,
    y_holdout: np.ndarray | pd.Series,
    params: dict,
    name: str,
    cat_features: list[str] | None = None,
) -> float:
    """Train LightGBM and evaluate on holdout. Returns holdout AUC."""
    import lightgbm

    model = LGBMClassifier(**params)

    fit_kwargs = {}
    if cat_features and isinstance(X_train, pd.DataFrame):
        available_cats = [c for c in cat_features if c in X_train.columns]
        if available_cats:
            fit_kwargs["categorical_feature"] = available_cats

    fit_kwargs["eval_set"] = [(X_holdout, y_holdout)]  # type: ignore[list-item]
    fit_kwargs["callbacks"] = [
        lightgbm.early_stopping(50, verbose=False),
        lightgbm.log_evaluation(0),
    ]

    model.fit(X_train, y_train, **fit_kwargs)
    preds = model.predict_proba(X_holdout)[:, 1]
    auc = roc_auc_score(y_holdout, preds)
    iterations = getattr(model, "best_iteration_", getattr(model, "n_estimators", 0))
    print(f"  {name:<55s} AUC={auc:.5f} ({iterations} iters)")

    return auc


def run_evaluation(
    quantum_train: pd.DataFrame,
    holdout: pd.DataFrame,
    full_train: pd.DataFrame,
    quantum_features: dict | None = None,
    quantum_features_full: dict | None = None,
) -> dict:
    """Run all experiments, ablation for each, and print combined summary.

    Expects all DataFrames to already have engineered features
    (call engineer_features before splitting).
    """
    results: dict[str, float] = {}
    ablations: dict[str, dict] = {}

    y_qt = quantum_train[TARGET].reset_index(drop=True)
    y_ho = holdout[TARGET].reset_index(drop=True)
    y_ft = full_train[TARGET].reset_index(drop=True)

    cats_in_ablation = [
        column for column in CAT_FEATURES if column in FEATURES_ABLATION_PRUNED
    ]

    # Classical feature DataFrames (engineered features already present)
    X_qt = quantum_train[FEATURES_ABLATION_PRUNED].reset_index(drop=True)
    X_ho = holdout[FEATURES_ABLATION_PRUNED].reset_index(drop=True)
    X_ft = full_train[FEATURES_ABLATION_PRUNED].reset_index(drop=True)
    classical_features = list(X_qt.columns)

    # ── Define experiments ──
    # Each: (key, label, X_train, y_train, X_holdout, params, cats,
    #        candidate_features, baseline_features)
    experiments: list[tuple] = [
        (
            "vanilla_small",
            "Exp 1: Vanilla (700 rows)",
            X_qt,
            y_qt,
            X_ho,
            LGB_PARAMS_SMALL,
            cats_in_ablation,
            None,
            None,
        ),
        (
            "vanilla_full",
            f"Exp 3: Vanilla ({len(full_train)} rows)",
            X_ft,
            y_ft,
            X_ho,
            LGB_PARAMS_FULL,
            cats_in_ablation,
            None,
            None,
        ),
    ]

    # Add quantum experiments if features available
    quantum_feature_names: list[str] = []
    if quantum_features is not None:
        quantum_run = quantum_features[0]
        quantum_feature_names = quantum_features.get(
            "feature_names",
            [f"qf_{i}" for i in range(quantum_run["Xq_train"].shape[1])],
        )

        X_qt_quantum = concat_quantum_features(
            X_qt, quantum_run["Xq_train"], quantum_feature_names
        )
        X_ho_quantum = concat_quantum_features(
            X_ho, quantum_run["Xq_test"], quantum_feature_names
        )
        X_qt_qonly = pd.DataFrame(
            quantum_run["Xq_train"], columns=quantum_feature_names
        )
        X_ho_qonly = pd.DataFrame(quantum_run["Xq_test"], columns=quantum_feature_names)

        experiments.insert(
            1,
            (
                "quantum_small",
                "Exp 2: Vanilla + Quantum (700 rows)",
                X_qt_quantum,
                y_qt,
                X_ho_quantum,
                LGB_PARAMS_SMALL,
                cats_in_ablation,
                quantum_feature_names,
                classical_features,
            ),
        )
        experiments.insert(
            2,
            (
                "quantum_only",
                "Exp 2b: Quantum Only (700 rows)",
                X_qt_qonly,
                y_qt,
                X_ho_qonly,
                LGB_PARAMS_SMALL,
                None,
                None,
                None,
            ),
        )

    if quantum_features_full is not None:
        quantum_full_run = quantum_features_full[0]
        X_ft_quantum = concat_quantum_features(
            X_ft, quantum_full_run["Xq_train"], quantum_feature_names
        )
        X_ho_quantum_full = concat_quantum_features(
            X_ho, quantum_full_run["Xq_test"], quantum_feature_names
        )

        experiments.append(
            (
                "quantum_full",
                f"Exp 4: Vanilla + Quantum ({len(full_train)} rows)",
                X_ft_quantum,
                y_ft,
                X_ho_quantum_full,
                LGB_PARAMS_FULL,
                cats_in_ablation,
                quantum_feature_names,
                classical_features,
            )
        )

    # ── Run experiments + ablation ──
    for (
        key,
        label,
        X_train,
        y_train,
        X_holdout,
        params,
        cats,
        candidate_features,
        baseline_features,
    ) in experiments:
        results[key] = train_evaluate_lgb(
            X_train,
            y_train,
            X_holdout,
            y_ho,
            params,
            label,
            cat_features=cats,
        )

        ablations[key] = run_ablation(
            X_train=X_train,
            y_train=y_train,
            X_holdout=X_holdout,
            y_holdout=y_ho,
            features=list(X_train.columns),
            params=params,
            label=label,
            cat_features=cats,
            candidate_features=candidate_features,
            baseline_features=baseline_features,
        )

    # ── Final Summary ──
    metric = "roc_auc"
    print(f"\n{'=' * 70}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 70}")

    baseline = results.get("vanilla_small", 0.0)
    for key, label, *_ in experiments:
        delta = results[key] - baseline
        marker = " <-- baseline" if key == "vanilla_small" else ""
        print(f"\n  {label:<55s} {results[key]:>8.5f} {delta:>+8.5f}{marker}")
        print_ablation_summary(ablations[key], metric)

    return results


# ── Ablation Study ────────────────────────────────────────────────────────


def run_ablation(
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: np.ndarray | pd.Series,
    features: list[str],
    params: dict,
    label: str,
    cat_features: list[str] | None = None,
    candidate_features: list[str] | None = None,
    baseline_features: list[str] | None = None,
) -> dict:
    """Run all three feature selection methods and return structured results.

    Args:
        X_train: Training data (must contain all columns in features).
        y_train: Training labels.
        X_holdout: Holdout data.
        y_holdout: Holdout labels.
        features: All features to evaluate.
        params: LightGBM parameters.
        label: Human-readable label for printing.
        cat_features: Categorical feature names for LightGBM.
        candidate_features: Features to test in add-one screening. If None, skipped.
        baseline_features: Baseline set for add-one screening. If None, skipped.

    Returns:
        Dict with keys: drop_results, forward_features, forward_score,
        screening_results (if candidate_features provided).
    """
    print(f"    Ablation: {label}...")

    seeds = [42, 123, 777]
    metric = "roc_auc"
    model = LGBMClassifier(**params)

    fit_kwargs: dict = {
        "eval_set": [(X_holdout, y_holdout)],
        "callbacks": [
            lightgbm.early_stopping(50, verbose=False),
            lightgbm.log_evaluation(0),
        ],
    }
    if cat_features:
        available_cats = [
            column for column in cat_features if column in X_train.columns
        ]
        if available_cats:
            fit_kwargs["categorical_feature"] = available_cats

    shared = (X_train, y_train, X_holdout, y_holdout)
    shared_kwargs = dict(seeds=seeds, metric=metric, model=model, fit_kwargs=fit_kwargs)

    drop_result = drop_one_ablation(*shared, features=features, **shared_kwargs)

    ordered_features = [
        entry["feature"]
        for entry in sorted(drop_result.feature_results, key=lambda x: x["delta"])
    ]
    forward_result = forward_selection(
        *shared,
        features_ordered=ordered_features,
        **shared_kwargs,
    )

    if baseline_features is None or candidate_features is None:
        helpful_ordered = [
            entry["feature"]
            for entry in sorted(drop_result.feature_results, key=lambda x: x["delta"])
        ]
        midpoint = len(helpful_ordered) // 2
        baseline_features = helpful_ordered[:midpoint]
        candidate_features = helpful_ordered[midpoint:]

    screening_result = greedy_add_one_screening(
        baseline_features=baseline_features,
        candidate_features=candidate_features,
        X_train=X_train,
        y_train=y_train,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        **shared_kwargs,
    )

    return {
        "drop_one": drop_result,
        "forward": forward_result,
        "screening": screening_result,
    }


def print_ablation_summary(ablation: dict, metric: str = "roc_auc") -> None:
    """Print one score per ablation method."""
    drop_result: SelectionResult = ablation["drop_one"]
    forward_result: SelectionResult = ablation["forward"]
    screening_result: SelectionResult = ablation["screening"]

    print(
        f"    Drop-one:          {metric}={drop_result.selected_score:.5f} "
        f"({len(drop_result.selected_features)} features)"
    )
    print(
        f"    Forward selection: {metric}={forward_result.selected_score:.5f} "
        f"({len(forward_result.selected_features)} features)"
    )
    print(
        f"    Add-one screening: {metric}={screening_result.selected_score:.5f} "
        f"({len(screening_result.selected_features)} features)"
    )


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Quantum Feature Extraction Experiment"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Generate quantum features locally instead of loading from Rimay",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit new job to Rimay API (default: load existing results)",
    )
    parser.add_argument(
        "--submit-only",
        action="store_true",
        help="Submit to Rimay and exit without waiting for results",
    )
    parser.add_argument(
        "--full-quantum",
        action="store_true",
        help="Generate quantum features for full ~630K dataset (takes ~3 min)",
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=500,
        help="Measurement shots per circuit (default: 500)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Independent runs on simulator (default: 1)",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data, engineer features, then split
    df = engineer_features(load_data())
    quantum_train, holdout, full_train = create_splits(df)

    quantum_train.to_csv(RESULTS_DIR / "quantum_train.csv", index=False)
    holdout.to_csv(RESULTS_DIR / "holdout.csv", index=False)

    quantum_features_full = None

    if args.local:
        quantum_features = generate_quantum_features_fast(
            quantum_train,
            holdout,
            RAW_FEATURES,
            TARGET,
        )
    elif args.submit:
        X_train_raw = quantum_train[RAW_FEATURES].reset_index(drop=True).astype(float)
        y_train_raw = quantum_train[[TARGET]].reset_index(drop=True).astype(int)
        X_test_raw = holdout[RAW_FEATURES].reset_index(drop=True).astype(float)
        y_test_raw = holdout[[TARGET]].reset_index(drop=True).astype(int)

        metadata, _, execution = submit_to_rimay(
            X_train_raw,
            y_train_raw,
            X_test_raw,
            y_test_raw,
            results_dir=RESULTS_DIR,
            num_shots=args.num_shots,
            num_runs=args.num_runs,
        )
        if args.submit_only:
            print("\n  Submit-only mode. Re-run without --submit to load results.")
            return

        wait_for_result(execution)
        quantum_features = download_quantum_features(
            metadata["output_datapool_id"],
            RESULTS_DIR,
            metadata["num_runs"],
        )
    else:
        # Default: load existing Rimay results
        metadata_path = RESULTS_DIR / "rimay_metadata.json"
        if not metadata_path.exists():
            print(f"ERROR: No Rimay results found at {metadata_path}")
            print(
                "  Run with --submit to submit a new job, or --local for local simulation."
            )
            sys.exit(1)
        with open(metadata_path) as file:
            metadata = json.load(file)
        quantum_features = download_quantum_features(
            metadata["output_datapool_id"],
            RESULTS_DIR,
            metadata["num_runs"],
        )

    # Full-data quantum features (~630K train + 300 holdout)
    if args.full_quantum:
        print(f"\n{'=' * 60}")
        print("  Generating quantum features for full dataset...")
        print(f"  ({len(full_train)} train + {len(holdout)} holdout)")
        print(f"{'=' * 60}")

        cache_train = RESULTS_DIR / "quantum_features_full_train.npy"
        cache_holdout = RESULTS_DIR / "quantum_features_full_holdout.npy"

        if cache_train.exists() and cache_holdout.exists():
            print("\n  Loading cached full quantum features...")
            Xq_ft = np.load(cache_train)
            Xq_ho = np.load(cache_holdout)
            print(f"    Train: {Xq_ft.shape}, Holdout: {Xq_ho.shape}")
            n_qubits = len(RAW_FEATURES)
            quantum_feature_names = [f"qf_Z{i}" for i in range(n_qubits)]
            quantum_feature_names += [f"qf_ZZ{i}_{i + 1}" for i in range(n_qubits - 1)]
        else:
            result = generate_quantum_features_fast(
                full_train,
                holdout,
                RAW_FEATURES,
                TARGET,
            )
            Xq_ft = result[0]["Xq_train"]
            Xq_ho = result[0]["Xq_test"]
            quantum_feature_names = result["feature_names"]
            np.save(cache_train, Xq_ft)
            np.save(cache_holdout, Xq_ho)
            print(f"\n  Cached to {RESULTS_DIR}/quantum_features_full_*.npy")

        quantum_features_full = {
            0: {
                "Xq_train": Xq_ft,
                "Xq_test": Xq_ho,
                "yq_train": full_train[TARGET].values,
                "yq_test": holdout[TARGET].values,
            },
            "feature_names": quantum_feature_names,
        }

    # Run evaluation (includes ablation and final summary)
    run_evaluation(
        quantum_train,
        holdout,
        full_train,
        quantum_features,
        quantum_features_full,
    )


if __name__ == "__main__":
    main()
