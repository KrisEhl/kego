"""Quantum Feature Extraction using Rimay API with Manual Approval Steps.

This script follows the professional workflow from run.ipynb:
  1. Prepares training/test data and uploads to input datapool (manual approval)
  2. Creates output datapool and submits quantum job to Rimay
  3. Waits for completion and downloads results
  4. Evaluates multiple models:
     - Raw features only (all 13)
     - Ablation-pruned classical features
     - Quantum features only
     - Combined ablation-pruned + quantum

Unlike test_quantum_features.py which uses local simulation, this uses
the actual Kipu Quantum Rimay API for production-grade quantum feature extraction.

Usage:
  # Full workflow with manual approval steps
  uv run python quantum_rimay_workflow.py

  # Skip quantum step, only test classical features
  uv run python quantum_rimay_workflow.py --skip-quantum

  # Skip upload, use existing metadata
  uv run python quantum_rimay_workflow.py --skip-upload

  # Skip submission, only download from existing output datapool
  uv run python quantum_rimay_workflow.py --skip-submit

  # Evaluate only (download + analysis, with metadata)
  uv run python quantum_rimay_workflow.py --evaluate-only

  # Evaluate only (download + analysis, with manual datapool ID)
  uv run python quantum_rimay_workflow.py --evaluate-only --output-datapool-id <datapool-id>

Environment variables (required):
  PLANQK_ACCESS_TOKEN     - PlanQK personal access token
  PLANQK_ORGANIZATION_ID  - PlanQK organization ID
  PLANQK_CONSUMER_KEY     - Service consumer key
  PLANQK_CONSUMER_SECRET  - Service consumer secret
  PLANQK_SERVICE_ENDPOINT - Rimay service endpoint URL

Data directory (optional):
  KEGO_PATH_DATA - override default data path (default: ../../data)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

# ── Constants ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = (
    Path(os.environ.get("KEGO_PATH_DATA", PROJECT_ROOT / "data"))
    / "playground"
    / "playground-series-s6e2"
)
RESULTS_DIR = Path(__file__).resolve().parent / "results"
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

# 24 ablation-pruned features (13 raw + 11 engineered)
FEATURES_ABLATION_PRUNED = [
    # All 13 raw features (most important)
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

# LightGBM params for small data (700 rows) — more regularization
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

# Rimay API constraints
MAX_FEATURES = 15
MAX_SAMPLES = 1000
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

    # Risk score based on abnormal thresholds
    df["risk_score"] = (
        (df["ST depression"] > 2).astype(int) * 2
        + (df["Slope of ST"] <= 1).astype(int)
        + (df["Number of vessels fluro"] >= 1).astype(int)
    )

    # Cross-feature interactions
    df["age_x_stdep"] = df["Age"] * (df["ST depression"] + 0.1)
    df["Cholesterol_dev_sex"] = df["Cholesterol"] * (df["Sex"] + 1)
    df["BP_dev_sex"] = df["BP"] * (df["Sex"] + 1)
    df["ST depression_dev_sex"] = df["ST depression"] * (df["Sex"] + 1)

    # Signal conflict
    df["signal_conflict"] = (df["Chest pain type"] >= 3).astype(int) ^ (
        df["ST depression"] <= 0.5
    ).astype(int)

    return df


def load_data() -> pd.DataFrame:
    """Load Heart Disease dataset."""
    return pd.read_csv(DATA_DIR / "train.csv")


def create_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create train and holdout splits."""
    df = df.copy()
    df = engineer_features(df)

    # Stratified split
    from sklearn.model_selection import train_test_split

    train, holdout = train_test_split(
        df,
        test_size=HOLDOUT_SAMPLE,
        random_state=42,
        stratify=df[TARGET],
    )

    # For Rimay, use a subset to stay under MAX_SAMPLES
    if len(train) > TRAIN_SAMPLE:
        train = train.sample(n=TRAIN_SAMPLE, random_state=42)

    return train, holdout


# ── Rimay API Workflow ────────────────────────────────────────────────────


def prepare_dataset(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Path:
    """Prepare dataset JSON file for upload."""
    # Encode target if categorical
    y_train_encoded = y_train.copy()
    y_test_encoded = y_test.copy()

    target_dtype = str(y_train_encoded[TARGET].dtype)

    # If target is object/string dtype, encode to integers
    if target_dtype in ("object", "str", "string"):
        unique_vals = y_train_encoded[TARGET].unique()
        mapping = {val: i for i, val in enumerate(sorted(unique_vals))}
        y_train_encoded[TARGET] = y_train_encoded[TARGET].map(mapping)
        y_test_encoded[TARGET] = y_test_encoded[TARGET].map(mapping)
        print(f"  Encoded target: {mapping}")

    dataset = {
        "training_tabular_data": X_train.to_dict(),
        "training_target_data": y_train_encoded.to_dict(),
        "test_tabular_data": X_test.to_dict(),
        "test_target_data": y_test_encoded.to_dict(),
    }

    dataset_path = RESULTS_DIR / "dataset.json"
    with open(dataset_path, "w") as f:
        json.dump(dataset, f)

    return dataset_path


def upload_data_and_wait_approval(
    dataset_path: Path,
) -> tuple[str, str]:
    """Upload dataset to input datapool and wait for user approval.

    Returns:
        (input_datapool_id, dataset_info_str)
    """
    from planqk.api.client import PlanqkApiClient

    access_token = os.environ["PLANQK_ACCESS_TOKEN"]
    org_id = os.environ["PLANQK_ORGANIZATION_ID"]

    api_client = PlanqkApiClient(
        access_token=access_token,
        organization_id=org_id,
    )

    # Create input datapool
    input_dp = api_client.api.data_pools.create_data_pool(name="Quantum Workflow Input")
    print(f"\n{'=' * 60}")
    print("  INPUT DATAPOOL CREATED")
    print(f"{'=' * 60}")
    print(f"  Datapool ID: {input_dp.id}")
    print(f"  Name: {input_dp.name}")

    # Upload file
    with open(dataset_path, "rb") as f:
        api_client.api.data_pools.add_data_pool_file(
            id=input_dp.id, file=("data.json", f)
        )
    print(f"  File uploaded: {dataset_path.name}")
    print(f"  File size: {dataset_path.stat().st_size / 1024:.1f} KB")

    info_str = f"""
    ┌─ INPUT DATAPOOL READY FOR APPROVAL ─────────────────────┐
    │                                                           │
    │  Datapool ID:  {input_dp.id}                 │
    │  Name:         Quantum Workflow Input                    │
    │  File:         data.json                                 │
    │  Size:         {dataset_path.stat().st_size / 1024:.1f} KB                                            │
    │                                                           │
    │  ✓ Dataset is ready to be processed                      │
    │  ✓ Please verify the data looks correct                  │
    │                                                           │
    └───────────────────────────────────────────────────────────┘
    """

    print(info_str)
    print("  Press ENTER to continue, or Ctrl+C to cancel...")
    input()

    return input_dp.id, info_str


def create_output_datapool_and_wait_approval(
    input_dp_id: str,
    num_shots: int = 100,
    num_runs: int = 1,
) -> tuple[str, dict]:
    """Create output datapool and wait for user approval before submission.

    Returns:
        (output_datapool_id, config_dict)
    """
    from planqk.api.client import PlanqkApiClient

    access_token = os.environ["PLANQK_ACCESS_TOKEN"]
    org_id = os.environ["PLANQK_ORGANIZATION_ID"]

    api_client = PlanqkApiClient(
        access_token=access_token,
        organization_id=org_id,
    )

    # Create output datapool
    output_dp = api_client.api.data_pools.create_data_pool(
        name="Quantum Workflow Output"
    )

    config = {
        "input_datapool_id": input_dp_id,
        "output_datapool_id": output_dp.id,
        "num_shots": num_shots,
        "num_runs": num_runs,
    }

    print(f"\n{'=' * 60}")
    print("  DATAPOOLS READY FOR APPROVAL")
    print(f"{'=' * 60}")

    info_str = f"""
    ┌─ QUANTUM JOB CONFIGURATION ─────────────────────────────┐
    │                                                           │
    │  INPUT DATAPOOL:                                         │
    │    ID: {input_dp_id}         │
    │    Purpose: Your data (700 train + 300 test)            │
    │                                                           │
    │  OUTPUT DATAPOOL:                                        │
    │    ID: {output_dp.id}         │
    │    Purpose: Quantum features (will be generated)        │
    │                                                           │
    │  QUANTUM JOB SETTINGS:                                   │
    │    Shots per run:    {num_shots}                                         │
    │    Number of runs:   {num_runs}                                          │
    │                                                           │
    │  ✓ Both datapools are ready                              │
    │  ✓ Please verify the IDs are correct                     │
    │                                                           │
    └───────────────────────────────────────────────────────────┘
    """

    print(info_str)
    print("  Press ENTER to submit job, or Ctrl+C to cancel...")
    input()

    return output_dp.id, config


def submit_quantum_job(
    config: dict,
) -> tuple[dict, object, object]:
    """Submit quantum job to Rimay service.

    Args:
        config: Dictionary with input_datapool_id, output_datapool_id, num_shots, num_runs

    Returns:
        (metadata_dict, service_client, execution_object)
    """
    from planqk.service.client import PlanqkServiceClient

    consumer_key = os.environ["PLANQK_CONSUMER_KEY"]
    consumer_secret = os.environ["PLANQK_CONSUMER_SECRET"]
    service_endpoint = os.environ["PLANQK_SERVICE_ENDPOINT"]

    print(f"\n{'=' * 60}")
    print("  SUBMITTING QUANTUM JOB")
    print(f"{'=' * 60}")
    print(f"  Input datapool:  {config['input_datapool_id']}")
    print(f"  Output datapool: {config['output_datapool_id']}")
    print(f"  Shots per run:   {config['num_shots']}")
    print(f"  Number of runs:  {config['num_runs']}")

    # Submit to service
    service_input = {
        "input_data_pool": {
            "id": config["input_datapool_id"],
            "ref": "DATAPOOL",
        },
        "output_data_pool": {
            "id": config["output_datapool_id"],
            "ref": "DATAPOOL",
        },
        "num_shots": config["num_shots"],
        "num_runs": config["num_runs"],
    }

    client = PlanqkServiceClient(
        service_endpoint=service_endpoint,
        access_key_id=consumer_key,
        secret_access_key=consumer_secret,
    )

    print("  Submitting to Rimay...")
    execution = client.run(request=service_input)
    print(f"  Execution ID: {execution.id}")
    print(f"  Status: {execution.status}")

    # Save metadata
    metadata = {
        "input_datapool_id": config["input_datapool_id"],
        "output_datapool_id": config["output_datapool_id"],
        "execution_id": execution.id if hasattr(execution, "id") else str(execution),
        "num_shots": config["num_shots"],
        "num_runs": config["num_runs"],
        "submitted_at": time.time(),
    }
    metadata_path = RESULTS_DIR / "rimay_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Metadata saved: {metadata_path}")

    return metadata, client, execution


def wait_for_quantum_result(execution) -> dict:
    """Wait for quantum job to complete."""
    print(f"\n{'=' * 60}")
    print("  WAITING FOR QUANTUM COMPUTATION")
    print(f"{'=' * 60}")
    print(f"  Execution ID: {execution.id}")
    print(f"  Initial status: {execution.status}")
    print("  Polling for completion (this may take a few minutes)...")

    start = time.time()
    result = execution.result()  # Blocks until done
    elapsed = time.time() - start

    execution.refresh()
    print(f"\n  ✓ Execution completed in {elapsed:.1f}s")
    print(f"  Final status: {execution.status}")

    if execution.status == "FAILED":
        print("\n  ERROR: Quantum job FAILED!")

        # Capture and save logs and result
        log_path = RESULTS_DIR / "rimay_execution_logs.txt"
        with open(log_path, "w") as log_file:
            log_file.write(f"Execution ID: {execution.id}\n")
            log_file.write(f"Status: {execution.status}\n")
            log_file.write(f"Elapsed: {elapsed:.1f}s\n")
            log_file.write("=" * 60 + "\n\n")

            # Print result object
            print("\n  Result Object:")
            result_str = json.dumps(
                result.dict() if hasattr(result, "dict") else result,
                indent=2,
                default=str,
            )
            print(result_str)
            log_file.write("RESULT OBJECT:\n")
            log_file.write(result_str + "\n\n")

            # Try to get execution logs
            log_file.write("=" * 60 + "\nEXECUTION LOGS:\n")
            try:
                logs = execution.logs()
                for entry in logs:
                    log_str = str(entry)
                    print(f"  LOG: {log_str}")
                    log_file.write(log_str + "\n")
            except Exception as e:
                error_msg = f"Could not retrieve logs: {e}"
                print(f"  {error_msg}")
                log_file.write(error_msg + "\n")

        print(f"\n  📋 Logs saved to: {log_path}")
        sys.exit(1)

    return result.dict() if hasattr(result, "dict") else result


def download_and_prepare_features(
    output_dp_id: str,
    num_runs: int = 1,
) -> dict:
    """Download quantum features from output datapool."""
    from planqk.api.client import PlanqkApiClient

    api_client = PlanqkApiClient(
        access_token=os.environ["PLANQK_ACCESS_TOKEN"],
        organization_id=os.environ["PLANQK_ORGANIZATION_ID"],
    )

    download_dir = RESULTS_DIR / "quantum_output"
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("  DOWNLOADING QUANTUM FEATURES")
    print(f"{'=' * 60}")
    print(f"  Output datapool: {output_dp_id}")

    files = api_client.api.data_pools.get_data_pool_files(id=output_dp_id)
    print(f"  Files in datapool: {len(files)}")

    for file_info in files:
        print(f"    - {file_info.name}")
        file_stream = api_client.api.data_pools.get_data_pool_file(
            id=output_dp_id, file_id=file_info.id
        )
        fpath = download_dir / file_info.name
        with open(fpath, "wb") as f:
            for chunk in file_stream:
                f.write(chunk)

    # Load quantum features
    results = {}
    for run_idx in range(num_runs):
        Xq_train = np.load(download_dir / f"1_Xq_train_{run_idx}.npy")
        Xq_test = np.load(download_dir / f"1_Xq_validation_{run_idx}.npy")
        yq_train = np.load(download_dir / f"1_yq_train_{run_idx}.npy")
        yq_test = np.load(download_dir / f"1_yq_validation_{run_idx}.npy")
        results[run_idx] = {
            "Xq_train": Xq_train,
            "Xq_test": Xq_test,
            "yq_train": yq_train,
            "yq_test": yq_test,
        }
        print(f"  Run {run_idx}: train={Xq_train.shape}, test={Xq_test.shape}")

    return results


# ── Model Evaluation ──────────────────────────────────────────────────────


def train_evaluate_lgb(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    name: str,
) -> float:
    """Train LightGBM and return test AUC."""
    import lightgbm

    model = LGBMClassifier(**LGB_PARAMS_SMALL)

    fit_kwargs = {"eval_set": [(X_test, y_test)]}
    fit_kwargs["callbacks"] = [
        lightgbm.early_stopping(50),
        lightgbm.log_evaluation(0),
    ]

    model.fit(X_train, y_train, **fit_kwargs)
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    # Feature importances
    importances = dict(
        zip(
            (
                X_train.columns
                if isinstance(X_train, pd.DataFrame)
                else [f"f{i}" for i in range(X_train.shape[1])]
            ),
            model.feature_importances_,
        )
    )
    ranked = sorted(importances.items(), key=lambda x: -x[1])

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Test AUC: {auc:.5f}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Train rows: {X_train.shape[0]}")
    print("  Top 10 features:")
    for i, (feat, imp) in enumerate(ranked[:10]):
        print(f"    {i + 1:2d}. {feat:<35s} {imp:>6.0f}")

    return auc


def run_evaluation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    quantum_features: dict | None = None,
) -> None:
    """Compare classical vs quantum-enhanced models with all feature configurations."""
    print(f"\n{'=' * 60}")
    print("  MODEL EVALUATION")
    print(f"{'=' * 60}")

    # Feature engineering
    train_eng = engineer_features(train_df)
    test_eng = engineer_features(test_df)

    # Encode target if categorical
    y_train = train_df[TARGET].values
    y_test = test_df[TARGET].values

    # Map categorical to numeric if needed
    if isinstance(y_train[0], str):
        unique_vals = sorted(set(y_train) | set(y_test))
        mapping = {val: i for i, val in enumerate(unique_vals)}
        y_train = np.array([mapping[v] for v in y_train])
        y_test = np.array([mapping[v] for v in y_test])

    results = {}

    # ── Experiment 1: Raw features only (all 13) ──
    X_train_raw = train_eng[RAW_FEATURES].reset_index(drop=True)
    X_test_raw = test_eng[RAW_FEATURES].reset_index(drop=True)

    print("\n  DEBUG: Data shapes for experiments")
    print(f"    y_train: {len(y_train)}, y_test: {len(y_test)}")
    print(f"    X_train_raw: {X_train_raw.shape}")
    print(f"    X_test_raw: {X_test_raw.shape}")

    auc_raw = train_evaluate_lgb(
        X_train_raw,
        y_train,
        X_test_raw,
        y_test,
        "Exp 1: Raw Features Only (all 13)",
    )
    results["raw_all"] = auc_raw

    # ── Experiment 2: Ablation-pruned features (13 raw + 11 engineered) ──
    X_train_classical = train_eng[FEATURES_ABLATION_PRUNED].reset_index(drop=True)
    X_test_classical = test_eng[FEATURES_ABLATION_PRUNED].reset_index(drop=True)

    auc_classical = train_evaluate_lgb(
        X_train_classical,
        y_train,
        X_test_classical,
        y_test,
        "Exp 2: Ablation-Pruned Features (13 raw + 11 engineered)",
    )
    results["classical"] = auc_classical

    # ── Experiment 3: Quantum features (if available) ──
    if quantum_features and 0 in quantum_features:
        Xq_train = quantum_features[0]["Xq_train"]
        Xq_test = quantum_features[0]["Xq_test"]

        print("\n  DEBUG: Quantum feature shapes")
        print(f"    Xq_train shape: {Xq_train.shape}, dtype: {Xq_train.dtype}")
        print(f"    Xq_test shape: {Xq_test.shape}, dtype: {Xq_test.dtype}")
        print(
            f"    Xq_train NaNs: {np.isnan(Xq_train).sum()}, Infs: {np.isinf(Xq_train).sum()}"
        )
        print(
            f"    Xq_train min: {np.nanmin(Xq_train):.4f}, max: {np.nanmax(Xq_train):.4f}"
        )
        print(
            f"    Xq_train mean: {np.nanmean(Xq_train):.4f}, std: {np.nanstd(Xq_train):.4f}"
        )
        print(f"    X_train_classical shape: {X_train_classical.shape}")

        X_train_combined = pd.concat(
            [
                X_train_classical,
                pd.DataFrame(
                    Xq_train,
                    columns=[f"qf_{i}" for i in range(Xq_train.shape[1])],
                ),
            ],
            axis=1,
        )
        X_test_combined = pd.concat(
            [
                X_test_classical,
                pd.DataFrame(
                    Xq_test,
                    columns=[f"qf_{i}" for i in range(Xq_test.shape[1])],
                ),
            ],
            axis=1,
        )

        auc_quantum = train_evaluate_lgb(
            X_train_combined,
            y_train,
            X_test_combined,
            y_test,
            "Exp 3: Ablation-Pruned + Quantum Features",
        )
        results["quantum"] = auc_quantum

        # ── Experiment 4: Quantum features only ──
        X_train_qonly = pd.DataFrame(
            Xq_train,
            columns=[f"qf_{i}" for i in range(Xq_train.shape[1])],
        )
        X_test_qonly = pd.DataFrame(
            Xq_test,
            columns=[f"qf_{i}" for i in range(Xq_test.shape[1])],
        )

        auc_qonly = train_evaluate_lgb(
            X_train_qonly,
            y_train,
            X_test_qonly,
            y_test,
            "Exp 4: Quantum Features Only",
        )
        results["quantum_only"] = auc_qonly

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Experiment':<50s} {'AUC':>8s} {'Delta':>8s}")
    print(f"  {'-' * 50} {'-' * 8} {'-' * 8}")

    baseline = results.get("raw_all", 0)
    for key, label in [
        ("raw_all", "Raw Features Only (all 13)"),
        ("classical", "Ablation-Pruned (best classical)"),
        ("quantum_only", "Quantum Features Only"),
        ("quantum", "Ablation-Pruned + Quantum"),
    ]:
        if key in results:
            delta = results[key] - baseline
            marker = " <-- baseline" if key == "raw_all" else ""
            print(f"  {label:<50s} {results[key]:>8.5f} {delta:>+8.5f}{marker}")


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Quantum Feature Extraction with Rimay API"
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip data upload, use existing metadata",
    )
    parser.add_argument(
        "--skip-submit",
        action="store_true",
        help="Skip job submission, use existing metadata",
    )
    parser.add_argument(
        "--skip-quantum",
        action="store_true",
        help="Skip quantum computation, only run classical models",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only download and evaluate (assumes job is complete)",
    )
    parser.add_argument(
        "--output-datapool-id",
        type=str,
        default=None,
        help="Output datapool ID (for --evaluate-only with manual datapool)",
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=100,
        help="Measurement shots per circuit (default: 100)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Independent runs (default: 1)",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load and split data (always needed for evaluation)
    print(f"\n{'=' * 60}")
    print("  LOADING DATA")
    print(f"{'=' * 60}")
    df = load_data()
    train, test = create_splits(df)
    print(f"  Train rows: {len(train)}")
    print(f"  Test rows:  {len(test)}")

    # If evaluate-only with manual datapool ID, skip everything and download directly
    if args.evaluate_only and args.output_datapool_id:
        print(f"\n{'=' * 60}")
        print("  EVALUATE-ONLY MODE")
        print(f"{'=' * 60}")
        print("  Downloading quantum features from datapool...")
        print(f"  Datapool ID: {args.output_datapool_id}\n")

        quantum_features = download_and_prepare_features(
            args.output_datapool_id, num_runs=args.num_runs
        )
    else:
        # ────────────────────────────────────────────────────────────────────────
        # STAGE 1: DATA UPLOAD (with manual approval)
        # ────────────────────────────────────────────────────────────────────────

        if args.evaluate_only:
            print("\n  [Skipping upload, loading from existing metadata]")
            metadata_path = RESULTS_DIR / "rimay_metadata.json"
            if not metadata_path.exists():
                print(f"ERROR: No metadata at {metadata_path}")
                print("\nUsage with manual datapool ID:")
                print(
                    "  uv run python quantum_rimay_workflow.py --evaluate-only --output-datapool-id <ID>"
                )
                sys.exit(1)
            with open(metadata_path) as f:
                metadata = json.load(f)
            input_dp_id = metadata["input_datapool_id"]
        elif args.skip_upload:
            print("\n  [Skipping upload, using existing metadata]")
            metadata_path = RESULTS_DIR / "rimay_metadata.json"
            if not metadata_path.exists():
                print(f"ERROR: No metadata at {metadata_path}")
                sys.exit(1)
            with open(metadata_path) as f:
                metadata = json.load(f)
            input_dp_id = metadata["input_datapool_id"]
        else:
            # Prepare and upload data
            X_train = train[RAW_FEATURES].reset_index(drop=True).astype(float)
            y_train = train[[TARGET]].reset_index(drop=True)
            X_test = test[RAW_FEATURES].reset_index(drop=True).astype(float)
            y_test = test[[TARGET]].reset_index(drop=True)

            dataset_path = prepare_dataset(X_train, y_train, X_test, y_test)
            print(f"\n  Dataset prepared: {dataset_path}")

            # Upload with user approval
            input_dp_id, _ = upload_data_and_wait_approval(dataset_path)

        # ────────────────────────────────────────────────────────────────────────
        # STAGE 2: CREATE & APPROVE DATAPOOLS & SUBMIT JOB (unless skip requested)
        # ────────────────────────────────────────────────────────────────────────

        quantum_features = None

        if args.skip_quantum:
            print(f"\n{'=' * 60}")
            print("  SKIPPING QUANTUM COMPUTATION")
            print(f"{'=' * 60}")
            print("  Running classical feature comparison only...\n")
        elif args.evaluate_only:
            print("\n  [Evaluate-only mode]")

            # Use manual datapool ID if provided
            if args.output_datapool_id:
                output_dp_id = args.output_datapool_id
                num_runs_param = args.num_runs
                print(f"  Using manual output datapool ID: {output_dp_id}")
            else:
                metadata_path = RESULTS_DIR / "rimay_metadata.json"
                if not metadata_path.exists():
                    print(f"ERROR: No metadata at {metadata_path}")
                    print(
                        "\nUsage: Provide output datapool ID with --output-datapool-id"
                    )
                    print(
                        "  uv run python quantum_rimay_workflow.py --evaluate-only --output-datapool-id <ID>"
                    )
                    sys.exit(1)
                with open(metadata_path) as f:
                    metadata = json.load(f)
                output_dp_id = metadata["output_datapool_id"]
                num_runs_param = metadata["num_runs"]
                print(f"  Loaded output datapool ID from metadata: {output_dp_id}")

            quantum_features = download_and_prepare_features(
                output_dp_id, num_runs=num_runs_param
            )
        elif args.skip_submit:
            print("\n  [Skipping job submission, using existing metadata]")
            metadata_path = RESULTS_DIR / "rimay_metadata.json"
            if not metadata_path.exists():
                print(f"ERROR: No metadata at {metadata_path}")
                sys.exit(1)
            with open(metadata_path) as f:
                metadata = json.load(f)
            output_dp_id = metadata["output_datapool_id"]

            quantum_features = download_and_prepare_features(
                output_dp_id, num_runs=metadata["num_runs"]
            )
        else:
            # Full workflow: upload data, create datapools, submit job, wait, download

            # Stage 2a: Create output datapool and wait for approval
            output_dp_id, config = create_output_datapool_and_wait_approval(
                input_dp_id,
                num_shots=args.num_shots,
                num_runs=args.num_runs,
            )

            # Stage 2b: Submit quantum job (after both datapools approved)
            metadata, client, execution = submit_quantum_job(config)
            output_dp_id = metadata["output_datapool_id"]

            # ────────────────────────────────────────────────────────────────────
            # STAGE 3: WAIT FOR COMPLETION
            # ────────────────────────────────────────────────────────────────────

            wait_for_quantum_result(execution)

            # ────────────────────────────────────────────────────────────────────
            # STAGE 4: DOWNLOAD & EVALUATE
            # ────────────────────────────────────────────────────────────────────

            metadata_path = RESULTS_DIR / "rimay_metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)

            quantum_features = download_and_prepare_features(
                output_dp_id, num_runs=metadata["num_runs"]
            )

    run_evaluation(train, test, quantum_features)

    print(f"\n{'=' * 60}")
    print("  ✓ WORKFLOW COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print("\n  Files created:")
    print("    - dataset.json               (your data)")
    print("    - rimay_metadata.json        (execution tracking)")
    print("    - quantum_output/            (quantum features)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
