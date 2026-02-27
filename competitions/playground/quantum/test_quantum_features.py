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
  # Small-data experiments only (default, fast)
  uv run python test_quantum_features.py

  # Full-data quantum features (~15 min for 630K rows)
  uv run python test_quantum_features.py --full-quantum

  # Rimay API (requires env vars, see below)
  uv run python test_quantum_features.py --rimay

  # Rimay submit only / evaluate only
  uv run python test_quantum_features.py --rimay --submit-only
  uv run python test_quantum_features.py --rimay --evaluate-only

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
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

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

    The holdout is sampled first (stratified), then quantum_train from the rest.
    full_train = everything except holdout.
    """
    rng = np.random.RandomState(42)

    # Stratified holdout sample (300 rows)
    holdout_idx = []
    for label in [0, 1]:
        class_idx = df.index[df[TARGET] == label].tolist()
        n_sample = int(HOLDOUT_SAMPLE * (len(class_idx) / len(df)))
        holdout_idx.extend(rng.choice(class_idx, size=n_sample, replace=False))
    holdout = df.loc[holdout_idx].reset_index(drop=True)

    # Everything except holdout = full_train
    remaining = df.drop(index=holdout_idx).reset_index(drop=True)

    # Stratified quantum_train sample (700 rows) from remaining
    qtrain_idx = []
    for label in [0, 1]:
        class_idx = remaining.index[remaining[TARGET] == label].tolist()
        n_sample = int(TRAIN_SAMPLE * (len(class_idx) / len(remaining)))
        qtrain_idx.extend(rng.choice(class_idx, size=n_sample, replace=False))
    quantum_train = remaining.loc[qtrain_idx].reset_index(drop=True)

    print(
        f"Splits: quantum_train={len(quantum_train)}, "
        f"holdout={len(holdout)}, full_train={len(remaining)}"
    )
    print(
        f"  quantum_train target dist: "
        f"{quantum_train[TARGET].value_counts().to_dict()}"
    )
    print(f"  holdout target dist: {holdout[TARGET].value_counts().to_dict()}")

    return quantum_train, holdout, remaining


# ── Rimay API ─────────────────────────────────────────────────────────────


def submit_to_rimay(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    num_shots: int = 500,
    num_runs: int = 1,
) -> dict:
    """Upload data to Rimay Simulator and return execution metadata."""
    from planqk.api.client import PlanqkApiClient
    from planqk.service.client import PlanqkServiceClient

    access_token = os.environ["PLANQK_ACCESS_TOKEN"]
    org_id = os.environ["PLANQK_ORGANIZATION_ID"]
    consumer_key = os.environ["PLANQK_CONSUMER_KEY"]
    consumer_secret = os.environ["PLANQK_CONSUMER_SECRET"]
    service_endpoint = os.environ["PLANQK_SERVICE_ENDPOINT"]

    total_samples = len(X_train) + len(X_test)
    n_features = X_train.shape[1]
    assert total_samples <= MAX_SAMPLES, (
        f"Total samples {total_samples} exceeds limit {MAX_SAMPLES}"
    )
    assert n_features <= MAX_FEATURES, (
        f"Features {n_features} exceeds limit {MAX_FEATURES}"
    )

    print(
        f"\nRimay submission: {len(X_train)} train + {len(X_test)} test = "
        f"{total_samples} samples, {n_features} features"
    )

    # Prepare dataset
    dataset = {
        "training_tabular_data": X_train.to_dict(),
        "training_target_data": y_train.to_dict(),
        "test_tabular_data": X_test.to_dict(),
        "test_target_data": y_test.to_dict(),
    }

    dataset_path = RESULTS_DIR / "dataset.json"
    with open(dataset_path, "w") as f:
        json.dump(dataset, f)
    print(f"  Dataset saved to {dataset_path}")

    # Create datapools (use api.data_pools, not deprecated data_pools)
    api_client = PlanqkApiClient(
        access_token=access_token,
        organization_id=org_id,
    )

    input_dp = api_client.api.data_pools.create_data_pool(
        name="Rimay Heart Disease Input"
    )
    print(f"  Input datapool: {input_dp.id}")

    with open(dataset_path, "rb") as f:
        api_client.api.data_pools.add_data_pool_file(
            id=input_dp.id, file=("data.json", f)
        )

    output_dp = api_client.api.data_pools.create_data_pool(
        name="Rimay Heart Disease Output"
    )
    print(f"  Output datapool: {output_dp.id}")

    # Submit to Rimay
    service_input = {
        "data": {
            "input_data_pool": {"id": input_dp.id, "ref": "DATAPOOL"},
            "output_data_pool": {"id": output_dp.id, "ref": "DATAPOOL"},
            "num_shots": num_shots,
            "num_runs": num_runs,
        }
    }

    client = PlanqkServiceClient(
        service_endpoint=service_endpoint,
        access_key_id=consumer_key,
        secret_access_key=consumer_secret,
    )

    print("  Submitting to Rimay Simulator...")
    execution = client.run(request=service_input)

    # Save metadata for later retrieval
    metadata = {
        "input_datapool_id": input_dp.id,
        "output_datapool_id": output_dp.id,
        "execution_id": execution.id if hasattr(execution, "id") else str(execution),
        "num_shots": num_shots,
        "num_runs": num_runs,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": n_features,
    }
    metadata_path = RESULTS_DIR / "rimay_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Metadata saved to {metadata_path}")

    return metadata, client, execution


def wait_for_result(execution) -> dict:
    """Poll until Rimay execution completes."""
    print(f"  Execution ID: {execution.id}")
    print(f"  Status: {execution.status}")
    print("  Waiting for completion (polling every 10s)...")
    start = time.time()

    # The SDK's result() method blocks until completion
    result = execution.result()
    elapsed = time.time() - start
    execution.refresh()
    print(f"  Execution completed in {elapsed:.1f}s (status: {execution.status})")

    if execution.status == "FAILED":
        print("\n  ERROR: Rimay execution FAILED.")
        print(f"  Execution ID: {execution.id}")
        try:
            logs = execution.logs()
            for entry in logs:
                print(f"  LOG: {entry}")
        except Exception:
            pass
        result_dict = result.dict() if hasattr(result, "dict") else result
        print(f"  Result: {result_dict}")
        sys.exit(1)

    result_dict = result.dict() if hasattr(result, "dict") else result
    return result_dict


def download_quantum_features(output_datapool_id: str, num_runs: int = 1) -> dict:
    """Download quantum features from output datapool."""
    from planqk.api.client import PlanqkApiClient

    api_client = PlanqkApiClient(
        access_token=os.environ["PLANQK_ACCESS_TOKEN"],
        organization_id=os.environ["PLANQK_ORGANIZATION_ID"],
    )

    download_dir = RESULTS_DIR / "quantum_output"
    download_dir.mkdir(parents=True, exist_ok=True)

    files = api_client.api.data_pools.get_data_pool_files(id=output_datapool_id)
    print(f"\n  Output datapool contains {len(files)} files:")

    for file_info in files:
        print(f"    - {file_info.name}")
        file_stream = api_client.api.data_pools.get_data_pool_file(
            id=output_datapool_id, file_id=file_info.id
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
        print(
            f"  Run {run_idx}: Xq_train={Xq_train.shape}, Xq_test={Xq_test.shape}"
        )

    return results


# ── Local Quantum Feature Extraction (Qiskit) ─────────────────────────────


def build_feature_circuit(n_qubits: int, x: np.ndarray):
    """Build a parameterized quantum circuit encoding classical features.

    Architecture (inspired by Rimay's spin-glass Hamiltonian approach):
      Layer 1 — Angle encoding: RY(x_i * pi) on qubit i
      Layer 2 — ZZ entanglement: CNOT-RZ(x_i * x_{i+1})-CNOT for adjacent pairs
      Layer 3 — Non-linear re-encoding: RY(x_i^2 * pi) on qubit i

    This creates a quantum state whose expectation values encode both
    individual feature information (single-body) and pairwise correlations
    (two-body) that classical feature engineering can't easily replicate.
    """
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)

    # Layer 1: angle encoding
    for i in range(n_qubits):
        qc.ry(float(x[i] * np.pi), i)

    # Layer 2: ZZ entanglement (nearest-neighbor spin-glass interactions)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
        qc.rz(float(x[i] * x[i + 1] * np.pi), i + 1)
        qc.cx(i, i + 1)

    # Layer 3: non-linear re-encoding (counterdiabatic-inspired)
    for i in range(n_qubits):
        qc.ry(float(x[i] ** 2 * np.pi), i)

    return qc


def build_observables(n_qubits: int):
    """Build Pauli Z observables for single-body and two-body measurements.

    Returns:
        observables: list of SparsePauliOp
        names: list of str labels for each observable
    """
    from qiskit.quantum_info import SparsePauliOp

    observables = []
    names = []

    # Single-body: <Z_i> for each qubit
    for i in range(n_qubits):
        pauli_str = ["I"] * n_qubits
        pauli_str[n_qubits - 1 - i] = "Z"  # Qiskit uses little-endian
        observables.append(SparsePauliOp("".join(pauli_str)))
        names.append(f"qf_Z{i}")

    # Two-body: <Z_i Z_j> for nearest-neighbor pairs
    for i in range(n_qubits - 1):
        pauli_str = ["I"] * n_qubits
        pauli_str[n_qubits - 1 - i] = "Z"
        pauli_str[n_qubits - 1 - (i + 1)] = "Z"
        observables.append(SparsePauliOp("".join(pauli_str)))
        names.append(f"qf_ZZ{i}_{i + 1}")

    return observables, names


def extract_quantum_features_local(
    X: np.ndarray,
    n_qubits: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract quantum features from classical data using Qiskit statevector simulation.

    For each sample, builds a parameterized quantum circuit, computes the
    statevector, and extracts expectation values of Z (single-body) and ZZ
    (two-body) observables.

    Args:
        X: Feature matrix (n_samples, n_features), should be scaled to [0, 1].
        n_qubits: Number of qubits (defaults to n_features).

    Returns:
        quantum_features: (n_samples, n_qubits + n_qubits-1) array
        feature_names: list of feature name strings
    """
    from qiskit.primitives import StatevectorEstimator

    n_samples, n_features = X.shape
    if n_qubits is None:
        n_qubits = n_features

    observables, names = build_observables(n_qubits)
    estimator = StatevectorEstimator()

    print(f"\n  Local quantum feature extraction ({n_qubits} qubits):")
    print(f"    Single-body observables: {n_qubits}")
    print(f"    Two-body observables: {n_qubits - 1}")
    print(f"    Total quantum features: {len(observables)}")
    print(f"    Samples to process: {n_samples}")

    quantum_features = np.zeros((n_samples, len(observables)))

    start = time.time()
    # Batch all circuits and observables for efficient execution
    pubs = []
    for idx in range(n_samples):
        circuit = build_feature_circuit(n_qubits, X[idx])
        for obs in observables:
            pubs.append((circuit, obs))

    # Run all in one call
    print(f"    Running {len(pubs)} circuit-observable pairs...")
    result = estimator.run(pubs)
    all_values = result.result()

    for idx in range(n_samples):
        for j in range(len(observables)):
            pub_idx = idx * len(observables) + j
            quantum_features[idx, j] = float(
                all_values[pub_idx].data.evs
            )

    elapsed = time.time() - start
    print(f"    Completed in {elapsed:.1f}s ({elapsed / n_samples * 1000:.1f}ms/sample)")

    return quantum_features, names


def generate_local_quantum_features(
    quantum_train: pd.DataFrame,
    holdout: pd.DataFrame,
) -> dict:
    """Generate quantum features for train and holdout using local simulation."""
    from sklearn.preprocessing import MinMaxScaler

    # Scale raw features to [0, 1] for quantum encoding
    scaler = MinMaxScaler()
    X_train_raw = quantum_train[RAW_FEATURES].values.astype(float)
    X_test_raw = holdout[RAW_FEATURES].values.astype(float)

    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    print("\nGenerating quantum features (train)...")
    Xq_train, qf_names = extract_quantum_features_local(X_train_scaled)

    print("\nGenerating quantum features (holdout)...")
    Xq_test, _ = extract_quantum_features_local(X_test_scaled)

    return {
        0: {
            "Xq_train": Xq_train,
            "Xq_test": Xq_test,
            "yq_train": quantum_train[TARGET].values,
            "yq_test": holdout[TARGET].values,
        },
        "feature_names": qf_names,
    }


# ── Fast Numpy Quantum Simulator ─────────────────────────────────────────


def _apply_ry(state: np.ndarray, qubit: int, theta: float, n_qubits: int) -> np.ndarray:
    """Apply RY(theta) gate to a qubit in the statevector.

    Qiskit uses little-endian ordering: qubit 0 is the least significant bit,
    which maps to the last axis (n_qubits-1) in the tensor representation.
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    dim = 2**n_qubits
    axis = n_qubits - 1 - qubit  # Map Qiskit qubit to tensor axis
    state = state.reshape([2] * n_qubits)
    state = np.moveaxis(state, axis, 0)
    new_0 = c * state[0] - s * state[1]
    new_1 = s * state[0] + c * state[1]
    state = np.stack([new_0, new_1], axis=0)
    state = np.moveaxis(state, 0, axis)
    return state.reshape(dim)


def _apply_zz_rotation(
    state: np.ndarray, qubit_a: int, qubit_b: int, theta: float, n_qubits: int
) -> np.ndarray:
    """Apply exp(-i * theta/2 * Z_a Z_b) — the CNOT-RZ-CNOT decomposition.

    This is diagonal in the computational basis: each basis state |...b_a...b_b...⟩
    picks up phase exp(-i * theta/2 * (-1)^{b_a} * (-1)^{b_b}).
    """
    dim = 2**n_qubits
    basis = np.arange(dim, dtype=np.int64)
    # Qiskit little-endian: qubit i = bit i
    z_a = 1.0 - 2.0 * ((basis >> qubit_a) & 1).astype(np.float64)
    z_b = 1.0 - 2.0 * ((basis >> qubit_b) & 1).astype(np.float64)
    phases = np.exp(-1j * theta / 2 * z_a * z_b)
    return state * phases


def _compute_expectations(state: np.ndarray, n_qubits: int) -> np.ndarray:
    """Compute ⟨Z_i⟩ and ⟨Z_i Z_j⟩ from statevector using bit operations."""
    probs = np.abs(state) ** 2
    dim = 2**n_qubits
    basis = np.arange(dim, dtype=np.int64)

    features = np.empty(n_qubits + n_qubits - 1, dtype=np.float64)

    # Single-body: ⟨Z_i⟩ = sum_k (-1)^bit(k,i) * |c_k|^2
    for i in range(n_qubits):
        signs = 1.0 - 2.0 * ((basis >> i) & 1).astype(np.float64)
        features[i] = np.dot(signs, probs)

    # Two-body: ⟨Z_i Z_j⟩ for nearest-neighbor pairs
    for idx, i in enumerate(range(n_qubits - 1)):
        j = i + 1
        signs = 1.0 - 2.0 * (((basis >> i) ^ (basis >> j)) & 1).astype(np.float64)
        features[n_qubits + idx] = np.dot(signs, probs)

    return features


def _process_sample(x: np.ndarray, n_qubits: int) -> np.ndarray:
    """Process a single sample through the quantum circuit. Used by multiprocessing."""
    dim = 2**n_qubits
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0

    # Layer 1: RY angle encoding
    for i in range(n_qubits):
        state = _apply_ry(state, i, float(x[i] * np.pi), n_qubits)

    # Layer 2: ZZ entanglement (nearest-neighbor)
    for i in range(n_qubits - 1):
        state = _apply_zz_rotation(
            state, i, i + 1, float(x[i] * x[i + 1] * np.pi), n_qubits
        )

    # Layer 3: RY non-linear re-encoding
    for i in range(n_qubits):
        state = _apply_ry(state, i, float(x[i] ** 2 * np.pi), n_qubits)

    return _compute_expectations(state, n_qubits)


def _process_chunk(args: tuple) -> np.ndarray:
    """Process a chunk of samples. Picklable for multiprocessing."""
    X_chunk, n_qubits = args
    n_samples = X_chunk.shape[0]
    n_out = n_qubits + n_qubits - 1
    result = np.zeros((n_samples, n_out), dtype=np.float64)
    for idx in range(n_samples):
        result[idx] = _process_sample(X_chunk[idx], n_qubits)
    return result


def extract_quantum_features_fast(
    X: np.ndarray,
    n_qubits: int | None = None,
    n_workers: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract quantum features using pure numpy statevector simulation.

    Same circuit as the Qiskit version (RY encoding → ZZ entanglement →
    RY re-encoding) but ~30x faster, with multiprocessing for large datasets.

    Args:
        X: Feature matrix (n_samples, n_features), should be scaled to [0, 1].
        n_qubits: Number of qubits (defaults to n_features).
        n_workers: Number of parallel workers (defaults to CPU count).

    Returns:
        quantum_features: (n_samples, n_qubits + n_qubits-1) array
        feature_names: list of feature name strings
    """
    from multiprocessing import Pool, cpu_count

    n_samples, n_features = X.shape
    if n_qubits is None:
        n_qubits = n_features
    if n_workers is None:
        n_workers = cpu_count()

    n_out = n_qubits + n_qubits - 1
    names = [f"qf_Z{i}" for i in range(n_qubits)]
    names += [f"qf_ZZ{i}_{i + 1}" for i in range(n_qubits - 1)]

    print(f"\n  Fast numpy quantum feature extraction ({n_qubits} qubits):")
    print(f"    Single-body: {n_qubits}, Two-body: {n_qubits - 1}")
    print(f"    Total quantum features: {n_out}")
    print(f"    Samples to process: {n_samples}")

    # Use multiprocessing for large datasets
    if n_samples > 1000 and n_workers > 1:
        print(f"    Using {n_workers} parallel workers")
        # Split into chunks
        chunk_size = max(1, n_samples // n_workers)
        chunks = []
        for i in range(0, n_samples, chunk_size):
            chunks.append((X[i : i + chunk_size], n_qubits))

        start = time.time()
        with Pool(n_workers) as pool:
            results = pool.map(_process_chunk, chunks)
        quantum_features = np.vstack(results)
    else:
        print(f"    Single-threaded mode")
        quantum_features = np.zeros((n_samples, n_out), dtype=np.float64)
        start = time.time()
        for idx in range(n_samples):
            quantum_features[idx] = _process_sample(X[idx], n_qubits)

    elapsed = time.time() - start
    print(
        f"    Completed in {elapsed:.1f}s "
        f"({elapsed / n_samples * 1000:.2f}ms/sample, "
        f"{n_samples / elapsed:.0f} samples/s)"
    )

    return quantum_features, names


def generate_quantum_features_fast(
    train_df: pd.DataFrame,
    holdout: pd.DataFrame,
) -> dict:
    """Generate quantum features using the fast numpy simulator."""
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    X_train_raw = train_df[RAW_FEATURES].values.astype(float)
    X_test_raw = holdout[RAW_FEATURES].values.astype(float)

    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    print(f"\nGenerating quantum features (train, {len(train_df)} rows)...")
    Xq_train, qf_names = extract_quantum_features_fast(X_train_scaled)

    print(f"\nGenerating quantum features (holdout, {len(holdout)} rows)...")
    Xq_test, _ = extract_quantum_features_fast(X_test_scaled)

    return {
        0: {
            "Xq_train": Xq_train,
            "Xq_test": Xq_test,
            "yq_train": train_df[TARGET].values,
            "yq_test": holdout[TARGET].values,
        },
        "feature_names": qf_names,
    }


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

    fit_kwargs["eval_set"] = [(X_holdout, y_holdout)]
    fit_kwargs["callbacks"] = [
        lightgbm.early_stopping(50),
        lightgbm.log_evaluation(0),
    ]

    model.fit(X_train, y_train, **fit_kwargs)
    preds = model.predict_proba(X_holdout)[:, 1]
    auc = roc_auc_score(y_holdout, preds)

    # Feature importances (top 10)
    importances = dict(zip(
        (X_train.columns if isinstance(X_train, pd.DataFrame)
         else [f"f{i}" for i in range(X_train.shape[1])]),
        model.feature_importances_,
    ))
    ranked = sorted(importances.items(), key=lambda x: -x[1])

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Holdout AUC: {auc:.5f}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Train rows: {X_train.shape[0]}")
    print(f"  Top 10 features:")
    for i, (feat, imp) in enumerate(ranked[:10]):
        print(f"    {i+1:2d}. {feat:<35s} {imp:>6.0f}")

    return auc


def run_evaluation(
    quantum_train: pd.DataFrame,
    holdout: pd.DataFrame,
    full_train: pd.DataFrame,
    quantum_features: dict | None = None,
    quantum_features_full: dict | None = None,
) -> dict:
    """Run all experiments and compare."""
    results = {}

    # Engineer features for all splits
    qt_eng = engineer_features(quantum_train)
    ho_eng = engineer_features(holdout)
    ft_eng = engineer_features(full_train)

    y_qt = quantum_train[TARGET]
    y_ho = holdout[TARGET]
    y_ft = full_train[TARGET]

    # Categorical features present in ablation-pruned set
    cats_in_ablation = [c for c in CAT_FEATURES if c in FEATURES_ABLATION_PRUNED]

    # ── Experiment 1: Vanilla LightGBM (small, 700 rows) ──
    X_qt = qt_eng[FEATURES_ABLATION_PRUNED]
    X_ho = ho_eng[FEATURES_ABLATION_PRUNED]

    auc1 = train_evaluate_lgb(
        X_qt, y_qt, X_ho, y_ho,
        LGB_PARAMS_SMALL,
        "Exp 1: Vanilla LightGBM (700 rows, ablation-pruned)",
        cat_features=cats_in_ablation,
    )
    results["vanilla_small"] = auc1

    # ── Experiment 2: Quantum LightGBM (small, 700 rows + quantum features) ──
    if quantum_features is not None:
        qf = quantum_features[0]  # Use run 0
        Xq_train = qf["Xq_train"]
        Xq_test = qf["Xq_test"]

        n_qf = Xq_train.shape[1]
        qf_names = [f"qf_{i}" for i in range(n_qf)]

        # Concatenate ablation-pruned features with quantum features
        X_qt_q = pd.concat([
            X_qt.reset_index(drop=True),
            pd.DataFrame(Xq_train, columns=qf_names),
        ], axis=1)
        X_ho_q = pd.concat([
            X_ho.reset_index(drop=True),
            pd.DataFrame(Xq_test, columns=qf_names),
        ], axis=1)

        auc2 = train_evaluate_lgb(
            X_qt_q, y_qt.reset_index(drop=True),
            X_ho_q, y_ho.reset_index(drop=True),
            LGB_PARAMS_SMALL,
            f"Exp 2: Quantum LightGBM (700 rows, ablation-pruned + {n_qf} quantum)",
            cat_features=cats_in_ablation,
        )
        results["quantum_small"] = auc2

        # Also test quantum features alone
        auc2b = train_evaluate_lgb(
            pd.DataFrame(Xq_train, columns=qf_names),
            y_qt.reset_index(drop=True),
            pd.DataFrame(Xq_test, columns=qf_names),
            y_ho.reset_index(drop=True),
            LGB_PARAMS_SMALL,
            f"Exp 2b: Quantum-Only LightGBM (700 rows, {n_qf} quantum features only)",
        )
        results["quantum_only"] = auc2b
    else:
        print("\n  [Skipping Exp 2: no quantum features available]")

    # ── Experiment 3: Vanilla LightGBM (full ~630K rows) ──
    X_ft = ft_eng[FEATURES_ABLATION_PRUNED]

    auc3 = train_evaluate_lgb(
        X_ft, y_ft, X_ho, y_ho,
        LGB_PARAMS_FULL,
        f"Exp 3: Vanilla LightGBM ({len(full_train)} rows, ablation-pruned)",
        cat_features=cats_in_ablation,
    )
    results["vanilla_full"] = auc3

    # ── Experiment 4: Quantum LightGBM (full ~630K rows + quantum features) ──
    if quantum_features_full is not None:
        qf_full = quantum_features_full[0]
        Xq_ft = qf_full["Xq_train"]
        Xq_ho_full = qf_full["Xq_test"]

        n_qf = Xq_ft.shape[1]
        qf_names = [f"qf_{i}" for i in range(n_qf)]

        X_ft_q = pd.concat([
            X_ft.reset_index(drop=True),
            pd.DataFrame(Xq_ft, columns=qf_names),
        ], axis=1)
        X_ho_q_full = pd.concat([
            X_ho.reset_index(drop=True),
            pd.DataFrame(Xq_ho_full, columns=qf_names),
        ], axis=1)

        auc4 = train_evaluate_lgb(
            X_ft_q, y_ft.reset_index(drop=True),
            X_ho_q_full, y_ho.reset_index(drop=True),
            LGB_PARAMS_FULL,
            f"Exp 4: Quantum LightGBM ({len(full_train)} rows, ablation-pruned + {n_qf} quantum)",
            cat_features=cats_in_ablation,
        )
        results["quantum_full"] = auc4
    else:
        print("\n  [Skipping Exp 4: no full quantum features available]")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Experiment':<55s} {'AUC':>8s} {'Delta':>8s}")
    print(f"  {'-'*55} {'-'*8} {'-'*8}")

    baseline = results.get("vanilla_small", 0)
    for key, label in [
        ("vanilla_small", "Exp 1: Vanilla (700 rows)"),
        ("quantum_small", "Exp 2: Vanilla + Quantum (700 rows)"),
        ("quantum_only", "Exp 2b: Quantum Only (700 rows)"),
        ("vanilla_full", f"Exp 3: Vanilla ({len(full_train)} rows)"),
        ("quantum_full", f"Exp 4: Vanilla + Quantum ({len(full_train)} rows)"),
    ]:
        if key in results:
            delta = results[key] - baseline
            marker = " <-- baseline" if key == "vanilla_small" else ""
            print(
                f"  {label:<55s} {results[key]:>8.5f} "
                f"{delta:>+8.5f}{marker}"
            )

    return results


# ── Ablation Study ────────────────────────────────────────────────────────


def _train_auc_quiet(
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: np.ndarray | pd.Series,
    params: dict,
    cat_features: list[str] | None = None,
) -> float:
    """Train LightGBM and return holdout AUC (no printing)."""
    import lightgbm

    model = LGBMClassifier(**params)
    fit_kwargs: dict = {}
    if cat_features and isinstance(X_train, pd.DataFrame):
        available_cats = [c for c in cat_features if c in X_train.columns]
        if available_cats:
            fit_kwargs["categorical_feature"] = available_cats
    fit_kwargs["eval_set"] = [(X_holdout, y_holdout)]
    fit_kwargs["callbacks"] = [
        lightgbm.early_stopping(50),
        lightgbm.log_evaluation(0),
    ]
    model.fit(X_train, y_train, **fit_kwargs)
    preds = model.predict_proba(X_holdout)[:, 1]
    return roc_auc_score(y_holdout, preds)


def run_ablation(
    X_train_classical: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_holdout_classical: pd.DataFrame,
    y_holdout: np.ndarray | pd.Series,
    Xq_train: np.ndarray,
    Xq_holdout: np.ndarray,
    qf_names: list[str],
    params: dict,
    label: str,
    cat_features: list[str] | None = None,
) -> list[str]:
    """Forward selection: greedily add quantum features that improve AUC."""
    print(f"\n{'='*60}")
    print(f"  ABLATION: {label}")
    print(f"{'='*60}")

    # Baseline: classical features only
    baseline_auc = _train_auc_quiet(
        X_train_classical, y_train,
        X_holdout_classical, y_holdout,
        params, cat_features,
    )
    print(f"\n  Baseline (classical only): {baseline_auc:.5f}")
    print(f"  Classical features: {X_train_classical.shape[1]}")
    print(f"  Candidate quantum features: {len(qf_names)}")

    # Forward selection
    selected: list[str] = []
    remaining = list(range(len(qf_names)))
    current_auc = baseline_auc

    print(f"\n  {'Step':<6s} {'Added Feature':<20s} {'AUC':>8s} {'Delta':>8s} {'Action':<10s}")
    print(f"  {'-'*6} {'-'*20} {'-'*8} {'-'*8} {'-'*10}")

    step = 0
    while remaining:
        best_auc = -1.0
        best_idx = -1

        for qi in remaining:
            # Build feature set: classical + selected + candidate
            sel_indices = [qf_names.index(s) for s in selected] + [qi]
            X_tr = pd.concat([
                X_train_classical.reset_index(drop=True),
                pd.DataFrame(
                    Xq_train[:, sel_indices],
                    columns=[qf_names[i] for i in sel_indices],
                ),
            ], axis=1)
            X_ho = pd.concat([
                X_holdout_classical.reset_index(drop=True),
                pd.DataFrame(
                    Xq_holdout[:, sel_indices],
                    columns=[qf_names[i] for i in sel_indices],
                ),
            ], axis=1)

            auc = _train_auc_quiet(
                X_tr, y_train, X_ho, y_holdout, params, cat_features,
            )
            if auc > best_auc:
                best_auc = auc
                best_idx = qi

        step += 1
        delta = best_auc - current_auc
        feat_name = qf_names[best_idx]

        if best_auc > current_auc:
            selected.append(feat_name)
            current_auc = best_auc
            remaining.remove(best_idx)
            action = "ADDED"
        else:
            remaining.remove(best_idx)
            action = "SKIPPED"

        print(
            f"  {step:<6d} {feat_name:<20s} {best_auc:>8.5f} "
            f"{delta:>+8.5f} {action:<10s}"
        )

    # Summary
    print(f"\n  {'─'*50}")
    print(f"  Baseline AUC:  {baseline_auc:.5f}")
    print(f"  Final AUC:     {current_auc:.5f} ({current_auc - baseline_auc:+.5f})")
    print(f"  Selected {len(selected)}/{len(qf_names)} quantum features:")
    for feat in selected:
        print(f"    + {feat}")

    if not selected:
        print("  (no quantum features improved AUC)")

    return selected


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Quantum Feature Extraction Experiment")
    parser.add_argument(
        "--rimay", action="store_true",
        help="Use Rimay API instead of local simulation",
    )
    parser.add_argument(
        "--full-quantum", action="store_true",
        help="Generate quantum features for full ~630K dataset (takes ~3 min)",
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run forward-selection ablation on quantum features",
    )
    parser.add_argument(
        "--submit-only", action="store_true",
        help="(Rimay) Submit and save metadata, don't wait for results",
    )
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="(Rimay) Skip API submission, load results from output datapool",
    )
    parser.add_argument(
        "--num-shots", type=int, default=500,
        help="Measurement shots per circuit (default: 500)",
    )
    parser.add_argument(
        "--num-runs", type=int, default=1,
        help="Independent runs on simulator (default: 1)",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load and split data
    df = load_data()
    quantum_train, holdout, full_train = create_splits(df)

    # Save splits for reproducibility
    quantum_train.to_csv(RESULTS_DIR / "quantum_train.csv", index=False)
    holdout.to_csv(RESULTS_DIR / "holdout.csv", index=False)
    print(f"Splits saved to {RESULTS_DIR}/")

    quantum_features_full = None

    if args.rimay:
        # ── Rimay API path ──
        if not args.evaluate_only:
            X_train_raw = quantum_train[RAW_FEATURES].reset_index(drop=True).astype(float)
            y_train_raw = quantum_train[[TARGET]].reset_index(drop=True).astype(int)
            X_test_raw = holdout[RAW_FEATURES].reset_index(drop=True).astype(float)
            y_test_raw = holdout[[TARGET]].reset_index(drop=True).astype(int)

            metadata, client, execution = submit_to_rimay(
                X_train_raw, y_train_raw,
                X_test_raw, y_test_raw,
                num_shots=args.num_shots,
                num_runs=args.num_runs,
            )

            if args.submit_only:
                print("\n  Submit-only mode. Run with --evaluate-only to get results.")
                return

            wait_for_result(execution)
            output_dp_id = metadata["output_datapool_id"]
            num_runs = metadata["num_runs"]
        else:
            metadata_path = RESULTS_DIR / "rimay_metadata.json"
            if not metadata_path.exists():
                print(f"ERROR: No metadata found at {metadata_path}")
                sys.exit(1)
            with open(metadata_path) as f:
                metadata = json.load(f)
            output_dp_id = metadata["output_datapool_id"]
            num_runs = metadata["num_runs"]

        quantum_features = download_quantum_features(output_dp_id, num_runs)
    else:
        # ── Local simulation ──
        # Small-data quantum features (700 train + 300 holdout) using fast numpy
        quantum_features = generate_quantum_features_fast(quantum_train, holdout)

        # Full-data quantum features (~630K train + 300 holdout)
        if args.full_quantum:
            print(f"\n{'='*60}")
            print(f"  Generating quantum features for full dataset...")
            print(f"  ({len(full_train)} train + {len(holdout)} holdout)")
            print(f"{'='*60}")

            # Save/load cache for the expensive full computation
            cache_train = RESULTS_DIR / "quantum_features_full_train.npy"
            cache_holdout = RESULTS_DIR / "quantum_features_full_holdout.npy"

            if cache_train.exists() and cache_holdout.exists():
                print("\n  Loading cached full quantum features...")
                Xq_ft = np.load(cache_train)
                Xq_ho = np.load(cache_holdout)
                print(f"    Train: {Xq_ft.shape}, Holdout: {Xq_ho.shape}")
                n_qubits = len(RAW_FEATURES)
                qf_names = [f"qf_Z{i}" for i in range(n_qubits)]
                qf_names += [f"qf_ZZ{i}_{i + 1}" for i in range(n_qubits - 1)]
            else:
                result = generate_quantum_features_fast(full_train, holdout)
                Xq_ft = result[0]["Xq_train"]
                Xq_ho = result[0]["Xq_test"]
                qf_names = result["feature_names"]
                # Cache for re-runs
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
                "feature_names": qf_names,
            }

    # Run evaluation
    run_evaluation(
        quantum_train, holdout, full_train,
        quantum_features, quantum_features_full,
    )

    # Ablation study
    if args.ablation:
        qt_eng = engineer_features(quantum_train)
        ho_eng = engineer_features(holdout)
        ft_eng = engineer_features(full_train)
        cats_in_ablation = [c for c in CAT_FEATURES if c in FEATURES_ABLATION_PRUNED]

        qf_names = quantum_features["feature_names"]

        # Small-data ablation
        run_ablation(
            qt_eng[FEATURES_ABLATION_PRUNED],
            quantum_train[TARGET],
            ho_eng[FEATURES_ABLATION_PRUNED],
            holdout[TARGET],
            quantum_features[0]["Xq_train"],
            quantum_features[0]["Xq_test"],
            qf_names,
            LGB_PARAMS_SMALL,
            "Small data (700 rows)",
            cat_features=cats_in_ablation,
        )

        # Full-data ablation
        if quantum_features_full is not None:
            selected = run_ablation(
                ft_eng[FEATURES_ABLATION_PRUNED],
                full_train[TARGET],
                ho_eng[FEATURES_ABLATION_PRUNED],
                holdout[TARGET],
                quantum_features_full[0]["Xq_train"],
                quantum_features_full[0]["Xq_test"],
                qf_names,
                LGB_PARAMS_FULL,
                f"Full data ({len(full_train)} rows)",
                cat_features=cats_in_ablation,
            )


if __name__ == "__main__":
    main()
