"""Quantum feature generation via Rimay API, Qiskit, or fast numpy simulation.

Provides three backends for generating quantum features from classical tabular data:
  1. Rimay API (default): submit to Kipu Quantum's cloud simulator
  2. Qiskit: local statevector simulation via qiskit.primitives
  3. Fast numpy: pure numpy statevector (~30x faster than Qiskit)

All backends use the same quantum circuit architecture:
  Layer 1 — Angle encoding: RY(x_i * pi) on qubit i
  Layer 2 — ZZ entanglement: CNOT-RZ(x_i * x_{i+1})-CNOT for adjacent pairs
  Layer 3 — Non-linear re-encoding: RY(x_i^2 * pi) on qubit i
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────

MAX_FEATURES = 15
MAX_SAMPLES = 1000


# ── Rimay API ────────────────────────────────────────────────────────────


def submit_to_rimay(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    results_dir: Path,
    num_shots: int = 500,
    num_runs: int = 1,
) -> tuple:
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

    dataset_path = results_dir / "dataset.json"
    with open(dataset_path, "w") as file:
        json.dump(dataset, file)
    print(f"  Dataset saved to {dataset_path}")

    # Create datapools
    api_client = PlanqkApiClient(
        access_token=access_token,
        organization_id=org_id,
    )

    input_dp = api_client.api.data_pools.create_data_pool(
        name="Rimay Heart Disease Input"
    )
    print(f"  Input datapool: {input_dp.id}")

    with open(dataset_path, "rb") as file:
        api_client.api.data_pools.add_data_pool_file(
            id=input_dp.id, file=("data.json", file)
        )

    output_dp = api_client.api.data_pools.create_data_pool(
        name="Rimay Heart Disease Output"
    )
    print(f"  Output datapool: {output_dp.id}")

    print(
        "\n  Both datapools must be approved on the PlanQK website before submitting."
    )
    print(f"    Input:  {input_dp.id}")
    print(f"    Output: {output_dp.id}")
    input("  Press Enter once both datapools are approved...")

    # Submit to Rimay
    service_input = {
        "input_data_pool": {"id": input_dp.id, "ref": "DATAPOOL"},
        "output_data_pool": {"id": output_dp.id, "ref": "DATAPOOL"},
        "num_shots": num_shots,
        "num_runs": num_runs,
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
    metadata_path = results_dir / "rimay_metadata.json"
    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=2, default=str)
    print(f"  Metadata saved to {metadata_path}")

    return metadata, client, execution


def wait_for_result(execution) -> dict:
    """Poll until Rimay execution completes."""
    print(f"  Execution ID: {execution.id}")
    print(f"  Status: {execution.status}")
    print("  Waiting for completion (polling every 10s)...")
    start = time.time()

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


def download_quantum_features(
    output_datapool_id: str,
    results_dir: Path,
    num_runs: int = 1,
) -> dict:
    """Download quantum features from output datapool."""
    from planqk.api.client import PlanqkApiClient

    api_client = PlanqkApiClient(
        access_token=os.environ["PLANQK_ACCESS_TOKEN"],
        organization_id=os.environ["PLANQK_ORGANIZATION_ID"],
    )

    download_dir = results_dir / "quantum_output"
    download_dir.mkdir(parents=True, exist_ok=True)

    files = api_client.api.data_pools.get_data_pool_files(id=output_datapool_id)
    print(f"\n  Output datapool contains {len(files)} files:")

    for file_info in files:
        print(f"    - {file_info.name}")
        file_stream = api_client.api.data_pools.get_data_pool_file(
            id=output_datapool_id, file_id=file_info.id
        )
        fpath = download_dir / file_info.name
        with open(fpath, "wb") as file:
            for chunk in file_stream:
                file.write(chunk)

    results = {}
    for run_index in range(num_runs):
        Xq_train = np.load(download_dir / f"1_Xq_train_{run_index}.npy")
        Xq_test = np.load(download_dir / f"1_Xq_validation_{run_index}.npy")
        yq_train = np.load(download_dir / f"1_yq_train_{run_index}.npy")
        yq_test = np.load(download_dir / f"1_yq_validation_{run_index}.npy")
        results[run_index] = {
            "Xq_train": Xq_train,
            "Xq_test": Xq_test,
            "yq_train": yq_train,
            "yq_test": yq_test,
        }
        print(f"  Run {run_index}: Xq_train={Xq_train.shape}, Xq_test={Xq_test.shape}")

    return results


# ── Local Quantum Feature Extraction (Qiskit) ───────────────────────────


def build_feature_circuit(n_qubits: int, x: np.ndarray):
    """Build a parameterized quantum circuit encoding classical features."""
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
    """Build Pauli Z observables for single-body and two-body measurements."""
    from qiskit.quantum_info import SparsePauliOp

    observables = []
    names = []

    for i in range(n_qubits):
        pauli_str = ["I"] * n_qubits
        pauli_str[n_qubits - 1 - i] = "Z"
        observables.append(SparsePauliOp("".join(pauli_str)))
        names.append(f"qf_Z{i}")

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
    """Extract quantum features using Qiskit statevector simulation."""
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
    pubs = []
    for index in range(n_samples):
        circuit = build_feature_circuit(n_qubits, X[index])
        for obs in observables:
            pubs.append((circuit, obs))

    print(f"    Running {len(pubs)} circuit-observable pairs...")
    result = estimator.run(pubs)
    all_values = result.result()

    for index in range(n_samples):
        for j in range(len(observables)):
            pub_index = index * len(observables) + j
            quantum_features[index, j] = float(all_values[pub_index].data.evs)

    elapsed = time.time() - start
    print(
        f"    Completed in {elapsed:.1f}s ({elapsed / n_samples * 1000:.1f}ms/sample)"
    )

    return quantum_features, names


def generate_local_quantum_features(
    quantum_train: pd.DataFrame,
    holdout: pd.DataFrame,
    raw_features: list[str],
    target: str,
) -> dict:
    """Generate quantum features for train and holdout using Qiskit simulation."""
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(
        quantum_train[raw_features].values.astype(float)
    )
    X_test_scaled = scaler.transform(holdout[raw_features].values.astype(float))

    print("\nGenerating quantum features (train)...")
    Xq_train, feature_names = extract_quantum_features_local(X_train_scaled)

    print("\nGenerating quantum features (holdout)...")
    Xq_test, _ = extract_quantum_features_local(X_test_scaled)

    return {
        0: {
            "Xq_train": Xq_train,
            "Xq_test": Xq_test,
            "yq_train": quantum_train[target].values,
            "yq_test": holdout[target].values,
        },
        "feature_names": feature_names,
    }


# ── Fast Numpy Quantum Simulator ────────────────────────────────────────


def _apply_ry(state: np.ndarray, qubit: int, theta: float, n_qubits: int) -> np.ndarray:
    """Apply RY(theta) gate to a qubit in the statevector."""
    cos_half = np.cos(theta / 2)
    sin_half = np.sin(theta / 2)
    dim = 2**n_qubits
    axis = n_qubits - 1 - qubit
    state = state.reshape([2] * n_qubits)
    state = np.moveaxis(state, axis, 0)
    new_0 = cos_half * state[0] - sin_half * state[1]
    new_1 = sin_half * state[0] + cos_half * state[1]
    state = np.stack([new_0, new_1], axis=0)
    state = np.moveaxis(state, 0, axis)
    return state.reshape(dim)


def _apply_zz_rotation(
    state: np.ndarray, qubit_a: int, qubit_b: int, theta: float, n_qubits: int
) -> np.ndarray:
    """Apply exp(-i * theta/2 * Z_a Z_b) rotation."""
    dim = 2**n_qubits
    basis = np.arange(dim, dtype=np.int64)
    z_a = 1.0 - 2.0 * ((basis >> qubit_a) & 1).astype(np.float64)
    z_b = 1.0 - 2.0 * ((basis >> qubit_b) & 1).astype(np.float64)
    phases = np.exp(-1j * theta / 2 * z_a * z_b)
    return state * phases


def _compute_expectations(state: np.ndarray, n_qubits: int) -> np.ndarray:
    """Compute <Z_i> and <Z_i Z_j> from statevector using bit operations."""
    probs = np.abs(state) ** 2
    dim = 2**n_qubits
    basis = np.arange(dim, dtype=np.int64)

    features = np.empty(n_qubits + n_qubits - 1, dtype=np.float64)

    for i in range(n_qubits):
        signs = 1.0 - 2.0 * ((basis >> i) & 1).astype(np.float64)
        features[i] = np.dot(signs, probs)

    for index, i in enumerate(range(n_qubits - 1)):
        j = i + 1
        signs = 1.0 - 2.0 * (((basis >> i) ^ (basis >> j)) & 1).astype(np.float64)
        features[n_qubits + index] = np.dot(signs, probs)

    return features


def _process_sample(x: np.ndarray, n_qubits: int) -> np.ndarray:
    """Process a single sample through the quantum circuit."""
    dim = 2**n_qubits
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0

    for i in range(n_qubits):
        state = _apply_ry(state, i, float(x[i] * np.pi), n_qubits)

    for i in range(n_qubits - 1):
        state = _apply_zz_rotation(
            state, i, i + 1, float(x[i] * x[i + 1] * np.pi), n_qubits
        )

    for i in range(n_qubits):
        state = _apply_ry(state, i, float(x[i] ** 2 * np.pi), n_qubits)

    return _compute_expectations(state, n_qubits)


def _process_chunk(args: tuple) -> np.ndarray:
    """Process a chunk of samples. Picklable for multiprocessing."""
    X_chunk, n_qubits = args
    n_samples = X_chunk.shape[0]
    n_out = n_qubits + n_qubits - 1
    result = np.zeros((n_samples, n_out), dtype=np.float64)
    for index in range(n_samples):
        result[index] = _process_sample(X_chunk[index], n_qubits)
    return result


def extract_quantum_features_fast(
    X: np.ndarray,
    n_qubits: int | None = None,
    n_workers: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract quantum features using pure numpy statevector simulation."""
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

    if n_samples > 1000 and n_workers > 1:
        print(f"    Using {n_workers} parallel workers")
        chunk_size = max(1, n_samples // n_workers)
        chunks = []
        for i in range(0, n_samples, chunk_size):
            chunks.append((X[i : i + chunk_size], n_qubits))

        start = time.time()
        with Pool(n_workers) as pool:
            results = pool.map(_process_chunk, chunks)
        quantum_features = np.vstack(results)
    else:
        print("    Single-threaded mode")
        quantum_features = np.zeros((n_samples, n_out), dtype=np.float64)
        start = time.time()
        for index in range(n_samples):
            quantum_features[index] = _process_sample(X[index], n_qubits)

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
    raw_features: list[str],
    target: str,
) -> dict:
    """Generate quantum features using the fast numpy simulator."""
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(train_df[raw_features].values.astype(float))
    X_test_scaled = scaler.transform(holdout[raw_features].values.astype(float))

    print(f"\nGenerating quantum features (train, {len(train_df)} rows)...")
    Xq_train, feature_names = extract_quantum_features_fast(X_train_scaled)

    print(f"\nGenerating quantum features (holdout, {len(holdout)} rows)...")
    Xq_test, _ = extract_quantum_features_fast(X_test_scaled)

    return {
        0: {
            "Xq_train": Xq_train,
            "Xq_test": Xq_test,
            "yq_train": train_df[target].values,
            "yq_test": holdout[target].values,
        },
        "feature_names": feature_names,
    }
