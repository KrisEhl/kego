#!/usr/bin/env bash
# Setup a new Ray worker node and connect it to the cluster head.
#
# Usage:
#   ./scripts/setup-ray-worker.sh <worker-ip> [head-ip]
#
# Example:
#   ./scripts/setup-ray-worker.sh 192.168.178.33 192.168.178.32
#
# Prerequisites:
#   - SSH access to the worker as user "kristian" (key-based auth)
#   - NVIDIA GPU drivers already installed on the worker
#   - curl available on the worker

set -euo pipefail

WORKER_IP="${1:?Usage: $0 <worker-ip> [head-ip]}"
export RAY_API_SERVER_IP="${2:-192.168.178.32}"
export RAY_API_SERVER_PORT="6379"
PYTHON_VERSION="3.13"
PROJECT_DIR="/home/kristian/projects/network-training"
SSH_USER="kristian"

echo "=== Setting up Ray worker at ${WORKER_IP} (head: ${RAY_API_SERVER_IP}:${RAY_API_SERVER_PORT}) ==="

# --- 0. Verify head node is running the Ray cluster ---
echo "Checking Ray cluster on head ${RAY_API_SERVER_IP}..."
if ! curl -sf --connect-timeout 5 "http://${RAY_API_SERVER_IP}:8265/api/version" >/dev/null 2>&1; then
    echo "ERROR: Ray cluster is not running on ${RAY_API_SERVER_IP}:8265"
    echo "Start the head node first:"
    echo "  ssh ${SSH_USER}@${RAY_API_SERVER_IP} 'cd ${PROJECT_DIR} && .venv/bin/ray start --head --port=${RAY_API_SERVER_PORT} --dashboard-host=0.0.0.0'"
    exit 1
fi
echo "Ray cluster is running on ${RAY_API_SERVER_IP}:8265"

ssh "${SSH_USER}@${WORKER_IP}" bash -s <<'REMOTE_SCRIPT'
set -euo pipefail

PYTHON_VERSION="3.13"
PROJECT_DIR="/home/kristian/projects/network-training"

# --- 1. Install uv if not present ---
if ! command -v uv &>/dev/null && [ ! -f "$HOME/.local/bin/uv" ]; then
    echo "[1/5] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "[1/5] uv already installed"
fi
export PATH="$HOME/.local/bin:$PATH"

# --- 2. Install Python via uv ---
echo "[2/5] Ensuring Python ${PYTHON_VERSION}..."
uv python install "${PYTHON_VERSION}"

# --- 3. Create project and venv ---
echo "[3/5] Setting up project venv..."
mkdir -p "${PROJECT_DIR}"
cd "${PROJECT_DIR}"

if [ ! -f "pyproject.toml" ]; then
    cat > pyproject.toml <<'TOML'
[project]
name = "network-training"
version = "0.1.0"
description = "Allows setup of ray cluster"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "ray[default]>=2.53.0",
]
TOML
fi

uv venv --python "${PYTHON_VERSION}" 2>/dev/null || true
uv sync

# --- 4. Install ML dependencies ---
echo "[4/5] Installing ML packages..."
uv pip install \
    kego==0.6.0 \
    catboost \
    lightgbm \
    xgboost \
    scikit-learn \
    pandas \
    numpy \
    torch \
    pytabkit \
    rtdl-revisiting-models \
    skorch

# --- 5. Verify GPU access ---
echo "[5/5] Verifying installation..."
.venv/bin/python -c "
import ray; print(f'  ray {ray.__version__}')
import torch; print(f'  torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
import xgboost; print(f'  xgboost {xgboost.__version__}')
import catboost; print(f'  catboost {catboost.__version__}')
import lightgbm; print(f'  lightgbm {lightgbm.__version__}')
import pytabkit;
import skorch; print(f'  skorch {skorch.__version__}')
"
echo "=== Dependencies installed ==="
REMOTE_SCRIPT

# --- 6. Download Kaggle competition data + copy external data from head ---
KAGGLE_COMPETITION="playground-series-s6e2"
DATA_DIR="${PROJECT_DIR}/data"

echo "=== Setting up competition data ==="
ssh "${SSH_USER}@${WORKER_IP}" bash -s <<REMOTE_DATA
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
cd "${PROJECT_DIR}"

# Copy kaggle credentials from head if not present
if [ ! -f "\$HOME/.kaggle/kaggle.json" ]; then
    echo "[6a] Copying Kaggle credentials from head..."
    mkdir -p "\$HOME/.kaggle"
    scp "${SSH_USER}@${RAY_API_SERVER_IP}:\$HOME/.kaggle/kaggle.json" "\$HOME/.kaggle/kaggle.json"
    chmod 600 "\$HOME/.kaggle/kaggle.json"
fi

# Download Kaggle competition data
COMP="${KAGGLE_COMPETITION}"
COMP_PREFIX="\${COMP%%-*}"
DATA_PATH="${DATA_DIR}/\${COMP_PREFIX}/\${COMP}"
if [ -d "\${DATA_PATH}" ] && [ "\$(ls -A \${DATA_PATH})" ]; then
    echo "[6b] Kaggle data already exists at \${DATA_PATH}, skipping download"
else
    echo "[6b] Downloading Kaggle competition data..."
    mkdir -p "${DATA_DIR}/\${COMP_PREFIX}"
    uv tool run kaggle competitions download -c "\${COMP}" -p "${DATA_DIR}/\${COMP_PREFIX}/"
    unzip -o "${DATA_DIR}/\${COMP_PREFIX}/\${COMP}.zip" -d "\${DATA_PATH}"
    rm -f "${DATA_DIR}/\${COMP_PREFIX}/\${COMP}.zip"
fi

# Copy external data (Heart_Disease_Prediction.csv) from head node
if [ ! -f "\${DATA_PATH}/Heart_Disease_Prediction.csv" ]; then
    echo "[6c] Copying Heart_Disease_Prediction.csv from head..."
    scp "${SSH_USER}@${RAY_API_SERVER_IP}:${DATA_DIR}/playground/${KAGGLE_COMPETITION}/Heart_Disease_Prediction.csv" \
        "\${DATA_PATH}/Heart_Disease_Prediction.csv"
else
    echo "[6c] Heart_Disease_Prediction.csv already exists, skipping"
fi

# Create data symlink (script expects data at parents[2]/data/)
ln -sf "${DATA_DIR}" "\$HOME/data" 2>/dev/null || true

echo "=== Data setup complete ==="
ls -lh "\${DATA_PATH}/"
REMOTE_DATA

# --- Start Ray worker and connect to head ---
echo "=== Starting Ray worker on ${WORKER_IP} ==="
ssh "${SSH_USER}@${WORKER_IP}" bash -s <<REMOTE_START
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
export RAY_API_SERVER_IP="${RAY_API_SERVER_IP}"
export RAY_API_SERVER_PORT="${RAY_API_SERVER_PORT}"
cd "${PROJECT_DIR}"

# Stop any existing Ray processes
.venv/bin/ray stop --force 2>/dev/null || true

# Start worker connected to head
RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 uv run ray start --address="\${RAY_API_SERVER_IP}:\${RAY_API_SERVER_PORT}"
echo "=== Ray worker started and connected to \${RAY_API_SERVER_IP}:\${RAY_API_SERVER_PORT} ==="
REMOTE_START

echo "=== Done! Check cluster status at http://${RAY_API_SERVER_IP}:8265 ==="
