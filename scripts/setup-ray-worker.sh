#!/usr/bin/env bash
# Setup a Ray worker node and connect it to the cluster head.
# This script runs DIRECTLY on the worker machine (no SSH).
#
# Usage:
#   ./scripts/setup-ray-worker.sh [head-ip]
#
# Example:
#   ./scripts/setup-ray-worker.sh 192.168.178.32
#
# Prerequisites:
#   - NVIDIA GPU drivers already installed
#   - SSH key set up for GitHub (git clone)
#   - curl available

set -euo pipefail

HEAD_IP="${1:-192.168.178.32}"
HEAD_PORT=6379
PROJECT_DIR="$HOME/projects/kego"
PYTHON_VERSION="3.13"
KAGGLE_COMPETITION="playground-series-s6e2"

echo "=== Setting up Ray worker (head: ${HEAD_IP}:${HEAD_PORT}) ==="

# --- 0. Verify head node is running the Ray cluster ---
echo "[0/8] Checking Ray cluster on head ${HEAD_IP}..."
if ! curl -sf --connect-timeout 5 "http://${HEAD_IP}:8265/api/version" >/dev/null 2>&1; then
    echo "ERROR: Ray cluster is not running on ${HEAD_IP}:8265"
    echo "Start the head node first:"
    echo "  cd ${PROJECT_DIR}/cluster && RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 uv run ray start --head --port=${HEAD_PORT} --node-ip-address ${HEAD_IP} --dashboard-host=0.0.0.0 --dashboard-port=8265 --ray-client-server-port=10001 --num-cpus=\$(expr \$(nproc --all) - 2)"
    exit 1
fi
echo "Ray cluster is running."

# --- 1. Install uv if not present ---
if ! command -v uv &>/dev/null && [ ! -f "$HOME/.local/bin/uv" ]; then
    echo "[1/8] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "[1/8] uv already installed"
fi
export PATH="$HOME/.local/bin:$PATH"

# --- 2. Install Python via uv ---
echo "[2/8] Ensuring Python ${PYTHON_VERSION}..."
uv python install "${PYTHON_VERSION}"

# --- 3. Clone or update kego repo ---
if [ ! -d "${PROJECT_DIR}" ]; then
    echo "[3/8] Cloning kego repo..."
    if ! ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
        echo "ERROR: SSH key not set up for GitHub. Add your SSH key first:"
        echo "  ssh-keygen -t ed25519 && cat ~/.ssh/id_ed25519.pub"
        echo "Then add the public key at https://github.com/settings/keys"
        exit 1
    fi
    mkdir -p "$(dirname "${PROJECT_DIR}")"
    git clone git@github.com:KrisEhl/kego.git "${PROJECT_DIR}"
else
    echo "[3/8] Updating kego repo..."
    git -C "${PROJECT_DIR}" pull
fi

# --- 4. Sync cluster workspace member (includes all ML deps) ---
echo "[4/7] Setting up cluster venv..."
cd "${PROJECT_DIR}/cluster"
uv sync

# --- 5. Verify GPU access and imports ---
echo "[5/7] Verifying installation..."
uv run python <<'PYCHECK'
import ray; print(f'  ray {ray.__version__}')
import torch; print(f'  torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
import xgboost; print(f'  xgboost {xgboost.__version__}')
import catboost; print(f'  catboost {catboost.__version__}')
import lightgbm; print(f'  lightgbm {lightgbm.__version__}')
import importlib.metadata; print(f'  pytabkit {importlib.metadata.version("pytabkit")}')
import skorch; print(f'  skorch {skorch.__version__}')
PYCHECK
echo "=== Dependencies verified ==="

# --- 6. Download Kaggle competition data + copy external data from head ---
echo "[6/7] Setting up competition data..."
DATA_DIR="${PROJECT_DIR}/data"
COMP_PREFIX="${KAGGLE_COMPETITION%%-*}"
DATA_PATH="${DATA_DIR}/${COMP_PREFIX}/${KAGGLE_COMPETITION}"

# Copy kaggle credentials from head if not present
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "  Copying Kaggle credentials from head..."
    mkdir -p "$HOME/.kaggle"
    scp "kristian@${HEAD_IP}:$HOME/.kaggle/kaggle.json" "$HOME/.kaggle/kaggle.json"
    chmod 600 "$HOME/.kaggle/kaggle.json"
fi

# Download Kaggle competition data
if [ -d "${DATA_PATH}" ] && [ "$(ls -A "${DATA_PATH}")" ]; then
    echo "  Kaggle data already exists at ${DATA_PATH}, skipping download"
else
    echo "  Downloading Kaggle competition data..."
    mkdir -p "${DATA_DIR}/${COMP_PREFIX}"
    uv tool run kaggle competitions download -c "${KAGGLE_COMPETITION}" -p "${DATA_DIR}/${COMP_PREFIX}/"
    unzip -o "${DATA_DIR}/${COMP_PREFIX}/${KAGGLE_COMPETITION}.zip" -d "${DATA_PATH}"
    rm -f "${DATA_DIR}/${COMP_PREFIX}/${KAGGLE_COMPETITION}.zip"
fi

# Copy external data from head node
if [ ! -f "${DATA_PATH}/Heart_Disease_Prediction.csv" ]; then
    echo "  Copying Heart_Disease_Prediction.csv from head..."
    scp "kristian@${HEAD_IP}:${DATA_DIR}/${COMP_PREFIX}/${KAGGLE_COMPETITION}/Heart_Disease_Prediction.csv" \
        "${DATA_PATH}/Heart_Disease_Prediction.csv"
else
    echo "  Heart_Disease_Prediction.csv already exists, skipping"
fi

echo "=== Data setup complete ==="
ls -lh "${DATA_PATH}/"

# --- 7. Start Ray worker ---
echo "[7/7] Starting Ray worker..."
cd "${PROJECT_DIR}/cluster"

# Stop any existing Ray processes
ray stop --force 2>/dev/null || true

RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 uv run ray start --address="${HEAD_IP}:${HEAD_PORT}"

echo "=== Done! Check cluster status at http://${HEAD_IP}:8265 ==="
