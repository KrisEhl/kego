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
HEAD_IP="${2:-192.168.178.32}"
HEAD_PORT="6379"
PYTHON_VERSION="3.13"
PROJECT_DIR="/home/kristian/projects/network-training"
SSH_USER="kristian"

echo "=== Setting up Ray worker at ${WORKER_IP} (head: ${HEAD_IP}:${HEAD_PORT}) ==="

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

# --- Start Ray worker and connect to head ---
echo "=== Starting Ray worker on ${WORKER_IP} ==="
ssh "${SSH_USER}@${WORKER_IP}" bash -s <<REMOTE_START
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
cd "${PROJECT_DIR}"

# Stop any existing Ray processes
.venv/bin/ray stop --force 2>/dev/null || true

# Start worker connected to head
.venv/bin/ray start --address="${HEAD_IP}:${HEAD_PORT}"
echo "=== Ray worker started and connected to ${HEAD_IP}:${HEAD_PORT} ==="
REMOTE_START

echo "=== Done! Check cluster status at http://${HEAD_IP}:8265 ==="
