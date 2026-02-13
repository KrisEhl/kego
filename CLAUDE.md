# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**kego** is a Python helper library for quick ML analysis of Kaggle competitions. It provides shared utilities (data handling, plotting, training) and competition-specific modules. Uses a uv workspace monorepo where each competition notebook project depends on the main `kego` package as an editable install.

## Commands

```bash
# Setup / install
make install              # uv sync + pre-commit install
make re-install           # Clean .venv and reinstall

# Download competition data
KAGGLE_COMPETITION=<name> make download-competition-data

# Scaffold a new competition
KAGGLE_COMPETITION=<name> make setup-new-competition

# Publish to PyPI
make publish

# Run pre-commit hooks manually
uv run pre-commit run --all-files

# Setup a Ray cluster worker (run on the worker machine)
./scripts/setup-ray-worker.sh [head-ip]

# Run a training script (example: RNA baseline)
uv run python notebooks/stanford/train_rna_baseline.py
```

## Architecture

### Core Library (`kego/`)

- **`train.py` / `trainer.py`** — Training orchestration. `Trainer` wraps `train_model()` and `train_model_split()`.
- **`models/model_base.py`** — Abstract `ModelBase` with required `train()` method.
- **`datasets/`** — DataFrame utilities, `split_dataset()` (3-way split with optional stratification), `build_xy()`, normalization.
- **`plotting/`** — Matplotlib wrappers: grid layouts (`create_axes_grid`), scatter, histograms, lines, heatmaps, timeseries.
- **`competitions/`** — Competition-specific code (RNA models/datasets, Ariel features/metrics, PII tokenization, ARC images).
- **`torch/`** — Device detection utilities (CUDA/CPU). Torch is an optional dependency—imported lazily via `imports.py`.
- **`constants.py`** — Type aliases, default paths. Data path overridden by `KEGO_PATH_DATA` env var.
- **`parallelize.py`** — Multiprocessing wrapper.

### Workspace Structure

Each competition lives in `notebooks/<competition>/` with its own `pyproject.toml` and dependencies. Workspace members are declared in the root `pyproject.toml` under `[tool.uv.workspace]`. Not all notebook directories are workspace members—only those needing extra dependencies.

### Ray Cluster (`cluster/`)

The `cluster/` workspace member provides a uv-managed venv with `ray[default]` for Ray cluster workers. On each worker node, `cd cluster && uv sync` creates the venv, then ML deps are installed via `uv pip install`. Worker setup is handled by `scripts/setup-ray-worker.sh`.

### Current Active Focus: Stanford RNA 3D Folding

- Goal: predict x,y,z coordinates for each RNA nucleotide
- Bidirectional LSTM baseline model in `kego/competitions/rna/`
- `RNADataset` handles variable-length sequences with custom `collate_fn` (padding)
- Training script: `notebooks/stanford/train_rna_baseline.py`

## Code Conventions

- Follow Google Python coding conventions
- Python 3.10+ type hints throughout
- Pre-commit enforces: black (formatting), isort (imports), flake8 (linting), mypy (type checking), autoflake (unused imports), nbstripout (clean notebook outputs)
- Use `%load_ext autoreload` / `%autoreload 2` in notebooks to pick up library changes

## Environment

- **Package manager**: uv (not pip/poetry)
- **Build backend**: hatchling
- **Data directory**: `./data/` (gitignored), override with `KEGO_PATH_DATA` env var
- **Competition env vars**: Set `FOLDER_COMPETITION` in `.env` for competition-specific data paths
