# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**kego** is a Python helper library for quick ML analysis of Kaggle competitions. It provides shared utilities (data handling, plotting, training) and competition-specific modules. Uses a uv workspace monorepo where each competition notebook project depends on the main `kego` package as an editable install.

## Commands

See the root `Makefile` for setup/install/download/publish targets. Ray cluster commands (run from `competitions/playground/`) are documented in full, with all variables, in `competitions/playground/README.md`.

## Architecture

### Core Library (`kego/`)

- **`train.py` / `trainer.py`** — Training orchestration. `Trainer` wraps `train_model()` and `train_model_split()`.
- **`models/model_base.py`** — Abstract `ModelBase` with required `train()` method.
- **`datasets/`** — DataFrame utilities, `split_dataset()` (3-way split with optional stratification), `build_xy()`, normalization.
- **`plotting/`** — Matplotlib wrappers: grid layouts (`create_axes_grid`), scatter, histograms, lines, heatmaps, timeseries.
- **`gpu/`** — Device detection utilities (CUDA/CPU). Torch is an optional dependency—imported lazily via `imports.py`.
- **`constants.py`** — Type aliases, default paths. Data path overridden by `KEGO_PATH_DATA` env var.
- **`parallelize.py`** — Multiprocessing wrapper.

### Workspace Structure

Each competition lives in `competitions/<competition>/` with its own `pyproject.toml` and dependencies. Workspace members are declared in the root `pyproject.toml` under `[tool.uv.workspace]`. Not all competition directories are workspace members—only those needing extra dependencies.

### Ray Cluster

Ray cluster tooling lives in `competitions/playground/` alongside the training script. On each worker node, `cd competitions/playground && uv sync` sets up the venv (includes Ray, all ML deps). Worker setup is handled by `competitions/playground/scripts-cluster/setup-ray-worker.sh`.

## Code Conventions

- Follow Google Python coding conventions
- **All imports at the top of files** — Standard library, third-party, then local imports. No inline imports except for:
  - Conditional imports (e.g., `tomllib`/`tomli` version compatibility)
  - Lazy imports to avoid circular dependencies (document why with a comment)
- Python 3.10+ type hints throughout
- Pre-commit enforces: black (formatting), isort (imports), flake8 (linting), mypy (type checking), autoflake (unused imports), nbstripout (clean notebook outputs)
- Use `%load_ext autoreload` / `%autoreload 2` in notebooks to pick up library changes

## Environment

- **Package manager**: uv (not pip/poetry)
- **Build backend**: hatchling
- **Data directory**: `./data/` (gitignored), override with `KEGO_PATH_DATA` env var
- **Competition env vars**: Set `FOLDER_COMPETITION` in `.env` for competition-specific data paths
