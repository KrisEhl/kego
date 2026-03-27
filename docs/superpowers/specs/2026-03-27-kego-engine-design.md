# kego Engine Design

**Date:** 2026-03-27
**Status:** Approved
**Scope:** Unified CLI for local iteration, cluster dispatch, Kaggle training/submission, and experiment tracking across all competitions in this repo.

---

## Problem

The current workflow has three disconnected pain points:

1. **Script → Kaggle submission** is manual: copy-paste Python code into Kaggle's web notebook editor
2. **Experiment tracking** is fragmented: cluster runs log to MLflow, local runs don't, Kaggle runs are untracked
3. **Compute dispatch** is per-competition: each competition has its own Makefile targets, no shared abstraction

---

## Solution: `kego` CLI

A single CLI command with a `--target` flag that dispatches the same Python script to any compute target. Training scripts stay pure Python with argparse — no changes required.

```bash
uv run kego <command> [args]
```

---

## Commands

### `kego run` — dispatch a training script

```bash
# Smoke test: injects --debug, runs locally, does not pollute experiment history
kego run train_cnn.py --debug

# Local run (CPU models, quick CV, full execution)
kego run train_cnn.py --fold 0 --epochs 5

# Single fold on cluster
kego run train_cnn.py --fold 0 --epochs 30 --target cluster --name soundscape-v8

# Fan out all folds as parallel cluster jobs
kego run train_cnn.py --folds 0,1,2,3 --target cluster --name soundscape-v8

# Train on Kaggle GPU (e.g. for competitions needing H100)
kego run training/train.py --target kaggle --gpu T4x2 --name nemotron-tir-v4
```

**`--debug` contract:** `kego run` forwards `--debug` to the script as an argparse flag. The script is responsible for handling it (subsample data, reduce epochs, etc.). Debug runs are tagged in MLflow and excluded from `kego ls` by default.

**`--folds` fan-out:** `--folds 0,1,2,3` dispatches one Ray job per fold in parallel. All folds with the same `--name` are grouped under one parent experiment. Scripts that already parallelize folds internally (via Ray remote functions) should use plain `--fold` and ignore `--folds`.

**Experiment ID:** assigned automatically at `kego run` time — a 6-character alphanumeric short hash (e.g. `a3f2b1`). Stored in MLflow metadata. `kego push` uses the ID to collect and rename checkpoints into a standard layout (`{name}_fold{N}_{id}.pt`) when packaging for upload — the training script does not need to know the ID.

### `kego ls` — list and compare experiments

```bash
kego ls
kego ls --competition birdclef-2026
kego ls --name soundscape-v8     # show all runs with this name (including re-runs)
```

Output:

```
ID       NAME                FOLDS    METRIC     STD    STATUS    TARGET    AGO
a3f2b1   soundscape-v8       4/4      0.8821     0.003  done      cluster   2h
c9d2e4   soundscape-v8       4/4      0.8803     0.002  done      cluster   5h
f1e8a2   soundscape-v7       4/4      0.8794     0.002  done      cluster   1d
```

The metric column header is the `primary_metric` defined in the competition's `kego.toml` (e.g. `fold_auc`, `cmAP`, `accuracy`).

Experiments are referenced by ID (unambiguous) or name (takes latest if ambiguous).

### `kego push` — upload checkpoints as Kaggle dataset

```bash
kego push --experiment a3f2b1 --competition birdclef-2026
kego push --experiment soundscape-v8 --competition birdclef-2026  # takes latest
```

- Collects fold checkpoints from the `checkpoint_dir` defined in the competition's `kego.toml`, matching the experiment ID
- Packages into a clean temp dir (no stale `.pt` files)
- Uploads as Kaggle dataset: `{kaggle_user}/{competition}-{experiment-name}`
- Embeds experiment ID and git hash in dataset metadata

### `kego submit` — generate notebook and submit to competition

```bash
kego submit --experiment a3f2b1 --competition birdclef-2026
```

- Generates a thin launcher notebook from the competition's `inference_notebook` script
- Patches `kernel-metadata.json` (dataset refs, GPU flag, etc.)
- Auto-references the dataset uploaded by `kego push` for the same experiment
- Runs `kaggle kernels push`, polls until complete, prints LB score if available

Typical full pipeline:

```bash
kego run train_cnn.py --folds 0,1,2,3 --target cluster --name soundscape-v8
kego push --experiment soundscape-v8 --competition birdclef-2026
kego submit --experiment soundscape-v8 --competition birdclef-2026
```

---

## Experiment Tracking

**MLflow** is the single tracking backend. One server on the cluster head node, always the same URI regardless of where the job runs.

### Metric emission (stdout convention)

No MLflow import required in training scripts. `kego run` parses stdout and logs to MLflow:

```python
# In your training script
print(f"KEGO_METRIC fold_auc {score:.4f}")
print(f"KEGO_METRIC val_loss {loss:.4f}")
print(f"KEGO_METRIC epoch {epoch}")
```

### Parameter logging

Two layers:

1. **Auto-capture:** all CLI args passed to the script are logged as MLflow params automatically — no code needed
2. **Explicit escape hatch:** for derived/computed values not in CLI args:

```python
print(f"KEGO_PARAM n_mels {N_MELS}")
print(f"KEGO_PARAM spec_cache specs_cache_224_htk")
```

### Tracking across targets

- **Local:** `kego run` logs directly to MLflow server
- **Cluster:** Ray job's stdout is captured by `kego run`; metrics forwarded to MLflow
- **Kaggle:** notebook phones home to MLflow via `MLFLOW_TRACKING_URI` env var injected at notebook generation time. If the cluster is unreachable from Kaggle, metrics are buffered in the notebook output and synced post-run via `kego sync --experiment <id>`

---

## Compute Targets

### Local

Executes `uv run python <script> <args>` in the competition's venv. `--debug` flag forwarded. No cluster required.

### Cluster (Ray)

Wraps `ray job submit` with correct runtime env. Folds dispatched as parallel jobs:

```bash
# kego run translates to:
ray job submit \
  --address http://192.168.178.32:8265 \
  --runtime-env-json '{"env_vars": {"KEGO_PATH_DATA": "...", "MLFLOW_TRACKING_URI": "..."}}' \
  -- uv run python train_cnn.py --fold 0 --epochs 30
```

### Kaggle

For `train_and_infer` competitions:
1. Packages script + kego lib as a Kaggle dataset
2. Generates a thin launcher notebook
3. Runs `kaggle kernels push` and polls status

---

## Configuration

### Root `kego.toml` (cluster + MLflow)

```toml
[cluster]
ray_address = "http://192.168.178.32:8265"
mlflow_uri = "http://192.168.178.32:5000"

[cluster.resources]
default = {num_gpus = 0.5}
heavy = {num_gpus = 1, resources = {"heavy_gpu": 1}}
```

### Per-competition `kego.toml`

**Inference-only pattern** (training is local/cluster, Kaggle only for submission):

```toml
[competition]
slug = "birdclef-2026"
kaggle_user = "aldisued"
enable_gpu = false
submit_file = "submission.csv"
pattern = "inference_only"
inference_notebook = "inference/kernel.py"
checkpoint_dir = "training/outputs"
primary_metric = "cmAP"
```

**Train-and-infer pattern** (Kaggle used for training + submission):

```toml
[competition]
slug = "nvidia-nemotron-model-reasoning-challenge"
kaggle_user = "aldisued"
enable_gpu = true
submit_file = "submission.parquet"
pattern = "train_and_infer"
training_notebook = "notebook/train.py"
inference_notebook = "notebook/infer.py"
checkpoint_dir = "notebook/outputs"
primary_metric = "accuracy"
```

---

## Project Structure

No changes to existing scripts or library. Only additions:

```
kego/
  cli/                        # NEW: kego CLI implementation
    __init__.py
    run.py                    # kego run
    ls.py                     # kego ls
    push.py                   # kego push
    submit.py                 # kego submit
    sync.py                   # kego sync (pull metrics from Kaggle output)
    targets/
      local.py
      cluster.py
      kaggle.py
    tracking.py               # stdout parser → MLflow
    notebook.py               # notebook generation from Python scripts
kego.toml                     # root config (cluster, MLflow)
competitions/
  birdclef-2026/
    kego.toml                 # competition config
    training/train_cnn.py     # unchanged
    inference/kernel.py       # unchanged
  nvidia-nemotron-*/
    kego.toml
    notebook/train.py         # unchanged
    notebook/infer.py         # unchanged
  playground/
    kego.toml
    train_s6e2_baseline.py    # unchanged
```

The `kego` CLI is registered as a script entry point in the root `pyproject.toml`:

```toml
[project.scripts]
kego = "kego.cli:main"
```

Available anywhere in the repo via `uv run kego`.

---

## Out of Scope

- Hyperparameter search / Optuna integration (existing Ray-based tuning is sufficient)
- Automatic model selection or ensembling (done in training scripts today)
- Web UI (MLflow UI covers experiment comparison)
