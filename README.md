# kego

Helpers for quick ML analysis of Kaggle competitions: a shared Python library
(data handling, ensembling, plotting, experiment tracking) plus a config-driven
training pipeline exposed as the `kego` CLI. Competition-specific code lives in
`competitions/<slug>/` as members of a uv workspace, each depending on the main
`kego` package as an editable install.

## Install

```bash
# As a dependency (PyPI)
uv add "kego[ml]"        # core is dependency-light; [ml] pulls pandas/sklearn/mlflow/kaggle/...

# Development setup (this repo)
make install             # uv sync + pre-commit install
make test                # pytest with coverage
```

## CLI

The pipeline drives a cached grid of (model x feature set x fold x seed):
features -> trainer -> prediction store -> ensembler -> submitter. Learners whose
predictions are already stored are skipped, so re-runs and re-ensembles are cheap.

```bash
kego run --config v1 --task <competition-slug>        # train grid + ensemble
kego run --model catboost --params learning_rate:0.01 # ad-hoc single model
kego run --config v1 --hp-tune --hp-params max_depth::3:9:int
kego ensemble --config v1                             # re-ensemble stored predictions
kego tune --config v1                                 # Optuna HP tuning
kego submit --config v1                               # submit to Kaggle
kego status                                           # check current training runs
kego submissions                                      # list Kaggle submissions
kego cache [status|prune]                             # manage the prediction cache
```

Configs are YAML resolved from `competitions/<task>/configs/<name>.yaml`
(defaults < YAML < `--params` dotlist overrides, via OmegaConf). See
`kego/pipeline/config.py` for the schema (`PipelineConfig`).

Simulation-style competitions (e.g. `pokemon-tcg-ai-battle`) add:

```bash
kego train-agent --agent mcts --variant small192_zacian --task pokemon-tcg-ai-battle  # self-play / policy training
kego battle --config mcts_vs_random                   # battle local agents
kego league / league matrix / league merge            # play Elo league / view / merge matrices
kego models --task <slug>                             # Elo standings from the model registry
kego models prune / unprune                           # retire or restore registry versions
kego sync [--list]                                    # replay checkpoint registrations queued while the hub was down
```

## Repository layout

- **`kego/`** — the library
  - `pipeline/` — config, trainer, prediction store, ensembler, tuner, submitter, CLI
  - `tracking/` — MLflow helpers, run tracker, model registry, agent league
  - `datasets/`, `preprocessing/`, `features/` — splits, target encoding, feature selection
  - `ensemble/` — hill climbing, stacking, blending, disagreement analysis
  - `models/` — sklearn-style wrappers and neural nets (FT-Transformer, ResNet)
  - `plotting/` — matplotlib wrappers (grids, histograms, timeseries, ...)
  - `fleet.py` / `dispatch.py` — machine registry (`fleet.toml`) + SSH job dispatch
  - `gpu/` — device benchmarks and monitoring
- **`competitions/`** — one directory per competition; workspace members are
  declared in the root `pyproject.toml` (only those needing extra dependencies)
- **`tests/`** — pytest suite (`make test`)

## Environment

| Variable | Purpose |
|---|---|
| `KEGO_PATH_DATA` | data directory (default `./data/`, gitignored) |
| `KEGO_MLFLOW` / `MLFLOW_TRACKING_URI` | MLflow tracking server |
| `KAGGLE_COMPETITION` | used by the data-download / scaffold make targets |

## Make targets

```bash
make install / re-install
KAGGLE_COMPETITION=<name> make download-competition-data   # download + scaffold workspace member
KAGGLE_COMPETITION=<name> make setup-new-competition
make test                  # pytest with coverage
make fleet-register        # register this machine in fleet.toml
make publish               # build + upload to PyPI
```

## Ray cluster

Cluster tooling (head/worker setup, job submission, log parsing) lives in
`competitions/playground/` — see its README for the full command reference.
