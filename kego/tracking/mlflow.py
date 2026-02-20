import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_predictions_from_runs(runs_df, tracking_uri):
    """Load and average predictions from a DataFrame of MLflow runs.

    Shared logic for --from-experiment and --from-ensemble.
    Groups by learner ID (model/feature_set/folds_nf) when params are available,
    falls back to bare model_name for backward compatibility.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    all_oof = {}
    all_holdout = {}
    all_test = {}
    seed_counts = {}

    for _, run in runs_df.iterrows():
        model_name = run.get("params.model")
        if model_name is None:
            continue

        # Build learner ID from params (backward compat: fall back to model_name)
        feature_set = run.get("params.feature_set", "")
        folds_n = run.get("params.folds_n", "")
        if feature_set and folds_n:
            learner_id = f"{model_name}/{feature_set}/{folds_n}f"
        else:
            learner_id = model_name

        artifact_dir = client.download_artifacts(run.run_id, "predictions")
        oof = np.load(os.path.join(artifact_dir, "oof.npy"))
        holdout = np.load(os.path.join(artifact_dir, "holdout.npy"))
        test = np.load(os.path.join(artifact_dir, "test.npy"))

        if learner_id not in all_oof:
            all_oof[learner_id] = np.zeros_like(oof)
            all_holdout[learner_id] = np.zeros_like(holdout)
            all_test[learner_id] = np.zeros_like(test)
            seed_counts[learner_id] = 0

        all_oof[learner_id] += oof
        all_holdout[learner_id] += holdout
        all_test[learner_id] += test
        seed_counts[learner_id] += 1

        seed = run.get("params.seed", "?")
        logger.info(f"  Loaded {learner_id} seed={seed}")

    # Average across seeds
    for name in all_oof:
        n = seed_counts[name]
        all_oof[name] /= n
        all_holdout[name] /= n
        all_test[name] /= n
        logger.info(f"{name}: averaged over {n} seed(s)")

    learner_names = list(all_oof.keys())
    logger.info(f"Total learners loaded: {len(learner_names)}")
    return learner_names, all_oof, all_holdout, all_test


def get_completed_fingerprints(experiment_name, tracking_uri):
    """Query MLflow for completed runs and return their config fingerprints.

    Returns:
        (set of fingerprint strings, runs DataFrame)
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.warning(f"Experiment '{experiment_name}' not found")
        return set(), pd.DataFrame()

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    # Filter out ensemble runs
    runs = runs[~runs["tags.mlflow.runName"].str.startswith("ensemble_", na=True)]
    # Only keep runs that have a config_fingerprint param
    fp_col = "params.config_fingerprint"
    if fp_col not in runs.columns:
        logger.warning(
            f"No runs with config_fingerprint in '{experiment_name}' "
            f"(old experiment without fingerprints?)"
        )
        return set(), runs

    has_fp = runs[fp_col].notna()
    fingerprints = set(runs.loc[has_fp, fp_col].tolist())
    logger.info(
        f"Experiment '{experiment_name}': {len(runs)} runs, "
        f"{len(fingerprints)} with fingerprints"
    )
    return fingerprints, runs[has_fp]


def load_predictions_from_mlflow(experiment_names, tracking_uri):
    """Load per-model averaged predictions from MLflow experiments."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)

    all_runs = []
    for exp_name in experiment_names:
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            logger.warning(f"Experiment '{exp_name}' not found, skipping")
            continue

        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        # Filter out ensemble runs (NOT LIKE not supported by MLflow API)
        runs = runs[~runs["tags.mlflow.runName"].str.startswith("ensemble_", na=True)]
        logger.info(f"Experiment '{exp_name}': {len(runs)} model runs")
        all_runs.append(runs)

    if not all_runs:
        return [], {}, {}, {}

    runs_df = pd.concat(all_runs, ignore_index=True)
    return load_predictions_from_runs(runs_df, tracking_uri)


def load_predictions_from_ensemble(ensemble_name, tracking_uri):
    """Load predictions from runs tagged with a named ensemble."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)

    tag_key = f"ensemble:{ensemble_name}"
    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=f"tags.`{tag_key}` = 'true'",
    )

    if runs.empty:
        logger.error(f"No runs found in ensemble '{ensemble_name}'")
        return [], {}, {}, {}

    # Filter out ensemble summary runs
    runs = runs[~runs["tags.mlflow.runName"].str.startswith("ensemble_", na=True)]
    logger.info(f"Ensemble '{ensemble_name}': {len(runs)} model runs")

    return load_predictions_from_runs(runs, tracking_uri)
