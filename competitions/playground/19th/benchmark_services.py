"""Benchmark 19th (and Rimay quantum features) against classical models by training size.

Each dataset has its own config (features, sample sizes, holdout, LightGBM settings) and its
own output folder, benchmark/<dataset>/. All methods on a dataset are scored on the same
holdout, which no model trains or early-stops on. Training samples are nested: each smaller
sample is contained in every larger one.

Datasets:
  heart  Playground S6E2 heart disease. The holdout is Rimay's 300 test rows plus 700 more,
         and the 700-row sample is exactly Rimay's training set (execution b9c72ba0), so
         scores also exist on the Rimay-comparable 300 rows (`auc_rimay_300`).
  fraud  Playground S6E11 credit card fraud (1.7% positive). Stratified samples and a
         stratified 100,000-row holdout.

Usage:
  uv run python benchmark_services.py --dataset fraud prepare
  uv run python benchmark_services.py --dataset fraud baseline [--sizes 200000]
  uv run python benchmark_services.py --dataset fraud score --mode high --variant fast --size 5000
  uv run python benchmark_services.py --dataset fraud compare --mode high --variant fast
  uv run python benchmark_services.py --dataset fraud summary
  uv run python benchmark_services.py --dataset heart score-rimay

Targets are stored as 0/1. A predictions CSV has columns `id` and `probability` (of target 1).
19th holdout predictions live at predictions/pred_<size>_<mode>_<variant>.csv and are scored as
service `19th_<mode>` (Fast) or `19th_<mode>_<variant>` (Turbo, Warpspeed).
"""

import argparse
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from kego.pipeline.tune import HPSpace, run_study

DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = DIRECTORY.parents[2]
DATA_ROOT = Path(os.environ.get("KEGO_PATH_DATA", PROJECT_ROOT / "data")) / "playground"
BENCHMARK_ROOT = DIRECTORY / "benchmark"
# Rimay's run (execution b9c72ba0) lives in the quantum project; its rows anchor the heart samples.
RIMAY_DIRECTORY = DIRECTORY.parent / "quantum" / "quantum_results"
LABEL_COLUMN = "label"
HOLDOUT_SCORE = "auc_holdout"
RIMAY_SCORE = "auc_rimay_300"
RANDOM_SEED = 42
DEFAULT_MODE = "standard"
DEFAULT_VARIANT = "fast"
TRAINING_MODES = ["instant", "standard", "high", "ultra"]
MODEL_VARIANTS = ["fast", "turbo", "warpspeed"]


@dataclass(frozen=True)
class DatasetConfig:
    """Everything that differs between benchmark datasets."""

    name: str
    competition: str
    target: str
    positive_label: object
    numeric_features: list[str]
    categorical_features: list[str]
    sample_sizes: list[int]
    holdout_rows: int
    baseline_seeds: int
    bootstrap_resamples: int
    lightgbm_params: dict
    early_stopping_rounds: int
    extra_score_columns: list[str] = field(default_factory=list)
    xgboost_trials: int = 60
    xgboost_tuning_seconds: int = 900

    @property
    def features(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]

    @property
    def data_directory(self) -> Path:
        return DATA_ROOT / self.competition

    @property
    def directory(self) -> Path:
        return BENCHMARK_ROOT / self.name

    @property
    def samples_directory(self) -> Path:
        return self.directory / "samples"

    @property
    def results_directory(self) -> Path:
        return self.directory / "results"

    @property
    def predictions_directory(self) -> Path:
        return self.directory / "predictions"

    @property
    def classical_scores_path(self) -> Path:
        return self.results_directory / "classical_scores.csv"

    @property
    def service_scores_path(self) -> Path:
        return self.results_directory / "service_scores.csv"

    @property
    def comparison_path(self) -> Path:
        return self.results_directory / "comparison_vs_classical.csv"

    @property
    def score_columns(self) -> list[str]:
        return [HOLDOUT_SCORE, *self.extra_score_columns]

    @property
    def xgboost_parameters_path(self) -> Path:
        return self.results_directory / "xgboost_tuned_parameters.json"

    @property
    def cache_directory(self) -> Path:
        return self.directory / "cache"

    def service_predictions_path(self, size: int, mode: str, variant: str) -> Path:
        return self.predictions_directory / f"pred_{size}_{mode}_{variant}.csv"


def service_name(mode: str, variant: str) -> str:
    """Service label in the score tables: 19th_<mode> for Fast, 19th_<mode>_<variant> otherwise."""
    return f"19th_{mode}" if variant == DEFAULT_VARIANT else f"19th_{mode}_{variant}"


HEART = DatasetConfig(
    name="heart",
    competition="playground-series-s6e2",
    target="Heart Disease",
    positive_label="Presence",
    numeric_features=[
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
    ],
    categorical_features=[],
    sample_sizes=[100, 250, 500, 700, 2000, 5000, 20000],
    holdout_rows=1000,
    baseline_seeds=5,
    bootstrap_resamples=2000,
    # Small-data settings from the quantum project, with more trees (early stopping picks the
    # count) and min_child_weight=1, since 10 blocks all splits below ~200 rows.
    lightgbm_params={
        "n_estimators": 2000,
        "max_depth": 3,
        "num_leaves": 7,
        "learning_rate": 0.1,
        "metric": "auc",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
        "verbosity": -1,
        "n_jobs": -1,
    },
    early_stopping_rounds=50,
    extra_score_columns=[RIMAY_SCORE],
)

FRAUD = DatasetConfig(
    name="fraud",
    competition="playground-series-s6e11",
    target="is_fraud",
    positive_label=1.0,
    numeric_features=[
        "amount_usd",
        "hours_since_last_txn",
        "txn_count_last_24h",
        "distance_from_home_km",
        "card_age_months",
        "customer_age",
        "account_balance_usd",
        "cvv_retry_count",
        "velocity_score",
        "time_of_day_hour",
        "day_of_week",
        "merchant_risk_score",
        "prior_disputes",
        "is_foreign_transaction",
        "is_new_merchant",
        "used_vpn",
        "ip_country_mismatch",
        "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt",
    ],
    categorical_features=["merchant_category", "card_type", "auth_method", "channel", "device_type"],
    sample_sizes=[1000, 5000, 20000, 100000, 200000, 599960],
    holdout_rows=100000,
    baseline_seeds=3,
    bootstrap_resamples=1000,
    lightgbm_params={
        "n_estimators": 5000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "metric": "auc",
        "verbosity": -1,
        "n_jobs": -1,
    },
    early_stopping_rounds=100,
)

DATASETS = {config.name: config for config in [HEART, FRAUD]}


def load_data(config: DatasetConfig) -> pd.DataFrame:
    """Load a competition's train.csv with the target as 0/1 and booleans as 0/1."""
    frame = pd.read_csv(config.data_directory / "train.csv")
    frame[config.target] = (frame[config.target] == config.positive_label).astype(int)
    for column in config.numeric_features:
        if frame[column].dtype == bool:
            frame[column] = frame[column].astype(int)
    return frame


def build_heart_samples(config: DatasetConfig) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """Nested samples with Rimay's 700 training rows first, holdout led by Rimay's 300 test rows."""
    full = load_data(config)
    rimay_train = pd.read_csv(RIMAY_DIRECTORY / "quantum_train.csv")
    rimay_holdout = pd.read_csv(RIMAY_DIRECTORY / "holdout.csv")
    columns = ["id", *config.features, config.target]

    remaining = full[~full["id"].isin(set(rimay_train["id"]) | set(rimay_holdout["id"]))]
    remaining = remaining.sample(frac=1.0, random_state=RANDOM_SEED)
    extra_holdout_rows = config.holdout_rows - len(rimay_holdout)
    extra_holdout = remaining.iloc[:extra_holdout_rows]
    pool_rest = remaining.iloc[extra_holdout_rows:]

    rimay_train_shuffled = rimay_train.sample(frac=1.0, random_state=RANDOM_SEED)
    training_pool = pd.concat([rimay_train_shuffled[columns], pool_rest[columns]], ignore_index=True)
    holdout = pd.concat([rimay_holdout[columns], extra_holdout[columns]], ignore_index=True)
    holdout["in_rimay_holdout"] = holdout.index < len(rimay_holdout)
    samples = {size: training_pool.iloc[:size] for size in config.sample_sizes}
    return samples, holdout


def build_stratified_samples(config: DatasetConfig) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """Stratified holdout, then nested samples that each keep the overall positive rate.

    Positives and negatives are shuffled separately; a sample of n rows takes the first
    round(n * rate) positives and the first remaining negatives, so samples nest.
    """
    full = load_data(config)[["id", *config.features, config.target]]
    pool, holdout = train_test_split(
        full, test_size=config.holdout_rows, random_state=RANDOM_SEED, stratify=full[config.target]
    )
    positive_rate = pool[config.target].mean()
    positives = pool[pool[config.target] == 1].sample(frac=1.0, random_state=RANDOM_SEED)
    negatives = pool[pool[config.target] == 0].sample(frac=1.0, random_state=RANDOM_SEED)
    samples = {}
    for size in config.sample_sizes:
        positive_count = min(len(positives), round(size * positive_rate))
        negative_count = size - positive_count
        if negative_count > len(negatives):
            raise ValueError(f"Sample size {size} is larger than the training pool ({len(pool)} rows)")
        sample = pd.concat([positives.iloc[:positive_count], negatives.iloc[:negative_count]])
        samples[size] = sample.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return samples, holdout.reset_index(drop=True)


SAMPLE_BUILDERS: dict[str, Callable[[DatasetConfig], tuple[dict[int, pd.DataFrame], pd.DataFrame]]] = {
    "heart": build_heart_samples,
    "fraud": build_stratified_samples,
}


def prepare(config: DatasetConfig) -> None:
    """Write nested training CSVs and the holdout (features and labels separately)."""
    config.samples_directory.mkdir(parents=True, exist_ok=True)
    samples, holdout = SAMPLE_BUILDERS[config.name](config)
    for size, sample in samples.items():
        sample.to_csv(config.samples_directory / f"train_{size}.csv", index=False)
        print(f"  train_{size}.csv  rows={len(sample)}  positives={int(sample[config.target].sum())}")
    label_columns = ["id", config.target] + [c for c in ["in_rimay_holdout"] if c in holdout.columns]
    holdout[["id", *config.features]].to_csv(config.samples_directory / "holdout_features.csv", index=False)
    holdout[label_columns].to_csv(config.samples_directory / "holdout_labels.csv", index=False)
    print(f"  holdout rows={len(holdout)}  positives={int(holdout[config.target].sum())}")


def load_sample(config: DatasetConfig, size: int) -> pd.DataFrame:
    """Load one training sample."""
    return pd.read_csv(config.samples_directory / f"train_{size}.csv")


def load_holdout(config: DatasetConfig) -> pd.DataFrame:
    """Load holdout features and labels with a binary label column."""
    features = pd.read_csv(config.samples_directory / "holdout_features.csv")
    labels = pd.read_csv(config.samples_directory / "holdout_labels.csv")
    holdout = features.merge(labels, on="id", validate="one_to_one")
    holdout[LABEL_COLUMN] = holdout[config.target].astype(int)
    return holdout


def score_probabilities(config: DatasetConfig, holdout: pd.DataFrame, probabilities: pd.Series) -> dict[str, float]:
    """AUC on the full holdout, plus the Rimay-comparable subset where the dataset has one."""
    scores = {HOLDOUT_SCORE: roc_auc_score(holdout[LABEL_COLUMN], probabilities)}
    if RIMAY_SCORE in config.extra_score_columns:
        mask = holdout["in_rimay_holdout"].to_numpy()
        scores[RIMAY_SCORE] = roc_auc_score(holdout[LABEL_COLUMN][mask], probabilities[mask])
    return scores


def lightgbm_features(config: DatasetConfig, frame: pd.DataFrame) -> pd.DataFrame:
    """Features with categoricals as pandas categories over the full category set."""
    features = frame[config.features].copy()
    for column in config.categorical_features:
        features[column] = features[column].astype("category")
    return features


def fit_lightgbm(config: DatasetConfig, frame: pd.DataFrame, labels: np.ndarray, seed: int) -> Callable:
    """Fit LightGBM, early-stopping on 20% of the training rows, never on the holdout."""
    features = lightgbm_features(config, frame)
    fit_features, stop_features, fit_labels, stop_labels = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    model = LGBMClassifier(**{**config.lightgbm_params, "random_state": seed})
    model.fit(
        fit_features,
        fit_labels,
        eval_set=[(stop_features, stop_labels)],
        callbacks=[lightgbm.early_stopping(config.early_stopping_rounds, verbose=False)],
    )
    categories = {column: fit_features[column].cat.categories for column in config.categorical_features}

    def predict(other: pd.DataFrame) -> np.ndarray:
        other_features = other[config.features].copy()
        for column, column_categories in categories.items():
            other_features[column] = pd.Categorical(other_features[column], categories=column_categories)
        return model.predict_proba(other_features)[:, 1]

    return predict


def fit_logistic_regression(config: DatasetConfig, frame: pd.DataFrame, labels: np.ndarray, seed: int) -> Callable:
    """Fit logistic regression on standardized numerics and one-hot categoricals."""
    transformers = [("numeric", StandardScaler(), config.numeric_features)]
    if config.categorical_features:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), config.categorical_features))
    model: Pipeline = make_pipeline(
        ColumnTransformer(transformers), LogisticRegression(max_iter=2000, random_state=seed)
    )
    model.fit(frame[config.features], labels)
    return lambda other: model.predict_proba(other[config.features])[:, 1]


XGBOOST_FIXED_PARAMETERS = {
    "n_estimators": 5000,
    "tree_method": "hist",
    "enable_categorical": True,
    "max_cat_to_onehot": 1,
    "eval_metric": "auc",
    "n_jobs": -1,
}


XGBOOST_SEARCH_SPACE = [
    HPSpace(name="max_depth", type="int", low=2, high=10),
    HPSpace(name="learning_rate", type="float", low=0.01, high=0.3, log=True),
    HPSpace(name="min_child_weight", type="float", low=0.5, high=100.0, log=True),
    HPSpace(name="subsample", type="float", low=0.5, high=1.0),
    HPSpace(name="colsample_bytree", type="float", low=0.3, high=1.0),
    HPSpace(name="reg_lambda", type="float", low=1e-3, high=30.0, log=True),
    HPSpace(name="reg_alpha", type="float", low=1e-3, high=10.0, log=True),
    HPSpace(name="gamma", type="float", low=0.0, high=5.0),
    HPSpace(name="max_bin", type="categorical", choices=[128, 256, 512]),
]


def fit_xgboost_with(
    config: DatasetConfig, features: pd.DataFrame, labels: np.ndarray, parameters: dict, seed: int
) -> tuple[XGBClassifier, float]:
    """Fit XGBoost, early-stopping on 20% of the given rows; return the model and its AUC on those rows."""
    fit_features, stop_features, fit_labels, stop_labels = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    model = XGBClassifier(
        **XGBOOST_FIXED_PARAMETERS,
        **parameters,
        early_stopping_rounds=config.early_stopping_rounds,
        random_state=seed,
    )
    model.fit(fit_features, fit_labels, eval_set=[(stop_features, stop_labels)], verbose=False)
    return model, roc_auc_score(stop_labels, model.predict_proba(stop_features)[:, 1])


def tuned_xgboost_parameters(config: DatasetConfig, features: pd.DataFrame, labels: np.ndarray) -> dict:
    """Optuna-tuned XGBoost parameters for one training sample, cached by sample size.

    Every trial trains on 80% of the sample and is scored on the other 20%; the holdout is
    never seen. The study stops after `xgboost_trials` trials or `xgboost_tuning_seconds`.
    """
    stored = json.loads(config.xgboost_parameters_path.read_text()) if config.xgboost_parameters_path.exists() else {}
    key = str(len(features))
    if key in stored:
        return stored[key]["parameters"]

    def objective(parameters: dict) -> float:
        _, score = fit_xgboost_with(config, features, labels, parameters, RANDOM_SEED)
        return score

    result = run_study(
        objective,
        XGBOOST_SEARCH_SPACE,
        name="xgboost_tuned",
        n_trials=config.xgboost_trials,
        timeout_seconds=config.xgboost_tuning_seconds,
        seed=RANDOM_SEED,
    )
    stored[key] = {
        "parameters": result.best_params,
        "inner_validation_auc": result.best_score,
        "trials": result.n_trials,
        "seconds": round(result.elapsed_seconds),
    }
    config.results_directory.mkdir(parents=True, exist_ok=True)
    config.xgboost_parameters_path.write_text(
        json.dumps(dict(sorted(stored.items(), key=lambda item: int(item[0]))), indent=2)
    )
    print(f"  xgboost_tuned size={key}: {result.n_trials} trials, inner AUC {result.best_score:.4f}")
    return result.best_params


def fit_xgboost_tuned(config: DatasetConfig, frame: pd.DataFrame, labels: np.ndarray, seed: int) -> Callable:
    """Fit XGBoost with parameters tuned on this sample, early-stopping on 20% of its rows."""
    features = lightgbm_features(config, frame)
    parameters = tuned_xgboost_parameters(config, features, labels)
    model, _ = fit_xgboost_with(config, features, labels, parameters, seed)
    categories = {column: features[column].cat.categories for column in config.categorical_features}

    def predict(other: pd.DataFrame) -> np.ndarray:
        other_features = other[config.features].copy()
        for column, column_categories in categories.items():
            other_features[column] = pd.Categorical(other_features[column], categories=column_categories)
        return model.predict_proba(other_features)[:, 1]

    return predict


CLASSICAL_MODELS: list[tuple[str, Callable]] = [
    ("lightgbm", fit_lightgbm),
    ("logistic_regression", fit_logistic_regression),
    ("xgboost_tuned", fit_xgboost_tuned),
]
CLASSICAL_MODEL_NAMES = [name for name, _ in CLASSICAL_MODELS]


def xgboost_submission(config: DatasetConfig) -> None:
    """Fit tuned XGBoost on every labeled row (holdout included) and predict the competition test set.

    Uses the parameters tuned on the largest training sample (no retuning) and averages the
    benchmark's seeds. Writes benchmark/<dataset>/submissions/submission_xgboost_tuned_all.csv.
    """
    stored = json.loads(config.xgboost_parameters_path.read_text())
    largest = max(stored, key=int)
    parameters = stored[largest]["parameters"]
    train = load_data(config)
    test = pd.read_csv(config.data_directory / "test.csv")
    for column in config.numeric_features:
        if test[column].dtype == bool:
            test[column] = test[column].astype(int)
    features = lightgbm_features(config, train)
    labels = train[config.target].to_numpy()
    test_features = test[config.features].copy()
    for column in config.categorical_features:
        test_features[column] = pd.Categorical(test_features[column], categories=features[column].cat.categories)
    seed_predictions = []
    for seed in range(config.baseline_seeds):
        model, stop_auc = fit_xgboost_with(config, features, labels, parameters, seed)
        seed_predictions.append(model.predict_proba(test_features)[:, 1])
        print(f"  seed {seed}: early-stop AUC {stop_auc:.4f}, {model.best_iteration + 1} trees")
    submissions_directory = config.directory / "submissions"
    submissions_directory.mkdir(parents=True, exist_ok=True)
    output_path = submissions_directory / "submission_xgboost_tuned_all.csv"
    pd.DataFrame({"id": test["id"], config.target: np.mean(seed_predictions, axis=0)}).to_csv(output_path, index=False)
    print(f"  Saved {output_path} ({len(test)} rows, parameters tuned at {largest} rows, {len(train)} training rows)")


def baseline(config: DatasetConfig, sizes: list[int] | None, models: list[str] | None) -> None:
    """Score classical models, averaged over seeds; only `sizes` x `models` are rerun, others are kept."""
    config.results_directory.mkdir(parents=True, exist_ok=True)
    holdout = load_holdout(config)
    sizes = sizes or config.sample_sizes
    models = models or CLASSICAL_MODEL_NAMES
    rows = []
    for size in sizes:
        sample = load_sample(config, size)
        labels = sample[config.target].astype(int).to_numpy()
        for model_name, fit_function in CLASSICAL_MODELS:
            if model_name not in models:
                continue
            seed_scores = []
            for seed in range(config.baseline_seeds):
                predict = fit_function(config, sample, labels, seed)
                seed_scores.append(score_probabilities(config, holdout, pd.Series(predict(holdout))))
            scores = pd.DataFrame(seed_scores)
            row = {"model": model_name, "size": size}
            for column in scores.columns:
                row[column] = scores[column].mean()
                row[f"{column}_std"] = scores[column].std()
            rows.append(row)
            print(f"  {model_name:<20s} size={size:>7d}  {HOLDOUT_SCORE}={row[HOLDOUT_SCORE]:.4f}")
    table = pd.DataFrame(rows)
    if config.classical_scores_path.exists():
        existing = pd.read_csv(config.classical_scores_path)
        rerun = existing["size"].isin(sizes) & existing["model"].isin(models)
        table = pd.concat([existing[~rerun], table], ignore_index=True)
    table.sort_values(["model", "size"]).to_csv(config.classical_scores_path, index=False)
    print(f"  Saved {config.classical_scores_path}")


def record_service_score(
    config: DatasetConfig, service: str, size: int, scores: Mapping[str, float | None], note: str
) -> None:
    """Upsert one service score row."""
    config.results_directory.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"service": service, "size": size, **scores, "note": note}])
    if config.service_scores_path.exists():
        existing = pd.read_csv(config.service_scores_path)
        existing = existing[~((existing["service"] == service) & (existing["size"] == size))]
        row = pd.concat([existing, row], ignore_index=True)
    row.sort_values(["service", "size"]).to_csv(config.service_scores_path, index=False)


def load_service_predictions(holdout: pd.DataFrame, predictions_path: Path) -> np.ndarray:
    """Service probabilities aligned to holdout row order."""
    predictions = pd.read_csv(predictions_path)[["id", "probability"]]
    merged = holdout[["id"]].merge(predictions, on="id", how="left", validate="one_to_one")
    missing = merged["probability"].isna().sum()
    if missing:
        raise ValueError(f"{missing} holdout rows have no prediction in {predictions_path}")
    return merged["probability"].to_numpy()


def score(config: DatasetConfig, mode: str, variant: str, size: int, note: str) -> None:
    """Score a 19th holdout predictions file (columns: id, probability)."""
    service = service_name(mode, variant)
    predictions_path = config.service_predictions_path(size, mode, variant)
    holdout = load_holdout(config)
    probabilities = pd.Series(load_service_predictions(holdout, predictions_path))
    scores = score_probabilities(config, holdout, probabilities)
    record_service_score(config, service, size, scores, note)
    print(f"  {service} size={size}: {scores}")


def score_rimay(config: DatasetConfig) -> None:
    """Score Rimay's quantum features with LightGBM on the 300 rows Rimay transformed.

    Rimay returns features only for the rows it was sent, so only the 300-row score exists.
    """
    if config.name != "heart":
        raise ValueError("Rimay was only run on the heart dataset")
    output = RIMAY_DIRECTORY / "quantum_output"
    quantum_train = np.load(output / "1_Xq_train_0.npy")
    quantum_test = np.load(output / "1_Xq_validation_0.npy")
    labels_train = np.load(output / "1_yq_train_0.npy").ravel().astype(int)
    labels_test = np.load(output / "1_yq_validation_0.npy").ravel().astype(int)
    raw_train = pd.DataFrame(np.load(output / "1_Xc_train.npy"), columns=config.features)
    raw_test = pd.DataFrame(np.load(output / "1_Xc_validation.npy"), columns=config.features)
    quantum_columns = [f"quantum_{index}" for index in range(quantum_train.shape[1])]
    quantum_only = DatasetConfig(**{**config.__dict__, "numeric_features": quantum_columns})
    raw_plus_quantum = DatasetConfig(**{**config.__dict__, "numeric_features": [*config.features, *quantum_columns]})

    feature_sets = {
        "rimay_quantum_only": (
            quantum_only,
            pd.DataFrame(quantum_train, columns=quantum_columns),
            pd.DataFrame(quantum_test, columns=quantum_columns),
        ),
        "rimay_raw_plus_quantum": (
            raw_plus_quantum,
            pd.concat([raw_train, pd.DataFrame(quantum_train, columns=quantum_columns)], axis=1),
            pd.concat([raw_test, pd.DataFrame(quantum_test, columns=quantum_columns)], axis=1),
        ),
    }
    for service, (feature_config, train_frame, test_frame) in feature_sets.items():
        seed_scores = []
        for seed in range(config.baseline_seeds):
            predict = fit_lightgbm(feature_config, train_frame, labels_train, seed)
            seed_scores.append(roc_auc_score(labels_test, predict(test_frame)))
        scores = {HOLDOUT_SCORE: None, RIMAY_SCORE: float(np.mean(seed_scores))}
        record_service_score(config, service, 700, scores, "LightGBM on Rimay features, 5 seeds, 500 shots")
        print(f"  {service}: {RIMAY_SCORE}={scores[RIMAY_SCORE]:.4f} (std {np.std(seed_scores):.4f})")


def classical_predictions_for(config: DatasetConfig, size: int, holdout: pd.DataFrame) -> dict[str, np.ndarray]:
    """Seed-averaged holdout predictions of each classical model, cached under benchmark/<dataset>/cache."""
    config.cache_directory.mkdir(parents=True, exist_ok=True)
    predictions = {}
    sample = None
    for model_name, fit_function in CLASSICAL_MODELS:
        cache_path = config.cache_directory / f"classical_{model_name}_{size}.npy"
        if cache_path.exists():
            predictions[model_name] = np.load(cache_path)
            continue
        if sample is None:
            sample = load_sample(config, size)
        sample_labels = sample[config.target].astype(int).to_numpy()
        seed_predictions = [
            fit_function(config, sample, sample_labels, seed)(holdout) for seed in range(config.baseline_seeds)
        ]
        predictions[model_name] = np.mean(seed_predictions, axis=0)
        np.save(cache_path, predictions[model_name])
    return predictions


def compare(config: DatasetConfig, mode: str, variant: str) -> None:
    """Paired bootstrap of a service against the better classical model at each size.

    The classical prediction is the mean over seeds, and the better of LightGBM and logistic
    regression is picked on the full holdout, which slightly favors the classical side.
    """
    holdout = load_holdout(config)
    labels = holdout[LABEL_COLUMN].to_numpy()
    generator = np.random.default_rng(0)
    resamples = [generator.integers(0, len(labels), len(labels)) for _ in range(config.bootstrap_resamples)]
    rows = []
    service = service_name(mode, variant)
    for size in config.sample_sizes:
        predictions_path = config.service_predictions_path(size, mode, variant)
        if not predictions_path.exists():
            print(f"  size={size}: no predictions at {predictions_path}, skipped")
            continue
        classical_predictions = classical_predictions_for(config, size, holdout)
        best_model = max(classical_predictions, key=lambda name: roc_auc_score(labels, classical_predictions[name]))
        classical = classical_predictions[best_model]
        service_probabilities = load_service_predictions(holdout, predictions_path)
        deltas = np.array(
            [
                roc_auc_score(labels[index], service_probabilities[index])
                - roc_auc_score(labels[index], classical[index])
                for index in resamples
            ]
        )
        low, high = np.percentile(deltas, [2.5, 97.5])
        rows.append(
            {
                "size": size,
                "service": service,
                "service_auc": roc_auc_score(labels, service_probabilities),
                "classical_model": best_model,
                "classical_auc": roc_auc_score(labels, classical),
                "lead": float(deltas.mean()),
                "lead_low": float(low),
                "lead_high": float(high),
                "probability_service_better": float((deltas > 0).mean()),
            }
        )
        print(f"  size={size:>7d}  lead={deltas.mean():+.4f}  95% [{low:+.4f}, {high:+.4f}]  vs {best_model}")
    config.results_directory.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    if config.comparison_path.exists():
        existing = pd.read_csv(config.comparison_path)
        table = pd.concat([existing[existing["service"] != service], table], ignore_index=True)
    table.sort_values(["service", "size"]).to_csv(config.comparison_path, index=False)
    print(f"  Saved {config.comparison_path}")


def summary(config: DatasetConfig) -> None:
    """Print classical and service scores side by side by training size."""
    columns = ["model", "size", *config.score_columns]
    tables = [pd.read_csv(config.classical_scores_path)[columns]]
    if config.service_scores_path.exists():
        services = pd.read_csv(config.service_scores_path).rename(columns={"service": "model"})
        tables.append(services[columns])
    combined = pd.concat(tables, ignore_index=True)
    for metric in config.score_columns:
        print(f"\n  {metric}")
        print(combined.pivot(index="size", columns="model", values=metric).round(4).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    baseline_parser = subparsers.add_parser("baseline")
    baseline_parser.add_argument("--sizes", type=int, nargs="+", help="Rerun only these sizes; keep the rest")
    baseline_parser.add_argument("--models", choices=CLASSICAL_MODEL_NAMES, nargs="+", help="Rerun only these models")
    subparsers.add_parser("score-rimay")
    subparsers.add_parser("summary")
    subparsers.add_parser("xgboost-submission")
    compare_parser = subparsers.add_parser("compare")
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--size", type=int, required=True)
    score_parser.add_argument("--note", default="")
    for service_parser in [compare_parser, score_parser]:
        service_parser.add_argument("--mode", choices=TRAINING_MODES, default=DEFAULT_MODE)
        service_parser.add_argument("--variant", choices=MODEL_VARIANTS, default=DEFAULT_VARIANT)
    arguments = parser.parse_args()
    config = DATASETS[arguments.dataset]
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if arguments.command == "prepare":
        prepare(config)
    elif arguments.command == "baseline":
        baseline(config, arguments.sizes, arguments.models)
    elif arguments.command == "score-rimay":
        score_rimay(config)
    elif arguments.command == "score":
        score(config, arguments.mode, arguments.variant, arguments.size, arguments.note)
    elif arguments.command == "compare":
        compare(config, arguments.mode, arguments.variant)
    elif arguments.command == "summary":
        summary(config)
    elif arguments.command == "xgboost-submission":
        xgboost_submission(config)


if __name__ == "__main__":
    main()
