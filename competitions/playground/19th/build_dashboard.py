"""Build dashboard_<dataset>.html: benchmark results embedded in a self-contained page.

Reads benchmark/<dataset>/results/*.csv, embeds them with the dataset's page text into
dashboard_template.html and writes dashboard_<dataset>.html next to this script. The page
needs no network: open it straight from disk.

`--dataset summary` builds dashboard_summary.html instead: both datasets on one condensed page
(19th standard mode against the classical models), without the Rimay quantum comparison.

Usage:
  uv run python build_dashboard.py --dataset heart
  uv run python build_dashboard.py --dataset fraud
  uv run python build_dashboard.py --dataset summary
"""

import argparse
import datetime
import json
import math
from pathlib import Path

import pandas as pd
from benchmark_services import DATASETS, HOLDOUT_SCORE, RIMAY_SCORE, DatasetConfig

DIRECTORY = Path(__file__).resolve().parent
TEMPLATE_PATH = DIRECTORY / "dashboard_template.html"
SUMMARY_TEMPLATE_PATH = DIRECTORY / "summary_template.html"
SUMMARY_DATASETS = {
    "heart": {
        "label": "Heart disease",
        "title": "Heart disease (Playground S6E2)",
        "context": "13 raw features, 44% positive, 1,000-row holdout. An easy dataset: logistic regression already "
        "reaches 0.954 with 100 rows.",
    },
    "fraud": {
        "label": "Card fraud",
        "title": "Card fraud (Playground S6E11)",
        "context": "24 raw features (5 categorical), 1.7% fraud, stratified 100,000-row holdout. 19th levels off near "
        "0.943 from 100,000 rows; the tree models keep improving.",
    },
}
SUMMARY_MODELS = ["19th_standard", "lightgbm", "logistic_regression", "xgboost_tuned"]
DATA_PLACEHOLDER = "__BENCHMARK_DATA__"
TITLE_PLACEHOLDER = "__PAGE_TITLE__"
MODEL_NAMES = {
    "19th_instant": "19th (instant)",
    "19th_standard": "19th (standard)",
    "19th_high": "19th (high)",
    "19th_ultra": "19th (ultra)",
    "lightgbm": "LightGBM",
    "logistic_regression": "Logistic regression",
    "xgboost_tuned": "XGBoost (tuned)",
    "rimay_quantum_only": "Rimay quantum features only",
    "rimay_raw_plus_quantum": "Raw + Rimay quantum features",
}
STANDARD_SERVICE = "19th_standard"
SHARED_METHOD = [
    "Training samples are nested: each smaller sample is part of every larger one.",
    "LightGBM early-stops on 20% of its own training rows. Logistic regression uses standardized numeric "
    "features and one-hot categoricals.",
    "XGBoost is tuned per training size with Optuna (60 trials or 15 minutes, via kego.pipeline.tune) on an "
    "80/20 split of that sample, then refit with 3 seeds; the holdout is never used for tuning.",
    "19th received the same raw columns as an event-probability task in standard mode; its Fast model was run "
    "locally on the holdout.",
    "The classical side of the bootstrap is whichever of LightGBM, logistic regression and tuned XGBoost scores "
    "higher on the full holdout, which slightly favors it.",
    "One 19th run per size. Classical baselines are untuned.",
]
PAGE_TEXT = {
    "heart": {
        "page_title": "19th vs Classical ML: Heart",
        "eyebrow": "Kaggle Playground S6E2 · heart disease · ROC AUC",
        "title": "19th vs classical ML on heart disease",
        "context": "13 raw features. Scored on a fixed 1,000-row holdout that no model trained or early-stopped on. "
        "19th in standard mode, one run per size; LightGBM, logistic regression and tuned XGBoost averaged over 5 seeds.",
        "curves_note": "Training size on a log scale. Logistic regression flattens near 0.964 from 2,000 rows while "
        "LightGBM keeps climbing and closes the gap.",
        "lead_note": "Paired bootstrap, 2,000 resamples of the holdout. A filled dot means the interval excludes "
        "zero (orange: classical ahead); a hollow dot means it crosses zero.",
        "method": [
            "The holdout is Rimay's 300 test rows plus 700 more rows; the 700-row sample is exactly the rows Rimay "
            "was trained on.",
            *SHARED_METHOD,
            "The dataset is easy: logistic regression reaches 0.954 with 100 rows.",
        ],
    },
    "fraud": {
        "page_title": "19th vs Classical ML: Fraud",
        "eyebrow": "Kaggle Playground S6E11 · credit card fraud · ROC AUC",
        "title": "19th vs classical ML on card fraud",
        "context": "24 raw features (5 categorical), 1.7% fraud. Scored on a fixed, stratified 100,000-row holdout "
        "(1,670 frauds) that no model trained or early-stopped on. 19th in standard mode, one run per size; "
        "LightGBM, logistic regression and tuned XGBoost averaged over 3 seeds.",
        "curves_note": "Training size on a log scale. The tree models need far more rows before they are "
        "competitive, but keep improving; 19th and logistic regression flatten from 100,000 rows. Tuned XGBoost "
        "ties 19th at 200,000 rows and passes it on the full data.",
        "lead_note": "Paired bootstrap, 1,000 resamples of the holdout. A filled dot means the interval excludes "
        "zero (orange: classical ahead); a hollow dot means it crosses zero.",
        "leaderboard_note": "Scored by Kaggle on its own test set. The holdout scores are from the benchmark: 19th "
        "trained on 599,960 rows; XGBoost was tuned and scored on 599,960 rows, then refit on all 699,960 for the "
        "submission. Public leaders score about 0.971 with feature engineering that no model here used.",
        "method": [
            "Samples and the holdout are stratified, so every sample keeps the 1.67% fraud rate.",
            *SHARED_METHOD,
            "Booleans were sent as 0/1 to every method. No feature engineering was done for any method.",
            "Instant, high and ultra modes were run up to 200,000 rows on the same preparations as standard.",
        ],
    },
}


def clean(value: object) -> object:
    """Turn NaN into None so the JSON is valid."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def records(frame: pd.DataFrame) -> list[dict]:
    """DataFrame rows as JSON-safe dictionaries."""
    return [{key: clean(value) for key, value in row.items()} for row in frame.to_dict(orient="records")]


def load_scores(config: DatasetConfig) -> pd.DataFrame:
    """Classical and service scores in one long table, with an auc_rimay_300 column always present."""
    classical = pd.read_csv(config.classical_scores_path)
    services = pd.read_csv(config.service_scores_path).rename(columns={"service": "model"})
    scores = pd.concat([classical, services], ignore_index=True)
    if RIMAY_SCORE not in scores.columns:
        scores[RIMAY_SCORE] = None
    scores = scores[["model", "size", HOLDOUT_SCORE, RIMAY_SCORE]]
    scores.insert(1, "model_name", scores["model"].map(MODEL_NAMES))
    scores["size"] = scores["size"].astype(int)
    return scores.sort_values(["model", "size"]).reset_index(drop=True)


def load_comparison(config: DatasetConfig) -> pd.DataFrame:
    """Bootstrap comparison of every 19th mode and variant against the best classical model."""
    comparison = pd.read_csv(config.comparison_path)
    comparison["classical_model"] = comparison["classical_model"].map(MODEL_NAMES)
    comparison["size"] = comparison["size"].astype(int)
    return comparison


def load_leaderboard(config: DatasetConfig) -> list[dict]:
    """Kaggle public leaderboard scores of submitted models, if any were recorded."""
    path = config.results_directory / "leaderboard.csv"
    return records(pd.read_csv(path)) if path.exists() else []


def build(config: DatasetConfig) -> None:
    text = PAGE_TEXT[config.name]
    comparison = load_comparison(config)
    payload = {
        "dataset": config.name,
        "generated": datetime.date.today().isoformat(),
        "text": text,
        "scores": records(load_scores(config)),
        # The headline, lead chart and table follow standard mode; other modes get their own table.
        "comparison": records(comparison[comparison["service"] == STANDARD_SERVICE]),
        "service_comparison": records(comparison),
        "leaderboard": load_leaderboard(config),
    }
    # "</" inside the JSON would close the <script> element early.
    embedded = json.dumps(payload).replace("</", "<\\/")
    template = TEMPLATE_PATH.read_text()
    for placeholder in [DATA_PLACEHOLDER, TITLE_PLACEHOLDER]:
        if template.count(placeholder) != 1:
            raise ValueError(f"{TEMPLATE_PATH.name} must contain {placeholder} exactly once")
    page = template.replace(TITLE_PLACEHOLDER, text["page_title"]).replace(DATA_PLACEHOLDER, embedded)
    output_path = DIRECTORY / f"dashboard_{config.name}.html"
    output_path.write_text(page)
    print(f"  Saved {output_path} ({len(payload['scores'])} scores, {len(payload['comparison'])} comparisons)")


def build_summary() -> None:
    """Write dashboard_summary.html: both datasets, 19th standard mode vs the classical models."""
    datasets = []
    leaderboard: list[dict] = []
    for name, text in SUMMARY_DATASETS.items():
        config = DATASETS[name]
        scores = load_scores(config)
        comparison = load_comparison(config)
        datasets.append(
            {
                "name": name,
                **text,
                "scores": records(scores[scores["model"].isin(SUMMARY_MODELS)].drop(columns=[RIMAY_SCORE])),
                "comparison": records(comparison[comparison["service"] == STANDARD_SERVICE]),
            }
        )
        leaderboard += load_leaderboard(config)
    payload = {"generated": datetime.date.today().isoformat(), "datasets": datasets, "leaderboard": leaderboard}
    embedded = json.dumps(payload).replace("</", "<\\/")
    template = SUMMARY_TEMPLATE_PATH.read_text()
    if template.count(DATA_PLACEHOLDER) != 1:
        raise ValueError(f"{SUMMARY_TEMPLATE_PATH.name} must contain {DATA_PLACEHOLDER} exactly once")
    output_path = DIRECTORY / "dashboard_summary.html"
    output_path.write_text(template.replace(DATA_PLACEHOLDER, embedded))
    print(f"  Saved {output_path} ({len(datasets)} datasets, {len(leaderboard)} leaderboard entries)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=[*sorted(DATASETS), "summary"], required=True)
    dataset = parser.parse_args().dataset
    if dataset == "summary":
        build_summary()
    else:
        build(DATASETS[dataset])


if __name__ == "__main__":
    main()
