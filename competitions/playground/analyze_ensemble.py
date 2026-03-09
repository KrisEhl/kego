"""Greedy ensemble member selection and leave-one-out analysis.

Tests whether each model actually improves the ensemble or just adds noise.

Modes:
  1. Greedy forward selection (default): Starting from empty, adds models one
     at a time picking the one with highest AUC gain.
  2. Leave-one-out (--check): Removes each model one at a time from an existing
     ensemble to measure each member's contribution.

Usage:
    python analyze_ensemble.py --from-experiment playground-s6e2-full
    python analyze_ensemble.py --from-ensemble submit-v1
    python analyze_ensemble.py --from-ensemble submit-v1 --check
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from kego.datasets.split import split_dataset  # noqa: E402
from kego.ensemble.analysis import (  # noqa: E402
    greedy_forward_selection,
    leave_one_out_analysis,
    print_forward_selection,
    print_leave_one_out,
)
from kego.tracking import (  # noqa: E402
    load_predictions_from_ensemble,
    load_predictions_from_mlflow,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = (
    Path(os.environ.get("KEGO_PATH_DATA", project_root / "data"))
    / "playground"
    / "playground-series-s6e2"
)
TARGET = "Heart Disease"


def main():
    parser = argparse.ArgumentParser(
        description="Greedy ensemble member selection and leave-one-out analysis"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--from-experiment",
        nargs="+",
        metavar="EXPERIMENT",
        help="Load predictions from MLflow experiment(s)",
    )
    source.add_argument(
        "--from-ensemble",
        type=str,
        metavar="ENSEMBLE",
        help="Load predictions from a curated ensemble (tagged runs)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Leave-one-out analysis instead of greedy forward selection",
    )
    args = parser.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        logger.error("MLFLOW_TRACKING_URI must be set")
        sys.exit(1)

    # --- Load predictions ---
    if args.from_ensemble:
        logger.info(f"Loading predictions from ensemble: {args.from_ensemble}")
        model_names, all_oof, all_holdout, _ = load_predictions_from_ensemble(
            args.from_ensemble, tracking_uri
        )
    else:
        logger.info(f"Loading predictions from experiments: {args.from_experiment}")
        model_names, all_oof, all_holdout, _ = load_predictions_from_mlflow(
            args.from_experiment, tracking_uri
        )

    if not model_names:
        logger.error("No predictions loaded, exiting")
        sys.exit(1)

    # --- Load holdout labels (same split as training) ---
    train_full = pd.read_csv(DATA_DIR / "train.csv")
    original = pd.read_csv(DATA_DIR / "Heart_Disease_Prediction.csv")

    train_full[TARGET] = train_full[TARGET].map({"Presence": 1, "Absence": 0})
    original[TARGET] = original[TARGET].map({"Presence": 1, "Absence": 0})

    original["id"] = -1
    train_full = pd.concat([train_full, original], ignore_index=True)

    train_split, holdout, _ = split_dataset(
        train_full,
        train_size=0.8,
        validate_size=0.2,
        stratify_column=TARGET,
    )
    holdout = holdout.reset_index(drop=True)
    holdout_labels = holdout[TARGET].values
    oof_labels = train_split[TARGET].values

    # --- Run analysis ---
    if args.check:
        full_auc, strategy, n_models, rows = leave_one_out_analysis(
            model_names, all_oof, all_holdout, holdout_labels, oof_labels=oof_labels
        )
        print_leave_one_out(full_auc, strategy, n_models, rows)
    else:
        display_rows, rejected_rows = greedy_forward_selection(
            model_names, all_oof, all_holdout, holdout_labels, oof_labels=oof_labels
        )
        print_forward_selection(display_rows, rejected_rows)


if __name__ == "__main__":
    main()
