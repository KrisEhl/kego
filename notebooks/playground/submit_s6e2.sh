#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPETITION="playground-series-s6e2"
SUBMISSION_FILE="$SCRIPT_DIR/submission.csv"
MESSAGE="${1:-XGBoost baseline with 10-fold CV}"

if [ ! -f "$SUBMISSION_FILE" ]; then
    echo "Error: $SUBMISSION_FILE not found. Run train_s6e2_baseline.py first."
    exit 1
fi

echo "Submitting $SUBMISSION_FILE to $COMPETITION..."
uv run kaggle c submit -c "$COMPETITION" -f "$SUBMISSION_FILE" -m "$MESSAGE"
