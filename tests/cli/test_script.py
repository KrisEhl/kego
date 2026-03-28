"""
Manual test script for kego run.

Usage:
    MLFLOW_TRACKING_URI=sqlite:///test.db uv run kego run tests/cli/test_script.py
    MLFLOW_TRACKING_URI=sqlite:///test.db uv run kego run tests/cli/test_script.py --debug
    MLFLOW_TRACKING_URI=sqlite:///test.db uv run kego run tests/cli/test_script.py --fold 2 --epochs 5 --name my-experiment
    MLFLOW_TRACKING_URI=sqlite:///test.db uv run kego ls
"""

import argparse
import time

parser = argparse.ArgumentParser(
    description="Fake training script for kego run testing"
)
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--backbone", type=str, default="efficientnet_b0")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

if args.debug:
    args.epochs = 2
    print("Debug mode: reduced to 2 epochs")

print(f"KEGO_PARAM backbone {args.backbone}")
print(f"KEGO_PARAM fold {args.fold}")

for epoch in range(args.epochs):
    time.sleep(0.1)
    auc = 0.70 + epoch * 0.02 + args.fold * 0.005
    print(f"KEGO_METRIC epoch {epoch}")
    print(f"KEGO_METRIC fold_auc {auc:.4f}")
    print(f"  epoch {epoch + 1}/{args.epochs}  fold_auc={auc:.4f}", flush=True)

print(f"Training complete. fold={args.fold} final_auc={auc:.4f}")
