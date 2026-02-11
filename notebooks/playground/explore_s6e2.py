"""Data exploration script for Playground Series S6E2 (Heart Disease prediction).

Prints data overview (types, shapes, missing values, duplicates, outliers)
and saves distribution/correlation plots to an output directory.

Usage:
    uv run python notebooks/playground/explore_s6e2.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from kego.constants import PATH_DATA
from kego.plotting.figures import create_axes_grid, save_figure
from kego.plotting.histograms import plot_histogram

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PATH_DATA / "playground" / "playground-series-s6e2"
OUTPUT_DIR = project_root / "notebooks" / "playground" / "plots_s6e2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

TARGET = "Heart Disease"

# Map target to numeric for analysis convenience
target_map = {"Presence": 1, "Absence": 0}
train[TARGET] = train[TARGET].map(target_map)

print("=" * 70)
print("SHAPE")
print("=" * 70)
print(f"  Train : {train.shape}")
print(f"  Test  : {test.shape}")

# ---------------------------------------------------------------------------
# Column overview
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("COLUMN TYPES & NON-NULL COUNTS (train)")
print("=" * 70)
print(train.dtypes.to_frame("dtype").join(train.count().to_frame("non_null")))

# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)
missing_train = train.isnull().sum()
missing_test = test.isnull().sum()
missing_df = pd.DataFrame({"train": missing_train, "test": missing_test})
missing_df["train_%"] = (missing_df["train"] / len(train) * 100).round(2)
missing_df["test_%"] = (missing_df["test"] / len(test) * 100).round(2)
print(missing_df[missing_df[["train", "test"]].sum(axis=1) > 0].to_string())
if missing_df[["train", "test"]].sum().sum() == 0:
    print("  No missing values.")

# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("DUPLICATES")
print("=" * 70)
feature_cols = [c for c in train.columns if c not in ("id", TARGET)]
n_dup_train = train.duplicated(subset=feature_cols).sum()
n_dup_test = test.duplicated(subset=feature_cols).sum()
print(
    f"  Train duplicate rows (excl. id & target): {n_dup_train} "
    f"({n_dup_train / len(train) * 100:.2f}%)"
)
print(
    f"  Test  duplicate rows (excl. id):          {n_dup_test} "
    f"({n_dup_test / len(test) * 100:.2f}%)"
)

# ---------------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS (train)")
print("=" * 70)
print(train.describe().T.to_string())

# ---------------------------------------------------------------------------
# Target distribution
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TARGET DISTRIBUTION (train)")
print("=" * 70)
target_counts = train[TARGET].value_counts().sort_index()
for val, cnt in target_counts.items():
    label = "Absence" if val == 0 else "Presence"
    print(f"  {label} ({val}): {cnt}  ({cnt / len(train) * 100:.2f}%)")

# ---------------------------------------------------------------------------
# Unique values per column (useful to spot low-cardinality / categorical)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("UNIQUE VALUES PER COLUMN (train)")
print("=" * 70)
for col in train.columns:
    n_unique = train[col].nunique()
    extra = ""
    if n_unique <= 10:
        extra = f"  values: {sorted(train[col].dropna().unique().tolist())}"
    print(f"  {col:30s}  {n_unique:>8}{extra}")

# ---------------------------------------------------------------------------
# Outliers via IQR method (numeric columns)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("OUTLIERS (IQR method, train)")
print("=" * 70)
numeric_cols = train.select_dtypes(include=[np.number]).columns.drop(
    ["id", TARGET], errors="ignore"
)

outlier_summary = []
for col in numeric_cols:
    q1 = train[col].quantile(0.25)
    q3 = train[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_out = ((train[col] < lower) | (train[col] > upper)).sum()
    pct = n_out / len(train) * 100
    outlier_summary.append((col, n_out, pct, lower, upper))
    if n_out > 0:
        print(
            f"  {col:30s}  {n_out:>8} ({pct:5.2f}%)  "
            f"bounds=[{lower:.2f}, {upper:.2f}]"
        )

if all(row[1] == 0 for row in outlier_summary):
    print("  No IQR outliers detected.")

# ---------------------------------------------------------------------------
# Zero / suspicious values
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ZERO-VALUE COUNTS (may indicate missing data encoded as 0)")
print("=" * 70)
for col in numeric_cols:
    n_zero = (train[col] == 0).sum()
    if n_zero > 0:
        print(f"  {col:30s}  {n_zero:>8} ({n_zero / len(train) * 100:.2f}%)")

# ---------------------------------------------------------------------------
# Correlation with target
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CORRELATION WITH TARGET (train)")
print("=" * 70)
corr_with_target = train[numeric_cols.tolist() + [TARGET]].corr()[TARGET].drop(TARGET)
corr_sorted = corr_with_target.abs().sort_values(ascending=False)
for col in corr_sorted.index:
    print(f"  {col:30s}  {corr_with_target[col]:+.4f}")

# ===========================================================================
# PLOTS
# ===========================================================================
print("\n" + "=" * 70)
print(f"SAVING PLOTS to {OUTPUT_DIR}")
print("=" * 70)

# --- 1. Feature distributions (train vs test overlay) ---
n_features = len(numeric_cols)
n_cols_grid = 4
n_rows_grid = int(np.ceil(n_features / n_cols_grid))

fig, axes_grid, _ = create_axes_grid(
    n_columns=n_cols_grid,
    n_rows=n_rows_grid,
    title="Feature Distributions (train)",
    figure_size=(n_cols_grid * 4, n_rows_grid * 3),
)

for idx, col in enumerate(numeric_cols):
    row, c = divmod(idx, n_cols_grid)
    ax = axes_grid[row][c]
    plot_histogram(col, df=train, axes=ax, label_x=col, font_size=10, title=col)

# hide unused axes
for idx in range(n_features, n_rows_grid * n_cols_grid):
    row, c = divmod(idx, n_cols_grid)
    axes_grid[row][c].set_visible(False)

save_figure(fig, filename=str(OUTPUT_DIR / "feature_distributions.png"), dpi=150)
plt.close(fig)
print("  Saved feature_distributions.png")

# --- 2. Target-split distributions ---
fig2, axes_grid2, _ = create_axes_grid(
    n_columns=n_cols_grid,
    n_rows=n_rows_grid,
    title="Distributions by Target Class",
    figure_size=(n_cols_grid * 4, n_rows_grid * 3),
)

for idx, col in enumerate(numeric_cols):
    row, c = divmod(idx, n_cols_grid)
    ax = axes_grid2[row][c]
    for label_val, color, label_name in [
        (0, "#1f77b4", "Absence"),
        (1, "#d62728", "Presence"),
    ]:
        subset = train.loc[train[TARGET] == label_val, col]
        plot_histogram(
            subset.values,
            axes=ax,
            label_x=col,
            font_size=10,
            title=col,
            alpha=0.5,
            color=color,
        )
    ax.legend(["Absence", "Presence"], fontsize=7)

for idx in range(n_features, n_rows_grid * n_cols_grid):
    row, c = divmod(idx, n_cols_grid)
    axes_grid2[row][c].set_visible(False)

save_figure(fig2, filename=str(OUTPUT_DIR / "distributions_by_target.png"), dpi=150)
plt.close(fig2)
print("  Saved distributions_by_target.png")

# --- 3. Correlation heatmap ---
corr_matrix = train[numeric_cols.tolist() + [TARGET]].corr()
fig3, ax3 = plt.subplots(figsize=(10, 8))
im = ax3.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax3.set_xticks(range(len(corr_matrix)))
ax3.set_yticks(range(len(corr_matrix)))
ax3.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=8)
ax3.set_yticklabels(corr_matrix.columns, fontsize=8)

for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        ax3.text(
            j,
            i,
            f"{corr_matrix.values[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=6,
            color="white" if abs(corr_matrix.values[i, j]) > 0.5 else "black",
        )

fig3.colorbar(im, fraction=0.046, pad=0.04)
ax3.set_title("Correlation Matrix", fontsize=12)
fig3.tight_layout()
save_figure(fig3, filename=str(OUTPUT_DIR / "correlation_matrix.png"), dpi=150)
plt.close(fig3)
print("  Saved correlation_matrix.png")

# --- 4. Box plots for outlier visualization ---
fig4, axes4 = plt.subplots(
    n_rows_grid, n_cols_grid, figsize=(n_cols_grid * 4, n_rows_grid * 3)
)
axes4 = np.atleast_2d(axes4)

for idx, col in enumerate(numeric_cols):
    row, c = divmod(idx, n_cols_grid)
    ax = axes4[row][c]
    train.boxplot(column=col, by=TARGET, ax=ax)
    ax.set_title(col, fontsize=10)
    ax.set_xlabel("")

for idx in range(n_features, n_rows_grid * n_cols_grid):
    row, c = divmod(idx, n_cols_grid)
    axes4[row][c].set_visible(False)

fig4.suptitle("Box Plots by Target Class", fontsize=14)
fig4.tight_layout()
save_figure(fig4, filename=str(OUTPUT_DIR / "boxplots_by_target.png"), dpi=150)
plt.close(fig4)
print("  Saved boxplots_by_target.png")

print("\nDone.")
