"""Deep EDA for Playground Series S6E2 (Heart Disease prediction).

Focus: outliers, categorical traps, synthetic data artifacts, train/test shift.

Prints detailed analysis and saves plots to an output directory.

Usage:
    uv run python notebooks/playground/explore_s6e2.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy import stats

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from kego.plotting.figures import create_axes_grid, save_figure

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DATA_DIR = (
    Path(os.environ.get("KEGO_PATH_DATA", project_root / "data"))
    / "playground"
    / "playground-series-s6e2"
)
OUTPUT_DIR = project_root / "notebooks" / "playground" / "plots_s6e2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Heart Disease"

CAT_FEATURES = [
    "Sex",
    "Chest pain type",
    "FBS over 120",
    "EKG results",
    "Exercise angina",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
]
CONT_FEATURES = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]


def section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
original = pd.read_csv(DATA_DIR / "Heart_Disease_Prediction.csv")

train[TARGET] = train[TARGET].map({"Presence": 1, "Absence": 0})
original[TARGET] = original[TARGET].map({"Presence": 1, "Absence": 0})

section("DATA SHAPES")
print(f"  Train:    {train.shape}")
print(f"  Test:     {test.shape}")
print(f"  Original: {original.shape}")
print(f"  Target rate (train):    {train[TARGET].mean():.4f}")
print(f"  Target rate (original): {original[TARGET].mean():.4f}")

# ===========================================================================
# 1. CATEGORICAL FEATURE DEEP DIVE
# ===========================================================================
section("CATEGORICAL FEATURE ANALYSIS — Value distributions & target rates")

for col in CAT_FEATURES:
    print(f"\n  --- {col} ---")
    vals = sorted(train[col].unique())
    print(
        f"  {'Value':>6s}  {'Count':>8s}  {'%':>7s}  {'Target%':>8s}  {'Test%':>7s}  {'Orig%':>7s}  Notes"
    )
    print(f"  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*20}")

    train_total = len(train)
    test_total = len(test)
    orig_total = len(original)

    for v in vals:
        t_cnt = (train[col] == v).sum()
        t_pct = t_cnt / train_total * 100
        t_target = train.loc[train[col] == v, TARGET].mean() * 100
        te_cnt = (test[col] == v).sum()
        te_pct = te_cnt / test_total * 100
        o_cnt = (original[col] == v).sum() if col in original.columns else 0
        o_pct = o_cnt / orig_total * 100 if orig_total > 0 else 0

        # Flag discrepancies
        notes = []
        if abs(t_pct - te_pct) > 2.0:
            notes.append(f"train/test shift {t_pct - te_pct:+.1f}%")
        if orig_total > 0 and abs(t_pct - o_pct) > 5.0:
            notes.append(f"synth/orig shift {t_pct - o_pct:+.1f}%")
        if t_pct < 1.0:
            notes.append("RARE (<1%)")

        print(
            f"  {v:>6}  {t_cnt:>8d}  {t_pct:>6.2f}%  {t_target:>7.2f}%  "
            f"{te_pct:>6.2f}%  {o_pct:>6.2f}%  {'  '.join(notes)}"
        )

    # Ordinal check: is target rate monotonic with value?
    target_rates = [train.loc[train[col] == v, TARGET].mean() for v in vals]
    is_monotonic_inc = all(a <= b for a, b in zip(target_rates, target_rates[1:]))
    is_monotonic_dec = all(a >= b for a, b in zip(target_rates, target_rates[1:]))
    if not is_monotonic_inc and not is_monotonic_dec and len(vals) > 2:
        print(f"  ** NON-MONOTONIC target rate: NOT truly ordinal!")
        print(
            f"     Target rates by value: {dict(zip(vals, [f'{r:.3f}' for r in target_rates]))}"
        )


# ===========================================================================
# 2. CONTINUOUS FEATURE OUTLIER ANALYSIS
# ===========================================================================
section("CONTINUOUS FEATURE OUTLIER ANALYSIS")

for col in CONT_FEATURES:
    q1, q3 = train[col].quantile(0.25), train[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    extreme_lower, extreme_upper = q1 - 3 * iqr, q3 + 3 * iqr

    mild = train[(train[col] < lower) | (train[col] > upper)]
    extreme = train[(train[col] < extreme_lower) | (train[col] > extreme_upper)]

    print(f"\n  --- {col} ---")
    print(f"  Range:          [{train[col].min():.1f}, {train[col].max():.1f}]")
    print(f"  IQR:            [{q1:.1f}, {q3:.1f}], IQR={iqr:.1f}")
    print(
        f"  Mild outliers:  {len(mild)} ({len(mild)/len(train)*100:.2f}%)  bounds=[{lower:.1f}, {upper:.1f}]"
    )
    print(
        f"  Extreme:        {len(extreme)} ({len(extreme)/len(train)*100:.2f}%)  bounds=[{extreme_lower:.1f}, {extreme_upper:.1f}]"
    )

    # Target rate in outliers vs non-outliers
    if len(mild) > 0:
        target_out = mild[TARGET].mean()
        target_in = train.loc[~train.index.isin(mild.index), TARGET].mean()
        print(
            f"  Target rate:    outliers={target_out:.4f}  vs  non-outliers={target_in:.4f}  (delta={target_out - target_in:+.4f})"
        )

    # Check for suspicious values (0s in columns that shouldn't have them)
    n_zero = (train[col] == 0).sum()
    if n_zero > 0 and col not in ("ST depression",):
        print(
            f"  ** Zero values: {n_zero} ({n_zero/len(train)*100:.2f}%) — possible missing data"
        )
        if col in original.columns:
            orig_zero = (original[col] == 0).sum()
            print(f"     Original data: {orig_zero}/{len(original)} zeros")


# ===========================================================================
# 3. SYNTHETIC vs ORIGINAL DATA COMPARISON
# ===========================================================================
section("SYNTHETIC vs ORIGINAL DATA COMPARISON")

all_features = CAT_FEATURES + CONT_FEATURES
for col in all_features:
    if col not in original.columns:
        continue
    synth = train[col]
    orig = original[col]

    if col in CAT_FEATURES:
        # Chi-squared test on distributions
        synth_dist = synth.value_counts(normalize=True).sort_index()
        orig_dist = orig.value_counts(normalize=True).sort_index()
        all_vals = sorted(set(synth_dist.index) | set(orig_dist.index))
        synth_pct = [synth_dist.get(v, 0) * 100 for v in all_vals]
        orig_pct = [orig_dist.get(v, 0) * 100 for v in all_vals]
        max_diff = max(abs(s - o) for s, o in zip(synth_pct, orig_pct))
        flag = " ** LARGE SHIFT" if max_diff > 5 else ""
        print(f"  {col:<25s}  max_diff={max_diff:.1f}%{flag}")
    else:
        # KS test for continuous features
        ks_stat, ks_p = stats.ks_2samp(synth, orig)
        flag = " ** SIGNIFICANT" if ks_p < 0.01 else ""
        print(
            f"  {col:<25s}  KS={ks_stat:.4f}  p={ks_p:.4g}"
            f"  synth_mean={synth.mean():.2f}  orig_mean={orig.mean():.2f}{flag}"
        )


# ===========================================================================
# 4. TRAIN vs TEST DISTRIBUTION SHIFT
# ===========================================================================
section("TRAIN vs TEST DISTRIBUTION SHIFT")

for col in all_features:
    if col not in test.columns:
        continue
    if col in CAT_FEATURES:
        tr_dist = train[col].value_counts(normalize=True).sort_index()
        te_dist = test[col].value_counts(normalize=True).sort_index()
        all_vals = sorted(set(tr_dist.index) | set(te_dist.index))
        max_diff = max(
            abs(tr_dist.get(v, 0) - te_dist.get(v, 0)) * 100 for v in all_vals
        )
        flag = " ** SHIFT" if max_diff > 1 else ""
        print(f"  {col:<25s}  max_cat_diff={max_diff:.2f}%{flag}")
    else:
        ks_stat, ks_p = stats.ks_2samp(train[col], test[col])
        mean_diff = train[col].mean() - test[col].mean()
        flag = " ** SIGNIFICANT" if ks_p < 0.01 else ""
        print(
            f"  {col:<25s}  KS={ks_stat:.4f}  p={ks_p:.4g}  mean_diff={mean_diff:+.3f}{flag}"
        )


# ===========================================================================
# 5. DUPLICATE ANALYSIS
# ===========================================================================
section("DUPLICATE ANALYSIS")

feature_cols = [c for c in train.columns if c not in ("id", TARGET)]
dup_mask = train.duplicated(subset=feature_cols, keep=False)
n_dup_rows = dup_mask.sum()
n_dup_groups = train[dup_mask].groupby(feature_cols).ngroups

print(f"  Rows involved in duplicates: {n_dup_rows} ({n_dup_rows/len(train)*100:.2f}%)")
print(f"  Unique duplicate groups: {n_dup_groups}")

# Check target consistency in duplicates
if n_dup_rows > 0:
    dup_groups = train[dup_mask].groupby(feature_cols)[TARGET]
    inconsistent = dup_groups.apply(lambda g: g.nunique() > 1)
    n_inconsistent = inconsistent.sum()
    print(
        f"  Groups with INCONSISTENT targets: {n_inconsistent} ({n_inconsistent/max(n_dup_groups,1)*100:.1f}%)"
    )
    if n_inconsistent > 0:
        print(f"  ** Same features, different labels — noisy labels!")

    # Size distribution of duplicate groups
    group_sizes = dup_groups.size()
    print(
        f"  Duplicate group sizes: min={group_sizes.min()}, max={group_sizes.max()}, "
        f"median={group_sizes.median():.0f}, mean={group_sizes.mean():.1f}"
    )


# ===========================================================================
# 6. RARE VALUE COMBINATIONS (cross-tabulation)
# ===========================================================================
section("RARE & SUSPICIOUS CROSS-TABULATIONS")

# Thallium x Chest pain type (the two strongest categorical predictors)
print("\n  Thallium x Chest pain type (count / target_rate):")
ct = pd.crosstab(train["Thallium"], train["Chest pain type"], margins=True)
ct_target = pd.crosstab(
    train["Thallium"], train["Chest pain type"], values=train[TARGET], aggfunc="mean"
)
print("  Counts:")
print(ct.to_string().replace("\n", "\n  "))
print("\n  Target rates:")
print(ct_target.round(3).to_string().replace("\n", "\n  "))

# Thallium x Exercise angina
print("\n\n  Thallium x Exercise angina (count / target_rate):")
ct2 = pd.crosstab(train["Thallium"], train["Exercise angina"], margins=True)
ct2_target = pd.crosstab(
    train["Thallium"], train["Exercise angina"], values=train[TARGET], aggfunc="mean"
)
print("  Counts:")
print(ct2.to_string().replace("\n", "\n  "))
print("\n  Target rates:")
print(ct2_target.round(3).to_string().replace("\n", "\n  "))

# Number of vessels fluro x Thallium
print("\n\n  Number of vessels fluro x Thallium (target_rate):")
ct3_target = pd.crosstab(
    train["Number of vessels fluro"],
    train["Thallium"],
    values=train[TARGET],
    aggfunc="mean",
)
print(ct3_target.round(3).to_string().replace("\n", "\n  "))


# ===========================================================================
# 7. CORRELATION MATRIX
# ===========================================================================
section("CORRELATION WITH TARGET")

numeric_cols = CONT_FEATURES + CAT_FEATURES
corr = train[numeric_cols + [TARGET]].corr()[TARGET].drop(TARGET)
for col in corr.abs().sort_values(ascending=False).index:
    print(f"  {col:<25s}  r={corr[col]:+.4f}")


# ===========================================================================
# PLOTS
# ===========================================================================
section(f"SAVING PLOTS to {OUTPUT_DIR}")


# --- Plot 1: Categorical value distributions with target rates ---
n_cats = len(CAT_FEATURES)
fig, axes, _ = create_axes_grid(
    n_columns=4,
    n_rows=2,
    title="Categorical Features: Distribution & Target Rate",
    figure_size=(20, 10),
)

for idx, col in enumerate(CAT_FEATURES):
    row, c = divmod(idx, 4)
    ax = axes[row][c]
    vals = sorted(train[col].unique())

    counts = [((train[col] == v).sum()) for v in vals]
    target_rates = [train.loc[train[col] == v, TARGET].mean() for v in vals]

    x = np.arange(len(vals))
    bars = ax.bar(x, counts, color="#4C72B0", alpha=0.7, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in vals], fontsize=8)
    ax.set_ylabel("Count", fontsize=8, color="#4C72B0")
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", labelsize=7)

    # Target rate on secondary axis
    ax2 = ax.twinx()
    ax2.plot(
        x, target_rates, "o-", color="#C44E52", markersize=5, linewidth=2, zorder=3
    )
    ax2.set_ylabel("Target Rate", fontsize=8, color="#C44E52")
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", labelsize=7)
    ax2.axhline(
        train[TARGET].mean(), color="#C44E52", linestyle="--", alpha=0.3, linewidth=1
    )

save_figure(fig, filename=str(OUTPUT_DIR / "categorical_traps.png"), dpi=150)
plt.close(fig)
print("  Saved categorical_traps.png")


# --- Plot 2: Continuous feature distributions with outlier regions ---
n_cont = len(CONT_FEATURES)
fig2, axes2, _ = create_axes_grid(
    n_columns=3,
    n_rows=2,
    title="Continuous Features: Distributions & Outlier Regions",
    figure_size=(18, 10),
)

for idx, col in enumerate(CONT_FEATURES):
    row, c = divmod(idx, 3)
    ax = axes2[row][c]

    q1, q3 = train[col].quantile(0.25), train[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    # Histogram split by target
    for label_val, color, label_name in [
        (0, "#4C72B0", "Absence"),
        (1, "#C44E52", "Presence"),
    ]:
        subset = train.loc[train[TARGET] == label_val, col]
        ax.hist(subset, bins=60, alpha=0.5, color=color, label=label_name, density=True)

    # Outlier region shading
    ax.axvspan(train[col].min() - 1, lower, alpha=0.1, color="red", zorder=0)
    ax.axvspan(upper, train[col].max() + 1, alpha=0.1, color="red", zorder=0)
    ax.axvline(lower, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(upper, color="red", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_title(f"{col}\nIQR outliers: [{lower:.0f}, {upper:.0f}]", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

# Hide unused subplot
if n_cont < 6:
    for idx in range(n_cont, 6):
        row, c = divmod(idx, 3)
        axes2[row][c].set_visible(False)

save_figure(fig2, filename=str(OUTPUT_DIR / "continuous_outliers.png"), dpi=150)
plt.close(fig2)
print("  Saved continuous_outliers.png")


# --- Plot 3: Train vs Test vs Original distributions (categorical) ---
fig3, axes3, _ = create_axes_grid(
    n_columns=4,
    n_rows=2,
    title="Train vs Test vs Original: Categorical Distributions",
    figure_size=(20, 10),
)

for idx, col in enumerate(CAT_FEATURES):
    row, c = divmod(idx, 4)
    ax = axes3[row][c]
    vals = sorted(set(train[col].unique()) | set(test[col].unique()))

    tr_pcts = [(train[col] == v).mean() * 100 for v in vals]
    te_pcts = [(test[col] == v).mean() * 100 for v in vals]
    orig_pcts = (
        [(original[col] == v).mean() * 100 for v in vals]
        if col in original.columns
        else [0] * len(vals)
    )

    x = np.arange(len(vals))
    w = 0.25
    ax.bar(x - w, tr_pcts, w, label="Train", color="#4C72B0", alpha=0.8)
    ax.bar(x, te_pcts, w, label="Test", color="#55A868", alpha=0.8)
    ax.bar(x + w, orig_pcts, w, label="Original", color="#C44E52", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in vals], fontsize=8)
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.set_ylabel("%", fontsize=8)
    ax.tick_params(labelsize=7)
    if idx == 0:
        ax.legend(fontsize=7)

save_figure(fig3, filename=str(OUTPUT_DIR / "train_test_original_cats.png"), dpi=150)
plt.close(fig3)
print("  Saved train_test_original_cats.png")


# --- Plot 4: Train vs Test continuous distributions ---
fig4, axes4, _ = create_axes_grid(
    n_columns=3,
    n_rows=2,
    title="Train vs Test vs Original: Continuous Distributions",
    figure_size=(18, 10),
)

for idx, col in enumerate(CONT_FEATURES):
    row, c = divmod(idx, 3)
    ax = axes4[row][c]

    bins = np.linspace(
        min(train[col].min(), test[col].min()),
        max(train[col].max(), test[col].max()),
        61,
    )
    ax.hist(
        train[col], bins=bins, alpha=0.5, color="#4C72B0", label="Train", density=True
    )
    ax.hist(
        test[col], bins=bins, alpha=0.5, color="#55A868", label="Test", density=True
    )
    if col in original.columns:
        ax.hist(
            original[col],
            bins=bins,
            alpha=0.4,
            color="#C44E52",
            label="Original",
            density=True,
        )

    ks_stat, ks_p = stats.ks_2samp(train[col], test[col])
    ax.set_title(f"{col}\nKS={ks_stat:.4f} (p={ks_p:.2g})", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

if n_cont < 6:
    for idx in range(n_cont, 6):
        row, c = divmod(idx, 3)
        axes4[row][c].set_visible(False)

save_figure(fig4, filename=str(OUTPUT_DIR / "train_test_original_cont.png"), dpi=150)
plt.close(fig4)
print("  Saved train_test_original_cont.png")


# --- Plot 5: Thallium deep dive (the dominant predictor) ---
fig5, axes5 = plt.subplots(1, 3, figsize=(18, 5))

# 5a: Thallium value counts by target
vals = sorted(train["Thallium"].unique())
x = np.arange(len(vals))
for i, (label_val, color, name) in enumerate(
    [(0, "#4C72B0", "Absence"), (1, "#C44E52", "Presence")]
):
    counts = [
        (train[(train["Thallium"] == v) & (train[TARGET] == label_val)].shape[0])
        for v in vals
    ]
    axes5[0].bar(x + i * 0.35, counts, 0.35, label=name, color=color, alpha=0.8)
axes5[0].set_xticks(x + 0.175)
axes5[0].set_xticklabels([str(v) for v in vals])
axes5[0].set_title("Thallium: Counts by Target")
axes5[0].legend()
axes5[0].set_xlabel("Thallium value")
axes5[0].set_ylabel("Count")

# 5b: Thallium x Chest pain type heatmap (target rate)
ct_target = pd.crosstab(
    train["Thallium"], train["Chest pain type"], values=train[TARGET], aggfunc="mean"
)
im = axes5[1].imshow(
    ct_target.values, cmap="RdYlGn_r", aspect="auto", norm=Normalize(vmin=0, vmax=1)
)
axes5[1].set_xticks(range(len(ct_target.columns)))
axes5[1].set_xticklabels(ct_target.columns.astype(str))
axes5[1].set_yticks(range(len(ct_target.index)))
axes5[1].set_yticklabels(ct_target.index.astype(str))
for i in range(len(ct_target.index)):
    for j in range(len(ct_target.columns)):
        v = ct_target.values[i, j]
        if not np.isnan(v):
            axes5[1].text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if v > 0.6 or v < 0.2 else "black",
            )
axes5[1].set_title("Thallium x Chest Pain: Target Rate")
axes5[1].set_xlabel("Chest pain type")
axes5[1].set_ylabel("Thallium")
plt.colorbar(im, ax=axes5[1], fraction=0.046)

# 5c: Thallium x Exercise angina heatmap (target rate)
ct2_target = pd.crosstab(
    train["Thallium"], train["Exercise angina"], values=train[TARGET], aggfunc="mean"
)
im2 = axes5[2].imshow(
    ct2_target.values, cmap="RdYlGn_r", aspect="auto", norm=Normalize(vmin=0, vmax=1)
)
axes5[2].set_xticks(range(len(ct2_target.columns)))
axes5[2].set_xticklabels(ct2_target.columns.astype(str))
axes5[2].set_yticks(range(len(ct2_target.index)))
axes5[2].set_yticklabels(ct2_target.index.astype(str))
for i in range(len(ct2_target.index)):
    for j in range(len(ct2_target.columns)):
        v = ct2_target.values[i, j]
        if not np.isnan(v):
            axes5[2].text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if v > 0.6 or v < 0.2 else "black",
            )
axes5[2].set_title("Thallium x Angina: Target Rate")
axes5[2].set_xlabel("Exercise angina")
axes5[2].set_ylabel("Thallium")
plt.colorbar(im2, ax=axes5[2], fraction=0.046)

fig5.suptitle(
    "Thallium Deep Dive (dominant predictor, values 3/6/7 are NOT ordinal)", fontsize=12
)
fig5.tight_layout()
save_figure(fig5, filename=str(OUTPUT_DIR / "thallium_deep_dive.png"), dpi=150)
plt.close(fig5)
print("  Saved thallium_deep_dive.png")


# --- Plot 6: Correlation matrix ---
all_cols = CONT_FEATURES + CAT_FEATURES + [TARGET]
corr_matrix = train[all_cols].corr()

fig6, ax6 = plt.subplots(figsize=(12, 10))
im6 = ax6.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax6.set_xticks(range(len(corr_matrix)))
ax6.set_yticks(range(len(corr_matrix)))
ax6.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=8)
ax6.set_yticklabels(corr_matrix.columns, fontsize=8)
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        v = corr_matrix.values[i, j]
        ax6.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=6,
            color="white" if abs(v) > 0.5 else "black",
        )
fig6.colorbar(im6, fraction=0.046, pad=0.04)
ax6.set_title("Feature Correlation Matrix", fontsize=12)
fig6.tight_layout()
save_figure(fig6, filename=str(OUTPUT_DIR / "correlation_matrix.png"), dpi=150)
plt.close(fig6)
print("  Saved correlation_matrix.png")


# --- Plot 7: Box plots for continuous features by target ---
fig7, axes7 = plt.subplots(1, len(CONT_FEATURES), figsize=(4 * len(CONT_FEATURES), 5))
for idx, col in enumerate(CONT_FEATURES):
    ax = axes7[idx]
    data_0 = train.loc[train[TARGET] == 0, col]
    data_1 = train.loc[train[TARGET] == 1, col]
    bp = ax.boxplot(
        [data_0, data_1], labels=["Absence", "Presence"], patch_artist=True, widths=0.6
    )
    bp["boxes"][0].set_facecolor("#4C72B0")
    bp["boxes"][1].set_facecolor("#C44E52")
    for box in bp["boxes"]:
        box.set_alpha(0.6)
    ax.set_title(col, fontsize=10)
    ax.tick_params(labelsize=8)

fig7.suptitle("Continuous Features by Target Class", fontsize=12)
fig7.tight_layout()
save_figure(fig7, filename=str(OUTPUT_DIR / "boxplots_by_target.png"), dpi=150)
plt.close(fig7)
print("  Saved boxplots_by_target.png")


# --- Plot 8: Risk score vs target (composite feature analysis) ---
# Build risk-related composite features to visualize
train_tmp = train.copy()
train_tmp["top4_sum"] = (
    train_tmp["Thallium"]
    + train_tmp["Chest pain type"]
    + train_tmp["Number of vessels fluro"]
    + train_tmp["Exercise angina"]
)
train_tmp["abnormal_count"] = (
    (train_tmp["Thallium"] >= 6).astype(int)
    + (train_tmp["Number of vessels fluro"] >= 1).astype(int)
    + (train_tmp["Chest pain type"] >= 3).astype(int)
    + (train_tmp["Exercise angina"] == 1).astype(int)
    + (train_tmp["Slope of ST"] >= 2).astype(int)
    + (train_tmp["ST depression"] > 1).astype(int)
    + (train_tmp["Sex"] == 1).astype(int)
)

fig8, axes8 = plt.subplots(1, 2, figsize=(14, 5))

for ax, col, title in [
    (axes8[0], "top4_sum", "top4_sum (Thallium+ChestPain+Vessels+Angina)"),
    (axes8[1], "abnormal_count", "abnormal_count (7 risk factors)"),
]:
    vals = sorted(train_tmp[col].unique())
    target_rates = [train_tmp.loc[train_tmp[col] == v, TARGET].mean() for v in vals]
    counts = [(train_tmp[col] == v).sum() for v in vals]

    x = np.arange(len(vals))
    bars = ax.bar(x, counts, color="#4C72B0", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in vals], fontsize=8)
    ax.set_ylabel("Count", color="#4C72B0")

    ax2 = ax.twinx()
    ax2.plot(x, target_rates, "o-", color="#C44E52", markersize=6, linewidth=2)
    ax2.set_ylabel("Target Rate", color="#C44E52")
    ax2.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)

fig8.suptitle("Composite Risk Scores: Distribution & Target Rate", fontsize=12)
fig8.tight_layout()
save_figure(fig8, filename=str(OUTPUT_DIR / "composite_risk_scores.png"), dpi=150)
plt.close(fig8)
print("  Saved composite_risk_scores.png")


# ===========================================================================
# 9. MISCLASSIFICATION ANALYSIS
# ===========================================================================
section("MISCLASSIFICATION ANALYSIS (LightGBM 5-fold OOF)")

from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

LGB_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 5,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "metric": "auc",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

features = [c for c in train.columns if c not in ("id", TARGET)]
X = train[features]
y = train[TARGET]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = np.zeros(len(train))

import lightgbm as lgb

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model = LGBMClassifier(**LGB_PARAMS)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        categorical_feature=CAT_FEATURES,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]

train["oof_prob"] = oof_probs
train["oof_pred"] = (oof_probs >= 0.5).astype(int)
train["correct"] = (train["oof_pred"] == train[TARGET]).astype(int)
train["error_type"] = "correct"
train.loc[(train[TARGET] == 1) & (train["oof_pred"] == 0), "error_type"] = "FN"
train.loc[(train[TARGET] == 0) & (train["oof_pred"] == 1), "error_type"] = "FP"
train["confidence"] = np.abs(oof_probs - 0.5)

n_correct = train["correct"].sum()
n_wrong = len(train) - n_correct
n_fp = (train["error_type"] == "FP").sum()
n_fn = (train["error_type"] == "FN").sum()
print(f"  Correct: {n_correct} ({n_correct/len(train)*100:.2f}%)")
print(f"  Wrong:   {n_wrong} ({n_wrong/len(train)*100:.2f}%)  FP={n_fp}  FN={n_fn}")

# Confident mistakes (prob > 0.8 wrong direction)
confident_wrong = train[(train["correct"] == 0) & (train["confidence"] > 0.3)]
print(
    f"  Confidently wrong (|prob-0.5|>0.3): {len(confident_wrong)} ({len(confident_wrong)/len(train)*100:.3f}%)"
)

# --- Per-feature: misclassification rate in outlier vs non-outlier regions ---
section("MISCLASSIFICATION vs OUTLIERS (continuous features)")

for col in CONT_FEATURES:
    q1, q3 = train[col].quantile(0.25), train[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    is_outlier = (train[col] < lower) | (train[col] > upper)
    n_out = is_outlier.sum()
    if n_out == 0:
        continue

    err_outlier = 1 - train.loc[is_outlier, "correct"].mean()
    err_normal = 1 - train.loc[~is_outlier, "correct"].mean()
    fp_outlier = (train.loc[is_outlier, "error_type"] == "FP").mean()
    fn_outlier = (train.loc[is_outlier, "error_type"] == "FN").mean()
    fp_normal = (train.loc[~is_outlier, "error_type"] == "FP").mean()
    fn_normal = (train.loc[~is_outlier, "error_type"] == "FN").mean()

    print(f"\n  --- {col} ({n_out} outliers, {n_out/len(train)*100:.2f}%) ---")
    print(
        f"  Error rate:  outlier={err_outlier:.4f}  normal={err_normal:.4f}  (ratio={err_outlier/max(err_normal,1e-9):.2f}x)"
    )
    print(f"  FP rate:     outlier={fp_outlier:.4f}  normal={fp_normal:.4f}")
    print(f"  FN rate:     outlier={fn_outlier:.4f}  normal={fn_normal:.4f}")

# --- Per-feature: misclassification rate by categorical value ---
section("MISCLASSIFICATION by CATEGORICAL VALUE")

for col in CAT_FEATURES:
    vals = sorted(train[col].unique())
    print(f"\n  --- {col} ---")
    print(
        f"  {'Value':>6s}  {'Count':>8s}  {'ErrRate':>8s}  {'FP%':>7s}  {'FN%':>7s}  {'ConfWrong':>10s}  Notes"
    )
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*20}")

    overall_err = 1 - train["correct"].mean()
    for v in vals:
        mask = train[col] == v
        subset = train[mask]
        n = len(subset)
        err_rate = 1 - subset["correct"].mean()
        fp_rate = (subset["error_type"] == "FP").mean()
        fn_rate = (subset["error_type"] == "FN").mean()
        conf_wrong = ((subset["correct"] == 0) & (subset["confidence"] > 0.3)).sum()

        notes = []
        if err_rate > overall_err * 1.5:
            notes.append(f"HIGH ERR ({err_rate/overall_err:.1f}x avg)")
        if n < len(train) * 0.01:
            notes.append("RARE")

        print(
            f"  {v:>6}  {n:>8d}  {err_rate:>7.4f}  {fp_rate:>6.4f}  {fn_rate:>6.4f}  "
            f"{conf_wrong:>10d}  {'  '.join(notes)}"
        )

# --- Cross-tab: misclassification by Thallium x Chest pain type ---
section("MISCLASSIFICATION HEATMAP: Thallium x Chest pain type")

ct_err = pd.crosstab(
    train["Thallium"],
    train["Chest pain type"],
    values=1 - train["correct"],
    aggfunc="mean",
)
print("  Error rate by Thallium x Chest pain type:")
print(ct_err.round(4).to_string().replace("\n", "\n  "))

ct_n = pd.crosstab(train["Thallium"], train["Chest pain type"])
ct_wrong = pd.crosstab(
    train["Thallium"],
    train["Chest pain type"],
    values=1 - train["correct"],
    aggfunc="sum",
)
print("\n  Wrong prediction count:")
print(ct_wrong.astype(int).to_string().replace("\n", "\n  "))

# --- Binned continuous features: misclassification rate ---
section("MISCLASSIFICATION by BINNED CONTINUOUS FEATURES")

for col in CONT_FEATURES:
    bins = pd.qcut(train[col], q=10, duplicates="drop")
    grouped = train.groupby(bins, observed=True).agg(
        count=(TARGET, "size"),
        target_rate=(TARGET, "mean"),
        error_rate=("correct", lambda x: 1 - x.mean()),
        conf_wrong=("correct", lambda x: 0),  # placeholder
    )
    # Recompute conf_wrong properly
    conf_wrong_by_bin = (
        train.assign(_bin=bins)
        .groupby("_bin", observed=True)
        .apply(lambda g: ((g["correct"] == 0) & (g["confidence"] > 0.3)).sum())
    )
    grouped["conf_wrong"] = conf_wrong_by_bin.values

    print(f"\n  --- {col} ---")
    print(
        f"  {'Bin':<25s}  {'Count':>7s}  {'Target%':>8s}  {'Err%':>7s}  {'ConfWrong':>10s}"
    )
    for interval, row in grouped.iterrows():
        print(
            f"  {str(interval):<25s}  {int(row['count']):>7d}  "  # type: ignore[index]
            f"{row['target_rate']:>7.4f}  {row['error_rate']:>6.4f}  {int(row['conf_wrong']):>10d}"  # type: ignore[index]
        )


# ===========================================================================
# MISCLASSIFICATION PLOTS
# ===========================================================================
section("MISCLASSIFICATION PLOTS")

# --- Plot 9: Error rate by categorical value (with dual axis) ---
fig9, axes9, _ = create_axes_grid(
    n_columns=4,
    n_rows=2,
    title="Misclassification Rate by Categorical Value",
    figure_size=(20, 10),
)

for idx, col in enumerate(CAT_FEATURES):
    row, c = divmod(idx, 4)
    ax = axes9[row][c]
    vals = sorted(train[col].unique())

    err_rates = [1 - train.loc[train[col] == v, "correct"].mean() for v in vals]
    counts = [(train[col] == v).sum() for v in vals]

    x = np.arange(len(vals))
    ax.bar(x, counts, color="#AAAAAA", alpha=0.5, zorder=1, label="Count")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in vals], fontsize=8)
    ax.set_ylabel("Count", fontsize=8, color="#888888")
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", labelsize=7)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        err_rates,
        "s-",
        color="#D62728",
        markersize=6,
        linewidth=2,
        zorder=3,
        label="Error Rate",
    )
    ax2.axhline(
        1 - train["correct"].mean(),
        color="#D62728",
        linestyle="--",
        alpha=0.3,
        linewidth=1,
    )
    ax2.set_ylabel("Error Rate", fontsize=8, color="#D62728")
    ax2.set_ylim(0, max(err_rates) * 1.3 if max(err_rates) > 0 else 0.1)
    ax2.tick_params(axis="y", labelsize=7)

save_figure(fig9, filename=str(OUTPUT_DIR / "misclass_by_categorical.png"), dpi=150)
plt.close(fig9)
print("  Saved misclass_by_categorical.png")

# --- Plot 10: Misclassification heatmaps (Thallium x others) ---
fig10, axes10 = plt.subplots(1, 3, figsize=(18, 5))

for ax, col2, title in [
    (axes10[0], "Chest pain type", "Thallium x Chest Pain"),
    (axes10[1], "Exercise angina", "Thallium x Angina"),
    (axes10[2], "Number of vessels fluro", "Thallium x Vessels"),
]:
    ct = pd.crosstab(
        train["Thallium"],
        train[col2],
        values=1 - train["correct"],
        aggfunc="mean",
    )
    im = ax.imshow(
        ct.values,
        cmap="Reds",
        aspect="auto",
        norm=Normalize(vmin=0, vmax=ct.values.max() * 1.1),
    )
    ax.set_xticks(range(len(ct.columns)))
    ax.set_xticklabels(ct.columns.astype(str))
    ax.set_yticks(range(len(ct.index)))
    ax.set_yticklabels(ct.index.astype(str))
    for i in range(len(ct.index)):
        for j in range(len(ct.columns)):
            v = ct.values[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.3f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if v > ct.values.max() * 0.6 else "black",
                )
    ax.set_title(f"{title}\n(error rate)", fontsize=10)
    ax.set_xlabel(col2)
    ax.set_ylabel("Thallium")
    plt.colorbar(im, ax=ax, fraction=0.046)

fig10.suptitle("Misclassification Heatmaps", fontsize=12)
fig10.tight_layout()
save_figure(fig10, filename=str(OUTPUT_DIR / "misclass_heatmaps.png"), dpi=150)
plt.close(fig10)
print("  Saved misclass_heatmaps.png")

# --- Plot 11: Continuous feature distributions for correct vs wrong ---
fig11, axes11, _ = create_axes_grid(
    n_columns=3,
    n_rows=2,
    title="Feature Distribution: Correct vs Misclassified",
    figure_size=(18, 10),
)

for idx, col in enumerate(CONT_FEATURES):
    row, c = divmod(idx, 3)
    ax = axes11[row][c]

    correct = train.loc[train["correct"] == 1, col]
    wrong = train.loc[train["correct"] == 0, col]

    bins = np.linspace(train[col].min(), train[col].max(), 61)
    ax.hist(
        correct,
        bins=bins,
        alpha=0.5,
        color="#55A868",
        label=f"Correct (n={len(correct)})",
        density=True,
    )
    ax.hist(
        wrong,
        bins=bins,
        alpha=0.5,
        color="#D62728",
        label=f"Wrong (n={len(wrong)})",
        density=True,
    )

    # Mark outlier boundaries
    q1, q3 = train[col].quantile(0.25), train[col].quantile(0.75)
    iqr = q3 - q1
    ax.axvline(q1 - 1.5 * iqr, color="red", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(q3 + 1.5 * iqr, color="red", linestyle="--", alpha=0.4, linewidth=1)

    ax.set_title(col, fontsize=10)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

if len(CONT_FEATURES) < 6:
    for idx in range(len(CONT_FEATURES), 6):
        row, c = divmod(idx, 3)
        axes11[row][c].set_visible(False)

save_figure(fig11, filename=str(OUTPUT_DIR / "misclass_continuous_dist.png"), dpi=150)
plt.close(fig11)
print("  Saved misclass_continuous_dist.png")

# --- Plot 12: Error rate by binned continuous feature ---
fig12, axes12, _ = create_axes_grid(
    n_columns=3,
    n_rows=2,
    title="Error Rate by Binned Feature Value",
    figure_size=(18, 10),
)

for idx, col in enumerate(CONT_FEATURES):
    row, c = divmod(idx, 3)
    ax = axes12[row][c]

    bins_cut = pd.qcut(train[col], q=10, duplicates="drop")
    grouped = train.groupby(bins_cut, observed=True).agg(
        count=(TARGET, "size"),
        error_rate=("correct", lambda x: 1 - x.mean()),
    )
    x = np.arange(len(grouped))
    bar_labels = [f"{iv.left:.0f}-{iv.right:.0f}" for iv in grouped.index]

    ax.bar(x, grouped["count"], color="#AAAAAA", alpha=0.5)
    ax.set_ylabel("Count", fontsize=8, color="#888888")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=6, rotation=45, ha="right")
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", labelsize=7)

    ax2 = ax.twinx()
    ax2.plot(x, grouped["error_rate"], "s-", color="#D62728", markersize=5, linewidth=2)
    ax2.axhline(1 - train["correct"].mean(), color="#D62728", linestyle="--", alpha=0.3)
    ax2.set_ylabel("Error Rate", fontsize=8, color="#D62728")
    ax2.tick_params(axis="y", labelsize=7)

if len(CONT_FEATURES) < 6:
    for idx in range(len(CONT_FEATURES), 6):
        row, c = divmod(idx, 3)
        axes12[row][c].set_visible(False)

save_figure(fig12, filename=str(OUTPUT_DIR / "misclass_binned_continuous.png"), dpi=150)
plt.close(fig12)
print("  Saved misclass_binned_continuous.png")

# Clean up temp columns
train.drop(
    columns=["oof_prob", "oof_pred", "correct", "error_type", "confidence"],
    inplace=True,
)

print("\nDone.")
