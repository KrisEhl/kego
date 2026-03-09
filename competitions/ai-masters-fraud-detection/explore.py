"""Quick EDA for AI Masters Fraud Detection (IEEE-CIS).

Covers: data overview, missing data, target distribution, feature groups,
train/test shift, correlations, and saves diagnostic plots.

Usage:
    uv run python competitions/ai-masters-fraud-detection/explore.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from kego.plotting.figures import create_axes_grid, save_figure

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DATA_DIR = (
    Path(os.environ.get("KEGO_PATH_DATA", project_root / "data"))
    / "ai"
    / "ai-masters-fraud-detection"
)
OUTPUT_DIR = project_root / "competitions" / "ai-masters-fraud-detection" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "isFraud"

# Feature groups
CAT_FEATURES = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "DeviceType",
    "DeviceInfo",
    "id_12",
    "id_15",
    "id_16",
    "id_23",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
]

NUMERIC_KEY_FEATURES = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
]

C_FEATURES = [f"C{i}" for i in range(1, 15)]
D_FEATURES = [f"D{i}" for i in range(1, 16)]
V_FEATURES = [f"V{i}" for i in range(1, 340)]
ID_NUMERIC = [f"id_{i:02d}" for i in range(1, 12)]


def section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
section("LOADING DATA")

train_txn = pd.read_csv(DATA_DIR / "train_transaction.csv")
train_id = pd.read_csv(DATA_DIR / "train_identity.csv")
test_txn = pd.read_csv(DATA_DIR / "test_transaction.csv")
test_id = pd.read_csv(DATA_DIR / "test_identity.csv")

train = train_txn.merge(train_id, on="TransactionID", how="left")
test = test_txn.merge(test_id, on="TransactionID", how="left")

del train_txn, train_id, test_txn, test_id

print(f"  Train: {train.shape}  ({train.memory_usage(deep=True).sum() / 1e6:.0f} MB)")
print(f"  Test:  {test.shape}  ({test.memory_usage(deep=True).sum() / 1e6:.0f} MB)")

# ===========================================================================
# 1. TARGET DISTRIBUTION
# ===========================================================================
section("TARGET DISTRIBUTION")

fraud_rate = train[TARGET].mean()
n_fraud = train[TARGET].sum()
n_legit = len(train) - n_fraud
print(f"  Total:     {len(train):,}")
print(f"  Fraud:     {n_fraud:,} ({fraud_rate * 100:.2f}%)")
print(f"  Legit:     {n_legit:,} ({(1 - fraud_rate) * 100:.2f}%)")
print(f"  Imbalance: 1:{n_legit / n_fraud:.0f}")

# ===========================================================================
# 2. MISSING DATA ANALYSIS
# ===========================================================================
section("MISSING DATA ANALYSIS")

train_missing = train.isnull().mean() * 100
test_missing = test.isnull().mean() * 100

# Group by missing rate
thresholds = [0, 1, 10, 25, 50, 75, 90, 100.01]
labels = ["<1%", "1-10%", "10-25%", "25-50%", "50-75%", "75-90%", ">90%"]
bins = pd.cut(train_missing, thresholds, labels=labels)
counts = bins.value_counts().sort_index()
print("\n  Train missing data distribution:")
for label, count in counts.items():
    print(f"    {label:>8s}: {count:>3d} features")

# Top missing features
print("\n  Top 30 missing features (train):")
top_missing = train_missing.sort_values(ascending=False).head(30)
for col, pct in top_missing.items():
    te_pct = test_missing.get(col, 0)
    shift = abs(pct - te_pct)
    flag = " ** SHIFT" if shift > 5 else ""
    print(f"    {col:<20s}  train={pct:>6.1f}%  test={te_pct:>6.1f}%{flag}")

# Features with different missing patterns train vs test
print("\n  Features with >5% train/test missing rate difference:")
missing_diff = (train_missing - test_missing).abs().sort_values(ascending=False)
for col, diff in missing_diff.head(15).items():
    if diff > 5:
        print(
            f"    {col:<20s}  diff={diff:>5.1f}%  "
            f"(train={train_missing[col]:.1f}%, test={test_missing[col]:.1f}%)"
        )

# Missing data correlations — do features go missing together?
print("\n  V-feature missing pattern groups (NaN correlation clusters):")
v_present = [c for c in V_FEATURES if c in train.columns]
v_null = train[v_present].isnull()
# Group V features by identical missing patterns
v_patterns: dict[tuple, list[str]] = {}
for col in v_present:
    pattern = tuple(v_null[col].values[:100])  # sample first 100 rows
    v_patterns.setdefault(pattern, []).append(col)
print(
    f"    {len(v_patterns)} distinct missing patterns across {len(v_present)} V-features"
)
for i, (_, cols) in enumerate(
    sorted(v_patterns.items(), key=lambda x: -len(x[1]))[:10]
):
    miss_rate = train[cols[0]].isnull().mean() * 100
    if len(cols) <= 6:
        print(
            f"    Group {i + 1}: {len(cols)} features ({miss_rate:.1f}% missing) — {cols}"
        )
    else:
        print(
            f"    Group {i + 1}: {len(cols)} features ({miss_rate:.1f}% missing) — "
            f"{cols[:3]} ... {cols[-3:]}"
        )

# ===========================================================================
# 3. IDENTITY TABLE COVERAGE
# ===========================================================================
section("IDENTITY TABLE COVERAGE")

has_identity = train[ID_NUMERIC[0]].notna()
n_with_id = has_identity.sum()
print(f"  Rows with identity info: {n_with_id:,} ({n_with_id / len(train) * 100:.1f}%)")
print(f"  Rows without:           {len(train) - n_with_id:,}")
fraud_with_id = train.loc[has_identity, TARGET].mean()
fraud_without_id = train.loc[~has_identity, TARGET].mean()
print(f"  Fraud rate WITH identity:    {fraud_with_id:.4f}")
print(f"  Fraud rate WITHOUT identity: {fraud_without_id:.4f}")

# ===========================================================================
# 4. CATEGORICAL FEATURE ANALYSIS
# ===========================================================================
section("CATEGORICAL FEATURE ANALYSIS")

# Only analyze cats that exist in train
cat_available = [c for c in CAT_FEATURES if c in train.columns]

for col in cat_available:
    n_unique = train[col].nunique()
    n_missing = train[col].isnull().sum()
    miss_pct = n_missing / len(train) * 100

    print(f"\n  --- {col} (unique={n_unique}, missing={miss_pct:.1f}%) ---")

    if n_unique > 20:
        # High cardinality — show top 10 + fraud rate
        print(f"    High cardinality ({n_unique} values). Top 10:")
        vc = train[col].value_counts().head(10)
        for val, cnt in vc.items():
            pct = cnt / len(train) * 100
            fraud = train.loc[train[col] == val, TARGET].mean() * 100
            print(
                f"    {str(val):>25s}  n={cnt:>7,}  ({pct:>5.2f}%)  fraud={fraud:.2f}%"
            )
        continue

    vals = train[col].dropna().unique()
    vals = sorted(vals, key=str)
    print(
        f"    {'Value':>20s}  {'Count':>8s}  {'%':>7s}  {'Fraud%':>7s}  {'Test%':>7s}  Notes"
    )
    print(f"    {'-' * 20}  {'-' * 8}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 20}")

    for v in vals:
        t_cnt = (train[col] == v).sum()
        t_pct = t_cnt / len(train) * 100
        t_fraud = train.loc[train[col] == v, TARGET].mean() * 100
        te_pct = (test[col] == v).sum() / len(test) * 100 if col in test.columns else 0

        notes = []
        if abs(t_pct - te_pct) > 2.0:
            notes.append(f"shift {t_pct - te_pct:+.1f}%")
        if t_pct < 1.0:
            notes.append("RARE")
        if t_fraud > fraud_rate * 100 * 2:
            notes.append(f"HIGH FRAUD ({t_fraud / (fraud_rate * 100):.1f}x)")

        print(
            f"    {str(v):>20s}  {t_cnt:>8,}  {t_pct:>6.2f}%  {t_fraud:>6.2f}%  "
            f"{te_pct:>6.2f}%  {'  '.join(notes)}"
        )


# ===========================================================================
# 5. NUMERIC KEY FEATURES
# ===========================================================================
section("NUMERIC KEY FEATURES — Distribution & Fraud Rate")

for col in NUMERIC_KEY_FEATURES:
    if col not in train.columns:
        continue
    miss_pct = train[col].isnull().mean() * 100
    print(f"\n  --- {col} (missing={miss_pct:.1f}%) ---")
    valid = train[col].dropna()
    print(
        f"    Range: [{valid.min():.2f}, {valid.max():.2f}]  "
        f"Mean={valid.mean():.2f}  Median={valid.median():.2f}  Std={valid.std():.2f}"
    )
    # Percentiles
    pcts = valid.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(
        f"    Pctls: 1%={pcts[0.01]:.1f}  5%={pcts[0.05]:.1f}  25%={pcts[0.25]:.1f}  "
        f"50%={pcts[0.5]:.1f}  75%={pcts[0.75]:.1f}  95%={pcts[0.95]:.1f}  99%={pcts[0.99]:.1f}"
    )
    # Fraud rate in top/bottom deciles
    if len(valid) > 100:
        low_q = valid.quantile(0.1)
        high_q = valid.quantile(0.9)
        fraud_low = train.loc[train[col] <= low_q, TARGET].mean()
        fraud_mid = train.loc[
            (train[col] > low_q) & (train[col] < high_q), TARGET
        ].mean()
        fraud_high = train.loc[train[col] >= high_q, TARGET].mean()
        print(
            f"    Fraud rate: bottom10%={fraud_low:.4f}  mid80%={fraud_mid:.4f}  top10%={fraud_high:.4f}"
        )


# ===========================================================================
# 6. C-FEATURES (counting features)
# ===========================================================================
section("C-FEATURES ANALYSIS (counting/aggregation features)")

c_available = [c for c in C_FEATURES if c in train.columns]
print(
    f"\n  {'Feature':<8s}  {'Miss%':>6s}  {'Unique':>7s}  "
    f"{'Mean':>8s}  {'Median':>8s}  {'Max':>10s}  {'CorrTarget':>11s}"
)
print(
    f"  {'-' * 8}  {'-' * 6}  {'-' * 7}  {'-' * 8}  {'-' * 8}  {'-' * 10}  {'-' * 11}"
)
for col in c_available:
    miss = train[col].isnull().mean() * 100
    nunique = train[col].nunique()
    mean = train[col].mean()
    median = train[col].median()
    max_val = train[col].max()
    corr = train[[col, TARGET]].corr().iloc[0, 1]
    print(
        f"  {col:<8s}  {miss:>5.1f}%  {nunique:>7d}  "
        f"{mean:>8.1f}  {median:>8.1f}  {max_val:>10.0f}  {corr:>+10.4f}"
    )


# ===========================================================================
# 7. D-FEATURES (timedelta features)
# ===========================================================================
section("D-FEATURES ANALYSIS (timedelta features)")

d_available = [c for c in D_FEATURES if c in train.columns]
print(
    f"\n  {'Feature':<8s}  {'Miss%':>6s}  {'Unique':>7s}  "
    f"{'Mean':>10s}  {'Median':>10s}  {'Max':>10s}  {'CorrTarget':>11s}"
)
print(
    f"  {'-' * 8}  {'-' * 6}  {'-' * 7}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 11}"
)
for col in d_available:
    miss = train[col].isnull().mean() * 100
    nunique = train[col].nunique()
    mean = train[col].mean()
    median = train[col].median()
    max_val = train[col].max()
    corr = train[[col, TARGET]].corr().iloc[0, 1]
    print(
        f"  {col:<8s}  {miss:>5.1f}%  {nunique:>7d}  "
        f"{mean:>10.1f}  {median:>10.1f}  {max_val:>10.0f}  {corr:>+10.4f}"
    )


# ===========================================================================
# 8. V-FEATURES SUMMARY (339 anonymous features)
# ===========================================================================
section("V-FEATURES SUMMARY")

v_available = [c for c in V_FEATURES if c in train.columns]
v_corr = train[v_available + [TARGET]].corr()[TARGET].drop(TARGET).abs()
v_missing = train[v_available].isnull().mean() * 100

print(f"\n  Total V-features: {len(v_available)}")
print(f"  With <1% missing: {(v_missing < 1).sum()}")
print(f"  With >50% missing: {(v_missing > 50).sum()}")
print(f"  With >75% missing: {(v_missing > 75).sum()}")

# Top 20 by target correlation
print("\n  Top 20 V-features by |correlation with target|:")
top_v = v_corr.sort_values(ascending=False).head(20)
for col, corr_val in top_v.items():
    miss = v_missing[col]
    actual_corr = train[[col, TARGET]].corr().iloc[0, 1]
    print(
        f"    {col:<8s}  |r|={corr_val:.4f}  (r={actual_corr:+.4f})  missing={miss:.1f}%"
    )


# ===========================================================================
# 9. TRAIN vs TEST DISTRIBUTION SHIFT
# ===========================================================================
section("TRAIN vs TEST DISTRIBUTION SHIFT")

# TransactionDT shift (temporal split)
print("\n  TransactionDT:")
print(f"    Train: [{train['TransactionDT'].min()}, {train['TransactionDT'].max()}]")
print(f"    Test:  [{test['TransactionDT'].min()}, {test['TransactionDT'].max()}]")
overlap = max(0, min(train["TransactionDT"].max(), test["TransactionDT"].max())) - max(
    train["TransactionDT"].min(), test["TransactionDT"].min()
)
print(
    f"    Overlap: {overlap} (temporal split = {'YES' if overlap <= 0 else 'partial'})"
)

# Numeric feature shifts
print("\n  Numeric feature KS-test (train vs test):")
shift_cols = NUMERIC_KEY_FEATURES + C_FEATURES + D_FEATURES
for col in shift_cols:
    if col not in train.columns or col not in test.columns:
        continue
    tr_vals = train[col].dropna()
    te_vals = test[col].dropna()
    if len(tr_vals) < 10 or len(te_vals) < 10:
        continue
    ks_stat, ks_p = stats.ks_2samp(tr_vals, te_vals)
    if ks_p < 0.01:
        mean_diff = tr_vals.mean() - te_vals.mean()
        print(
            f"    {col:<20s}  KS={ks_stat:.4f}  p={ks_p:.2g}  "
            f"mean_diff={mean_diff:+.3f}  ** SIGNIFICANT"
        )

# Categorical shifts
print("\n  Categorical feature shifts (train vs test):")
for col in cat_available:
    if col not in test.columns:
        continue
    tr_dist = train[col].value_counts(normalize=True, dropna=False).sort_index()
    te_dist = test[col].value_counts(normalize=True, dropna=False).sort_index()
    all_vals = sorted(set(tr_dist.index) | set(te_dist.index), key=str)
    max_diff = max(abs(tr_dist.get(v, 0) - te_dist.get(v, 0)) * 100 for v in all_vals)
    if max_diff > 2:
        print(f"    {col:<20s}  max_category_diff={max_diff:.2f}%  ** SHIFT")


# ===========================================================================
# 10. CORRELATION WITH TARGET
# ===========================================================================
section("TOP 40 FEATURES BY |CORRELATION WITH TARGET|")

numeric_cols = [
    c
    for c in train.select_dtypes(include=[np.number]).columns
    if c not in ("TransactionID", TARGET)
]
corr_all = train[numeric_cols + [TARGET]].corr()[TARGET].drop(TARGET)
top_corr = corr_all.abs().sort_values(ascending=False).head(40)
for col in top_corr.index:
    print(f"  {col:<20s}  r={corr_all[col]:+.4f}")


# ===========================================================================
# 11. DUPLICATE ANALYSIS
# ===========================================================================
section("DUPLICATE ANALYSIS")

# Check for duplicate TransactionIDs
dup_ids = train["TransactionID"].duplicated().sum()
print(f"  Duplicate TransactionIDs: {dup_ids}")

# Check near-duplicate rows (same card + amount + time)
dup_cols = ["card1", "TransactionAmt", "addr1"]
available_dup_cols = [c for c in dup_cols if c in train.columns]
if available_dup_cols:
    dup_mask = train.duplicated(subset=available_dup_cols, keep=False)
    n_dup = dup_mask.sum()
    print(
        f"  Same (card1, TransactionAmt, addr1): {n_dup:,} rows "
        f"({n_dup / len(train) * 100:.1f}%)"
    )
    if n_dup > 0:
        fraud_dup = train.loc[dup_mask, TARGET].mean()
        fraud_nodup = train.loc[~dup_mask, TARGET].mean()
        print(
            f"    Fraud rate: duplicates={fraud_dup:.4f}  "
            f"non-duplicates={fraud_nodup:.4f}"
        )


# ===========================================================================
# PLOTS
# ===========================================================================
section(f"SAVING PLOTS to {OUTPUT_DIR}")


# --- Plot 1: Target distribution ---
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))
axes1[0].bar(["Legit", "Fraud"], [n_legit, n_fraud], color=["#4C72B0", "#C44E52"])
axes1[0].set_title("Target Distribution (count)")
axes1[0].set_ylabel("Count")
for i, (label, val) in enumerate(zip(["Legit", "Fraud"], [n_legit, n_fraud])):
    axes1[0].text(i, val, f"{val:,}", ha="center", va="bottom", fontsize=9)

axes1[1].bar(
    ["Legit", "Fraud"],
    [100 - fraud_rate * 100, fraud_rate * 100],
    color=["#4C72B0", "#C44E52"],
)
axes1[1].set_title("Target Distribution (%)")
axes1[1].set_ylabel("%")
fig1.suptitle("Target: isFraud", fontsize=12)
fig1.tight_layout()
save_figure(fig1, filename=str(OUTPUT_DIR / "01_target_distribution.png"), dpi=150)
plt.close(fig1)
print("  Saved 01_target_distribution.png")


# --- Plot 2: Missing data heatmap (grouped) ---
feature_groups = {
    "Card": [f"card{i}" for i in range(1, 7)],
    "Addr": ["addr1", "addr2"],
    "Email": ["P_emaildomain", "R_emaildomain"],
    "C": C_FEATURES,
    "D": D_FEATURES,
    "M": [f"M{i}" for i in range(1, 10)],
    "ID": [f"id_{i:02d}" for i in range(1, 39)] + ["DeviceType", "DeviceInfo"],
    "V1-V50": [f"V{i}" for i in range(1, 51)],
    "V51-V100": [f"V{i}" for i in range(51, 101)],
    "V101-V200": [f"V{i}" for i in range(101, 201)],
    "V201-V339": [f"V{i}" for i in range(201, 340)],
}

fig2, ax2 = plt.subplots(figsize=(14, 6))
group_labels = []
group_miss_train = []
group_miss_test = []
for group_name, cols in feature_groups.items():
    available = [c for c in cols if c in train.columns]
    if available:
        group_labels.append(f"{group_name}\n({len(available)})")
        group_miss_train.append(train[available].isnull().mean().mean() * 100)
        te_available = [c for c in available if c in test.columns]
        group_miss_test.append(
            test[te_available].isnull().mean().mean() * 100 if te_available else 0
        )

x = np.arange(len(group_labels))
w = 0.35
ax2.bar(x - w / 2, group_miss_train, w, label="Train", color="#4C72B0", alpha=0.8)
ax2.bar(x + w / 2, group_miss_test, w, label="Test", color="#55A868", alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(group_labels, fontsize=8)
ax2.set_ylabel("Average Missing %")
ax2.set_title("Missing Data by Feature Group")
ax2.legend()
ax2.grid(axis="y", alpha=0.3)
fig2.tight_layout()
save_figure(fig2, filename=str(OUTPUT_DIR / "02_missing_by_group.png"), dpi=150)
plt.close(fig2)
print("  Saved 02_missing_by_group.png")


# --- Plot 3: Key categorical features — distribution & fraud rate ---
plot_cats = ["ProductCD", "card4", "card6", "M4", "DeviceType", "id_15"]
plot_cats = [c for c in plot_cats if c in train.columns and train[c].nunique() <= 15]
n_plot_cats = len(plot_cats)
n_cols_p3 = min(3, n_plot_cats)
n_rows_p3 = (n_plot_cats + n_cols_p3 - 1) // n_cols_p3

fig3, axes3, _ = create_axes_grid(
    n_columns=n_cols_p3,
    n_rows=n_rows_p3,
    title="Key Categorical Features: Distribution & Fraud Rate",
    figure_size=(7 * n_cols_p3, 5 * n_rows_p3),
)

for idx, col in enumerate(plot_cats):
    row, c = divmod(idx, n_cols_p3)
    ax = axes3[row][c] if n_rows_p3 > 1 else axes3[c]
    vals = sorted(train[col].dropna().unique(), key=str)

    counts = [(train[col] == v).sum() for v in vals]
    fraud_rates = [train.loc[train[col] == v, TARGET].mean() for v in vals]

    x_pos = np.arange(len(vals))
    ax.bar(x_pos, counts, color="#4C72B0", alpha=0.7, zorder=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(v) for v in vals], fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Count", fontsize=8, color="#4C72B0")
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", labelsize=7)

    ax2 = ax.twinx()
    ax2.plot(
        x_pos, fraud_rates, "o-", color="#C44E52", markersize=5, linewidth=2, zorder=3
    )
    ax2.set_ylabel("Fraud Rate", fontsize=8, color="#C44E52")
    ax2.set_ylim(0, max(fraud_rates) * 1.5 if max(fraud_rates) > 0 else 0.1)
    ax2.axhline(fraud_rate, color="#C44E52", linestyle="--", alpha=0.3, linewidth=1)
    ax2.tick_params(axis="y", labelsize=7)

save_figure(fig3, filename=str(OUTPUT_DIR / "03_categorical_fraud_rates.png"), dpi=150)
plt.close(fig3)
print("  Saved 03_categorical_fraud_rates.png")


# --- Plot 4: TransactionAmt distribution (log scale) ---
fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))

# 4a: Full distribution
axes4[0].hist(
    np.log1p(train["TransactionAmt"]),
    bins=100,
    color="#4C72B0",
    alpha=0.7,
    density=True,
)
axes4[0].set_title("log1p(TransactionAmt) — All")
axes4[0].set_xlabel("log1p(Amount)")

# 4b: By fraud label
for label_val, color, name in [(0, "#4C72B0", "Legit"), (1, "#C44E52", "Fraud")]:
    subset = train.loc[train[TARGET] == label_val, "TransactionAmt"]
    axes4[1].hist(
        np.log1p(subset), bins=100, alpha=0.5, color=color, label=name, density=True
    )
axes4[1].set_title("log1p(TransactionAmt) — By Target")
axes4[1].legend()

# 4c: Train vs test
axes4[2].hist(
    np.log1p(train["TransactionAmt"]),
    bins=100,
    alpha=0.5,
    color="#4C72B0",
    label="Train",
    density=True,
)
axes4[2].hist(
    np.log1p(test["TransactionAmt"]),
    bins=100,
    alpha=0.5,
    color="#55A868",
    label="Test",
    density=True,
)
axes4[2].set_title("log1p(TransactionAmt) — Train vs Test")
axes4[2].legend()

fig4.suptitle("Transaction Amount Distribution", fontsize=12)
fig4.tight_layout()
save_figure(fig4, filename=str(OUTPUT_DIR / "04_transaction_amt.png"), dpi=150)
plt.close(fig4)
print("  Saved 04_transaction_amt.png")


# --- Plot 5: TransactionDT (temporal patterns) ---
fig5, axes5 = plt.subplots(2, 2, figsize=(16, 10))

# Time in days
train["_day"] = train["TransactionDT"] // 86400
test["_day"] = test["TransactionDT"] // 86400
train["_hour"] = (train["TransactionDT"] % 86400) // 3600

# 5a: Daily transaction count
day_counts = train.groupby("_day").size()
axes5[0, 0].plot(day_counts.index, day_counts.values, color="#4C72B0", linewidth=0.8)
axes5[0, 0].set_title("Daily Transaction Count (train)")
axes5[0, 0].set_xlabel("Day")

# 5b: Daily fraud rate
day_fraud = train.groupby("_day")[TARGET].mean()
axes5[0, 1].plot(day_fraud.index, day_fraud.values, color="#C44E52", linewidth=0.8)
axes5[0, 1].axhline(fraud_rate, color="gray", linestyle="--", alpha=0.5)
axes5[0, 1].set_title("Daily Fraud Rate (train)")
axes5[0, 1].set_xlabel("Day")

# 5c: Hourly fraud rate
hour_fraud = train.groupby("_hour")[TARGET].mean()
axes5[1, 0].bar(hour_fraud.index, hour_fraud.values, color="#C44E52", alpha=0.7)
axes5[1, 0].axhline(fraud_rate, color="gray", linestyle="--", alpha=0.5)
axes5[1, 0].set_title("Fraud Rate by Hour of Day")
axes5[1, 0].set_xlabel("Hour")

# 5d: Train vs test day ranges
train_day_counts = train.groupby("_day").size()
test_day_counts = test.groupby("_day").size()
axes5[1, 1].fill_between(
    train_day_counts.index,
    train_day_counts.values,
    alpha=0.5,
    color="#4C72B0",
    label="Train",
)
axes5[1, 1].fill_between(
    test_day_counts.index,
    test_day_counts.values,
    alpha=0.5,
    color="#55A868",
    label="Test",
)
axes5[1, 1].set_title("Train vs Test Temporal Coverage")
axes5[1, 1].legend()

# Clean up temp columns
train.drop(columns=["_day", "_hour"], inplace=True)
test.drop(columns=["_day"], inplace=True)

fig5.suptitle("Temporal Patterns", fontsize=12)
fig5.tight_layout()
save_figure(fig5, filename=str(OUTPUT_DIR / "05_temporal_patterns.png"), dpi=150)
plt.close(fig5)
print("  Saved 05_temporal_patterns.png")


# --- Plot 6: Top V-features by correlation — distribution by target ---
top_v_names = v_corr.sort_values(ascending=False).head(12).index.tolist()

fig6, axes6, _ = create_axes_grid(
    n_columns=4,
    n_rows=3,
    title="Top 12 V-Features by |Correlation| — Distribution by Target",
    figure_size=(20, 12),
)

for idx, col in enumerate(top_v_names):
    row, c = divmod(idx, 4)
    ax = axes6[row][c]
    for label_val, color, name in [(0, "#4C72B0", "Legit"), (1, "#C44E52", "Fraud")]:
        subset = train.loc[train[TARGET] == label_val, col].dropna()
        ax.hist(subset, bins=50, alpha=0.5, color=color, label=name, density=True)
    corr_val = corr_all.get(col, 0)
    ax.set_title(f"{col} (r={corr_val:+.3f})", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

save_figure(fig6, filename=str(OUTPUT_DIR / "06_top_v_features.png"), dpi=150)
plt.close(fig6)
print("  Saved 06_top_v_features.png")


# --- Plot 7: Correlation heatmap of top features ---
top_features = corr_all.abs().sort_values(ascending=False).head(20).index.tolist()
corr_matrix = train[top_features + [TARGET]].corr()

fig7, ax7 = plt.subplots(figsize=(14, 12))
im7 = ax7.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax7.set_xticks(range(len(corr_matrix)))
ax7.set_yticks(range(len(corr_matrix)))
ax7.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=7)
ax7.set_yticklabels(corr_matrix.columns, fontsize=7)
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        v = corr_matrix.values[i, j]
        ax7.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=5,
            color="white" if abs(v) > 0.5 else "black",
        )
fig7.colorbar(im7, fraction=0.046, pad=0.04)
ax7.set_title("Correlation Matrix — Top 20 Features + Target", fontsize=12)
fig7.tight_layout()
save_figure(fig7, filename=str(OUTPUT_DIR / "07_correlation_matrix.png"), dpi=150)
plt.close(fig7)
print("  Saved 07_correlation_matrix.png")


# --- Plot 8: D-features missing pattern & fraud rate ---
fig8, axes8 = plt.subplots(1, 2, figsize=(14, 5))

# 8a: Missing rate per D-feature
d_miss = [train[c].isnull().mean() * 100 for c in d_available]
axes8[0].bar(range(len(d_available)), d_miss, color="#4C72B0", alpha=0.7)
axes8[0].set_xticks(range(len(d_available)))
axes8[0].set_xticklabels(d_available, rotation=45, ha="right", fontsize=8)
axes8[0].set_ylabel("Missing %")
axes8[0].set_title("D-Features: Missing Rate")

# 8b: Fraud rate when D-feature is NaN vs present
fraud_when_nan = []
fraud_when_present = []
for col in d_available:
    mask_nan = train[col].isnull()
    if mask_nan.sum() > 0 and (~mask_nan).sum() > 0:
        fraud_when_nan.append(train.loc[mask_nan, TARGET].mean())
        fraud_when_present.append(train.loc[~mask_nan, TARGET].mean())
    else:
        fraud_when_nan.append(0)
        fraud_when_present.append(0)

x = np.arange(len(d_available))
w = 0.35
axes8[1].bar(
    x - w / 2, fraud_when_present, w, label="Present", color="#55A868", alpha=0.7
)
axes8[1].bar(x + w / 2, fraud_when_nan, w, label="NaN", color="#C44E52", alpha=0.7)
axes8[1].axhline(fraud_rate, color="gray", linestyle="--", alpha=0.5)
axes8[1].set_xticks(x)
axes8[1].set_xticklabels(d_available, rotation=45, ha="right", fontsize=8)
axes8[1].set_ylabel("Fraud Rate")
axes8[1].set_title("D-Features: Fraud Rate (NaN vs Present)")
axes8[1].legend()

fig8.suptitle("D-Features (Timedelta) Analysis", fontsize=12)
fig8.tight_layout()
save_figure(fig8, filename=str(OUTPUT_DIR / "08_d_features.png"), dpi=150)
plt.close(fig8)
print("  Saved 08_d_features.png")


print("\nDone.")
