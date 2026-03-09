## Competition

IEEE-CIS Fraud Detection — predict whether an online transaction is fraudulent.

- Binary classification, metric: AUC-ROC
- 417,559 train rows / 172,981 test rows / 434 features (after join)
- Two source tables joined on `TransactionID`: transaction (394 cols) + identity (41 cols)

## Data Overview

| Property | Value |
|---|---|
| Fraud rate | 3.53% (1:27 imbalance) |
| Memory (loaded) | ~1.9 GB train, ~0.8 GB test |
| Train/test split | Temporal (no time overlap) |
| Identity coverage | 26.5% of transactions have identity data |

### Feature Groups

| Group | Columns | Missing | Notes |
|---|---|---|---|
| Transaction | TransactionDT, TransactionAmt | 0% | Time in seconds from reference; amount heavily right-skewed |
| ProductCD | 1 col | 0% | 5 categories (W dominates at 72%) |
| Card | card1-6 | 0-2% | Card metadata; card1 is high-cardinality numeric |
| Address | addr1, addr2 | 11.5% | addr2 nearly constant (87 for 99%+) |
| Distance | dist1, dist2 | 62%, 93% | Highly skewed; dist2 nearly always missing |
| Email | P_emaildomain, R_emaildomain | 16%, 75% | 59-60 unique domains each |
| C1-C14 | 14 cols | 0% | Counting/aggregation features; low correlation with target |
| D1-D15 | 15 cols | 0-94% | Timedelta features; negative correlation with fraud |
| M1-M9 | 9 cols | 31-67% | Match/verification flags (T/F) |
| V1-V339 | 339 cols | 0-84% | Anonymous Vesta-engineered features; strongest signal |
| Identity | id_01-id_38 | 74-99% | Device/browser/OS fingerprinting |
| Device | DeviceType, DeviceInfo | 74%, 78% | Desktop vs mobile; 1551 unique DeviceInfo values |

## Key EDA Findings

### Target & Imbalance

- 14,721 fraud (3.53%) vs 402,838 legit (96.47%)
- Fraud rate WITH identity data: 7.35% vs WITHOUT: 2.15% -- identity presence itself is a signal

### Missing Data is Structured, Not Random

- Only 48 features have <1% missing; 289 have >10% missing
- **V-features form 9 distinct NaN-pattern groups** where entire blocks go missing together (e.g., V95-V321 are always present; V138-V339 are 84% missing together)
- M-features show massive train/test missing rate shift: M7/M8/M9 are 67% missing in train but only 39% in test (27% shift)
- NaN patterns likely indicate what data was available for each transaction type -- missingness itself is a feature

### Strongest Predictors

Top features by |correlation with target|:

| Feature | r | Missing |
|---|---|---|
| V257 | +0.370 | 76% |
| V244 | +0.360 | 76% |
| V242 | +0.355 | 76% |
| V246 | +0.351 | 76% |
| V201 | +0.327 | 74% |
| V200 | +0.314 | 74% |
| V233 | +0.311 | 76% |
| V188 | +0.308 | 74% |

The most predictive features (V200s, V240s) are all in the 74-76% missing group. When present, they carry strong fraud signal.

### Categorical Feature Highlights

- **ProductCD**: Type C has 11.2% fraud (3.2x baseline), while dominant type W has only 2.1%
- **card6**: Credit cards have 6.6% fraud vs debit at 2.4%
- **M4**: Value M2 has 11.0% fraud (3.1x baseline)
- **DeviceType**: Mobile has 9.9% fraud (2.8x baseline) vs desktop 6.0%
- **id_23** (proxy detection): `IP_PROXY:ANONYMOUS` has 11.1% fraud
- **id_35**: False (failed match) has 11.7% fraud (3.3x baseline)
- **Email domains**: outlook.com has 8.9% fraud (purchaser) and 14.8% (recipient) -- highest among major providers

### Temporal Patterns

- Train covers days 1-122, test covers days 122-183 -- clean temporal split, no overlap
- Fraud rate varies by hour of day (higher at night/early morning)
- D-features (timedelta) are negatively correlated with fraud: newer/shorter intervals = more fraud

### Train/Test Distribution Shift

Significant shifts detected across almost all feature groups:
- **M-features**: Massive shift (M1/M2/M3/M7/M8/M9 shift 23-27%)
- **ProductCD**: W shifts from 72% to 80% in test
- **All C-features**: Statistically significant KS-test shifts
- **All D-features**: Significant shifts (expected given temporal split)
- **Identity features**: 5-9% shifts across most id_ columns

This confirms time-based CV is mandatory for reliable local validation.

### Duplicates

- 0 duplicate TransactionIDs
- 63.3% of rows share the same (card1, TransactionAmt, addr1) tuple -- lots of repeat cardholders
- Fraud rate slightly higher in duplicate groups (3.8% vs 3.1%)

## Modeling Considerations

1. **Time-based CV** -- temporal split means random CV will leak future information
2. **NaN as feature** -- create `is_null` binary features before imputation, especially for V and M groups
3. **V-feature reduction** -- 339 V-features can be reduced to ~80-120 by keeping one representative per NaN-group + correlation cluster
4. **UID construction** -- `card1 + addr1` approximates unique cardholders for per-user aggregations
5. **Memory optimization** -- downcast dtypes before feature engineering (1.9 GB baseline)
6. **Class imbalance** -- use stratified sampling and AUC-focused objectives

## Scripts

```bash
# Run EDA (prints analysis + saves 8 plots to plots/)
uv run python competitions/ai-masters-fraud-detection/explore.py
```

### Generated Plots

| # | Plot | Shows |
|---|---|---|
| 01 | target_distribution | Fraud vs legit count and percentage |
| 02 | missing_by_group | Average missing % by feature group (train vs test) |
| 03 | categorical_fraud_rates | Key categoricals with fraud rate overlay |
| 04 | transaction_amt | Amount distribution: overall, by target, train vs test |
| 05 | temporal_patterns | Daily counts, daily fraud rate, hourly fraud rate, train/test coverage |
| 06 | top_v_features | Top 12 V-features by correlation, distribution by target |
| 07 | correlation_matrix | Top 20 features + target correlation heatmap |
| 08 | d_features | D-feature missing rates and fraud rate (NaN vs present) |
