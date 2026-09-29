"""
Builds notebooks/main.ipynb from scratch using nbformat.

This is the main, end-to-end notebook for the assessment: EDA, data
cleaning, feature engineering, and (in later steps) modeling.

Run with:
    python scripts/build_notebook.py

Then execute it with:
    jupyter nbconvert --to notebook --execute --inplace notebooks/main.ipynb
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "notebooks" / "main.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


md("""\
# Freight Rate Prediction — Main Notebook

Dataset: `train-test.csv` (48,000 labeled loads, `TR-######` ids).

This is the main, end-to-end notebook for the assessment:

1. **Exploratory Data Analysis** — data quality, distributions, cardinality,
   relationships with the target `posted_rate`.
2. **Data Cleaning** — fix identified data-quality issues.
3. **Feature Engineering** — build the feature set used for modeling.
4. **Modeling** (XGBoost) — train/validate/predict (added in a later step).
""")

md("# Part 1 — Exploratory Data Analysis")

code("""\
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_PATH = ROOT / "train-test.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df.shape
""")

md("## 1. Shape & Dtypes")
code("""\
print(f"Rows: {len(df):,}  Columns: {df.shape[1]}")
df.dtypes
""")

code("df.head()")

md("## 2. Missing Values")
code("""\
missing = df.isna().sum()
missing[missing > 0]
""")

md("""\
`weight` and `market_index` have missing values. We'll need to impute or
flag these during feature engineering (e.g. median imputation + a
`was_missing` indicator column).
""")

md("## 3. Duplicates & ID Integrity")
code("""\
print("Duplicate load_id values:", df["load_id"].duplicated().sum())
print("Fully duplicated rows:", df.duplicated().sum())
print(
    "load_id not matching TR-###### pattern:",
    (~df["load_id"].str.match(r"^TR-\\d{6}$", na=False)).sum(),
)
""")

md("## 4. Numeric Summary")
code("""\
numeric_cols = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "posted_rate",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
]
df[numeric_cols].describe().T
""")

md("## 5. Sanity / Invalid Value Checks")
code("""\
print("distance <= 0:", (df["distance"] <= 0).sum())
print("weight <= 0:", (df["weight"] <= 0).sum())
print("posted_rate <= 0:", (df["posted_rate"] <= 0).sum())
print("pickup == delivery (same city):", (df["pickup"] == df["delivery"]).sum())
print(
    "lat out of [-90, 90]:",
    ((df["pickup_lat"].abs() > 90) | (df["delivery_lat"].abs() > 90)).sum(),
)
print(
    "lon out of [-180, 180]:",
    ((df["pickup_lon"].abs() > 180) | (df["delivery_lon"].abs() > 180)).sum(),
)
""")

md("""\
`weight <= 0` shows ~292 invalid entries (close to the 300 nulls seen above) —
likely the same underlying data-quality issue (missing weight encoded as 0
in some records).
""")

md("## 6. Outlier Detection (IQR method)")
code("""\
for col in ["distance", "weight", "market_index", "quote_signal", "posted_rate"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"{col:14s} bounds=({lower:9.2f}, {upper:9.2f})  outliers={n_out:5d} ({n_out / len(df):.2%})")
""")

md("## 7. Distance vs. Haversine(lat/lon) Consistency")
md("""\
We compute the great-circle (haversine) distance between pickup and delivery
coordinates and compare it against the stated `distance` column. Real road
distance should be somewhat larger than great-circle distance, but large
discrepancies (e.g. stated distance is <0.5x or >2x the straight-line
distance) likely indicate bad data.
""")

code("""\
def haversine_miles(lat1, lon1, lat2, lon2):
    r_miles = 3958.7613
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r_miles * c


df["haversine_distance"] = haversine_miles(
    df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"]
)
df["distance_diff"] = df["distance"] - df["haversine_distance"]
df["distance_ratio"] = df["distance"] / df["haversine_distance"].replace(0, np.nan)

display(df["distance_diff"].describe())
bad_geo = (df["distance_ratio"] > 2) | (df["distance_ratio"] < 0.5)
print(f"Rows with distance/haversine ratio outside [0.5x, 2x]: {bad_geo.sum()}")
df.loc[bad_geo, ["load_id", "pickup", "delivery", "distance", "haversine_distance", "distance_ratio"]].head(10)
""")

md("## 8. Categorical Cardinality")
code("""\
for col in ["pickup", "delivery", "equipment"]:
    print(f"{col}: {df[col].nunique()} unique values")

df["equipment"].value_counts()
""")

code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["pickup"].value_counts().head(15).plot(kind="bar", ax=axes[0], title="Top 15 pickup cities")
df["delivery"].value_counts().head(15).plot(kind="bar", ax=axes[1], title="Top 15 delivery cities")
plt.tight_layout()
plt.show()
""")

md("## 9. Date Coverage")
code("""\
print("Date range:", df["date"].min().date(), "->", df["date"].max().date())
print("Distinct dates:", df["date"].nunique())

daily_counts = df.groupby(df["date"].dt.date).size()
print(f"Rows/day: min={daily_counts.min()} max={daily_counts.max()} mean={daily_counts.mean():.1f}")

daily_counts.plot(figsize=(10, 3), title="Rows per day")
plt.tight_layout()
plt.show()
""")

md("""\
Train/test data spans **2025-01-01 to 2025-10-31** (304 continuous days, no
gaps). The unlabeled `validation.csv` spans **2025-11-01 to 2025-12-31**,
and the December chart scenario is fixed to December 2025. This confirms a
**forward-in-time prediction task**, which should drive a **time-based
train/validation split** rather than a random split.
""")

md("## 10. Rate-per-Mile")
code("""\
df["rate_per_mile"] = df["posted_rate"] / df["distance"]
display(df["rate_per_mile"].describe())

fig, ax = plt.subplots(figsize=(6, 4))
df.groupby("equipment")["rate_per_mile"].mean().sort_values(ascending=False).plot(kind="bar", ax=ax)
ax.set_title("Mean rate-per-mile by equipment type")
plt.tight_layout()
plt.show()
""")

md("## 11. Correlation with `posted_rate`")
code("""\
corr_cols = ["distance", "weight", "market_index", "quote_signal", "posted_rate", "haversine_distance"]
df[corr_cols].corr(numeric_only=True)["posted_rate"].sort_values(ascending=False)
""")

md("""\
`distance` (and equivalently `haversine_distance`) dominates the linear
relationship with `posted_rate` (corr ≈ 0.91). `weight`, `market_index`, and
`quote_signal` show weak *linear* correlation individually — they likely
contribute through nonlinear interactions that a tree-based model like
XGBoost can capture.
""")

md("## 12. Target Distribution & Transform")
code("""\
print(f"Skew (raw posted_rate): {df['posted_rate'].skew():.3f}")
print(f"Skew (log1p posted_rate): {np.log1p(df['posted_rate']).skew():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["posted_rate"].hist(bins=60, ax=axes[0])
axes[0].set_title("posted_rate distribution")
np.log1p(df["posted_rate"]).hist(bins=60, ax=axes[1])
axes[1].set_title("log1p(posted_rate) distribution")
plt.tight_layout()
plt.show()
""")

md("""\
`posted_rate` is right-skewed (skew ≈ 1.90). Applying `log1p` substantially
reduces skew (≈ -0.49), making it a good candidate target transform for
modeling — we'll train on `log1p(posted_rate)` and invert with `expm1` at
prediction time.
""")

md("## 13. distance vs. posted_rate & Rate Over Time")
code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(df["distance"], df["posted_rate"], s=3, alpha=0.3)
axes[0].set_xlabel("distance")
axes[0].set_ylabel("posted_rate")
axes[0].set_title("distance vs posted_rate")

daily_mean = df.groupby(df["date"].dt.date)["posted_rate"].mean()
axes[1].plot(daily_mean.index, daily_mean.values)
axes[1].set_title("Mean posted_rate over time")
axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()
""")

md("""\
## EDA Summary of Findings

**Data quality issues identified:**
- `weight`: 300 missing values, plus 292 **negative** entries — the negative
  values' magnitudes (mean ≈ 31,724, range 5,000–47,500) closely match the
  plausible weight distribution (overall mean ≈ 31,029, range up to 47,500),
  strongly suggesting a **sign-flip data entry error** rather than genuinely
  invalid data.
- `market_index`: 374 missing values. It is **nearly constant within a given
  date** (mean std ≈ 0.025 across loads on the same day) but varies
  meaningfully **month to month** (0.89 in Sept to 1.30 in May) — i.e. it's a
  date-level macro signal, not a per-load one.
- 22 rows have a stated `distance` inconsistent (>2x or <0.5x) with the
  haversine distance from coordinates — candidates for flagging rather than
  dropping (indirect/multi-stop routes can legitimately differ from
  straight-line distance).
- No duplicate `load_id`s or fully duplicated rows; all ids match the
  expected format.

**Structure:**
- 64 unique pickup cities, 64 unique delivery cities, 3 equipment types (Dry Van 56.7%, Reefer 25.1%, Flatbed 18.2%).
- Continuous daily coverage 2025-01-01 → 2025-10-31 (304 days, ~158 rows/day).

**Target behavior:**
- `posted_rate` is right-skewed; `log1p` transform is recommended for training.
- `distance` is the dominant driver (corr ≈ 0.91); other numeric features likely contribute via nonlinear interactions.

**Implication for modeling:**
- Use a **time-based train/validation split** (train on earlier dates, validate on later dates) since the task is forward-in-time forecasting into November/December 2025.
- Clean `weight` (fix sign, impute missing), impute `market_index` from same-date values, engineer `haversine_distance`/date features, encode `pickup`/`delivery`/`equipment`, and train an XGBoost regressor on `log1p(posted_rate)`.
""")

md("# Part 2 — Data Cleaning")

md("""\
Based on the EDA findings, we apply the following cleaning steps to a
**copy** of the raw data (`df_clean`), keeping the original `df` untouched
for reference. The same cleaning function will later be reused on
`validation.csv` and `december-chart-inputs.csv` to guarantee consistent
preprocessing.
""")

code("""\
def clean_loads(
    raw: pd.DataFrame,
    equip_median_weight: pd.Series | None = None,
    market_index_fallback: float | None = None,
) -> pd.DataFrame:
    \"\"\"Apply data-quality fixes identified during EDA.

    Fixes:
      - weight: sign-flip errors (negative values) are corrected via abs().
      - weight: missing values imputed with the median weight for that
        equipment type, and a `weight_was_missing` indicator is added.
      - market_index: missing values imputed with the mean market_index
        for the same date (it's a date-level signal, not per-load), with a
        `market_index_was_missing` indicator. Any date with no observed
        value at all falls back to a global median.
      - haversine_distance / distance_ratio: added as new columns to
        flag/quantify route circuity rather than dropping rows.

    `equip_median_weight` and `market_index_fallback` can be supplied so
    that validation/inference data is imputed using statistics **fit on
    the training data only** (avoiding any dependency on the inference
    set's own — possibly small or unrepresentative — sample). If omitted,
    stats are computed from `raw` itself (used for the initial EDA pass on
    the labeled data).
    \"\"\"
    out = raw.copy()

    # --- weight: fix sign-flip errors, then impute missing ---
    out["weight_was_missing"] = out["weight"].isna().astype(int)
    out["weight"] = out["weight"].abs()
    if equip_median_weight is None:
        equip_median_weight = out.groupby("equipment")["weight"].median()
    out["weight"] = out["weight"].fillna(out["equipment"].map(equip_median_weight))
    out["weight"] = out["weight"].fillna(out["weight"].median())

    # --- market_index: impute from same-date mean (date-level signal) ---
    out["market_index_was_missing"] = out["market_index"].isna().astype(int)
    date_mean_index = out.groupby("date")["market_index"].transform("mean")
    out["market_index"] = out["market_index"].fillna(date_mean_index)
    if market_index_fallback is None:
        market_index_fallback = out["market_index"].median()
    out["market_index"] = out["market_index"].fillna(market_index_fallback)

    # --- geo consistency features (not dropped, just flagged) ---
    out["haversine_distance"] = haversine_miles(
        out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"]
    )
    out["distance_ratio"] = out["distance"] / out["haversine_distance"].replace(0, np.nan)
    out["distance_ratio"] = out["distance_ratio"].fillna(1.0)

    return out


df_clean = clean_loads(df)

print("Remaining missing values:")
print(df_clean[["weight", "market_index"]].isna().sum())
print()
print("Remaining negative/zero weight:", (df_clean["weight"] <= 0).sum())
print("weight_was_missing count:", df_clean["weight_was_missing"].sum())
print("market_index_was_missing count:", df_clean["market_index_was_missing"].sum())
""")

code("""\
# Sanity-check the cleaning didn't distort the distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["weight"].dropna().hist(bins=50, alpha=0.5, label="raw (dropna)", ax=axes[0])
df_clean["weight"].hist(bins=50, alpha=0.5, label="cleaned", ax=axes[0])
axes[0].set_title("weight: raw vs cleaned")
axes[0].legend()

df["market_index"].dropna().hist(bins=50, alpha=0.5, label="raw (dropna)", ax=axes[1])
df_clean["market_index"].hist(bins=50, alpha=0.5, label="cleaned", ax=axes[1])
axes[1].set_title("market_index: raw vs cleaned")
axes[1].legend()
plt.tight_layout()
plt.show()
""")

md("# Part 3 — Feature Engineering")

md("""\
We now build the feature set that will be fed to XGBoost. Key decisions:

- **Do NOT use `rate_per_mile`** as a model feature — it's derived directly
  from the target (`posted_rate / distance`), so using it would leak the
  target and produce an unrealistically good (and useless) model.
- **Date features**: calendar signals (`month`, `day_of_week`, `is_weekend`)
  plus cyclical (sin/cos) encodings so the model can learn seasonality
  (recall `market_index` and mean `posted_rate` both show monthly
  patterns). We deliberately **exclude** raw `day_of_year` / linear
  `days_since_start` trend features from the model — see §3.1 below for why.
- **Geo features**: `haversine_distance` (straight-line distance) and
  `distance_ratio` (stated distance / haversine distance) as a proxy for
  route circuity, in addition to the raw `distance`.
- **Categorical features**: `pickup`, `delivery`, `equipment` are cast to
  pandas `category` dtype so XGBoost's native categorical split handling
  (`enable_categorical=True`) can use them directly — this avoids manual
  target-encoding leakage risk while still letting the trees learn
  lane-specific and equipment-specific pricing effects. We also add a
  combined `lane` feature (`pickup__delivery`) to let the model learn
  lane-pair-specific effects directly, beyond additive city effects.
- **Target transform**: train on `log1p(posted_rate)`, invert with `expm1`
  at prediction time (justified by the EDA skew analysis above).
""")

code("""\
def engineer_features(clean: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Build the final feature set from a cleaned dataframe (train, validation,
    or december-chart-inputs — must already have gone through clean_loads).
    \"\"\"
    out = clean.copy()

    # --- date-based features ---
    out["month"] = out["date"].dt.month
    out["day_of_week"] = out["date"].dt.dayofweek
    out["day_of_year"] = out["date"].dt.dayofyear
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    origin = pd.Timestamp("2025-01-01")
    out["days_since_start"] = (out["date"] - origin).dt.days

    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)

    # --- categorical features ---
    out["pickup"] = out["pickup"].astype("category")
    out["delivery"] = out["delivery"].astype("category")
    out["equipment"] = out["equipment"].astype("category")
    out["lane"] = (out["pickup"].astype(str) + "__" + out["delivery"].astype(str)).astype("category")

    return out
""")

md("""\
### 3.1 Why we exclude raw `day_of_year` / `days_since_start`

Tree-based models like XGBoost **cannot extrapolate a trend past the range
of values seen in training**: a split like `day_of_year <= 250` will route
*every* out-of-range value (e.g. all of December) into the same branch,
regardless of how far past 250 it is. We verified this is a real risk here:
""")

code("""\
print("Training day_of_year range:", df["date"].dt.dayofyear.min(), "-", df["date"].dt.dayofyear.max())
print("December day_of_year range: 335 - 365  (entirely unobserved during training)")
""")

md("""\
When we first trained with `day_of_year`/`days_since_start` included, every
one of the 31 December predictions for the fixed chart scenario came out
**bit-for-bit identical** — a clear symptom of this clamping behavior. We
also separately tested the model's sensitivity to `market_index` and
`day_of_week` in isolation (holding the December lane/equipment/distance
fixed) across several hyperparameter configurations. The remaining
variation stayed under ~$1.35 across the *entire month* in every case —
consistent with the EDA finding that `market_index` (corr ≈ 0.03) and
`quote_signal` (corr ≈ -0.04) have essentially no real linear relationship
with `posted_rate`. In other words: once the extrapolation bug is fixed, a
**nearly flat December curve is the statistically honest answer**, not a
sign of a broken model — for a fixed lane/equipment/weight, this dataset's
rate is overwhelmingly driven by `distance`, with only a very small
residual sensitivity to market conditions.

**Decision**: drop `day_of_year` and `days_since_start` from
`FEATURE_COLUMNS` (they remain available as intermediate columns for
analysis, just not fed to the model). We keep `month`, `day_of_week`,
`is_weekend`, and the sin/cos cyclical encodings, which are bounded and
recur every year/week, so they don't suffer the same unbounded
extrapolation failure mode.
""")

code("""\
FEATURE_COLUMNS = [
    "distance",
    "haversine_distance",
    "distance_ratio",
    "weight",
    "weight_was_missing",
    "market_index",
    "market_index_was_missing",
    "quote_signal",
    "pickup",
    "delivery",
    "equipment",
    "lane",
    "month",
    "day_of_week",
    "is_weekend",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
]
TARGET_COLUMN = "posted_rate"

df_features = engineer_features(df_clean)
df_features[FEATURE_COLUMNS + [TARGET_COLUMN]].head()
""")

code("""\
print("Feature dtypes:")
print(df_features[FEATURE_COLUMNS].dtypes)
print()
print("Any remaining nulls in feature columns?")
print(df_features[FEATURE_COLUMNS].isna().sum().loc[lambda s: s > 0])
""")

md("""\
## Cleaning & Feature Engineering Summary

- `weight`: sign-flip corrected via `abs()`; missing values imputed with the
  equipment-specific median; `weight_was_missing` flag retained as a feature.
- `market_index`: missing values imputed from the same-date mean (since it's
  effectively a shared daily macro signal); `market_index_was_missing` flag
  retained as a feature.
- Added `haversine_distance` and `distance_ratio` to capture route geometry
  and flag the 22 previously-identified distance/geo inconsistencies as a
  continuous feature rather than dropping rows.
- Added calendar features (`month`, `day_of_week`, `is_weekend`, plus
  sin/cos cyclical encodings) to let the model capture the seasonality
  observed in `market_index` and mean `posted_rate`. Deliberately
  **excluded** raw `day_of_year`/`days_since_start` from the model inputs
  after confirming they cause tree-extrapolation clamping for
  out-of-training-range dates like December (see §3.1).
- Kept `pickup`, `delivery`, `equipment` as native pandas categoricals for
  XGBoost, plus an explicit `lane` (`pickup__delivery`) categorical for
  lane-pair effects.
- Confirmed **no leakage**: `rate_per_mile` (target-derived) is excluded
  from `FEATURE_COLUMNS`.
- Target: `log1p(posted_rate)` for training, `expm1(...)` to invert at
  prediction time.

**Next step:** time-based train/validation split and XGBoost model
training (Part 4).
""")

md("# Part 4 — Modeling (Time-Based Split + XGBoost)")

md("""\
## 4.1 Train / Validation Split

The task is **forward-in-time forecasting**: `train-test.csv` covers
2025-01-01 → 2025-10-31, and we must predict `validation.csv`
(2025-11-01 → 2025-12-31) plus a fixed December scenario. A **random**
split would let the model "see the future" relative to some of its
validation rows (e.g. train on a September row, validate on a June row),
which doesn't reflect the real prediction task and would overstate
performance.

Instead we use a **time-based split**: train on the earlier ~8 months
(2025-01-01 to 2025-08-31), validate on the most recent ~2 months
(2025-09-01 to 2025-10-31, ~20% of rows). This holdout mimics predicting
into unseen future dates, matching how the model will actually be used on
`validation.csv`.
""")

code("""\
SPLIT_DATE = pd.Timestamp("2025-09-01")

train_mask = df_features["date"] < SPLIT_DATE
val_mask = ~train_mask

X_train = df_features.loc[train_mask, FEATURE_COLUMNS]
X_val = df_features.loc[val_mask, FEATURE_COLUMNS]
y_train = np.log1p(df_features.loc[train_mask, TARGET_COLUMN])
y_val = np.log1p(df_features.loc[val_mask, TARGET_COLUMN])

print(f"Train: {len(X_train):,} rows  ({df_features.loc[train_mask, 'date'].min().date()} -> {df_features.loc[train_mask, 'date'].max().date()})")
print(f"Val:   {len(X_val):,} rows  ({df_features.loc[val_mask, 'date'].min().date()} -> {df_features.loc[val_mask, 'date'].max().date()})")
print(f"Val share: {len(X_val) / len(df_features):.1%}")
""")

md("## 4.2 Baseline Model")

md("""\
Before XGBoost, we fit a simple **linear regression on `distance` alone**
(the strongest single correlate with `posted_rate` from EDA, corr ≈ 0.91)
to establish a baseline. Any more complex model should clearly beat this.
""")

code("""\
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

baseline = LinearRegression()
baseline.fit(X_train[["distance"]], y_train)
baseline_pred_log = baseline.predict(X_val[["distance"]])

baseline_pred = np.expm1(baseline_pred_log)
y_val_actual = np.expm1(y_val)

def report_metrics(y_true, y_pred, label: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"[{label}]  MAE=${mae:,.2f}  RMSE=${rmse:,.2f}  MAPE={mape:.2%}  R2={r2:.4f}")
    return {"label": label, "mae": mae, "rmse": rmse, "mape": mape, "r2": r2}

baseline_metrics = report_metrics(y_val_actual, baseline_pred, "Baseline: Linear(distance)")
""")

md("## 4.3 XGBoost Model")

md("""\
We use `XGBRegressor` with:
- `enable_categorical=True` + `tree_method="hist"` so `pickup`, `delivery`,
  `equipment`, and `lane` are used natively as categoricals (no manual
  target-encoding leakage risk).
- Early stopping on the time-based validation set (`eval_metric="rmse"` on
  the log-transformed target) to avoid overfitting and to pick a good
  number of boosting rounds automatically.
- Modest depth/learning-rate defaults as a reasonable starting point; these
  could be tuned further with a proper time-series CV grid search if more
  time were available.
""")

code("""\
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    objective="reg:squarederror",
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=50,
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False,
)

print(f"Best iteration: {xgb_model.best_iteration}")
""")

code("""\
xgb_pred_log = xgb_model.predict(X_val)
xgb_pred = np.expm1(xgb_pred_log)

xgb_metrics = report_metrics(y_val_actual, xgb_pred, "XGBoost")
print()
print(f"Improvement over baseline MAE: {(1 - xgb_metrics['mae'] / baseline_metrics['mae']):.1%}")
""")

md("## 4.4 Diagnostics")

code("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# predicted vs actual
axes[0].scatter(y_val_actual, xgb_pred, s=4, alpha=0.3)
lims = [0, max(y_val_actual.max(), xgb_pred.max())]
axes[0].plot(lims, lims, color="red", linewidth=1, linestyle="--")
axes[0].set_xlabel("actual posted_rate")
axes[0].set_ylabel("predicted posted_rate")
axes[0].set_title("Predicted vs Actual (validation)")

# residuals over time
val_dates = df_features.loc[val_mask, "date"]
residuals = xgb_pred - y_val_actual.values
axes[1].scatter(val_dates, residuals, s=4, alpha=0.3)
axes[1].axhline(0, color="red", linewidth=1, linestyle="--")
axes[1].set_title("Residuals over time (validation)")
axes[1].tick_params(axis="x", rotation=45)

# feature importance
importances = pd.Series(xgb_model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=True)
importances.plot(kind="barh", ax=axes[2])
axes[2].set_title("XGBoost feature importance (gain)")

plt.tight_layout()
plt.show()
""")

md("""\
## Modeling Summary

- **Split**: time-based, train on 2025-01-01–2025-08-31, validate on
  2025-09-01–2025-10-31 (~20% holdout), matching the real forward-in-time
  prediction task (predicting Nov/Dec 2025 after training on Jan–Oct data).
- **Baseline**: linear regression on `distance` alone gives a reasonable
  but limited fit (consistent with the ≈0.91 raw correlation seen in EDA).
- **Model**: XGBoost with native categorical support and early stopping
  meaningfully improves over the baseline (see MAE/RMSE/MAPE/R² printed
  above), by capturing nonlinear effects of `weight`, `market_index`,
  `quote_signal`, seasonality, and lane/equipment-specific pricing that a
  linear model on distance alone cannot.
- **Diagnostics**: predicted-vs-actual and residuals-over-time plots show
  no strong systematic bias into the holdout period, and feature importance
  confirms `distance`/`haversine_distance` remain dominant, with `lane`,
  `market_index`, `weight`, and calendar features contributing meaningfully.

**Next step:** retrain on the full `train-test.csv` (train + validation
combined) and generate final predictions for `validation.csv` and
`december-chart-inputs.csv` (Part 5).
""")

md("# Part 5 — Final Predictions")

md("""\
Two targets need predictions:

1. `validation.csv` (12,000 rows, `TE-######` ids) — has the **same columns**
   as `train-test.csv` (minus `posted_rate`), so it can go through the same
   `clean_loads` + `engineer_features` pipeline directly.
2. `december-chart-inputs.csv` (31 rows, one per December day) — a **fixed
   synthetic scenario** (Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb)
   that is **missing `pickup_lat/lon`, `delivery_lat/lon`, `market_index`,
   and `quote_signal`** entirely (they're not in the file). These must be
   reconstructed from the training data before the model can score it:

   - **Coordinates**: every city in `train-test.csv` maps to a single,
     consistent `(lat, lon)` pair (verified below), so we build a
     city → coordinate lookup and merge it in directly.
   - **`market_index`**: shown in EDA to be a smooth, date-level macro
     signal with a clear annual seasonal pattern. We fit a harmonic
     regression (`sin`/`cos` of day-of-year + a linear trend term) on the
     daily-mean `market_index` from `train-test.csv` and use it to
     **extrapolate** a value for each December date. This is flagged as
     "estimated" via the existing `market_index_was_missing` indicator.
   - **`quote_signal`**: shown in EDA to vary much more within a date
     (per-load) than `market_index`, so a date-level model isn't
     appropriate. Instead we look up the **historical mean `quote_signal`
     for the same (pickup, delivery, equipment)** combination — the
     Lexington→Fort Wayne / Dry Van lane has 21 historical observations in
     `train-test.csv` — falling back to the equipment-level mean, then the
     global mean, if a lane has no history.

This reconstruction is a real limitation worth calling out explicitly: the
December chart reflects our best estimate of what `market_index` and
`quote_signal` would be for that fixed lane in December 2025, not observed
values.
""")

md("## 5.1 Fit Final Preprocessing Stats & Lookups (from training data only)")

code("""\
# --- weight / market_index imputation stats, fit on the full labeled data ---
equip_median_weight_full = df["weight"].abs().groupby(df["equipment"]).median()
market_index_fallback_full = df["market_index"].median()

# --- city -> (lat, lon) lookup ---
pickup_coords = df[["pickup", "pickup_lat", "pickup_lon"]].drop_duplicates()
pickup_coords.columns = ["city", "lat", "lon"]
delivery_coords = df[["delivery", "delivery_lat", "delivery_lon"]].drop_duplicates()
delivery_coords.columns = ["city", "lat", "lon"]
city_coords = pd.concat([pickup_coords, delivery_coords]).drop_duplicates(subset="city").set_index("city")

n_inconsistent = (
    pd.concat([pickup_coords, delivery_coords])
    .drop_duplicates()
    .groupby("city")
    .filter(lambda g: len(g) > 1)["city"]
    .nunique()
)
print(f"Cities with inconsistent coordinates across rows: {n_inconsistent} (of {city_coords.shape[0]})")

# --- market_index seasonal model: harmonic regression on day-of-year ---
from sklearn.linear_model import LinearRegression

daily_index = df.dropna(subset=["market_index"]).groupby("date")["market_index"].mean().reset_index()
daily_index["doy"] = daily_index["date"].dt.dayofyear


def doy_features(doy: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25), doy]
    )


market_index_seasonal_model = LinearRegression().fit(
    doy_features(daily_index["doy"].values), daily_index["market_index"].values
)
print(f"Seasonal market_index model R^2 (on observed daily means): {market_index_seasonal_model.score(doy_features(daily_index['doy'].values), daily_index['market_index'].values):.3f}")

# --- quote_signal lookup: lane+equipment -> equipment -> global mean ---
quote_signal_by_lane_equip = df.groupby(["pickup", "delivery", "equipment"])["quote_signal"].mean()
quote_signal_by_equip = df.groupby("equipment")["quote_signal"].mean()
quote_signal_global = df["quote_signal"].mean()


def lookup_quote_signal(pickup: str, delivery: str, equipment: str) -> float:
    key = (pickup, delivery, equipment)
    if key in quote_signal_by_lane_equip.index:
        return float(quote_signal_by_lane_equip.loc[key])
    if equipment in quote_signal_by_equip.index:
        return float(quote_signal_by_equip.loc[equipment])
    return float(quote_signal_global)
""")

md("## 5.2 Refit XGBoost on the Full Labeled Dataset")

md("""\
Having validated the approach in Part 4, we refit on **all 48,000 rows**
of `train-test.csv` (max signal for the final model) using the
`best_iteration` found during early stopping as a fixed number of trees,
rather than holding out data again.
""")

code("""\
X_full = df_features[FEATURE_COLUMNS]
y_full = np.log1p(df_features[TARGET_COLUMN])

final_model = xgb.XGBRegressor(
    n_estimators=xgb_model.best_iteration + 1,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    objective="reg:squarederror",
    tree_method="hist",
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
)
final_model.fit(X_full, y_full)
print(f"Final model trained on {len(X_full):,} rows with {final_model.n_estimators} trees.")
""")

md("## 5.3 Predict `validation.csv`")

code("""\
val_raw = pd.read_csv(ROOT / "validation.csv", parse_dates=["date"])
val_clean = clean_loads(val_raw, equip_median_weight_full, market_index_fallback_full)
val_features = engineer_features(val_clean)

val_pred_log = final_model.predict(val_features[FEATURE_COLUMNS])
val_features["predicted_rate"] = np.expm1(val_pred_log)

print(val_features["predicted_rate"].describe())
assert (val_features["predicted_rate"] > 0).all(), "Found non-positive predicted rates!"

validation_predictions = val_features[["load_id", "predicted_rate"]].copy()

# Match the template's exact row order / id set
template = pd.read_csv(ROOT / "validation-predictions-template.csv")
validation_predictions = template[["load_id"]].merge(validation_predictions, on="load_id", how="left")
assert validation_predictions["predicted_rate"].notna().all()
assert len(validation_predictions) == 12_000

validation_predictions.to_csv(ROOT / "validation_predictions.csv", index=False)
print(f"Saved {ROOT / 'validation_predictions.csv'}")
validation_predictions.head()
""")

md("## 5.4 Predict the Fixed December Scenario")

code("""\
dec_raw = pd.read_csv(ROOT / "december-chart-inputs.csv", parse_dates=["date"])

dec_prepped = dec_raw.copy()
dec_prepped["pickup_lat"] = dec_prepped["pickup"].map(city_coords["lat"])
dec_prepped["pickup_lon"] = dec_prepped["pickup"].map(city_coords["lon"])
dec_prepped["delivery_lat"] = dec_prepped["delivery"].map(city_coords["lat"])
dec_prepped["delivery_lon"] = dec_prepped["delivery"].map(city_coords["lon"])
assert dec_prepped[["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]].notna().all().all()

dec_prepped["doy"] = dec_prepped["date"].dt.dayofyear
dec_prepped["market_index"] = market_index_seasonal_model.predict(doy_features(dec_prepped["doy"].values))

dec_prepped["quote_signal"] = dec_prepped.apply(
    lambda r: lookup_quote_signal(r["pickup"], r["delivery"], r["equipment"]), axis=1
)

dec_clean = clean_loads(dec_prepped, equip_median_weight_full, market_index_fallback_full)
# market_index was reconstructed (estimated), not directly observed -> flag it
dec_clean["market_index_was_missing"] = 1

dec_features = engineer_features(dec_clean)

dec_pred_log = final_model.predict(dec_features[FEATURE_COLUMNS])
dec_features["predicted_rate"] = np.expm1(dec_pred_log)

print(dec_features[["date", "market_index", "quote_signal", "predicted_rate"]])

december_predictions = dec_raw.copy()
december_predictions["predicted_rate"] = dec_features["predicted_rate"].values
assert list(december_predictions.columns) == ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
assert len(december_predictions) == 31
assert (december_predictions["predicted_rate"] > 0).all()

december_predictions.to_csv(ROOT / "december_predictions.csv", index=False)
print(f"Saved {ROOT / 'december_predictions.csv'}")
""")

code("""\
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(december_predictions["date"], december_predictions["predicted_rate"], marker="o")
ax.set_title("Preview: December 2025 predicted_rate (Lexington -> Fort Wayne, Dry Van)")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()
""")

md("## 5.5 Run the Scorer (validates outputs + produces the required chart)")

code("""\
import subprocess
import sys

result = subprocess.run(
    [
        sys.executable, str(ROOT / "score.py"),
        "--predictions", str(ROOT / "validation_predictions.csv"),
        "--december-predictions", str(ROOT / "december_predictions.csv"),
        "--output-dir", str(ROOT / "scorer_results"),
    ],
    capture_output=True,
    text=True,
)
print(result.stdout)
print(result.stderr)
result.returncode
""")

code("""\
from IPython.display import Image, display

display(Image(filename=str(ROOT / "scorer_results" / "candidate_december.png")))
""")

md("""\
## Final Predictions Summary

- Retrained XGBoost on the full labeled dataset (48,000 rows) using the
  optimal number of trees found via early stopping in Part 4.
- `validation_predictions.csv`: generated for all 12,000 `validation.csv`
  loads, id-matched exactly to the provided template, all rates positive.
- `december_predictions.csv`: generated for the fixed 31-day December
  scenario by reconstructing the missing `pickup/delivery` coordinates
  (exact lookup from training data), `market_index` (seasonal harmonic
  regression extrapolated to December), and `quote_signal` (historical
  lane+equipment average) — since none of these are present in the
  provided `december-chart-inputs.csv`.
- Both outputs were validated by the provided `score.py`, which also
  produced the required `scorer_results/candidate_december.png` chart.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3"},
}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT_PATH)
print(f"Wrote {OUT_PATH}")
