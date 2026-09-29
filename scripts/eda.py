"""
Exploratory Data Analysis on data/train_test.csv (train-test.csv).

Run with:
    python scripts/eda.py

Outputs:
    - Console summary of shape, dtypes, missingness, duplicates, cardinality,
      distance/coordinate consistency, and target distribution.
    - reports/eda_plots/*.png charts.
    - reports/eda_summary.txt (same summary as printed to console).
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "train-test.csv"
REPORT_DIR = ROOT / "reports"
PLOT_DIR = REPORT_DIR / "eda_plots"


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in miles between two (lat, lon) points."""
    r_miles = 3958.7613
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r_miles * c


def section(title: str, buf: StringIO) -> None:
    line = f"\n{'=' * 10} {title} {'=' * 10}\n"
    print(line)
    buf.write(line)


def emit(text: str, buf: StringIO) -> None:
    print(text)
    buf.write(text + "\n")


def main() -> None:
    buf = StringIO()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    # ---------------------------------------------------------------- shape
    section("SHAPE & DTYPES", buf)
    emit(f"Rows: {len(df):,}  Columns: {df.shape[1]}", buf)
    emit(str(df.dtypes), buf)

    # ------------------------------------------------------------ missing
    section("MISSING VALUES", buf)
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        emit("No missing values in any column.", buf)
    else:
        emit(str(missing), buf)

    # ---------------------------------------------------------- duplicates
    section("DUPLICATES", buf)
    dup_ids = df["load_id"].duplicated().sum()
    dup_rows = df.duplicated().sum()
    emit(f"Duplicate load_id values: {dup_ids}", buf)
    emit(f"Fully duplicated rows: {dup_rows}", buf)

    # ------------------------------------------------------- id continuity
    section("LOAD_ID FORMAT CHECK", buf)
    bad_ids = ~df["load_id"].str.match(r"^TR-\d{6}$", na=False)
    emit(f"load_id values not matching TR-###### pattern: {bad_ids.sum()}", buf)

    # ----------------------------------------------------- numeric summary
    section("NUMERIC SUMMARY", buf)
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
    emit(str(df[numeric_cols].describe().T), buf)

    # -------------------------------------------------------- sanity ranges
    section("SANITY / INVALID VALUE CHECKS", buf)
    emit(f"distance <= 0: {(df['distance'] <= 0).sum()}", buf)
    emit(f"weight <= 0: {(df['weight'] <= 0).sum()}", buf)
    emit(f"posted_rate <= 0: {(df['posted_rate'] <= 0).sum()}", buf)
    emit(f"pickup == delivery (same city): {(df['pickup'] == df['delivery']).sum()}", buf)
    emit(
        f"lat out of [-90,90] range: "
        f"{((df['pickup_lat'].abs() > 90) | (df['delivery_lat'].abs() > 90)).sum()}",
        buf,
    )
    emit(
        f"lon out of [-180,180] range: "
        f"{((df['pickup_lon'].abs() > 180) | (df['delivery_lon'].abs() > 180)).sum()}",
        buf,
    )

    # ------------------------------------------------- outliers (IQR-based)
    section("OUTLIERS (IQR method, 1.5x whisker)", buf)
    for col in ["distance", "weight", "market_index", "quote_signal", "posted_rate"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        emit(f"{col}: bounds=({lower:.2f}, {upper:.2f})  outliers={n_out}  ({n_out / len(df):.2%})", buf)

    # --------------------------------------------- distance vs coordinates
    section("DISTANCE vs HAVERSINE(lat/lon) CONSISTENCY", buf)
    df["haversine_distance"] = haversine_miles(
        df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"]
    )
    df["distance_diff"] = df["distance"] - df["haversine_distance"]
    df["distance_ratio"] = df["distance"] / df["haversine_distance"].replace(0, np.nan)
    emit(str(df["distance_diff"].describe()), buf)
    emit(
        f"Rows where stated distance is > 2x or < 0.5x haversine distance "
        f"(possible bad geo data): {((df['distance_ratio'] > 2) | (df['distance_ratio'] < 0.5)).sum()}",
        buf,
    )

    # -------------------------------------------------------- categorical
    section("CATEGORICAL CARDINALITY", buf)
    for col in ["pickup", "delivery", "equipment"]:
        emit(f"{col}: {df[col].nunique()} unique values", buf)
    emit("\nequipment value counts:", buf)
    emit(str(df["equipment"].value_counts()), buf)
    emit("\nTop 10 pickup cities:", buf)
    emit(str(df["pickup"].value_counts().head(10)), buf)
    emit("\nTop 10 delivery cities:", buf)
    emit(str(df["delivery"].value_counts().head(10)), buf)

    # -------------------------------------------------------------- dates
    section("DATE COVERAGE", buf)
    emit(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}", buf)
    emit(f"Distinct dates: {df['date'].nunique()}", buf)
    daily_counts = df.groupby(df["date"].dt.date).size()
    emit(f"Rows per day: min={daily_counts.min()} max={daily_counts.max()} mean={daily_counts.mean():.1f}", buf)

    # ------------------------------------------------------ rate per mile
    section("RATE-PER-MILE", buf)
    df["rate_per_mile"] = df["posted_rate"] / df["distance"]
    emit(str(df["rate_per_mile"].describe()), buf)

    # ------------------------------------------------------- correlations
    section("CORRELATION WITH posted_rate", buf)
    corr_cols = ["distance", "weight", "market_index", "quote_signal", "posted_rate", "haversine_distance"]
    corr = df[corr_cols].corr(numeric_only=True)["posted_rate"].sort_values(ascending=False)
    emit(str(corr), buf)

    # ------------------------------------------------------------- target
    section("TARGET DISTRIBUTION (posted_rate) SKEW", buf)
    emit(f"Skew (raw): {df['posted_rate'].skew():.3f}", buf)
    emit(f"Skew (log1p): {np.log1p(df['posted_rate']).skew():.3f}", buf)

    # -------------------------------------------------------------- plots
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    df["posted_rate"].hist(bins=60, ax=axes[0, 0])
    axes[0, 0].set_title("posted_rate distribution")

    np.log1p(df["posted_rate"]).hist(bins=60, ax=axes[0, 1])
    axes[0, 1].set_title("log1p(posted_rate) distribution")

    axes[1, 0].scatter(df["distance"], df["posted_rate"], s=3, alpha=0.3)
    axes[1, 0].set_xlabel("distance")
    axes[1, 0].set_ylabel("posted_rate")
    axes[1, 0].set_title("distance vs posted_rate")

    daily_mean = df.groupby(df["date"].dt.date)["posted_rate"].mean()
    axes[1, 1].plot(daily_mean.index, daily_mean.values)
    axes[1, 1].set_title("mean posted_rate over time")
    axes[1, 1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "overview.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    equip_means = df.groupby("equipment")["rate_per_mile"].mean().sort_values(ascending=False)
    equip_means.plot(kind="bar", ax=ax2)
    ax2.set_title("Mean rate-per-mile by equipment type")
    fig2.tight_layout()
    fig2.savefig(PLOT_DIR / "rate_per_mile_by_equipment.png", dpi=150)
    plt.close(fig2)

    section("DONE", buf)
    emit(f"Plots saved to: {PLOT_DIR}", buf)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "eda_summary.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
