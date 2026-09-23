"""
data_processing.py
==================
Flight Booking Analysis – Data Cleaning, EDA, and Insight Generation.

Responsibilities:
  - Load raw CSV data
  - Validate and clean the dataset
  - Engineer derived features
  - Compute aggregated statistics used by the dashboard and prediction module
  - Return clean DataFrames and insight dictionaries consumed by app.py

Dataset columns:
  airline, flight, source_city, departure_time, stops, arrival_time,
  destination_city, class, duration, days_left, price
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Flight_Booking.csv")

# Ordered mapping for stop categories
STOPS_ORDER = {"zero": 0, "one": 1, "two_or_more": 2}

# Ordered mapping for time-of-day categories
TIME_ORDER = {
    "Early_Morning": 0,
    "Morning": 1,
    "Afternoon": 2,
    "Evening": 3,
    "Night": 4,
    "Late_Night": 5,
}

# Price-tier bucket edges (INR)
PRICE_BINS = [0, 5_000, 10_000, 20_000, 40_000, 80_000, 200_000]
PRICE_LABELS = ["<5K", "5K-10K", "10K-20K", "20K-40K", "40K-80K", ">80K"]


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV file and perform basic dtype casting."""
    df = pd.read_csv(path, index_col=0)

    # Ensure numeric columns are cast correctly
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df["days_left"] = pd.to_numeric(df["days_left"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# 2. Data Cleaning
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform full data cleaning pipeline.

    Steps:
      1. Strip whitespace from all string columns
      2. Drop exact duplicate rows
      3. Remove rows with null values in critical columns
      4. Remove extreme price outliers (beyond 3 × IQR)
      5. Add ordinal encodings and derived features
    """
    df = df.copy()

    # --- 2.1 Strip whitespace ---
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # --- 2.2 Drop duplicates ---
    before = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = before - len(df)

    # --- 2.3 Drop rows with NaN in critical columns ---
    critical_cols = ["airline", "source_city", "destination_city",
                     "class", "duration", "days_left", "price", "stops"]
    df.dropna(subset=critical_cols, inplace=True)

    # --- 2.4 Remove extreme price outliers using IQR ---
    q1 = df["price"].quantile(0.25)
    q3 = df["price"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr
    df = df[(df["price"] >= lower) & (df["price"] <= upper)]

    # --- 2.5 Ordinal encodings ---
    df["stops_num"] = df["stops"].map(STOPS_ORDER).fillna(1).astype(int)
    df["departure_time_num"] = df["departure_time"].map(TIME_ORDER).fillna(0).astype(int)
    df["arrival_time_num"] = df["arrival_time"].map(TIME_ORDER).fillna(0).astype(int)

    # --- 2.6 Derived features ---
    # Route string
    df["route"] = df["source_city"] + " → " + df["destination_city"]

    # Price tier bucket
    df["price_tier"] = pd.cut(
        df["price"], bins=PRICE_BINS, labels=PRICE_LABELS, right=False
    )

    # Days-left bucket for trend analysis
    df["days_bucket"] = pd.cut(
        df["days_left"],
        bins=[0, 7, 14, 21, 30, 49],
        labels=["1-7", "8-14", "15-21", "22-30", "31-49"],
        right=True,
    )

    # Class binary flag
    df["is_business"] = (df["class"] == "Business").astype(int)

    return df


# ---------------------------------------------------------------------------
# 3. Missing Value Report
# ---------------------------------------------------------------------------

def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarising missing values per column."""
    total = df.isnull().sum()
    pct = (total / len(df) * 100).round(2)
    report = pd.DataFrame({"Missing Count": total, "Missing %": pct})
    report = report[report["Missing Count"] > 0].sort_values("Missing Count", ascending=False)
    return report


# ---------------------------------------------------------------------------
# 4. Duplicate Report
# ---------------------------------------------------------------------------

def duplicate_report(df_raw: pd.DataFrame) -> dict:
    """Return statistics about duplicate rows in the raw dataset."""
    n_dups = df_raw.duplicated().sum()
    return {
        "total_rows": len(df_raw),
        "duplicate_rows": int(n_dups),
        "duplicate_pct": round(n_dups / len(df_raw) * 100, 2),
    }


# ---------------------------------------------------------------------------
# 5. Outlier Report
# ---------------------------------------------------------------------------

def outlier_report(df: pd.DataFrame, col: str = "price") -> dict:
    """Return IQR-based outlier statistics for a numeric column."""
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    return {
        "column": col,
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "lower_fence": round(lower, 2),
        "upper_fence": round(upper, 2),
        "n_outliers": len(outliers),
        "outlier_pct": round(len(outliers) / len(df) * 100, 2),
    }


# ---------------------------------------------------------------------------
# 6. KPI Computation
# ---------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute headline KPIs displayed on the dashboard."""
    most_popular_airline = df["airline"].value_counts().idxmax()
    most_popular_route = df["route"].value_counts().idxmax()
    avg_duration_h = df["duration"].mean()
    avg_duration_fmt = f"{int(avg_duration_h)}h {int((avg_duration_h % 1) * 60)}m"

    return {
        "total_flights": len(df),
        "avg_price": round(df["price"].mean(), 0),
        "median_price": round(df["price"].median(), 0),
        "most_popular_airline": most_popular_airline,
        "most_popular_route": most_popular_route,
        "avg_duration": round(avg_duration_h, 2),
        "avg_duration_fmt": avg_duration_fmt,
        "economy_count": int((df["class"] == "Economy").sum()),
        "business_count": int((df["class"] == "Business").sum()),
        "total_airlines": df["airline"].nunique(),
        "total_routes": df["route"].nunique(),
    }


# ---------------------------------------------------------------------------
# 7. Aggregated Analysis DataFrames
# ---------------------------------------------------------------------------

def airline_price_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, median, min, max price per airline and class."""
    return (
        df.groupby(["airline", "class"])["price"]
        .agg(["mean", "median", "min", "max", "count"])
        .round(0)
        .reset_index()
        .rename(columns={"mean": "avg_price", "median": "median_price",
                         "min": "min_price", "max": "max_price",
                         "count": "flight_count"})
    )


def stops_price_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Average price per number-of-stops category."""
    result = (
        df.groupby("stops")["price"]
        .agg(["mean", "median", "count"])
        .round(0)
        .reset_index()
        .rename(columns={"mean": "avg_price", "median": "median_price", "count": "flight_count"})
    )
    result["stops_order"] = result["stops"].map(STOPS_ORDER).fillna(1)
    return result.sort_values("stops_order").drop(columns="stops_order")


def duration_price_corr(df: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    """Bin duration into equal-width buckets and return mean price per bucket."""
    df2 = df.copy()
    df2["duration_bin"] = pd.cut(df2["duration"], bins=bins)
    result = (
        df2.groupby("duration_bin", observed=True)["price"]
        .mean()
        .round(0)
        .reset_index()
    )
    result["duration_mid"] = result["duration_bin"].apply(lambda x: round(x.mid, 2))
    return result.dropna()


def days_left_price_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Average price per days-left value for trend analysis."""
    return (
        df.groupby("days_left")["price"]
        .mean()
        .round(0)
        .reset_index()
        .rename(columns={"price": "avg_price"})
        .sort_values("days_left")
    )


def route_analysis(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Top routes by volume and their average prices."""
    return (
        df.groupby("route")
        .agg(flight_count=("price", "count"), avg_price=("price", "mean"))
        .round(0)
        .reset_index()
        .sort_values("flight_count", ascending=False)
        .head(top_n)
    )


def expensive_routes(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Top routes by average price (min 50 flights)."""
    result = (
        df.groupby("route")
        .agg(flight_count=("price", "count"), avg_price=("price", "mean"))
        .round(0)
        .reset_index()
    )
    return result[result["flight_count"] >= 50].sort_values("avg_price", ascending=False).head(top_n)


def class_price_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Price statistics split by travel class."""
    return (
        df.groupby("class")["price"]
        .agg(["mean", "median", "min", "max", "std", "count"])
        .round(0)
        .reset_index()
        .rename(columns={"mean": "avg_price", "median": "median_price",
                         "min": "min_price", "max": "max_price",
                         "std": "std_price", "count": "flight_count"})
    )


def departure_time_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Average price and volume per departure time slot."""
    result = (
        df.groupby("departure_time")
        .agg(avg_price=("price", "mean"), flight_count=("price", "count"))
        .round(0)
        .reset_index()
    )
    result["time_order"] = result["departure_time"].map(TIME_ORDER).fillna(99)
    return result.sort_values("time_order").drop(columns="time_order")


def price_distribution_data(df: pd.DataFrame) -> pd.DataFrame:
    """Price tier distribution counts."""
    dist = df["price_tier"].value_counts().reset_index()
    dist.columns = ["price_tier", "count"]
    # Preserve bucket order
    dist["order"] = dist["price_tier"].map({v: i for i, v in enumerate(PRICE_LABELS)})
    return dist.sort_values("order").drop(columns="order")


def airline_market_share(df: pd.DataFrame) -> pd.DataFrame:
    """Airline-wise flight count and market share %."""
    counts = df["airline"].value_counts().reset_index()
    counts.columns = ["airline", "flight_count"]
    counts["share_pct"] = (counts["flight_count"] / counts["flight_count"].sum() * 100).round(2)
    return counts


# ---------------------------------------------------------------------------
# 8. Feature Engineering for Prediction
# ---------------------------------------------------------------------------

def prepare_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical features and return a feature matrix
    suitable for scikit-learn estimators.

    Returns a DataFrame with numeric columns only (no 'price' column).
    The target column 'price' is retained as-is in the original df.
    """
    feature_cols = [
        "airline", "source_city", "destination_city",
        "departure_time", "arrival_time", "class",
        "stops_num", "duration", "days_left",
    ]
    df_feat = df[feature_cols].copy()

    # One-hot encode categorical columns
    cat_cols = ["airline", "source_city", "destination_city",
                "departure_time", "arrival_time", "class"]
    df_encoded = pd.get_dummies(df_feat, columns=cat_cols, drop_first=True)

    return df_encoded


# ---------------------------------------------------------------------------
# 9. Full Pipeline Entry Point
# ---------------------------------------------------------------------------

def run_pipeline(path: str = DATA_PATH) -> dict:
    """
    Run the full data processing pipeline and return a dictionary of
    clean data and pre-computed statistics for use by the dashboard.
    """
    df_raw = load_data(path)
    dup_info = duplicate_report(df_raw)
    missing_raw = missing_value_report(df_raw)

    df = clean_data(df_raw)

    return {
        # Core data
        "df": df,
        "df_raw": df_raw,
        # Reports
        "duplicate_info": dup_info,
        "missing_report": missing_raw,
        "outlier_report": outlier_report(df),
        # KPIs
        "kpis": compute_kpis(df),
        # Aggregations
        "airline_price_stats": airline_price_stats(df),
        "stops_price_stats": stops_price_stats(df),
        "duration_price_corr": duration_price_corr(df),
        "days_left_trend": days_left_price_trend(df),
        "route_analysis": route_analysis(df),
        "expensive_routes": expensive_routes(df),
        "class_comparison": class_price_comparison(df),
        "departure_analysis": departure_time_analysis(df),
        "price_distribution": price_distribution_data(df),
        "market_share": airline_market_share(df),
    }
