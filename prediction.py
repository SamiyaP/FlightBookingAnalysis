"""
prediction.py
=============
Flight Booking Analysis – Fare Prediction Module.

This module:
  1. Trains a Random Forest Regressor on the cleaned flight dataset.
  2. Serialises the trained model + preprocessor to disk (models/ directory).
  3. Exposes a predict_fare() function for single-flight price prediction.
  4. Provides model evaluation metrics and feature importance retrieval.

Model Pipeline
--------------
  Raw categorical features → OneHotEncoder
  Numeric features         → StandardScaler
  Combined features        → RandomForestRegressor

Usage
-----
    from prediction import train_model, predict_fare, load_model

    # One-time training (or call train_and_save())
    pipeline, metrics = train_model(df_clean)

    # Predict for a new flight
    price = predict_fare(pipeline, {
        "airline": "Vistara",
        "source_city": "Delhi",
        "destination_city": "Mumbai",
        "departure_time": "Morning",
        "arrival_time": "Afternoon",
        "class": "Economy",
        "stops_num": 0,
        "duration": 2.25,
        "days_left": 15,
    })
"""

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fare_predictor.pkl")

CATEGORICAL_FEATURES = [
    "airline",
    "source_city",
    "destination_city",
    "departure_time",
    "arrival_time",
    "class",
]

NUMERIC_FEATURES = [
    "stops_num",
    "duration",
    "days_left",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# ---------------------------------------------------------------------------
# 1. Preprocessor
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - One-hot encodes categorical columns (handle_unknown='ignore' so
        unseen categories at inference time don't raise an error)
      - Scales numeric columns with StandardScaler
    """
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, CATEGORICAL_FEATURES),
            ("num", num_transformer, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


# ---------------------------------------------------------------------------
# 2. Model Training
# ---------------------------------------------------------------------------

def train_model(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    n_estimators: int = 150,
    max_depth: int = 20,
) -> tuple:
    """
    Train a Random Forest pipeline on the cleaned DataFrame.

    Parameters
    ----------
    df            : cleaned DataFrame from data_processing.clean_data()
    test_size     : fraction of data to hold out for evaluation
    random_state  : reproducibility seed
    n_estimators  : number of trees in the Random Forest
    max_depth     : maximum depth of each tree

    Returns
    -------
    (pipeline, metrics_dict)
      pipeline     : fitted sklearn Pipeline
      metrics_dict : dict with MAE, RMSE, R2, MAPE
    """
    # --- Verify required columns exist ---
    missing = [c for c in ALL_FEATURES + ["price"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")

    X = df[ALL_FEATURES].copy()
    y = df["price"].copy()

    # --- Train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # --- Build pipeline ---
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=random_state,
        )),
    ])

    # --- Fit ---
    pipeline.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = pipeline.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100

    metrics = {
        "MAE":  round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2":   round(r2, 4),
        "MAPE": round(mape, 2),
        "train_size": len(X_train),
        "test_size":  len(X_test),
    }

    return pipeline, metrics


# ---------------------------------------------------------------------------
# 3. Save / Load Model
# ---------------------------------------------------------------------------

def save_model(pipeline, path: str = MODEL_PATH) -> None:
    """Serialise the trained pipeline to disk using pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)


def load_model(path: str = MODEL_PATH):
    """Deserialise the trained pipeline from disk. Returns None if not found."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def train_and_save(df: pd.DataFrame, **kwargs) -> tuple:
    """
    Convenience wrapper: train, evaluate, save, and return (pipeline, metrics).
    """
    pipeline, metrics = train_model(df, **kwargs)
    save_model(pipeline)
    return pipeline, metrics


# ---------------------------------------------------------------------------
# 4. Inference
# ---------------------------------------------------------------------------

def predict_fare(pipeline, input_data: dict) -> float:
    """
    Predict the ticket price for a single flight.

    Parameters
    ----------
    pipeline   : fitted sklearn Pipeline (from train_model or load_model)
    input_data : dict with keys matching ALL_FEATURES

    Returns
    -------
    float : predicted price in INR
    """
    df_input = pd.DataFrame([input_data])
    # Ensure all expected columns are present
    for col in ALL_FEATURES:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[ALL_FEATURES]
    prediction = pipeline.predict(df_input)[0]
    return round(float(prediction), 0)


# ---------------------------------------------------------------------------
# 5. Feature Importance Extraction
# ---------------------------------------------------------------------------

def get_feature_importances(pipeline) -> tuple:
    """
    Extract feature names and importances from the trained pipeline.

    Returns
    -------
    (feature_names, importances) : (list[str], list[float])
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor    = pipeline.named_steps["regressor"]

    # Recover feature names after one-hot encoding
    cat_encoder  = preprocessor.named_transformers_["cat"]
    cat_features = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_names    = cat_features + NUMERIC_FEATURES

    importances  = regressor.feature_importances_.tolist()
    return all_names, importances


# ---------------------------------------------------------------------------
# 6. Route Recommendation Insights
# ---------------------------------------------------------------------------

def route_recommendations(df: pd.DataFrame, top_n: int = 5) -> list:
    """
    Generate simple rule-based route recommendations based on
    price-to-duration efficiency.

    Returns a list of dicts with recommendation details.
    """
    # Best value = lowest price per hour of flight for Economy class
    eco = df[df["class"] == "Economy"].copy()
    eco["price_per_hour"] = eco["price"] / eco["duration"]

    route_stats = (
        eco.groupby("route")
        .agg(
            avg_price=("price", "mean"),
            avg_duration=("duration", "mean"),
            avg_pph=("price_per_hour", "mean"),
            flight_count=("price", "count"),
        )
        .reset_index()
    )

    # Only consider routes with enough data
    route_stats = route_stats[route_stats["flight_count"] >= 50]
    best = route_stats.sort_values("avg_pph").head(top_n)

    recommendations = []
    for _, row in best.iterrows():
        recommendations.append({
            "route": row["route"],
            "avg_price": round(row["avg_price"], 0),
            "avg_duration_hrs": round(row["avg_duration"], 2),
            "price_per_hour": round(row["avg_pph"], 0),
            "flight_count": int(row["flight_count"]),
            "insight": (
                f"Best value route: ₹{row['avg_price']:,.0f} avg for "
                f"{row['avg_duration']:.1f}h flight "
                f"(₹{row['avg_pph']:,.0f}/hr)"
            ),
        })
    return recommendations


# ---------------------------------------------------------------------------
# CLI entry point – run to train & save the model
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_processing import run_pipeline

    print("Loading and processing data …")
    results = run_pipeline()
    df = results["df"]

    print(f"Training model on {len(df):,} records …")
    pipeline, metrics = train_and_save(df)

    print("\n✅ Model trained and saved!")
    print(f"   MAE  : ₹{metrics['MAE']:,.2f}")
    print(f"   RMSE : ₹{metrics['RMSE']:,.2f}")
    print(f"   R²   : {metrics['R2']:.4f}")
    print(f"   MAPE : {metrics['MAPE']:.2f}%")
