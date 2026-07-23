from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "Advertising.csv"
MODEL_PATH = ROOT_DIR / "models" / "sales_model.pkl"
RESULTS_PATH = ROOT_DIR / "results" / "model_results.csv"

FEATURE_COLUMNS = ["TV_Ad_Budget", "Radio_Ad_Budget", "Newspaper_Ad_Budget"]
REGION_OPTIONS = ["North", "South", "East", "West"]


def load_dataset() -> pd.DataFrame:
    """Load the advertising dataset from disk."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


def load_model():
    """Load the trained regression model from disk."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def load_model_results() -> pd.DataFrame | None:
    """Load saved model evaluation results if available."""

    if not RESULTS_PATH.exists():
        return None
    return pd.read_csv(RESULTS_PATH)


def get_dataset_summary(dataset: pd.DataFrame) -> dict[str, object]:
    """Build a lightweight summary for the home page."""

    return {
        "rows": int(dataset.shape[0]),
        "columns": int(dataset.shape[1]),
        "avg_sales": round(float(dataset["Sales"].mean()), 2),
        "max_sales": round(float(dataset["Sales"].max()), 2),
        "regions": REGION_OPTIONS,
    }


def format_currency(value: float) -> str:
    """Format a numeric value as USD currency."""

    return f"${value:,.2f}"

