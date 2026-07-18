from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

NUMERIC_COLUMNS = ["TV_Ad_Budget", "Radio_Ad_Budget", "Newspaper_Ad_Budget"]
CATEGORICAL_COLUMNS = ["Region"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
TARGET_COLUMN = "Sales"


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load the advertising dataset from disk."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def clean_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataset and standardize model inputs."""

    cleaned = dataframe.copy()

    if "Campaign_ID" in cleaned.columns:
        cleaned = cleaned.drop(columns=["Campaign_ID"])

    if "Newspaper_Ad_Budget" in cleaned.columns:
        cleaned["Newspaper_Ad_Budget"] = (
            cleaned["Newspaper_Ad_Budget"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace("USD", "", regex=False)
            .str.strip()
        )

    for column in NUMERIC_COLUMNS + [TARGET_COLUMN]:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if "Region" in cleaned.columns:
        cleaned["Region"] = cleaned["Region"].astype(str).str.strip().str.title()
        cleaned.loc[cleaned["Region"].isin(["Nan", "None", ""]), "Region"] = pd.NA

    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def split_features_target(dataframe: pd.DataFrame):
    """Separate model features from the target variable."""

    missing_columns = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in dataframe.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    features = dataframe[FEATURE_COLUMNS].copy()
    target = dataframe[TARGET_COLUMN].copy()
    return features, target


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing block shared by every model."""

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_COLUMNS),
            ("categorical", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )


def build_model_pipelines(random_state: int = 42) -> dict[str, Pipeline]:
    """Create the model pipelines required by the project."""

    return {
        "Linear Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", LinearRegression()),
            ]
        ),
        "Decision Tree Regressor": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", DecisionTreeRegressor(random_state=random_state)),
            ]
        ),
        "Random Forest Regressor": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", RandomForestRegressor(random_state=random_state, n_estimators=200)),
            ]
        ),
    }


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    """Compute the required regression metrics."""

    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": mse ** 0.5,
        "R2_Score": r2_score(y_true, y_pred),
    }


def train_and_evaluate_models(
    features,
    target,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Train all requested models and return the fitted pipelines and metrics."""

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    fitted_models: dict[str, Pipeline] = {}
    result_rows: list[dict[str, Any]] = []

    for model_name, pipeline in build_model_pipelines(random_state=random_state).items():
        fitted_pipeline = pipeline.fit(x_train, y_train)
        predictions = fitted_pipeline.predict(x_test)
        metrics = evaluate_predictions(y_test, predictions)

        fitted_models[model_name] = fitted_pipeline
        result_rows.append({"Model": model_name, **metrics})

    results = pd.DataFrame(result_rows)
    return fitted_models, results, x_train, x_test, y_train, y_test


def choose_best_model(results: pd.DataFrame) -> str:
    """Choose the best model using the lowest RMSE, then highest R2 score."""

    if results.empty:
        raise ValueError("No model results were provided.")

    sorted_results = results.sort_values(by=["RMSE", "MAE", "MSE", "R2_Score"], ascending=[True, True, True, False])
    return str(sorted_results.iloc[0]["Model"])


def save_model(model: Pipeline, file_path: str | Path) -> None:
    """Persist a fitted pipeline to disk."""

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_results(results: pd.DataFrame, file_path: str | Path) -> None:
    """Persist the evaluation table to disk."""

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)
