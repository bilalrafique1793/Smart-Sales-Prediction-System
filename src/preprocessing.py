from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

NUMERIC_COLUMNS = ["TV_Ad_Budget", "Radio_Ad_Budget", "Newspaper_Ad_Budget", "Sales"]
TEXT_COLUMNS = ["Region"]
DROP_COLUMNS = ["Campaign_ID"]


def _clean_text_value(value: object) -> object:
    if pd.isna(value):
        return pd.NA

    text = str(value).replace("USD", "").replace("$", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else pd.NA


def _standardize_region(value: object) -> object:
    cleaned_value = _clean_text_value(value)
    if pd.isna(cleaned_value):
        return pd.NA
    return str(cleaned_value).title()


def clean_dataset(dataframe: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Clean the raw dataset and standardize column types."""

    cleaned = dataframe.copy()

    for column in DROP_COLUMNS:
        if column in cleaned.columns:
            cleaned = cleaned.drop(columns=column)

    for column in cleaned.select_dtypes(include="object").columns:
        cleaned[column] = cleaned[column].map(_clean_text_value)

    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").astype(float)

    if "Region" in cleaned.columns:
        cleaned["Region"] = cleaned["Region"].map(_standardize_region)

    if "Sales" in cleaned.columns:
        cleaned = cleaned.dropna(subset=["Sales"])

    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()

    return cleaned.reset_index(drop=True)


def dataset_quality_report(dataframe: pd.DataFrame) -> dict[str, object]:
    """Summarize missing values, duplicates, and data-type issues."""

    expected_numeric = [column for column in NUMERIC_COLUMNS if column in dataframe.columns]
    expected_text = [column for column in TEXT_COLUMNS if column in dataframe.columns]

    incorrect_numeric_dtypes = {
        column: str(dataframe[column].dtype)
        for column in expected_numeric
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    }
    incorrect_text_dtypes = {
        column: str(dataframe[column].dtype)
        for column in expected_text
        if pd.api.types.is_numeric_dtype(dataframe[column])
    }

    return {
        "missing_values": dataframe.isna().sum(),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "incorrect_dtypes": {**incorrect_numeric_dtypes, **incorrect_text_dtypes},
        "dtypes": dataframe.dtypes,
    }


def save_cleaned_dataset(dataframe: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist the cleaned dataset to disk."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    return path
