from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load the advertising dataset from disk."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def dataset_overview(dataframe: pd.DataFrame) -> dict[str, object]:
    """Return the standard dataset overview requested for the notebook."""

    info_buffer = StringIO()
    dataframe.info(buf=info_buffer)
    return {
        "shape": dataframe.shape,
        "head": dataframe.head(),
        "info": info_buffer.getvalue().strip(),
        "describe": dataframe.describe(include="all").transpose(),
    }
