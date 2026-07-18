from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from common import clean_dataset, evaluate_predictions, load_dataset, split_features_target


def main() -> None:
    """Evaluate the saved best model on the held-out test split."""

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "Advertising.csv"
    model_path = project_root / "models" / "sales_model.pkl"

    try:
        raw_data = load_dataset(dataset_path)
        cleaned_data = clean_dataset(raw_data)
        features, target = split_features_target(cleaned_data)

        _, x_test, _, y_test = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=42,
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Saved model not found at {model_path}. Run src/train_model.py first."
            )

        model = joblib.load(model_path)
        predictions = model.predict(x_test)
        metrics = evaluate_predictions(y_test, predictions)

        print("Best model evaluation:")
        print(pd.DataFrame([metrics]).to_string(index=False))
    except Exception as exc:
        raise SystemExit(f"Evaluation failed: {exc}") from exc


if __name__ == "__main__":
    main()
