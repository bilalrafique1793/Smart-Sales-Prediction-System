from __future__ import annotations

from pathlib import Path

from common import (
    clean_dataset,
    choose_best_model,
    load_dataset,
    save_model,
    save_results,
    split_features_target,
    train_and_evaluate_models,
)


def main() -> None:
    """Train the candidate models, evaluate them, and save the best one."""

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "Advertising.csv"
    model_path = project_root / "models" / "sales_model.pkl"
    results_path = project_root / "results" / "model_results.csv"

    try:
        raw_data = load_dataset(dataset_path)
        cleaned_data = clean_dataset(raw_data)
        features, target = split_features_target(cleaned_data)

        fitted_models, results, *_ = train_and_evaluate_models(features, target)
        best_model_name = choose_best_model(results)
        best_model = fitted_models[best_model_name]

        save_model(best_model, model_path)
        save_results(results, results_path)

        print("Model comparison:")
        print(results.to_string(index=False))
        print()
        print(f"Best model: {best_model_name}")
        print(f"Saved model to: {model_path}")
        print(f"Saved metrics to: {results_path}")
    except Exception as exc:
        raise SystemExit(f"Training failed: {exc}") from exc


if __name__ == "__main__":
    main()
