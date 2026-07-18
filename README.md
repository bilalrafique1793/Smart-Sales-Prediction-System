# Smart Sales Prediction System - Machine Learning Module

This module handles only model development and evaluation for the Smart Sales Prediction System.

## What it does

- Loads `Advertising.csv`
- Removes `Campaign_ID`
- Cleans numeric and categorical fields
- One-hot encodes `Region`
- Trains these models:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- Evaluates each model with:
  - MAE
  - MSE
  - RMSE
  - R² Score
- Saves the best fitted model to `models/sales_model.pkl`
- Saves model comparison metrics to `results/model_results.csv`

## Files

- `src/common.py` - reusable data cleaning, preprocessing, training, and saving helpers
- `src/train_model.py` - trains all candidate models and saves the best one
- `src/evaluate_model.py` - evaluates the saved best model on the test split

## Requirements

Install the following Python packages if they are not already available:

- `pandas`
- `scikit-learn`
- `joblib`

## Run

Train models and generate outputs:

```bash
python src/train_model.py
```

Evaluate the saved best model:

```bash
python src/evaluate_model.py
```

## Notes

- Missing target values are removed before training.
- Missing feature values are handled through median or most-frequent imputation inside the pipeline.
- The saved model includes preprocessing, so it can be used directly for prediction.
