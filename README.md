# Smart Sales Prediction System - Data Analysis Module

This module covers the data analysis workflow for the Smart Sales Prediction System using `Advertising.csv`.

## What it does

- Loads the dataset with Pandas.
- Displays the shape, head, info, and descriptive statistics.
- Removes `Campaign_ID`.
- Cleans currency markers, extra spaces, and inconsistent region values.
- Converts advertising budgets and sales to numeric floats.
- Checks missing values, duplicate rows, and incorrect data types.
- Saves the cleaned dataset to `data/cleaned_advertising.csv`.
- Generates exploratory data analysis charts in `outputs/graphs/`.
- Summarizes the main insights in markdown inside the notebook.

## Files

- `src/data_loader.py` - dataset loading and overview helpers
- `src/preprocessing.py` - cleaning, quality checks, and cleaned-data export
- `src/eda.py` - chart generation and markdown insight helpers
- `sales_analysis.ipynb` - walkthrough notebook for the full analysis

## Requirements

Install these packages if they are not already available:

- `pandas`
- `matplotlib`
- `seaborn`
- `ipython`
- `jupyter`

## Run

Open the notebook and run all cells:

```bash
jupyter notebook sales_analysis.ipynb
```

Or run the analysis logic from a Python session using the functions in `src/`.

## Outputs

- `data/cleaned_advertising.csv`
- `outputs/graphs/histogram.png`
- `outputs/graphs/boxplot.png`
- `outputs/graphs/correlation_heatmap.png`
- `outputs/graphs/pairplot.png`
- `outputs/graphs/sales_distribution.png`
- `outputs/graphs/tv_vs_sales.png`
- `outputs/graphs/radio_vs_sales.png`
- `outputs/graphs/newspaper_vs_sales.png`
- `outputs/graphs/region_vs_sales.png`
