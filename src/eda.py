from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Markdown

NUMERIC_COLUMNS = ["TV_Ad_Budget", "Radio_Ad_Budget", "Newspaper_Ad_Budget", "Sales"]


def _prepare_numeric_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    available_columns = [column for column in NUMERIC_COLUMNS if column in dataframe.columns]
    return dataframe[available_columns].dropna()


def _save_current_figure(file_path: Path) -> Path:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    return file_path


def create_eda_plots(dataframe: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Create and save all requested EDA plots."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated_files: dict[str, Path] = {}
    numeric_frame = _prepare_numeric_frame(dataframe)

    sns.set_theme(style="whitegrid", context="notebook")

    plt.figure(figsize=(12, 8))
    numeric_frame.hist(bins=20, edgecolor="black", layout=(2, 2))
    generated_files["histogram"] = _save_current_figure(output_path / "histogram.png")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=numeric_frame, orient="h", palette="Set2")
    plt.title("Boxplot of Numeric Advertising Metrics")
    generated_files["boxplot"] = _save_current_figure(output_path / "boxplot.png")

    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_frame.corr(), annot=True, cmap="YlGnBu", fmt=".2f", square=True)
    plt.title("Correlation Heatmap")
    generated_files["correlation_heatmap"] = _save_current_figure(output_path / "correlation_heatmap.png")

    pairplot = sns.pairplot(numeric_frame, diag_kind="hist", corner=True)
    pairplot.fig.suptitle("Pairplot", y=1.02)
    pairplot.fig.savefig(output_path / "pairplot.png", dpi=300, bbox_inches="tight")
    plt.close(pairplot.fig)
    generated_files["pairplot"] = output_path / "pairplot.png"

    plt.figure(figsize=(10, 6))
    sns.histplot(numeric_frame["Sales"], kde=True, color="#2a6f97")
    plt.title("Sales Distribution")
    plt.xlabel("Sales")
    generated_files["sales_distribution"] = _save_current_figure(output_path / "sales_distribution.png")

    scatter_specs = [
        ("TV_Ad_Budget", "TV vs Sales", "tv_vs_sales.png", "#0b6e4f"),
        ("Radio_Ad_Budget", "Radio vs Sales", "radio_vs_sales.png", "#c77dff"),
        ("Newspaper_Ad_Budget", "Newspaper vs Sales", "newspaper_vs_sales.png", "#f4a261"),
    ]
    for column, title, filename, color in scatter_specs:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=dataframe, x=column, y="Sales", color=color)
        plt.title(title)
        generated_files[column.lower()] = _save_current_figure(output_path / filename)

    if "Region" in dataframe.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=dataframe, x="Region", y="Sales", color="#9ecae1")
        plt.title("Region vs Sales")
        generated_files["region_vs_sales"] = _save_current_figure(output_path / "region_vs_sales.png")

    return generated_files


def generate_insights_markdown(dataframe: pd.DataFrame) -> Markdown:
    """Summarize the strongest observations as markdown."""

    numeric_frame = _prepare_numeric_frame(dataframe)
    correlation = numeric_frame.corr(numeric_only=True)["Sales"].drop("Sales", errors="ignore")
    strongest_driver = correlation.abs().idxmax() if not correlation.empty else "N/A"
    strongest_corr = correlation.loc[strongest_driver] if strongest_driver != "N/A" else 0.0

    region_summary = (
        dataframe.groupby("Region", dropna=True)["Sales"].mean().sort_values(ascending=False)
        if "Region" in dataframe.columns and "Sales" in dataframe.columns
        else pd.Series(dtype=float)
    )
    top_region = region_summary.index[0] if not region_summary.empty else "N/A"
    top_region_sales = region_summary.iloc[0] if not region_summary.empty else 0.0

    insights = [
        "## Key Insights",
        "",
        f"- **{strongest_driver}** has the strongest relationship with sales at **{strongest_corr:.2f}**.",
        f"- **{top_region}** has the highest average sales at **{top_region_sales:.2f}**.",
        "- The sales distribution should be checked for skewness and outliers because advertising response is rarely perfectly symmetric.",
        "- Compare the scatterplots to see whether budget increases appear to produce diminishing returns.",
    ]
    return Markdown("\n".join(insights))
