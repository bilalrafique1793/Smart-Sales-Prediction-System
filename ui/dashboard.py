from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from ui.common import load_model_results


def render_dashboard_page(dataset) -> None:
    """Render the dashboard with preview, summary, charts, and model accuracy."""

    st.markdown("## Dashboard")
    st.write(
        "Review the dataset, sales distribution, and model performance in a single place."
    )

    with st.expander("Dataset preview"):
        st.dataframe(dataset.head(12), use_container_width=True)

    with st.expander("Statistics summary"):
        st.dataframe(dataset.describe(), use_container_width=True)

    st.markdown("### Sales distribution and advertising trends")
    chart_cols = st.columns(2, gap="large")
    with chart_cols[0]:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(dataset["Sales"], bins=12, kde=True, color="#38bdf8", ax=ax)
        ax.set_title("Sales Distribution")
        ax.set_xlabel("Sales")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)

    with chart_cols[1]:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(data=dataset, x="TV_Ad_Budget", y="Sales", hue="Region", palette="bright", ax=ax)
        ax.set_title("TV Spend vs Sales")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("### Regional performance")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=dataset, x="Region", y="Sales", estimator="mean", palette="rocket", ax=ax)
    ax.set_title("Average Sales by Region")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Model accuracy")
    results = load_model_results()
    if results is None or results.empty:
        st.warning("No model metrics are available yet. Ensure models/results.csv exists.")
        return

    metric_cols = st.columns(3)
    best_model = results.loc[results["RMSE"].idxmin()]
    metric_cols[0].metric("Best Model", best_model["Model"])
    metric_cols[1].metric("RMSE", f"{best_model['RMSE']:.2f}")
    metric_cols[2].metric("R² Score", f"{best_model['R2_Score']:.2f}")

    with st.expander("Full model evaluation table"):
        st.dataframe(results, use_container_width=True)
