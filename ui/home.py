from __future__ import annotations

import streamlit as st

from ui.common import format_currency, get_dataset_summary


def render_home_page(dataset) -> None:
    """Render the home page with project information and overview."""

    st.markdown("## Smart Sales Prediction System")
    st.write(
        "A polished forecasting interface that combines advertising budget inputs with a trained sales prediction model."
    )

    st.markdown("### Key Goals")
    st.write(
        "- Estimate sales impact before campaign execution\n"
        "- Surface actionable advertising insights\n"
        "- Provide a clean, modern decision-support dashboard"
    )

    summary = get_dataset_summary(dataset)
    with st.container():
        col1, col2, col3, col4 = st.columns(4, gap="large")
        col1.metric("Rows", summary["rows"], delta="Dataset")
        col2.metric("Columns", summary["columns"], delta="Fields")
        col3.metric("Average Sales", format_currency(summary["avg_sales"]), delta="Mean")
        col4.metric("Highest Sales", format_currency(summary["max_sales"]), delta="Peak")

    st.markdown("### Dataset Overview")
    st.info(
        "The dataset includes advertising budgets for TV, radio, newspaper, and a sales outcome target alongside regional segmentation."
    )

    st.markdown("### Project Workflow")
    workflow_steps = [
        "Data ingestion and cleaning",
        "Feature engineering and encoding",
        "Model inference for sales forecasting",
        "Interactive dashboard and result review",
    ]
    st.write("\n".join(f"- {step}" for step in workflow_steps))

    st.markdown("### Regions Covered")
    st.write(", ".join(summary["regions"]))
