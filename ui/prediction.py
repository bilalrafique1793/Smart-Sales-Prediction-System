from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.common import FEATURE_COLUMNS, REGION_OPTIONS, format_currency, load_model


def render_prediction_page() -> None:
    """Render the prediction form and model output section."""

    st.markdown("## Predict Sales")
    st.write(
        "Enter campaign budgets and select the target region to estimate expected sales using the trained model."
    )

    with st.form("prediction_form", clear_on_submit=False):
        left_col, right_col = st.columns(2)
        with left_col:
            tv_budget = st.number_input(
                "TV Advertisement Budget",
                min_value=0.0,
                value=25000.0,
                step=500.0,
                format="%.2f",
            )
            radio_budget = st.number_input(
                "Radio Advertisement Budget",
                min_value=0.0,
                value=12000.0,
                step=500.0,
                format="%.2f",
            )
        with right_col:
            newspaper_budget = st.number_input(
                "Newspaper Advertisement Budget",
                min_value=0.0,
                value=5000.0,
                step=500.0,
                format="%.2f",
            )
            region = st.selectbox("Region", REGION_OPTIONS)

        submitted = st.form_submit_button("Predict Sales")

    if submitted:
        invalid_fields = [
            name
            for name, value in [
                ("TV Advertisement Budget", tv_budget),
                ("Radio Advertisement Budget", radio_budget),
                ("Newspaper Advertisement Budget", newspaper_budget),
            ]
            if value <= 0
        ]

        if invalid_fields:
            st.error(f"Please enter positive values for: {', '.join(invalid_fields)}")
            return

        try:
            model = load_model()
        except FileNotFoundError as exc:
            st.error(str(exc))
            return

        input_frame = pd.DataFrame(
            [
                {
                    "TV_Ad_Budget": tv_budget,
                    "Radio_Ad_Budget": radio_budget,
                    "Newspaper_Ad_Budget": newspaper_budget,
                    "Region": region,
                }
            ]
        )
        prediction = model.predict(input_frame[FEATURE_COLUMNS + ["Region"]])[0]

        st.success("Prediction completed successfully")
        st.metric("Predicted Sales", format_currency(float(prediction)))
        st.info(
            "The app uses a saved regression pipeline loaded from models/sales_model.pkl."
        )
