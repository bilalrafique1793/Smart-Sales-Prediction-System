from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui.common import load_dataset
from ui.dashboard import render_dashboard_page
from ui.home import render_home_page
from ui.prediction import render_prediction_page

ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "assets"


def apply_custom_css() -> None:
    """Load and inject the custom stylesheet for a polished UI."""

    css_path = ASSETS_DIR / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_sidebar() -> str:
    """Create the sidebar navigation for the app."""

    st.sidebar.title("Smart Sales AI")
    st.sidebar.caption("Forecast sales from advertising campaigns")

    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Prediction", "Dashboard"],
        index=0,
        key="page_nav",
    )

    st.sidebar.markdown("---")
    st.sidebar.info("Upload-ready UI for the trained sales model.")
    return page


def main() -> None:
    """Run the Streamlit application."""

    st.set_page_config(page_title="Smart Sales Prediction", page_icon="📈", layout="wide")
    apply_custom_css()

    try:
        dataset = load_dataset()
        if dataset.empty:
            st.error("The advertising dataset does not contain any rows.")
            return
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    page = render_sidebar()

    if page == "Home":
        render_home_page(dataset)
    elif page == "Prediction":
        render_prediction_page()
    else:
        render_dashboard_page(dataset)


if __name__ == "__main__":
    main()
