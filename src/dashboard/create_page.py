import streamlit as st
import pandas as pd
import numpy as np
from typing import List
from src.data import get_daily_prices_list
from src.data import pivot_data
from src.viz import create_performance_plot
from typing import Callable


def setup_page(dashboard_config: dict) -> None:
    """Configure the Streamlit page settings."""
    st.set_page_config(
        **dashboard_config["page_config"],
    )
    st.title(dashboard_config["title"])
    # st.markdown(dashboard_config["description"])

    # st.sidebar.header("Controls")
    st.markdown(dashboard_config["style_string"], unsafe_allow_html=True)

    # Transformation selector
    # st.sidebar.header("Transformation Options")
    transformation_option = st.sidebar.selectbox(
        "Weights Transformation",
        ["x", "x²", "x^3", "exp(x)"],
        help="Choose the transformation function for calculating ratios\n",
    )

    # Define transformation functions
    def get_transformation(option: str):
        if option == "x":
            return lambda x: x
        elif option == "x²":
            return lambda x: x**2
        elif option == "x^3":
            return lambda x: x**3
        elif option == "exp(x)":
            return lambda x: np.exp(x)-1
        else:
            return lambda x: x  # default

    transformation = get_transformation(transformation_option)

    # add a link to my page on the sidebar
    st.sidebar.markdown(
        "<a id='homepage-link' href='https://www.noWei.us' target='_blank'>Homepage:  <b><i>noWei.us</i></b></a>",
        unsafe_allow_html=True,
    )
    # link to my linkedin page
    st.sidebar.markdown(
        "<a id='linkedin-link'  href='https://www.linkedin.com/in/weiyang2048/' target='_blank'>LinkedIn: <b><i>weiyang2048</i></b></a>",
        unsafe_allow_html=True,
    )

    return transformation


def show_market_performance(
    equity_config: dict,
    portfolio_config: dict,
    dashboard_config: dict,
) -> None:
    """Function to show the market performance dashboard."""
    transformation = setup_page(dashboard_config)

    # Symbol selection using tabs instead of sidebar radio
    symbol_types = [key for key in portfolio_config.keys()]
    tabs = st.tabs(symbol_types)

    # Create content for each tab
    for i, symbol_type in enumerate(symbol_types):
        with tabs[i]:
            symbols = portfolio_config[symbol_type]
            period = "7y"

            # Load and process data
            df_pivot = pivot_data(list(symbols), period, streamlit=True)

            # Create and display plot
            look_back_days = dashboard_config["look_back_days"]
            colors_dict = {
                symbol: equity_config.get(symbol, {}).get("color", "snow") for symbol in symbols
            }
            line_styles_dict = {
                symbol: equity_config.get(symbol, {}).get("line_style", "solid")
                for symbol in symbols
            }
            fig, df_normalized = create_performance_plot(
                df_pivot,
                symbols,
                look_back_days,
                colors_dict,
                line_styles_dict,
                equity_config,
                transformation,
            )
            st.plotly_chart(
                fig, use_container_width=True, config=dashboard_config["plotly_config"]
            )

            # Display normalized data option
            if st.checkbox("Show Normalized Data", key=f"raw_data_{symbol_type}"):
                st.dataframe(df_normalized)
