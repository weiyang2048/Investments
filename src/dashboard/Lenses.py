from src.configurations.yaml import register_resolvers
import streamlit as st
from typing import Callable
from loguru import logger
from omegaconf import OmegaConf

# from src.dashboard.create_page import setup_page, show_market_performance
import hydra
import os
import json
import numpy as np
from src.data import pivot_data
from src.viz.viz import create_performance_plot
from src.configurations.style_picker import get_random_style
from src.dashboard.create_page import setup_page_and_sidebar


def sidebar():

    # @ lense selector
    lense_option = st.sidebar.selectbox(
        "Lense",
        ["main", "sectoral", "zoo"],
        help="Choose the lense to display\n",
    )

    # @ Transformation selector
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
            return lambda x: np.exp(x) - 1
        else:
            return lambda x: x  # default

    transformation = get_transformation(transformation_option)

    return transformation, lense_option


def show_market_performance(
    equity_config: dict,
    portfolio_config: dict,
    dashboard_config: dict,
    transformation: Callable[[float], float] = lambda x: np.exp(x),
) -> None:
    """Function to show the market performance dashboard."""
    # transformation = sidebar()

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
            colors_dict = {symbol: equity_config.get(symbol, {}).get("color", get_random_style("color")) for symbol in symbols}
            line_styles_dict = {symbol: equity_config.get(symbol, {}).get("line_style", get_random_style("line_style")) for symbol in symbols}
            fig, df_normalized = create_performance_plot(
                df_pivot,
                symbols,
                look_back_days,
                colors_dict,
                line_styles_dict,
                equity_config,
                transformation,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Display normalized data option
            # if st.checkbox("Show Normalized Data", key=f"raw_data_{symbol_type}"):
            st.dataframe(df_pivot)


if __name__ == "__main__":

    if hydra.core.global_hydra.GlobalHydra().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_resolvers()

    with hydra.initialize(version_base=None, config_path="../../conf"):
        config = hydra.compose(
            config_name="main",
            # overrides=[
            #     # "+style_conf=main",
            # "portfolio=regions",
            # "~tickers.insurance_stocks",
            # ],
        )
    transformation, lense_option = setup_page_and_sidebar(config["style_conf"], add_to_sidebar=sidebar)
    st.title(lense_option)

    # OmegaConf.resolve(config)

    show_market_performance(config["tickers"], config["lenses"][lense_option], config["style_conf"], transformation)

    if "weiya" in os.path.expanduser("~"):
        config_dict = OmegaConf.to_container(config, resolve=True)
        config_json = json.dumps(config_dict, indent=2)
        st.subheader("Current Configuration (JSON)")
        st.json(config_json)
