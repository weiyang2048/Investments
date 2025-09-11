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

    # lense_option = st.selectbox(
    #     "Lense",
    #     ["main", "sectoral", "zoo"],
    #     help="Choose the lense to display\n",
    #     key="lense_option",
    # )
    col1, col2 = st.columns(2)

    with col1:
        initial_lookback_days = st.number_input(
            "Initial Lookback Days",
            min_value=1,
            max_value=3650,
            value=5,
            step=1,
            help="Ingrese el número inicial de días para el período de análisis. (Español: 'días de retroceso') / Entrez le nombre initial de jours pour la période d'analyse. (Français: 'jours de retour en arrière')",
            key="lookback_days_input",
        )
    with col2:
        lookback_factor = st.number_input(
            "Lookback Factor",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Ingrese el factor para multiplicar los días de retroceso. (Español: 'factor') / Entrez le facteur pour multiplier les jours de retour en arrière. (Français: 'facteur')",
            key="lookback_factor_input",
        )

    return transformation, initial_lookback_days, lookback_factor


def show_market_performance(
    equity_config: dict,
    portfolio_config: dict,
    transformation: Callable[[float], float] = lambda x: np.exp(x),
    initial_lookback_days: int = 5,
    lookback_factor: int = 3,
) -> None:
    """Function to show the market performance dashboard."""
    # transformation = sidebar()

    # Symbol selection using tabs instead of sidebar radio
    symbol_types = [key for key in portfolio_config.keys()]
    tabs = st.tabs(symbol_types)

    # Create content for each tab
    # Prompt user for initial lookback days and a factor to multiply

    # Generate look_back_days list based on user input
    look_back_days = [int(initial_lookback_days * (lookback_factor**i)) for i in range(6)]
    for i, symbol_type in enumerate(symbol_types):
        with tabs[i]:
            symbols = portfolio_config[symbol_type]
            period = f"{look_back_days[-1]}d"

            # Load and process data
            df_pivot = pivot_data(list(symbols), period, streamlit=True)

            # Create and display plot
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
    transformation, initial_lookback_days, lookback_factor = setup_page_and_sidebar(config["style_conf"], add_to_sidebar=sidebar)
    lense_option = "main"
    st.title(lense_option)

    show_market_performance(config["tickers"], config["lenses"][lense_option], transformation, initial_lookback_days, lookback_factor)

    if "weiya" in os.path.expanduser("~"):
        config_dict = OmegaConf.to_container(config, resolve=True)
        config_json = json.dumps(config_dict, indent=2)
        st.subheader("Current Configuration (JSON)")
        st.json(config_json)
