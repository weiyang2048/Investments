from re import T
from src.configurations.yaml import register_resolvers
import streamlit as st

import hydra
import numpy as np
from src.data import pivot_data
from src.performence.aggregation import aggregate_performance
from src.viz.viz import create_performance_plot, create_momentum_plot
from src.configurations.style_picker import get_random_style
from src.dashboard.create_page import setup_page_and_sidebar
import pandas as pd
from src.association import pivoted_to_corr
from src.viz.cmaps import custom_cmap
from src.viz.streamlit_display import (
    display_dataframe,
    display_table_of_contents,
    display_section_header,
)


def parse_custom_symbols(symbols_text):
    """Parse comma-separated symbols and clean them."""
    if not symbols_text or symbols_text.strip() == "":
        return []

    # Split by comma and clean each symbol
    symbols = [symbol.strip().upper() for symbol in symbols_text.split(",")]
    # Remove empty strings
    symbols = [symbol for symbol in symbols if symbol]
    return symbols


def sidebar(config):
    lenses = config["lenses"]
    lense_option = st.sidebar.radio(
        "Lense",
        lenses.keys(),
        help="Choose the lense to display\n",
        key="lense_option",
    )

    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("Custom Symbols")
    custom_symbols_text = st.sidebar.text_input(
        "Custom Symbols",
        placeholder="Enter symbols separated by commas (e.g., AAPL, MSFT, GOOGL)",
        help="Enter custom symbols separated by commas to analyze alongside the selected lens",
        key="custom_symbols_input",
    )
    custom_symbols = parse_custom_symbols(custom_symbols_text)

    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    target_return = st.sidebar.slider(
        "Target Return",
        min_value=1.1,
        max_value=2.5,
        value=config["target_return"],
        step=0.05,
        help="Target annualized return for momentum threshold calculation.",
        key="target_return_input",
    )
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("Correlation Denoising")
    marchenko_pastur = st.sidebar.checkbox(
        "Marchenko Pastur",
        value=True,
        key="marchenko_pastur_input",
    )

    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("Prices")
    show_performance_plot = st.sidebar.checkbox(
        "Show Prices",
        value=False,
        key="show_performance_plot_input",
    )
    initial_lookback_days = st.sidebar.number_input(
        "Initial Lookback Days",
        min_value=1,
        max_value=3650,
        value=7,
        step=1,
        help="Enter the initial number of days for the analysis period.",
        key="lookback_days_input",
    )
    lookback_factor = st.sidebar.number_input(
        "Lookback Factor",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Enter the factor to multiply the lookback days.",
        key="lookback_factor_input",
    )
    return marchenko_pastur, initial_lookback_days, lookback_factor, lense_option, target_return, show_performance_plot, custom_symbols


def show_market_performance(
    equity_config: dict,
    portfolio_config: dict,
    marchenko_pastur: bool = True,
    initial_lookback_days: int = 5,
    lookback_factor: int = 3,
    target_return: float = 1.4,
    show_performance_plot: bool = True,
    custom_symbols: list = None,
) -> None:
    """Function to show the market performance dashboard."""
    # transformation = sidebar()

    # Symbol selection using tabs instead of sidebar radio
    symbol_types = [key for key in portfolio_config.keys()]

    # Add custom symbols tab if custom symbols are provided
    if custom_symbols:
        symbol_types = symbol_types + ["Custom Symbols"]

    symbol_types = symbol_types + ["Summary"]
    tabs = st.tabs(symbol_types)

    # Generate look_back_days list based on user input
    look_back_days = [int(initial_lookback_days * (lookback_factor**i)) for i in range(6)]
    momentum_summaries = dict()
    dfs = dict()
    for i, symbol_type in enumerate(symbol_types):
        with tabs[i]:
            if symbol_type == "Summary":

                # Display momentum summaries
                for idx, symbol_type in enumerate(momentum_summaries.keys()):
                    momentum_df = momentum_summaries[symbol_type]
                    display_dataframe(momentum_df, symbol_type, "Momentum Combined")
                for symbol_type in dfs.keys():
                    df_pivot = dfs[symbol_type].copy()
                    pivoted_to_corr(df_pivot, plot=True, streamlit=True, marchenko_pastur=marchenko_pastur)
            elif symbol_type == "Custom Symbols":
                if custom_symbols:
                    symbols = custom_symbols
                    period = f"{look_back_days[-1]}d"

                    # Load and process data
                    df_pivot = pivot_data(list(symbols), period, streamlit=True)
                    dfs[symbol_type] = df_pivot
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
                    )
                    momentum_fig, momentum_combined = create_momentum_plot(
                        df_pivot,
                        symbols,
                        window_sizes=[7, 30, 90, 180, 360],
                        colors_dict=colors_dict,
                        line_styles_dict=line_styles_dict,
                        equity_config=equity_config,
                        target_return=target_return,
                    )

                    if show_performance_plot:
                        st.plotly_chart(fig, config={"displayModeBar": False})

                    display_section_header("Momentum")
                    momentum_summaries[symbol_type] = momentum_combined

                    display_dataframe(
                        momentum_combined,
                        symbol_type,
                        "Momentum Combined",
                    )

                    st.plotly_chart(momentum_fig, config={"displayModeBar": False})

                    # @ aggregations
                    melt_df = df_pivot.melt(id_vars=["Date"], var_name="Symbol", value_name="Price")
                    melt_df.sort_values(by=["Symbol", "Date"], inplace=True, ascending=True)

                    display_section_header("Correlation")
                    pivoted_to_corr(df_pivot, plot=True, streamlit=True, marchenko_pastur=marchenko_pastur)

                    stats_df = aggregate_performance(melt_df)
                    styled_stats = stats_df.style.background_gradient(cmap=custom_cmap, axis=0, vmin=-0.2, vmax=0.2, gmap=None).format("{:.2%}")
                    display_dataframe(styled_stats, centered=True)
                else:
                    st.info("No custom symbols provided. Please enter symbols in the sidebar.")
            else:
                symbols = portfolio_config[symbol_type]
                period = f"{look_back_days[-1]}d"

                # Load and process data
                df_pivot = pivot_data(list(symbols), period, streamlit=True)
                dfs[symbol_type] = df_pivot
                # Create and display plot
                colors_dict = {symbol: equity_config.get(symbol, {}).get("color", get_random_style("color")) for symbol in symbols}
                line_styles_dict = {symbol: equity_config.get(symbol, {}).get("line_style", get_random_style("line_style")) for symbol in symbols}

                momentum_fig, momentum_combined = create_momentum_plot(
                    df_pivot,
                    symbols,
                    window_sizes=[7, 30, 90, 180, 360],
                    colors_dict=colors_dict,
                    line_styles_dict=line_styles_dict,
                    equity_config=equity_config,
                    target_return=target_return,
                )

                display_section_header("Momentum")
                momentum_summaries[symbol_type] = momentum_combined

                display_dataframe(
                    momentum_combined,
                    symbol_type,
                    "Momentum Combined",
                )

                st.plotly_chart(momentum_fig, config={"displayModeBar": False})

                # @ aggregations
                melt_df = df_pivot.melt(id_vars=["Date"], var_name="Symbol", value_name="Price")
                melt_df.sort_values(by=["Symbol", "Date"], inplace=True, ascending=True)

                display_section_header("Correlation")
                pivoted_to_corr(df_pivot, plot=True, streamlit=True, marchenko_pastur=marchenko_pastur)

                display_section_header("Performance")
                stats_df = aggregate_performance(melt_df)
                styled_stats = stats_df.style.background_gradient(cmap=custom_cmap, axis=0, vmin=-0.2, vmax=0.2, gmap=None).format("{:.2%}")
                display_dataframe(styled_stats, centered=True)

                fig, df_normalized = create_performance_plot(
                    df_pivot,
                    symbols,
                    look_back_days,
                    colors_dict,
                    line_styles_dict,
                    equity_config,
                )

                if show_performance_plot:
                    st.plotly_chart(fig, config={"displayModeBar": False})

    # Bottom Table of Contents
    display_table_of_contents(
        sections=[
            "Momentum",
            "Correlation",
            "Performance",
        ]
    )


if __name__ == "__main__":

    if hydra.core.global_hydra.GlobalHydra().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_resolvers()

    with hydra.initialize(version_base=None, config_path="../../conf"):
        config = hydra.compose(config_name="main")
    marchenko_pastur, initial_lookback_days, lookback_factor, lense_option, target_return, show_performance_plot, custom_symbols = (
        setup_page_and_sidebar(config["style_conf"], add_to_sidebar=lambda: sidebar(config))
    )
    st.title(lense_option)

    # Table of Contents
    display_table_of_contents(
        sections=[
            "Momentum",
            "Correlation",
            "Performance",
        ]
    )

    show_market_performance(
        config["tickers"],
        config["lenses"][lense_option],
        marchenko_pastur,
        initial_lookback_days,
        lookback_factor,
        target_return,
        show_performance_plot,
        custom_symbols,
    )
