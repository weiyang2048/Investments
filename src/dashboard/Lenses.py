from re import T
from src.configurations.yaml import register_resolvers
import streamlit as st

import hydra
import numpy as np
from src.data import pivot_data
from src.performence.aggregation import aggregate_performance
from src.viz.viz import create_performance_plot, create_momentum_plot, create_momentum_ranking_display
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

# Constants
WINDOW_SIZES = [7, 30, 90, 180, 360]


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


def _create_style_dicts(symbols, equity_config):
    """Create color and line style dictionaries for symbols."""
    colors_dict = {symbol: equity_config.get(symbol, {}).get("color", get_random_style("color")) for symbol in symbols}
    line_styles_dict = {symbol: equity_config.get(symbol, {}).get("line_style", get_random_style("line_style")) for symbol in symbols}
    return colors_dict, line_styles_dict


def _process_symbol_data(symbols, period, streamlit=True):
    """Load and process data for given symbols."""
    return pivot_data(list(symbols), period, streamlit=streamlit)


def _create_melted_dataframe(df_pivot):
    """Create melted dataframe for performance analysis."""
    melt_df = df_pivot.melt(id_vars=["Date"], var_name="Symbol", value_name="Price")
    melt_df.sort_values(by=["Symbol", "Date"], inplace=True, ascending=True)
    return melt_df


def _display_performance_section(melt_df):
    """Display performance statistics with styling."""
    stats_df = aggregate_performance(melt_df)
    styled_stats = stats_df.style.background_gradient(cmap=custom_cmap, axis=0, vmin=-0.2, vmax=0.2, gmap=None).format("{:.2%}")
    display_dataframe(styled_stats, centered=True)


def _process_symbol_tab(
    symbols, symbol_type, look_back_days, equity_config, target_return, show_performance_plot, marchenko_pastur, dfs, momentum_summaries
):
    """Process a single symbol tab - handles both custom and regular symbols."""
    period = f"{look_back_days[-1]}d"

    # Load and process data
    df_pivot = _process_symbol_data(symbols, period)
    dfs[symbol_type] = df_pivot

    # Create style dictionaries
    colors_dict, line_styles_dict = _create_style_dicts(symbols, equity_config)

    # Create momentum plot
    # target_return = target_return + 0.15 if "rh" in symbol_type else target_return

    momentum_result = create_momentum_plot(
        df_pivot,
        symbols,
        window_sizes=WINDOW_SIZES,
        colors_dict=colors_dict,
        line_styles_dict=line_styles_dict,
        equity_config=equity_config,
        target_return=target_return,
    )
    momentum_fig = momentum_result["figure"]
    momentum_combined = momentum_result["momentum_combined"]

    # Display momentum section
    display_section_header("Momentum")
    st.write("target_return", target_return)

    # Create momentum ranking first
    momentum_ranking = _create_momentum_ranking(df_pivot, symbols, equity_config)
    
    momentum_summaries[symbol_type] = momentum_combined
    
    # Add momentum ranking as first row and sort columns
    momentum_combined_with_ranking = _add_momentum_ranking_to_momentum_df(momentum_combined, momentum_ranking)
    display_dataframe(momentum_combined_with_ranking, symbol_type, "Momentum Combined")

    # Display momentum ranking
    display_dataframe(momentum_ranking, symbol_type, "am")

    st.plotly_chart(momentum_fig, config={"displayModeBar": False})

    # Create performance plot
    performance_result = create_performance_plot(
        df_pivot,
        symbols,
        look_back_days,
        colors_dict,
        line_styles_dict,
        equity_config,
    )
    fig = performance_result["figure"]
    df_normalized = performance_result["normalized_data"]

    # Display performance plot if requested
    if show_performance_plot:
        st.plotly_chart(fig, config={"displayModeBar": False})

    # Process data for correlation and performance analysis
    melt_df = _create_melted_dataframe(df_pivot)

    # Display correlation section
    display_section_header("Correlation")
    pivoted_to_corr(df_pivot, plot=True, streamlit=True, marchenko_pastur=marchenko_pastur)

    # Display performance section
    if symbol_type != "Custom Symbols":
        display_section_header("Performance")
    _display_performance_section(melt_df)


def _add_momentum_ranking_to_momentum_df(momentum_df, momentum_ranking):
    """Sort momentum_df columns by momentum_ranking values."""
    # Get the agg_momentum values from momentum_ranking (transposed format)
    if 'am' in momentum_ranking.index:
        ranking_values = momentum_ranking.loc['am']
    else:
        # If not transposed, get from the agg_momentum column
        ranking_values = momentum_ranking.set_index('Symbol')['am']
    
    # Sort columns by the momentum ranking values (descending order)
    sorted_columns = ranking_values.sort_values(ascending=False).index
    return momentum_df[sorted_columns]


def _create_momentum_ranking(df_pivot, symbols, equity_config):
    """Create momentum ranking display for given symbols."""
    return create_momentum_ranking_display(
        df_pivot,
        symbols,
        window_sizes=WINDOW_SIZES,
        equity_config=equity_config,
    )


def _get_momentum_ranking_for_symbol_type(symbol_type, dfs, equity_config):
    """Get momentum ranking for a symbol type if it exists in dfs."""
    if symbol_type in dfs.keys():
        df_pivot = dfs[symbol_type].copy()
        symbols = [col for col in df_pivot.columns if col != "Date"]
        return _create_momentum_ranking(df_pivot, symbols, equity_config)
    return None


def _display_summary_tab(momentum_summaries, dfs, marchenko_pastur, equity_config):
    """Display the summary tab with all momentum summaries and correlations."""
    # Display momentum summaries
    for symbol_type in momentum_summaries.keys():
        momentum_df = momentum_summaries[symbol_type]
        momentum_ranking = _get_momentum_ranking_for_symbol_type(symbol_type, dfs, equity_config)
        
        if momentum_ranking is not None:
            # Sort columns by momentum ranking
            momentum_df_sorted = _add_momentum_ranking_to_momentum_df(momentum_df, momentum_ranking)
            display_dataframe(momentum_df_sorted, symbol_type, "Momentum Combined")
            display_dataframe(momentum_ranking, symbol_type, "Momentum Ranking")
        else:
            display_dataframe(momentum_df, symbol_type, "Momentum Combined")

    # Display correlations
    for symbol_type in dfs.keys():
        df_pivot = dfs[symbol_type].copy()
        pivoted_to_corr(df_pivot, plot=True, streamlit=True, marchenko_pastur=marchenko_pastur)


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
                _display_summary_tab(momentum_summaries, dfs, marchenko_pastur, equity_config)
            elif symbol_type == "Custom Symbols":
                if custom_symbols:
                    _process_symbol_tab(
                        custom_symbols,
                        symbol_type,
                        look_back_days,
                        equity_config,
                        target_return,
                        show_performance_plot,
                        marchenko_pastur,
                        dfs,
                        momentum_summaries,
                    )
                else:
                    st.info("No custom symbols provided. Please enter symbols in the sidebar.")
            else:
                symbols = portfolio_config[symbol_type]
                _process_symbol_tab(
                    symbols,
                    symbol_type,
                    look_back_days,
                    equity_config,
                    target_return,
                    show_performance_plot,
                    marchenko_pastur,
                    dfs,
                    momentum_summaries,
                )

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
