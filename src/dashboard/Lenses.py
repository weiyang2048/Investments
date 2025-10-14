from src.configurations.yaml import register_resolvers
import streamlit as st

import hydra
import numpy as np
from src.data import pivot_data
from src.viz.viz import create_performance_plot, create_momentum_plot, create_momentum_ranking_display
from src.configurations.style_picker import get_random_style
from src.dashboard.create_page import setup_page_and_sidebar
import pandas as pd
from src.association import pivoted_to_corr
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
    st.sidebar.markdown("Plots")
    show_performance_plot = st.sidebar.checkbox(
        "Show Prices",
        value=False,
        key="show_performance_plot_input",
    )
    show_momentum_plot = st.sidebar.checkbox(
        "Show Momentum Plot",
        value=False,
        key="show_momentum_plot_input",
    )
    show_correlation_plot = st.sidebar.checkbox(
        "Show Correlation Plot",
        value=False,
        key="show_correlation_plot_input",
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
    return (
        marchenko_pastur,
        initial_lookback_days,
        lookback_factor,
        lense_option,
        target_return,
        show_performance_plot,
        show_momentum_plot,
        show_correlation_plot,
        custom_symbols,
    )


def _process_and_prepare_data(symbols, period, equity_config, streamlit=True):
    """Load, process data and create style dictionaries for given symbols."""
    # Load and process data
    df_pivot = pivot_data(list(symbols), period, streamlit=streamlit)
    
    # Create style dictionaries
    colors_dict = {symbol: equity_config.get(symbol, {}).get("color", get_random_style("color")) for symbol in symbols}
    line_styles_dict = {symbol: equity_config.get(symbol, {}).get("line_style", get_random_style("line_style")) for symbol in symbols}
    
    return df_pivot, colors_dict, line_styles_dict




def _process_symbol_tab(
    symbols,
    symbol_type,
    look_back_days,
    equity_config,
    target_return,
    show_performance_plot,
    show_momentum_plot,
    show_correlation_plot,
    marchenko_pastur,
    dfs,
    momentum_summaries,
):
    """Process a single symbol tab - handles both custom and regular symbols."""
    period = f"{look_back_days[-1]}d"

    # Load, process data and create style dictionaries
    df_pivot, colors_dict, line_styles_dict = _process_and_prepare_data(symbols, period, equity_config)
    dfs[symbol_type] = df_pivot

    # Display momentum section
    display_section_header("Momentum")
    st.write("target_return", target_return)

    # Create momentum ranking first (always needed for data analysis)
    momentum_ranking, _ = _create_and_sort_momentum_data(df_pivot)

    # Only create momentum plot if requested
    if show_momentum_plot:
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

        # momentum_combined data is computed but not displayed

        # Display momentum plot
        st.plotly_chart(momentum_fig, config={"displayModeBar": False})
    else:
        # Still compute momentum data for analysis but don't create plot
        from src.data import normalize_prices, compute_momentum

        df_norm = normalize_prices(df_pivot)
        _, momentum_combined = compute_momentum(df_norm, WINDOW_SIZES, target_return=target_return)

        # momentum_combined data is computed but not displayed

    # Store momentum data for summary (moved outside conditional)
    momentum_summaries[symbol_type] = momentum_combined

    # Display momentum ranking
    display_dataframe(momentum_ranking, symbol_type, "am", vmin=0.1)

    # Create performance plot only if requested
    if show_performance_plot:
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

        # Display performance plot
        st.plotly_chart(fig, config={"displayModeBar": False})

    # Display correlation section
    display_section_header("Correlation")
    if show_correlation_plot:
        pivoted_to_corr(df_pivot, plot=True, streamlit=True, marchenko_pastur=marchenko_pastur)
    else:
        # Still compute correlation matrix for data analysis but don't display plot
        corr_matrix = pivoted_to_corr(df_pivot, plot=False, streamlit=True, marchenko_pastur=marchenko_pastur)
        corr_matrix = corr_matrix.round(0).astype(int)
        display_dataframe(corr_matrix, symbol_type, "Correlation Matrix")

    # Performance section removed as requested


def _create_and_sort_momentum_data(df_pivot, momentum_df=None):
    """Create momentum ranking and sort momentum dataframe by ranking values."""
    # Create momentum ranking
    momentum_ranking = create_momentum_ranking_display(
        df_pivot,
        window_sizes=WINDOW_SIZES,
    )
    
    # Sort momentum_df columns by momentum_ranking values if provided
    if momentum_df is not None:
        # Get the agg_momentum values from momentum_ranking (transposed format)
        if "am" in momentum_ranking.index:
            ranking_values = momentum_ranking.loc["am"]
        else:
            # If not transposed, get from the agg_momentum column
            ranking_values = momentum_ranking.set_index("Symbol")["am"]
        
        # Sort columns by the momentum ranking values (descending order)
        sorted_columns = ranking_values.sort_values(ascending=False).index
        momentum_df_sorted = momentum_df[sorted_columns]
        return momentum_ranking, momentum_df_sorted
    
    return momentum_ranking, None


def _get_momentum_ranking_for_symbol_type(symbol_type, dfs):
    """Get momentum ranking for a symbol type if it exists in dfs."""
    if symbol_type in dfs.keys():
        df_pivot = dfs[symbol_type].copy()
        momentum_ranking, _ = _create_and_sort_momentum_data(df_pivot)
        return momentum_ranking
    return None


def _display_summary_tab(momentum_summaries, dfs, marchenko_pastur, show_correlation_plot):
    """Display the summary tab with all momentum summaries and correlations."""
    # Display momentum summaries
    for symbol_type in momentum_summaries.keys():
        momentum_df = momentum_summaries[symbol_type]
        momentum_ranking = _get_momentum_ranking_for_symbol_type(symbol_type, dfs)

        if momentum_ranking is not None:
            # Only display momentum ranking table
            display_dataframe(momentum_ranking, symbol_type, "Momentum Ranking")

    # Display correlations
    for symbol_type in dfs.keys():
        df_pivot = dfs[symbol_type].copy()
        if show_correlation_plot:
            pivoted_to_corr(df_pivot, plot=True, streamlit=True, marchenko_pastur=marchenko_pastur)
        else:
            # Still compute correlation matrix for data analysis but don't display plot
            corr_matrix = pivoted_to_corr(df_pivot, plot=False, streamlit=True, marchenko_pastur=marchenko_pastur)
            display_dataframe(corr_matrix, symbol_type, "Correlation Matrix")


def show_market_performance(
    equity_config: dict,
    portfolio_config: dict,
    marchenko_pastur: bool = True,
    initial_lookback_days: int = 5,
    lookback_factor: int = 3,
    target_return: float = 1.4,
    show_performance_plot: bool = True,
    show_momentum_plot: bool = False,
    show_correlation_plot: bool = False,
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
                _display_summary_tab(momentum_summaries, dfs, marchenko_pastur, show_correlation_plot)
            elif symbol_type == "Custom Symbols":
                if custom_symbols:
                    _process_symbol_tab(
                        custom_symbols,
                        symbol_type,
                        look_back_days,
                        equity_config,
                        target_return,
                        show_performance_plot,
                        show_momentum_plot,
                        show_correlation_plot,
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
                    show_momentum_plot,
                    show_correlation_plot,
                    marchenko_pastur,
                    dfs,
                    momentum_summaries,
                )

    # Bottom Table of Contents
    display_table_of_contents(
        sections=[
            "Momentum",
            "Performance",
            "Correlation",
        ]
    )


if __name__ == "__main__":

    if hydra.core.global_hydra.GlobalHydra().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_resolvers()

    with hydra.initialize(version_base=None, config_path="../../conf"):
        config = hydra.compose(config_name="main")
    (
        marchenko_pastur,
        initial_lookback_days,
        lookback_factor,
        lense_option,
        target_return,
        show_performance_plot,
        show_momentum_plot,
        show_correlation_plot,
        custom_symbols,
    ) = setup_page_and_sidebar(config["style_conf"], add_to_sidebar=lambda: sidebar(config))
    st.title(lense_option)

    # Table of Contents
    display_table_of_contents(
        sections=[
            "Momentum",
            "Performance",
            "Correlation",
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
        show_momentum_plot,
        show_correlation_plot,
        custom_symbols,
    )
