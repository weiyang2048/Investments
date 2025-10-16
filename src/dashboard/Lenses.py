import time
from src.configurations.yaml import register_resolvers
import streamlit as st

import hydra
import numpy as np
from src.data import pivot_data
from src.viz.viz import create_combined_performance_momentum_plot, create_momentum_ranking_display
from src.configurations.style_picker import get_random_style
from src.dashboard.create_page import setup_page_and_sidebar
import pandas as pd
from src.association import pivoted_to_corr
from src.viz.streamlit_display import (
    display_dataframe,
    display_table_of_contents,
    display_section_header,
)
from src.data import normalize_prices, compute_momentum

# Window sizes are now computed dynamically from initial_lookback_days and lookback_factor


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
        "Lenses",
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
    initial_lookback_days = st.sidebar.number_input(
        "Initial Lookback Days",
        min_value=1,
        max_value=3650,
        value=6,
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
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    show_combined_plot = st.sidebar.checkbox(
        "Show Momentum Plot",
        value=True,
        key="show_combined_plot_input",
    )

    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("Correlation Denoising")
    marchenko_pastur = st.sidebar.checkbox(
        "Marchenko Pastur",
        value=True,
        key="marchenko_pastur_input",
    )
    show_correlation_plot = st.sidebar.checkbox(
        "Show Correlation Plot",
        value=True,
        key="show_correlation_plot_input",
    )
    return (
        marchenko_pastur,
        initial_lookback_days,
        lookback_factor,
        lense_option,
        show_combined_plot,
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
    show_combined_plot,
    show_correlation_plot,
    marchenko_pastur,
    dfs,
    momentum_summaries,
    log_col,
):
    """Process a single symbol tab - handles both custom and regular symbols."""
    period = f"{look_back_days[-1]}d"

    # Load, process data and create style
    with st.spinner("Downloading and Processing data..."):
        df_pivot, colors_dict, line_styles_dict = _process_and_prepare_data(symbols, period, equity_config)
    log_col.success(f"Data Loaded at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    dfs[symbol_type] = df_pivot

    # Display momentum section
    display_section_header("Momentum")

    # Create momentum ranking first (always needed for data analysis)
    with st.spinner("Computing momentum ranking..."):
        momentum_ranking, _ = _create_and_sort_momentum_data(df_pivot, look_back_days)

    # Display momentum ranking table first
    display_dataframe(momentum_ranking, symbol_type, "am", vmin=0.1)

    # Create combined plot if requested (after the table)
    if show_combined_plot:
        with st.spinner("Creating combined performance & momentum plot..."):
            combined_result = create_combined_performance_momentum_plot(
                df_pivot,
                symbols,
                look_back_days,
                colors_dict,
                line_styles_dict,
                equity_config,
                momentum_ranking=momentum_ranking,
            )
            combined_fig = combined_result["figure"]
            momentum_combined = combined_result["momentum_combined"]

        # Display combined plot
        st.plotly_chart(combined_fig, config={"displayModeBar": False})
    else:
        # Still compute momentum data for analysis but don't create plot
        df_norm = normalize_prices(df_pivot)
        with st.spinner("Computing momentum..."):
            _, momentum_combined = compute_momentum(df_norm, look_back_days)

    # Store momentum data for summary (moved outside conditional)
    momentum_summaries[symbol_type] = momentum_combined

    # Display correlation section
    display_section_header("Correlation")
    
    # Always compute and display correlation matrix
    with st.spinner("Computing correlation matrix..."):
        corr_matrix = pivoted_to_corr(df_pivot, plot=False, streamlit=True, marchenko_pastur=marchenko_pastur)
        corr_matrix = corr_matrix.round(0).astype(int)
        display_dataframe(corr_matrix, symbol_type, "Correlation Matrix")
    
    # Show correlation plot if requested (in addition to the table)
    if show_correlation_plot:
        with st.spinner("Computing correlation plot..."):
            pivoted_to_corr(df_pivot, plot=True, streamlit=True, marchenko_pastur=marchenko_pastur)

    # Performance section removed as requested


def _create_and_sort_momentum_data(df_pivot, window_sizes, momentum_df=None):
    """Create momentum ranking and sort momentum dataframe by ranking values."""
    # Create momentum ranking
    momentum_ranking = create_momentum_ranking_display(
        df_pivot,
        window_sizes=window_sizes,
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


def show_market_performance(
    equity_config: dict,
    portfolio_config: dict,
    marchenko_pastur: bool = True,
    initial_lookback_days: int = 5,
    lookback_factor: int = 3,
    show_combined_plot: bool = True,
    show_correlation_plot: bool = False,
    custom_symbols: list = None,
) -> None:
    """Function to show the market performance dashboard."""
    # First selection: Symbol types (previously handled by tabs)
    symbol_types = [key for key in portfolio_config.keys()]

    # Add custom symbols option if custom symbols are provided
    if custom_symbols:
        symbol_types = symbol_types + ["Custom Symbols"]

    # Summary removed per request

    # Symbol type selection (previously handled by tabs)
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        selected_symbol_type = st.radio(
            "Select Lense",
            symbol_types,
            key="lense_selector",
            help="Choose which symbol type to analyze",
            horizontal=True,
            label_visibility="visible",
        )

    # Generate look_back_days list based on user input
    look_back_days = [int(initial_lookback_days * (lookback_factor**i)) for i in range(6)]
    momentum_summaries = dict()
    dfs = dict()

    # Process the selected symbol type
    if selected_symbol_type == "Custom Symbols":
        if custom_symbols:
            _process_symbol_tab(
                custom_symbols,
                selected_symbol_type,
                look_back_days,
                equity_config,
                show_combined_plot,
                show_correlation_plot,
                marchenko_pastur,
                dfs,
                momentum_summaries,
                log_col=col3,
            )
        else:
            st.info("No custom symbols provided. Please enter symbols in the sidebar.")
    else:
        symbols = portfolio_config[selected_symbol_type]
        _process_symbol_tab(
            symbols,
            selected_symbol_type,
            look_back_days,
            equity_config,
            show_combined_plot,
            show_correlation_plot,
            marchenko_pastur,
            dfs,
            momentum_summaries,
            log_col=col3,
        )

    # Bottom Table of Contents
    display_table_of_contents(
        sections=[
            "Momentum",
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
        show_combined_plot,
        show_correlation_plot,
        custom_symbols,
    ) = setup_page_and_sidebar(config["style_conf"], add_to_sidebar=lambda: sidebar(config))

    st.title(lense_option)

    # Table of Contents
    display_table_of_contents(
        sections=[
            "Momentum",
            "Correlation",
        ]
    )

    show_market_performance(
        config["tickers"],
        config["lenses"][lense_option],
        marchenko_pastur,
        initial_lookback_days,
        lookback_factor,
        show_combined_plot,
        show_correlation_plot,
        custom_symbols,
    )
