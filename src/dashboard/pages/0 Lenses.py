import time
from src.configurations.yaml import register_resolvers
import streamlit as st

import hydra
from src.data import pivot_data
from src.data.FearGreed import FearGreed
from src.viz.viz import create_combined_performance_momentum_plot, create_momentum_ranking_display, create_price_ratio_plot
from src.configurations.style_picker import get_random_style
from src.dashboard.create_page import setup_page_and_sidebar
import pandas as pd
from src.association import pivoted_to_corr
from src.viz.streamlit_display import (
    display_dataframe,
    display_table_of_contents,
    display_section_header,
)

pd.set_option("display.max_rows", None)

# Window sizes are now computed dynamically from initial_lookback_days and lookback_factor


def display_fear_and_greed_info():
    """Display fear and greed index information at the top of the dashboard."""
    fear_greed = FearGreed()
    
    # Get both stock and crypto fear and greed data
    traditional_data = fear_greed.get_fear_and_greed()
    crypto_data = fear_greed.get_crypto_fear_and_greed()
    
    # Create a unified display with both metrics in one box
    stock_value = traditional_data["value"]
    stock_desc = traditional_data["description"]
    stock_update = traditional_data["last_update_est_str"]
    
    crypto_value = crypto_data["value"]
    crypto_desc = crypto_data["description"]
    crypto_update = crypto_data["last_update_est_str"]
    
    # Calculate colors for both metrics
    normalized_stock = stock_value / 100.0
    red_stock = int(255 * (1 - normalized_stock))
    green_stock = int(255 * normalized_stock)
    color_hex_stock = f"#{red_stock:02x}{green_stock:02x}{50:02x}"
    
    if crypto_value is not None:
        normalized_crypto = crypto_value / 100.0
        red_crypto = int(255 * (1 - normalized_crypto))
        green_crypto = int(255 * normalized_crypto)
        color_hex_crypto = f"#{red_crypto:02x}{green_crypto:02x}{50:02x}"
    else:
        color_hex_crypto = "#808080"  # Gray for error
        crypto_value = "N/A"
        crypto_desc = "Error"
    
    # Return combined HTML string
    combined_html = f"""
    <div style="text-align:center;padding:15px;border-radius:10px;background-color:#f0f2f6;"> 
        <div style="margin-bottom:10px;">
            <a href="https://www.cnn.com/markets/fear-and-greed" target="_blank" style="color:{color_hex_stock};text-decoration:none;margin-left:5px; font-weight: bold;">{stock_value} {stock_desc} (S&P 500)</a>
            |
            <a href="https://feargreedmeter.com/" target="_blank" style="color:{color_hex_crypto};text-decoration:none;color:#2895f7;font-size:12px;">🗺️</a>
            |
            <a href="https://alternative.me/crypto/fear-and-greed-index/" target="_blank" style="color:{color_hex_crypto};text-decoration:none;margin-left:5px; font-weight: bold;">{crypto_value} {crypto_desc} (Crypto)</a>
        </div>
        <div>
             <a href="https://finance.yahoo.com/markets/stocks/most-active/" target="_blank" style="color:#2895f7;text-decoration:none;">yStock</a> | <a href="https://finance.yahoo.com/markets/etfs/top-performing/?start=0&count=25" target="_blank" style="color:#2895f7;text-decoration:none;">yETF</a> | <a href="https://www.etfrc.com/funds/overlap.php" target="_blank" style="color:#2895f7;text-decoration:none;">Overlap</a> | <a href="https://totalrealreturns.com/" target="_blank" style="color:#2895f7;text-decoration:none;">TotalReturns</a> | <a href="https://www.portfoliovisualizer.com/optimize-portfolio#analysisResults" target="_blank" style="color:#2895f7;text-decoration:none;">PortfolioVisualizer</a>
        </div>
    </div>
    """
    cols = st.columns(3)
    with cols[1]:
        st.markdown(combined_html, unsafe_allow_html=True)


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
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("Correlation Denoising")
    marchenko_pastur = st.sidebar.checkbox(
        "Marchenko Pastur",
        value=True,
        key="marchenko_pastur_input",
    )
    return (
        marchenko_pastur,
        initial_lookback_days,
        lookback_factor,
        lense_option,
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
    st.write(len(symbols), "tickers",  look_back_days[-1], "max_lookback_days")
    display_section_header("Momentum")

    # Create momentum ranking first (always needed for data analysis)
    with st.spinner("Computing momentum ranking..."):
        momentum_ranking, _ = _create_and_sort_momentum_data(df_pivot, look_back_days, sort_column="combined_score")

    # Display momentum ranking table first
    display_dataframe(momentum_ranking, symbol_type, "am", vmin=-0.1, vmax=1, hide_rows=["combined_score"])
    num_symbols = len(df_pivot.columns) - 1  # Subtract 1 for Date column
    # Always create and display combined plot
    if num_symbols > 100:
        st.info(f"Number of symbols ({num_symbols}) is greater than 100. Displaying first 50 symbols.")
        symbols = list(momentum_ranking.columns[:50])
        momentum_ranking = momentum_ranking[symbols]
        df_pivot = df_pivot[["Date"] + symbols]
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

    st.plotly_chart(combined_fig, config={"displayModeBar": False})

    # Add ratio plot - first symbol is denominator, others are numerators
    if len(symbols) >= 2:
        display_section_header("Ratio")
        
        with st.spinner("Creating price ratio plot..."):
            denominator_symbol = symbols[0]
            numerator_symbols = symbols[1:]  # All other symbols are numerators
            ratio_fig = create_price_ratio_plot(
                df_pivot,
                denominator_symbol,
                numerator_symbols,
                colors_dict,
                line_styles_dict,
                equity_config,
                look_back_days,
                momentum_ranking=momentum_ranking,
                top_n=5
            )
            st.plotly_chart(ratio_fig, config={"displayModeBar": False})

    # Display correlation section
    display_section_header("Correlation")

    # Show correlation plot (clustermap) first and compute correlation matrix
    with st.spinner("Computing correlation plot..."):
        corr_matrix = pivoted_to_corr(df=df_pivot, plot=True, streamlit=True, marchenko_pastur=marchenko_pastur)
    
    # Display correlation matrix in a collapsible expander (collapsed by default)
    if "combined_score" in momentum_ranking.index and len(momentum_ranking.columns) > 1:
    # Get top symbols by combined_score (momentum)
        combined_scores = momentum_ranking.loc["combined_score"]
        top_n = min(20, len(combined_scores))
        top_symbols = combined_scores.nlargest(top_n).index.tolist()
        
        if len(top_symbols) > 1:
            with st.expander(f"📊 Top {len(top_symbols)} Symbols by Momentum Correlation", expanded=False):
                with st.spinner(f"Computing correlation for top {len(top_symbols)} symbols..."):
                    # Filter df_pivot to top symbols
                    top_df_pivot = df_pivot[["Date"] + top_symbols].copy()
                    
                    # Create correlation plot for top symbols
                    top_corr_matrix = pivoted_to_corr(
                        df=top_df_pivot, 
                        plot=True, 
                        streamlit=True, 
                        marchenko_pastur=marchenko_pastur
                    )
                    

    with st.expander("Correlation Matrix", expanded=False):
        corr_matrix = corr_matrix.round(0).astype(int)
        display_dataframe(corr_matrix, symbol_type, "Correlation Matrix")
    
    # Add correlation plot for top 20 symbols by momentum in expander

    # Performance section removed as requested


def _create_and_sort_momentum_data(df_pivot, window_sizes, momentum_df=None, sort_column="combined_score"):
    """Create momentum ranking and sort momentum dataframe by ranking values."""
    # Create momentum ranking
    momentum_ranking = create_momentum_ranking_display(
        df_pivot,
        window_sizes=window_sizes,
    )

    # Sort momentum_df columns by momentum_ranking values if provided
    if momentum_df is not None:
        # Get the agg_momentum values from momentum_ranking (transposed format)
        if sort_column in momentum_ranking.index:
            ranking_values = momentum_ranking.loc[sort_column]
        else:
            # If not transposed, get from the agg_momentum column
            ranking_values = momentum_ranking.set_index("Symbol")[sort_column]

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
    custom_symbols: list = None,
) -> None:
    """Function to show the market performance dashboard."""
    # Display Fear & Greed Index at the top

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
    look_back_days = [int(initial_lookback_days * (lookback_factor**i)) for i in range(5)]
    momentum_summaries = dict()
    dfs = dict()

    # Process the selected symbol type
    if selected_symbol_type == "Custom Symbols" and not custom_symbols:
        st.info("No custom symbols provided. Please enter symbols in the sidebar.")
    else:
        symbols = (
            custom_symbols
            if selected_symbol_type == "Custom Symbols"
            else portfolio_config[selected_symbol_type]
        )
        print(symbols)
        _process_symbol_tab(
            symbols,
            selected_symbol_type,
            look_back_days,
            equity_config,
            marchenko_pastur,
            dfs,
            momentum_summaries,
            log_col=col3,
        )

    # Bottom Table of Contents
    sections = ["Momentum", "Ratio", "Correlation"]
    
    display_table_of_contents(sections=sections)


if __name__ == "__main__":
    import platform
    print(platform.system()=="Windows")
    if hydra.core.global_hydra.GlobalHydra().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_resolvers()

    with hydra.initialize(version_base=None, config_path="../../../conf"):
        config = hydra.compose(config_name="main")
    (
        marchenko_pastur,
        initial_lookback_days,
        lookback_factor,
        lense_option,
        custom_symbols,
    ) = setup_page_and_sidebar(config["style_conf"], add_to_sidebar=lambda: sidebar(config))

    st.title(lense_option)

    # Display Fear & Greed Index at the top
    display_fear_and_greed_info()

    # Table of Contents
    sections = ["Momentum", "Ratio", "Correlation"]
    
    display_table_of_contents(sections=sections)

    show_market_performance(
        config["tickers"],
        config["lenses"][lense_option],
        marchenko_pastur,
        initial_lookback_days,
        lookback_factor,
        custom_symbols,
    )

#  * lines : 11-28-25 16:34 390