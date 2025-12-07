import time
from src.configurations.yaml import register_resolvers
import streamlit as st

import hydra
from src.data.TICKER import TICKERS
from src.data.FearGreed import FearGreed
from src.viz.viz import create_momentum_ranking_display, create_price_plot
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

# Window sizes are configured in conf/main.yaml as lookback_days


def display_fear_and_greed_info():
    """Display fear and greed index information at the top of the dashboard."""
    fear_greed = FearGreed()
    
    # Get both stock and crypto fear and greed data
    traditional_data = fear_greed.get_fear_and_greed()
    crypto_data = fear_greed.get_crypto_fear_and_greed()
    
    # Create a unified display with both metrics in one box
    stock_value = traditional_data["value"]
    stock_desc = traditional_data["description"]
    
    crypto_value = crypto_data["value"]
    crypto_desc = crypto_data["description"]
    
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
    # Get page names from lenses.pages (main, stocks, etfs)
    # These are the top-level options in the sidebar
    pages = config["lenses"].get("pages", {})
    page_names = list(pages.keys())
    
    selected_page = st.sidebar.radio(
        "Pages",
        page_names,
        help="Choose the page to display\n",
        key="page_option",
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
    st.sidebar.markdown("Correlation Denoising")
    marchenko_pastur = st.sidebar.checkbox(
        "Marchenko Pastur",
        value=True,
        key="marchenko_pastur_input",
    )
    return (
        marchenko_pastur,
        selected_page,
        custom_symbols,
    )


def _process_and_prepare_data(symbols, period, equity_config):
    """Load, process data and create style dictionaries for given symbols."""
    # Load and process data using TICKERS class
    ticker_obj = TICKERS(list(symbols), period=period, normalize=False)
    df_prices = ticker_obj.prices.copy()

    # Create style dictionaries
    colors_dict = {symbol: equity_config.get(symbol, {}).get("color", get_random_style("color")) for symbol in symbols}
    line_styles_dict = {symbol: equity_config.get(symbol, {}).get("line_style", get_random_style("line_style")) for symbol in symbols}

    return df_prices, colors_dict, line_styles_dict, ticker_obj


def _process_symbol_tab(
    symbols,
    symbol_type,
    look_back_days,
    equity_config,
    marchenko_pastur,
    log_col,
    metrics_order=None,
):
    """Process a single symbol tab - handles both custom and regular symbols."""
    period = f"{look_back_days[-1]}d"

    # Load, process data and create style
    with st.spinner("Downloading and Processing data..."):
        df_prices, colors_dict, line_styles_dict, ticker_obj = _process_and_prepare_data(symbols, period, equity_config)
    log_col.success(f"Data Loaded at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Display momentum section
    st.write(len(symbols), "tickers",  look_back_days[-1], "max_lookback_days")
    display_section_header("Momentum")

    # Create momentum ranking first (always needed for data analysis)
    with st.spinner("Computing momentum ranking..."):
        momentum_ranking, _ = _create_and_sort_momentum_data(df_prices, look_back_days, sort_column="combined_score", metrics_order=metrics_order, ticker_obj=ticker_obj)

    # Add yield row for Income lens
    if symbol_type == "Income":
        with st.spinner("Fetching dividend yields..."):
            yields = _fetch_dividend_yields(symbols)
            # Create yield row as a Series with same columns as momentum_ranking
            yield_row = pd.Series(
                {symbol: yields.get(symbol, None) for symbol in momentum_ranking.columns},
                name="yield"
            )
            # Find the index of "m" row
            if "m" in momentum_ranking.index:
                m_index = momentum_ranking.index.get_loc("m")
                # Insert yield row after "m"
                momentum_ranking = pd.concat([
                    momentum_ranking.iloc[:m_index + 1],
                    pd.DataFrame([yield_row], index=["yield"]),
                    momentum_ranking.iloc[m_index + 1:]
                ])
            else:
                # If "m" not found, append at the beginning
                momentum_ranking = pd.concat([
                    pd.DataFrame([yield_row], index=["yield"]),
                    momentum_ranking
                ])

    # Display momentum ranking table first
    display_dataframe(momentum_ranking, symbol_type, "am", vmin=-0.1, vmax=1, hide_rows=["combined_score"])

    # Add price plot - show only top 5 ranked symbols
    display_section_header("Price Performance")
    if "combined_score" in momentum_ranking.index and len(momentum_ranking.columns) > 0:
        combined_scores = momentum_ranking.loc["combined_score"]
        top_5_symbols = combined_scores.nlargest(5).index.tolist()
    else:
        top_5_symbols = symbols[:5] if len(symbols) >= 5 else symbols
    
    with st.spinner("Creating price performance plot for top 5 symbols..."):
        price_fig = create_price_plot(
            df_prices,
            top_5_symbols,
            look_back_days,
            colors_dict,
            line_styles_dict,
            equity_config,
        )
        st.plotly_chart(price_fig, config={"displayModeBar": False})

    # Display correlation section
    display_section_header("Correlation")

    # Show correlation plot (clustermap) first and compute correlation matrix
    with st.spinner("Computing correlation plot..."):
        corr_matrix = pivoted_to_corr(df=df_prices, plot=True, streamlit=True, marchenko_pastur=marchenko_pastur)
    
    # Display correlation matrix in a collapsible expander (collapsed by default)
    if "combined_score" in momentum_ranking.index and len(momentum_ranking.columns) > 1:
    # Get top symbols by combined_score (momentum)
        combined_scores = momentum_ranking.loc["combined_score"]
        top_n = min(20, len(combined_scores))
        top_symbols = combined_scores.nlargest(top_n).index.tolist()
        
        if len(top_symbols) > 1:
            with st.expander(f"📊 Top {len(top_symbols)} Symbols by Momentum Correlation", expanded=False):
                with st.spinner(f"Computing correlation for top {len(top_symbols)} symbols..."):
                    # Filter df_prices to top symbols
                    top_df_prices = df_prices[top_symbols].copy()
                    
                    # Create correlation plot for top symbols
                    pivoted_to_corr(
                        df=top_df_prices, 
                        plot=True, 
                        streamlit=True, 
                        marchenko_pastur=marchenko_pastur
                    )
                    

    with st.expander("Correlation Matrix", expanded=False):
        corr_matrix = corr_matrix.round(0).astype(int)
        display_dataframe(corr_matrix, symbol_type, "Correlation Matrix")
    
    # Add correlation plot for top 20 symbols by momentum in expander



def _create_and_sort_momentum_data(df_prices, window_sizes, momentum_df=None, sort_column="combined_score", metrics_order=None, ticker_obj=None):
    """Create momentum ranking and sort momentum dataframe by ranking values."""
    # Create momentum ranking
    momentum_ranking = create_momentum_ranking_display(
        df_prices,
        window_sizes=window_sizes,
        metrics_order=metrics_order,
        ticker_obj=ticker_obj,
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
    custom_symbols: list = None,
    lookback_days: list = None,
    metrics_order: list = None,
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

    # Use calendar days for lookback periods from config
    if lookback_days is None:
        lookback_days = [14, 50, 100, 200, 400]  # Default fallback
    look_back_days = lookback_days

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
            log_col=col3,
            metrics_order=metrics_order,
        )

    # Bottom Table of Contents
    sections = ["Momentum", "Price Performance", "Correlation"]
    
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
        selected_page,
        custom_symbols,
    ) = setup_page_and_sidebar(config["style_conf"], add_to_sidebar=lambda: sidebar(config))

    st.title(selected_page)

    # Display Fear & Greed Index at the top
    display_fear_and_greed_info()

    # Table of Contents
    sections = ["Momentum", "Price Performance", "Correlation"]
    
    display_table_of_contents(sections=sections)

    # Get the selected page's lens options
    pages = config["lenses"].get("pages", {})
    selected_page_lenses = pages.get(selected_page, {})
    
    # Get metrics_order separately (it's configuration, not a lens option)
    metrics_order = config.get("lenses", {}).get("metrics_order", None)
    
    show_market_performance(
        config["tickers"],
        selected_page_lenses,
        marchenko_pastur,
        custom_symbols,
        config.get("lookback_days", [14, 50, 100, 200, 400]),
        metrics_order=metrics_order,
    )

#  * lines : 11-28-25 16:34 390