import streamlit as st
import pandas as pd
from typing import List
from src.data import get_daily_prices
from src.viz import create_performance_plot


def setup_page(dashboard_config: dict) -> None:
    """Configure the Streamlit page settings."""
    st.set_page_config(
        **dashboard_config["page_config"],
    )
    st.title(dashboard_config["title"])
    st.markdown(dashboard_config["description"])
    st.sidebar.header("Controls")
    st.markdown(dashboard_config["style_string"], unsafe_allow_html=True)


@st.cache_data
def load_data(symbols: List[str], period: str = "10y") -> pd.DataFrame:
    """
    Load and pivot price data for the given symbols.

    Args:
        symbols: List of symbols to load data for
        period: Time period to load (e.g., "1y", "5y")

    Returns:
        Pivoted DataFrame with dates as index and symbols as columns
    """
    df = get_daily_prices(symbols, period)
    df.reset_index(inplace=True)
    return df.pivot(index="Date", columns="Symbol", values="Close").reset_index()


def show_market_performance(
    equity_config: dict, portfolio_config: dict, dashboard_config: dict
) -> None:
    """Function to show the market performance dashboard."""
    setup_page(dashboard_config)

    # Symbol selection
    symbol_type = st.sidebar.radio(
        "Select Symbol Type", [key for key in portfolio_config.keys()], index=0
    )
    symbols = portfolio_config[symbol_type]
    # Period selection
    period = st.sidebar.selectbox("Select Time Period", ["1y", "2y", "5y"], index=2)

    # Load and process data
    df_pivot = load_data(list(symbols), period)

    # Create and display plot
    look_back_days = dashboard_config["look_back_days"]
    colors_dict = {
        symbol: equity_config.get(symbol, {}).get("color", "snow") for symbol in symbols
    }
    line_styles_dict = {
        symbol: equity_config.get(symbol, {}).get("line_style", "solid") for symbol in symbols
    }
    fig = create_performance_plot(
        df_pivot, symbols, look_back_days, colors_dict, line_styles_dict, equity_config
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display raw data option
    if st.checkbox("Show Raw Data"):
        st.dataframe(df_pivot)
