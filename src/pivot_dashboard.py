from conf.config_loader import load_portfolios, load_config
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from src.data import get_daily_prices
from src.viz import create_performance_plot
from conf import load_config, get_symbols


def setup_page() -> None:
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="Market Performance Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/weiyang2048/Investments",
            "Report a bug": "https://bug.example.com",
            "About": "# Market Performance Dashboard\nCompare normalized performance across different time periods",
        },
    )
    st.title("Equity Market Dashboard")
    st.markdown("Compare normalized performance across different time periods")
    st.sidebar.header("Controls")


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


def main(etf_config: dict, portfolio_config: dict) -> None:
    """Main function to run the dashboard."""
    setup_page()

    # Symbol selection
    symbol_type = st.sidebar.radio(
        "Select Symbol Type", ["Markets", "Sectors", "Regions"], index=0
    )
    symbols = get_symbols(symbol_type, portfolio_config)

    # Period selection
    period = st.sidebar.selectbox(
        "Select Time Period", ["1y", "2y", "5y", "10y"], index=2
    )

    # Load and process data
    df_pivot = load_data(symbols, period)

    # Create and display plot
    look_back_days = [10, 30, 90, 180, 360][::-1]
    colors_dict = {symbol: etf_config["etfs"][symbol]["color"] for symbol in symbols}
    fig = create_performance_plot(df_pivot, symbols, look_back_days, colors_dict)
    st.plotly_chart(fig, use_container_width=True)

    # Display raw data option
    if st.checkbox("Show Raw Data"):
        st.dataframe(df_pivot)


if __name__ == "__main__":
    etf_config = load_config("etf")
    portfolio_config = load_portfolios("portfolio")
    main(etf_config, portfolio_config)
