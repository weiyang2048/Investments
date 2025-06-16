from conf.config_loader import load_portfolios_conf, load_config
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from src.data import get_daily_prices
from src.viz import create_performance_plot
from conf import load_config, get_symbols, load_dashboard_conf


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
    etf_config: dict, portfolio_config: dict, dashboard_config: dict
) -> None:
    """Function to show the market performance dashboard."""
    setup_page(dashboard_config)
 
    # Symbol selection
    symbol_type = st.sidebar.radio(
        "Select Symbol Type", [key for key in portfolio_config.keys()], index=0
    )
    symbols = get_symbols(symbol_type, portfolio_config)

    # Period selection
    period = st.sidebar.selectbox("Select Time Period", ["1y", "2y", "5y", "10y"], index=3)

    # Load and process data
    df_pivot = load_data(symbols, period) 

    # Create and display plot
    look_back_days = dashboard_config["look_back_days"]  
    colors_dict = {
        symbol: etf_config["etfs"].get(symbol, {}).get("color", "snow") for symbol in symbols
    }
    line_styles_dict = {
        symbol: etf_config["etfs"].get(symbol, {}).get("line_style", "solid") for symbol in symbols
    }
    fig = create_performance_plot(df_pivot, symbols, look_back_days, colors_dict, line_styles_dict)
    st.plotly_chart(fig, use_container_width=True)
 
    # Display raw data option  
    if st.checkbox("Show Raw Data"):    
        st.dataframe(df_pivot) 
    
  
if __name__ == "__main__":    
    etf_config = load_config("etf") 
    portfolio_config = load_portfolios_conf("portfolio")  
    dashboard_config = load_dashboard_conf("dashboard")    
    show_market_performance(etf_config, portfolio_config, dashboard_config)                 