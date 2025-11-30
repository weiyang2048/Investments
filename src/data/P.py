import yfinance as yf
import pandas as pd
import streamlit as st
from typing import List, Union
def get_tickers_close_prices(tickers: List[str], period: str = "5y", normalize: bool = False) -> pd.DataFrame:
    """
    Fetch close prices for multiple tickers using yfinance.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols to fetch (e.g., ["BTC-USD", "IOO"]).
    period : str, optional
        Period to fetch data for (default: "5y"). Options: "1d", "5d", "1mo", 
        "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max".
    
    Returns
    -------
    pd.DataFrame
        DataFrame with close prices for each ticker. Columns are ticker symbols,
        index is datetime. Timezone is removed to match liquidity index format.
    
    Examples
    --------
    >>> df = get_tickers_close_prices(["BTC-USD", "IOO"], period="5y")
    >>> print(df.head())
    """
    ensembles = yf.Tickers(tickers)
    ensembles_hist = ensembles.history(period=period)
    
    # Extract Close price columns
    close_prices = [x for x in ensembles_hist.columns if "Close" in x]
    ensembles_hist_close = ensembles_hist[close_prices]
    
    # Clean column names: remove "Close" prefix, keep ticker symbol
    ensembles_hist_close.columns = [x[1] for x in close_prices]
    
    # Align timezone with liquidity index (use tz-naive)
    if hasattr(ensembles_hist_close.index, "tz") and ensembles_hist_close.index.tz is not None:
        ensembles_hist_close = ensembles_hist_close.tz_convert(None)
    
    ensembles_hist_close.dropna(inplace=True, how="any")
    if normalize:
        ensembles_hist_close = ensembles_hist_close.div(ensembles_hist_close.iloc[0], axis=1)

    return ensembles_hist_close

@st.cache_resource(show_spinner=True)
def st_get_tickers_close_prices(tickers: List[str], period: str = "5y") -> pd.DataFrame:
    """
    Fetch close prices for multiple tickers using yfinance and cache the result.
    """
    return get_tickers_close_prices(tickers, period)