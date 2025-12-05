from typing import List
import pandas as pd
import yfinance as yf
from loguru import logger
import streamlit as st
import numpy as np


class Tickers:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.ticker = yf.Ticker(symbols)

    def get_daily_prices(self, period: str = "1mo") -> pd.DataFrame:
        self.df = self.ticker.history(period=period)


def get_daily_prices(symbol: str, period: str = "1mo") -> pd.DataFrame:
    """
    Get daily price data for a single symbol using yfinance
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    return df


@st.cache_data(ttl="10min")
def get_daily_prices_streamlit(symbol: str, period: str = "1mo") -> pd.DataFrame:
    """
    Get daily price data for a single symbol using yfinance
    """
    symbol = symbol.replace(".", "-")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    return df


def get_daily_prices_list(symbols: List[str], period: str = "1mo", streamlit: bool = False) -> pd.DataFrame:
    """
    Get daily price data for multiple symbols using yfinance

    Args:
        symbols: List of stock symbols
        period: Time period to fetch (e.g. "1mo", "1y", "max")

    Returns:
        DataFrame with daily prices for all symbols
    """
    dfs = []
    for symbol in symbols:
        if streamlit:
            df = get_daily_prices_streamlit(symbol, period)
        else:
            df = get_daily_prices(symbol, period)
        df["Symbol"] = symbol
        dfs.append(df)
        logger.opt(ansi=True).log(
            "data",
            f"Downloaded {symbol} data for the last {period}, df.shape: {df.shape}",
        )
    if not dfs:
        return pd.DataFrame()

    combined_df = pd.concat(dfs)
    combined_df = combined_df.assign(
        **{
            "Year": lambda df: df.index.year,
            "Month": lambda df: df.index.month,
        }
    )
    combined_df = combined_df[["Symbol"] + [col for col in combined_df.columns if col != "Symbol"]]

    return combined_df

def normalize_prices(df: pd.DataFrame, time_column: str = "Date") -> pd.DataFrame:
    """Normalize price data to start at 1.0."""
    df = df.copy()
    # sort by time column, ascending
    df.sort_values(time_column, inplace=True, ascending=True)
    # fill backward
    df = df.bfill()
    symbols = df.select_dtypes(include=[np.number]).columns

    df[symbols] = df[symbols].div(df[symbols].iloc[0], axis=1)
    return df

