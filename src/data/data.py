from datetime import datetime
from typing import List, Dict
import pandas as pd
import yfinance as yf
from loguru import logger
from functools import lru_cache, cache
import streamlit as st
import numpy as np


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
            f"Downloaded {symbol} data for the last {period}",
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


def pivot_data(symbols: List[str], period: str = "1mo", streamlit: bool = False) -> pd.DataFrame:
    """
    Transforms daily price data into a pivot table format.

    Args:
        symbols (List[str]): A list of stock symbols to fetch data for.
        period (str): The time period for which to fetch data (default is "1mo").

    Returns:
        pd.DataFrame: A DataFrame pivoted to have dates as rows and symbols as columns,
                      with closing prices as values.
    """
    df = get_daily_prices_list(symbols, period, streamlit)
    df.reset_index(inplace=True)
    return df.pivot(index="Date", columns="Symbol", values="Close").reset_index()


def normalize_prices(df: pd.DataFrame, time_column: str = "Date") -> pd.DataFrame:
    """Normalize price data to start at 1.0."""
    df = df.copy()
    # sort by time column, ascending
    df.sort_values(time_column, inplace=True, ascending=True)
    # fill backward
    df = df.fillna(method="bfill")
    symbols = df.select_dtypes(include=[np.number]).columns

    df[symbols] = df[symbols].div(df[symbols].iloc[0], axis=1)
    return df


def compute_momentum(df: pd.DataFrame, window_sizes: List[int] = [7, 30, 90, 180, 360], time_column: str = "Date") -> Dict[int, pd.DataFrame]:
    """
    Compute momentum for different window sizes.

    Args:
        df: DataFrame with price data (normalized or raw)
        window_sizes: List of window sizes in days for momentum calculation
        time_column: Name of the time column

    Returns:
        Dictionary mapping window size to momentum DataFrame
    """
    momentum_data = {}
    symbols = df.select_dtypes(include=[np.number]).columns

    for window in window_sizes:
        # Compute momentum: (current_price / price_window_days_ago) - 1
        momentum = df[symbols].rolling(window=window).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
        momentum[time_column] = df[time_column]
        momentum = momentum.dropna()
        momentum_data[window] = momentum

    return momentum_data


if __name__ == "__main__":
    df = get_daily_prices_list(["GDE", "AIVI", "DOL", "DGRW"], "5y")
