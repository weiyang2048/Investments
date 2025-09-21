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
    df = df.bfill()
    symbols = df.select_dtypes(include=[np.number]).columns

    df[symbols] = df[symbols].div(df[symbols].iloc[0], axis=1)
    return df


def compute_momentum(
    df: pd.DataFrame, window_sizes: List[int] = [7, 30, 90, 180, 360], time_column: str = "Date", target_return: float = 1.3
) -> tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    """
    Compute momentum for different window sizes and count threshold crossings.

    Args:
        df: DataFrame with price data (normalized or raw)
        window_sizes: List of window sizes in days for momentum calculation
        time_column: Name of the time column
        target_return: Target annualized return for threshold calculation

    Returns:
        Tuple of (momentum_data_dict, momentum_summary_dataframe)
    """
    momentum_data = {}
    symbols = df.select_dtypes(include=[np.number]).columns

    # Track momentum threshold counts for each symbol
    momentum_counts_long = {symbol: 0 for symbol in symbols}
    momentum_counts_mid = {symbol: 0 for symbol in symbols}
    momentum_counts_short = {symbol: 0 for symbol in symbols}

    for window in window_sizes:
        # Compute momentum: (current_price / price_window_days_ago) - 1
        momentum = df[symbols].rolling(window=window).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
        momentum[time_column] = df[time_column]
        momentum = momentum.dropna()
        momentum_data[window] = momentum

        # Get the last window*3 rows for better analysis
        display_rows = min(window * 3, len(momentum))
        momentum_display = momentum.tail(display_rows)

        # Calculate threshold: (1+y1)^(252/window) = target_return
        y1_threshold = target_return ** (window / 252) - 1

        # Count symbols with momentum above threshold
        for symbol in symbols:
            if symbol in momentum_display.columns:
                last_momentum = momentum_display[symbol].iloc[-1]
                if last_momentum > y1_threshold:
                    momentum_counts_long[symbol] += 1
                    if window <= 90:
                        momentum_counts_mid[symbol] += 1
                    if window <= 7:
                        momentum_counts_short[symbol] += 1

    # Create momentum summary DataFrame
    momentum_summary_long = pd.DataFrame(list(momentum_counts_long.items()), columns=["Symbol", "Momentum_Count_Long"])
    momentum_summary_long = momentum_summary_long.sort_values(by="Momentum_Count_Long", ascending=False)
    momentum_summary_mid = pd.DataFrame(list(momentum_counts_mid.items()), columns=["Symbol", "Momentum_Count_Mid"])
    momentum_summary_mid = momentum_summary_mid.sort_values(by="Momentum_Count_Mid", ascending=False)
    momentum_summary_short = pd.DataFrame(list(momentum_counts_short.items()), columns=["Symbol", "Momentum_Count_Short"])
    momentum_summary_short = momentum_summary_short.sort_values(by="Momentum_Count_Short", ascending=False)

    # Combine all momentum summaries into a single DataFrame
    # First, ensure all DataFrames have the same columns by getting the union of all symbols
    all_symbols = set(momentum_summary_long["Symbol"]) | set(momentum_summary_mid["Symbol"]) | set(momentum_summary_short["Symbol"])

    # Create a combined DataFrame with all symbols
    combined_data = {}
    for symbol in all_symbols:
        long_count = (
            momentum_summary_long[momentum_summary_long["Symbol"] == symbol]["Momentum_Count_Long"].iloc[0]
            if symbol in momentum_summary_long["Symbol"].values
            else 0
        )
        mid_count = (
            momentum_summary_mid[momentum_summary_mid["Symbol"] == symbol]["Momentum_Count_Mid"].iloc[0]
            if symbol in momentum_summary_mid["Symbol"].values
            else 0
        )
        short_count = (
            momentum_summary_short[momentum_summary_short["Symbol"] == symbol]["Momentum_Count_Short"].iloc[0]
            if symbol in momentum_summary_short["Symbol"].values
            else 0
        )

        combined_data[symbol] = {"Momentum_S": short_count, "Momentum_M": mid_count, "Momentum_L": long_count}

    # Create the combined DataFrame
    combined_data = pd.DataFrame(combined_data)
    column_sums = combined_data.sum()
    combined_data = combined_data[column_sums.sort_values(ascending=False).index]
    # momentum_combined = combined_data

    return momentum_data, combined_data


if __name__ == "__main__":
    df = get_daily_prices_list(["GDE", "AIVI", "DOL", "DGRW"], "5y")
