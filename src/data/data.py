from datetime import datetime
from typing import List, Dict
import pandas as pd
import yfinance as yf
from loguru import logger
from functools import lru_cache, cache
import streamlit as st
import numpy as np


class Ticker:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
    
    def get_daily_prices(self, period: str = "1mo") -> pd.DataFrame:
        return self.ticker.history(period=period)

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
        Tuple of (momentum_data_dict, momentum_combined_dict) where:
        - momentum_data_dict: {window_size: momentum_dataframe}
        - momentum_combined_dict: {window_size: {symbol: count}}
    """
    momentum_data = {}
    momentum_combined = {}
    symbols = df.select_dtypes(include=[np.number]).columns

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

        # Initialize counts for this window
        window_counts = {symbol: 0 for symbol in symbols}

        # Count symbols with momentum above threshold
        for symbol in symbols:
            if symbol in momentum_display.columns:
                last_momentum = momentum_display[symbol].iloc[-1]
                if last_momentum > y1_threshold:
                    window_counts[symbol] = 1

        momentum_combined[window] = window_counts
    momentum_combined = pd.DataFrame(momentum_combined).T
    # cumulative sum of the counts
    momentum_combined = momentum_combined.cumsum()
    # sort columns by the sum of the column counts
    col_sums = momentum_combined.sum(axis=0)
    momentum_combined = momentum_combined[col_sums.sort_values(ascending=False).index]
    return momentum_data, momentum_combined


def compute_annualized_momentum_sum(df: pd.DataFrame, window_sizes: List[int] = [7, 30, 90, 180, 360], time_column: str = "Date") -> pd.DataFrame:
    """
    Compute the sum of annualized momentum across all window sizes for each symbol.

    Args:
        df: DataFrame with price data (normalized or raw)
        window_sizes: List of window sizes in days for momentum calculation
        time_column: Name of the time column

    Returns:
        DataFrame with symbols and their summed annualized momentum, ranked by total momentum
    """
    symbols = df.select_dtypes(include=[np.number]).columns
    am = {}

    for symbol in symbols:
        total_annualized_momentum = 0
        total_weight = 0
        for window in window_sizes:
            # Compute momentum: (current_price / price_window_days_ago) - 1
            momentum = df[symbol].rolling(window=window).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
            momentum = momentum.dropna()

            if len(momentum) > 0:
                # Get the last value for this window
                last_momentum = momentum.iloc[-1]
                # Annualize the momentum: (1 + momentum)^(252/window) - 1, cap at 2
                annualized_momentum = min((1 + last_momentum) ** (252 / window) - 1, 1)
                weight = 1 / np.log(window)
                annualized_momentum = annualized_momentum * weight
                total_weight += weight
                total_annualized_momentum += annualized_momentum

        # Weighted average of annualized momentum by window size
        am[symbol] = total_annualized_momentum / total_weight if total_weight != 0 else 0

    # Create DataFrame and rank by total annualized momentum
    result_df = pd.DataFrame([{"Symbol": symbol, "am": np.round(am, 2)} for symbol, am in am.items()]).round(2)

    # Sort by momentum sum in descending order and add rank
    result_df = result_df.sort_values("am", ascending=False).reset_index(drop=True)
    result_df["Rank"] = range(1, len(result_df) + 1)

    return result_df[["Rank", "Symbol", "am"]]
