from datetime import datetime
from typing import List, Dict
import pandas as pd
import yfinance as yf
from loguru import logger
from functools import lru_cache, cache
import streamlit as st
import numpy as np


class Ticker:
    """
    attributes:
        * symbol: str
        * ticker: yfinance.Ticker
            * actions: pd.DataFrame; columns: Dividends, Stock Splits, [ETF] Capital Gains; index: Dates
            * analyst_price_targets: [STOCK] dict; keys: current, high, low, mean, median
            * balance_sheet, balancesheet: [STOCK] pd.DataFrame; columns: 5 Dates (yearly); index: financials 76 items
            * calendar : [STOCK] dict; keys: Dividend/Ex-Dividend/Earnings Date, Earnings/Revem=nue High/Low/Average
            * cash_flow, cashflow : [STOCK] pd.DataFrame; columns: 5 Dates (yearly); index: financials x items
            * dividends: pd.Series; index: Dates
            * earnings_dates: [STOCK] pd.DataFrame; columns: EPS Estimate, Reported EPS, Surprise(%); index: Earnings Dates
            * earnings_estimate: [STOCK] pd.DataFrame; columns: avg, low, high, yearAgoEps, numberOfAnalysts, growth; index: period, 0q, +1q, 0y, +1y
            * earnings_history: [STOCK] pd.DataFrame; columns: epsActual, epsEstimate, epsDifference, surprisePercent; index: quarter 4 yearly
            * eps_revisions: [STOCK] pd.DataFrame; columns : upLast7days	upLast30days	downLast30days	downLast7Days; index : period, 0q, +1q, 0y, +1y
            * eps_trend: columns current	7daysAgo	30daysAgo	60daysAgo	90daysAg; index period : 0q, +1q, 0y, +1y
    """

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


def _compute_momentum_for_symbol(df: pd.DataFrame, symbol: str, window: int) -> pd.Series:
    """
    Compute momentum for a single symbol and window size.

    Args:
        df: DataFrame with price data
        symbol: Symbol to compute momentum for
        window: Window size in days

    Returns:
        Series with momentum values
    """
    if len(df[symbol]) < window:
        return pd.Series(dtype=float)

    momentum = df[symbol].rolling(window=window).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
    return momentum.dropna()


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
        # Compute momentum for all symbols at once
        momentum = df[symbols].rolling(window=window).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
        momentum[time_column] = df[time_column]
        momentum = momentum.dropna()
        momentum_data[window] = momentum

        # Get the last window*3 rows for better analysis
        display_rows = min(window, len(momentum))
        momentum_display = momentum.tail(display_rows)

        # Calculate threshold: (1+y1)^(252/window) = target_return
        y1_threshold = target_return ** (window / 252) - 1

        # Initialize counts for this window
        window_counts = {symbol: 0 for symbol in symbols}

        # Count symbols with momentum above threshold
        for symbol in symbols:
            if symbol in momentum_display.columns:
                if len(momentum_display[symbol]) == 0:
                    window_counts[symbol] = 0
                    continue
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
        DataFrame with symbols, individual window annualized momentum columns,
        weighted average momentum, acceleration (moving average of momentum differences), and rank
    """
    symbols = df.select_dtypes(include=[np.number]).columns
    am = {}
    individual_momentum = {}
    individual_accelerations = {}

    # Calculate acceleration for all window sizes except the last one
    acceleration_windows = window_sizes[:-1]  # Exclude the last window

    for symbol in symbols:
        total_annualized_momentum = 0
        total_weight = 0
        symbol_momentum = {}
        symbol_accelerations = {}

        for window in window_sizes:
            # Use the shared momentum calculation function
            momentum = _compute_momentum_for_symbol(df, symbol, window)

            if len(momentum) > 0:
                last_momentum = momentum.iloc[-1]
                # Annualize the momentum: (1 + momentum)^(252/window) - 1, cap at 2
                annualized_momentum = min((1 + last_momentum) ** (252 / window) - 1, 1)
                symbol_momentum[f"m{window}"] = np.round(annualized_momentum, 4)

                weight = 1 / np.log(window)
                weighted_momentum = annualized_momentum * weight
                total_weight += weight
                total_annualized_momentum += weighted_momentum
            else:
                symbol_momentum[f"m{window}"] = np.nan

            # Calculate acceleration for this window if it's not the last one
            if window in acceleration_windows:
                if len(momentum) >= 3:  # Need at least 3 values for meaningful moving average
                    # Calculate momentum differences (A[i] - A[i-1])
                    momentum_diffs = momentum.diff().dropna()
                    
                    # Use a smaller window for moving average
                    ma_window = 2
                    
                    # Calculate moving average of momentum differences
                    acceleration = momentum_diffs.rolling(window=ma_window).mean().iloc[-1]
                    symbol_accelerations[f"a{window}"] = np.round(acceleration, 4)
                else:
                    symbol_accelerations[f"a{window}"] = np.nan

        # Weighted average of annualized momentum by window size
        am[symbol] = total_annualized_momentum / total_weight if total_weight != 0 else 0
        individual_momentum[symbol] = symbol_momentum
        individual_accelerations[symbol] = symbol_accelerations

    # Create DataFrame with individual momentum and acceleration columns
    result_data = []
    for symbol in symbols:
        row = {"Symbol": symbol, "am": np.round(am[symbol], 4)}
        row.update(individual_momentum[symbol])
        row.update(individual_accelerations[symbol])
        result_data.append(row)

    result_df = pd.DataFrame(result_data)

    # Sort by weighted average momentum in descending order and add rank
    result_df = result_df.sort_values("am", ascending=False).reset_index(drop=True)
    result_df["Rank"] = range(1, len(result_df) + 1)

    # Create column order: Rank, Symbol, then momentum-acceleration pairs, then weighted average
    ordered_columns = ["Rank", "Symbol"]
    
    # Add momentum-acceleration pairs for each window (except last)
    for window in window_sizes:
        ordered_columns.append(f"m{window}")
        if window in acceleration_windows:
            ordered_columns.append(f"a{window}")
    
    # Add weighted average at the end
    ordered_columns.append("am")
    
    result_df = result_df[ordered_columns]

    return result_df
