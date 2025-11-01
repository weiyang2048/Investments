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

