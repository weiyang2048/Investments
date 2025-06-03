from typing import List
import pandas as pd
import yfinance as yf
from functools import lru_cache
from loguru import logger


def get_daily_prices(symbols: List[str], period: str = "1mo") -> pd.DataFrame:
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
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        df["Symbol"] = symbol
        dfs.append(df)
        logger.opt(ansi=True).log(
            "data",
            f"Downloaded {symbol} data for the last {period}",
        )
    if not dfs:
        return pd.DataFrame()

    combined_df = pd.concat(dfs)
    return combined_df


if __name__ == "__main__":
    df = get_daily_prices(["GDE", "AIVI", "DOL", "DGRW"], "5y")
