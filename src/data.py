from datetime import datetime
from typing import List
import pandas as pd
import yfinance as yf
from loguru import logger
from functools import lru_cache, cache


@cache
def get_daily_prices(symbol: str, period: str = "1mo") -> pd.DataFrame:
    """
    Get daily price data for a single symbol using yfinance
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    return df


def get_daily_prices_list(symbols: List[str], period: str = "1mo") -> pd.DataFrame:
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
    # Symbol is the first column
    combined_df = combined_df[["Symbol"] + [col for col in combined_df.columns if col != "Symbol"]]
    # if time before 9:30 est, get the ticker.info['preMarketPrice']
    # if time after 9:30 est, get the ticker.info['regularMarketPrice']
    # nyse = yf.Market('NYSE')
    # nyse = nyse.status
    # if nyse.get("status") == "closed":
    #     if "WILL_OPEN" in nyse.get("yfit_market_status","").upper():
    #         today = datetime.now().date()
    #         for symbol in symbols:
    #             ticker = yf.Ticker(symbol)
    #             pre_market_price = ticker.info['preMarketPrice']
    #             # add a row to the combined_df
    #             combined_df = pd.concat([combined_df, pd.DataFrame({
    #                 "Symbol": symbol,
    #                 "Date": today,
    #                 "Close": pre_market_price
    #             })], ignore_index=True)

    return combined_df


if __name__ == "__main__":
    df = get_daily_prices_list(["GDE", "AIVI", "DOL", "DGRW"], "5y")
