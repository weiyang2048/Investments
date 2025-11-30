from typing import List
from src.data.P import get_tickers_close_prices
import yfinance as yf

import pandas as pd
from typing import Union
import src.indicators.INDICT as indict


class TICKERS:
    """
    tickers = ["SPY", "QQQ"]
    ticker = TICKERS(tickers, period="486d", normalize=True)
    # ticker.prices #@ P
    # ticker.get_ema(period=50) #@ EMA_50
    # ticker.get_rsi() #@ RSI_14
    """
    def __init__(self, tickers: List[str], period: str = "5y", normalize: bool = False):
        self.tickers = tickers
        self.period = period
        self.normalize = normalize
        self.prices = get_tickers_close_prices(tickers, period, normalize=self.normalize)
        self.emas = {}
        self.rsis = {}
        for ticker in tickers:
            ticker_df = self.prices[[ticker]].copy()
            ticker_df.rename(columns={ticker: "P"}, inplace=True)
            setattr(self, ticker, ticker_df)

    def get_ema(self, period: int) -> pd.Series:
        if period not in self.emas:
            df = pd.DataFrame(index=self.prices.index)
            for ticker in self.tickers:
                df[ticker] = indict.compute_ema(getattr(self, ticker)["P"], period).rename(f"EMA_{period}")
                getattr(self, ticker)[f"EMA_{period}"] = df[ticker]
            self.emas[period] = df
        return self.emas[period]

    def get_rsi(self, period: int = 14) -> pd.Series:
        if period not in self.rsis:
            df = pd.DataFrame(index=self.prices.index)
            for ticker in self.tickers:
                df[ticker] = indict.compute_rsi(getattr(self, ticker)["P"], period).rename(f"RSI_{period}")
                getattr(self, ticker)[f"RSI_{period}"] = df[ticker]
            self.rsis[period] = df
        return self.rsis[period]

    def is_strong(self):
        if "EMA_10" not in self.emas:
            self.get_ema(10)
        if "EMA_20" not in self.emas:
            self.get_ema(20)
        for ticker in self.tickers:
            getattr(self, ticker)["Strong"] = (
                (getattr(self, ticker)["EMA_10"] > getattr(self, ticker)["EMA_20"])
            ).astype(int)
        self.strong = pd.DataFrame(index=self.prices.index)
        for ticker in self.tickers:
            self.strong[ticker] = getattr(self, ticker)["Strong"]
        return self.strong
