from typing import List
from src.data.P import get_tickers_close_prices
import yfinance as yf

import pandas as pd
import numpy as np
from typing import Union
import src.indicators as indict
from tqdm import tqdm

class TICKERS:
    """
    tickers = ["SPY", "QQQ"]
    ticker = TICKERS(tickers, period="486d", normalize=True)
    # ✅ ticker.prices #@ P
    # ticker.get_ema(period=50) #@ EMA_50, df
    # ticker.get_rsi() #@ RSI_14, df
    # ticker.calculate_price_pct_change(days_back) #@ pct_changes, dict
    """

    def __init__(self, tickers: List[str], period: str = "5y", normalize: bool = False):
        self.tickers = tickers
        self.tickers_info = {}
        # self.tickers_info = {ticker: yf.Ticker(ticker).info for ticker in tickers}
        # for ticker in tickers:
            # setattr(self, ticker, yf.Ticker(ticker))
        self.period = period
        self.normalize = normalize
        self.prices = get_tickers_close_prices(tickers, period, normalize=self.normalize)
        self.emas = {}
        self.rsis = {}
        self.momentums = {}
        self.pct_changes = {ticker: {} for ticker in tickers}
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

    def get_momentum(self, period: int = 14, log: bool = True) -> pd.DataFrame:
        """
        Calculate momentum for all tickers using calendar dates from the index.
        
        When log=True: calculates log momentum log(P_t / P_{t-period})
        When log=False: calculates regular momentum (P_t / P_{t-period}) - 1
        
        Uses the latest index date as the current date and calculates momentum based on
        calendar days (period days ago) rather than row positions.
        
        Args:
            period: Number of calendar days to look back (default: 14)
            log: If True, calculate log momentum; if False, calculate regular momentum (default: True)
            
        Returns:
            DataFrame with momentum values for each ticker, indexed by date
        """
        cache_key = (period, log)
        
        if cache_key not in self.momentums:
            if self.prices.empty:
                return pd.DataFrame()
            
            # Get the latest index date as current date
            latest_date = self.prices.index.max()
            earliest_required_date = latest_date - pd.Timedelta(days=period)
            
            # Check if we have enough data based on calendar dates
            if self.prices.index.min() > earliest_required_date:
                raise ValueError(
                    f"Need data from at least {period} calendar days ago. "
                    f"Latest date: {latest_date.date()}, earliest available: {self.prices.index.min().date()}, "
                    f"required: {earliest_required_date.date()}"
                )
            
            df = pd.DataFrame(index=self.prices.index)
            
            for ticker in self.tickers:
                if ticker not in self.prices.columns:
                    continue
                
                prices_series = self.prices[ticker].dropna()
                
                if prices_series.empty:
                    df[ticker] = np.nan
                    continue
                
                # Check for non-positive prices
                if (prices_series <= 0).any():
                    raise ValueError(f"All prices must be positive for {ticker}")
                
                # Calculate momentum using calendar dates
                momentum = pd.Series(index=self.prices.index, dtype=float)
                momentum[:] = np.nan
                
                for current_date in prices_series.index:
                    # Find the date that is 'period' calendar days ago
                    target_date = current_date - pd.Timedelta(days=period)
                    
                    # Find the closest available price on or before the target date
                    # Use forward fill to get the most recent available price
                    available_dates = prices_series.index[prices_series.index <= target_date]
                    
                    if len(available_dates) == 0:
                        # No data available before target date
                        momentum[current_date] = np.nan
                        continue
                    
                    # Get the price at the target date (or closest before it)
                    past_price = prices_series.loc[available_dates[-1]]
                    current_price = prices_series.loc[current_date]
                    
                    if pd.isna(past_price) or pd.isna(current_price) or past_price <= 0:
                        momentum[current_date] = np.nan
                        continue
                    
                    # Calculate momentum
                    if log:
                        # Log momentum: log(P_t / P_{t-period})
                        momentum[current_date] = np.log(current_price / past_price)
                    else:
                        # Regular momentum: (P_t / P_{t-period}) - 1
                        momentum[current_date] = (current_price / past_price) - 1
                
                df[ticker] = momentum
                
                # Also store in individual ticker dataframe
                attr_name = f"log_momentum_{period}" if log else f"momentum_{period}"
                getattr(self, ticker)[attr_name] = momentum
            
            self.momentums[cache_key] = df
        
        return self.momentums[cache_key]

    def is_strong(self):
        if "EMA_10" not in self.emas:
            self.get_ema(10)
        if "EMA_20" not in self.emas:
            self.get_ema(20)
        for ticker in self.tickers:
            getattr(self, ticker)["Strong"] = ((getattr(self, ticker)["EMA_10"] > getattr(self, ticker)["EMA_20"])).astype(int)
        self.strong = pd.DataFrame(index=self.prices.index)
        for ticker in self.tickers:
            self.strong[ticker] = getattr(self, ticker)["Strong"]
        return self.strong

    def calculate_price_pct_change(self, days_back: int) -> dict:
        """
        Calculate percent change in price over a specific number of days for all tickers.

        Args:
            days_back: Number of days to look back for the percent change calculation

        Returns:
            Dictionary mapping ticker to percent change value
        """
        if self.prices.empty:
            return {}

        # Check if already calculated for all tickers
        all_calculated = all(days_back in self.pct_changes.get(ticker, {}) for ticker in self.tickers if ticker in self.prices.columns)
        if all_calculated:
            # Return cached values
            return {
                ticker: self.pct_changes[ticker][days_back]
                for ticker in self.tickers
                if ticker in self.prices.columns and days_back in self.pct_changes.get(ticker, {})
            }

        end_date = min(pd.Timestamp.today(), self.prices.index.max())

        # Filter to get data up to end_date
        mask = self.prices.index <= end_date
        price_filtered = self.prices.loc[mask].copy()

        if price_filtered.empty:
            return {}

        # Calculate for all tickers at once
        pct_changes = {}

        # For 1 day change, use last 2 prices
        if days_back <= 1:
            if len(price_filtered) < 2:
                return {}

            for ticker in self.tickers:
                if ticker not in price_filtered.columns:
                    continue

                latest = price_filtered[ticker].iloc[-1]
                previous = price_filtered[ticker].iloc[-2]

                if pd.isna(latest) or pd.isna(previous) or previous == 0:
                    pct_change = 0.0
                else:
                    pct_change = ((latest - previous) / previous) * 100

                pct_changes[ticker] = pct_change
                self.pct_changes[ticker][days_back] = pct_change
        else:
            # For longer periods, find price from days_back ago
            target_date = end_date - pd.Timedelta(days=days_back)
            sorted_prices = price_filtered.sort_index()
            past_prices = sorted_prices[sorted_prices.index <= target_date]

            if past_prices.empty:
                return {}

            for ticker in self.tickers:
                if ticker not in price_filtered.columns:
                    continue

                latest = price_filtered[ticker].iloc[-1]
                if latest is None or pd.isna(latest):
                    pct_change = 0.0
                else:
                    if ticker not in past_prices.columns:
                        pct_change = 0.0
                    else:
                        past_price = past_prices[ticker].iloc[-1]
                        if pd.isna(past_price) or past_price == 0:
                            pct_change = 0.0
                        else:
                            pct_change = ((latest - past_price) / past_price) * 100

                pct_changes[ticker] = pct_change
                self.pct_changes[ticker][days_back] = pct_change

        return pct_changes

    def get_accelerators(self, rsi_period: int = 14, top_n: int = 7) -> List[dict]:
        """
        Find accelerators: tickers with 1 day & 1 week change in top N, and (positive RSI delta or RSI > 50).

        Args:
            rsi_period: RSI period to use (default: 14)
            top_n: Number of top performers to consider (default: 7)

        Returns:
            List of dictionaries with accelerator information:
            - ticker: ticker symbol
            - rsi: current RSI value
            - rsi_delta: change in RSI from previous period
            - pct_1d: 1 day percent change
            - pct_1w: 1 week percent change
        """
        if self.prices.empty:
            return []

        # Ensure RSI is calculated
        rsi_df = self.get_rsi(period=rsi_period)

        if rsi_df.empty:
            return []

        # Calculate latest RSI and RSI delta
        latest_rsi = rsi_df.iloc[-1].dropna()

        # Calculate RSI delta (last RSI - previous RSI)
        if len(rsi_df) >= 2:
            previous_rsi = rsi_df.iloc[-2].dropna()
            rsi_delta = latest_rsi - previous_rsi.reindex(latest_rsi.index, fill_value=0)
        else:
            rsi_delta = pd.Series(0.0, index=latest_rsi.index)

        # Calculate 1 day and 1 week percent changes for all tickers
        pct_change_1d = self.calculate_price_pct_change(1)
        pct_change_1w = self.calculate_price_pct_change(7)

        # Find top N for 1 day and 1 week
        pct_1d_series = pd.Series(pct_change_1d)
        pct_1w_series = pd.Series(pct_change_1w)
        top_n_1d = set(pct_1d_series.nlargest(top_n).index) if len(pct_1d_series) >= top_n else set(pct_1d_series.nlargest(len(pct_1d_series)).index)
        top_n_1w = set(pct_1w_series.nlargest(top_n).index) if len(pct_1w_series) >= top_n else set(pct_1w_series.nlargest(len(pct_1w_series)).index)

        # Find accelerators
        accelerators = []
        for ticker in self.tickers:
            if (
                ticker in top_n_1d
                and ticker in top_n_1w
                and ticker in latest_rsi.index
                and ticker in rsi_delta.index
                and (rsi_delta[ticker] > 0 or latest_rsi[ticker] > 50)
            ):
                accelerators.append(
                    {
                        "ticker": ticker,
                        "rsi": latest_rsi[ticker],
                        "rsi_delta": rsi_delta[ticker],
                        "pct_1d": pct_change_1d.get(ticker, 0),
                        "pct_1w": pct_change_1w.get(ticker, 0),
                    }
                )

        return accelerators

    def get_tickers_info(self) -> dict:
        if not self.tickers_info:
            self.tickers_info = {ticker: yf.Ticker(ticker).info for ticker in tqdm(self.tickers)}
        return self.tickers_info
