from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import fear_and_greed
import pandas as pd
import yfinance as yf
from loguru import logger
from functools import lru_cache, cache
import streamlit as st
import numpy as np
import pytz
import yaml
import os
from scipy import stats
from src.data.TICKER import TICKERS


@st.cache_data(ttl="1h")
def get_fear_and_greed_data() -> dict:
    """
    Get fear and greed index data with caching for Streamlit.
    
    Returns:
        dict: Dictionary containing value, description, and last_update_est_str
    """
    fng_data = fear_and_greed.get()
    est = pytz.timezone("US/Eastern")
    last_update_est = fng_data.last_update.astimezone(est)
    value = fng_data.value
    description = fng_data.description
    last_update_est_str = last_update_est.strftime("%H:%M:%S  %Y-%m-%d %Z")
    return {"value": value, "description": description, "last_update_est_str": last_update_est_str}


def compute_symbol_metrics(df: pd.DataFrame, window_sizes: List[int] = [7, 30, 90, 180, 360], time_column: str = "Date", metrics_order: List[str] = None, ticker_obj=None) -> pd.DataFrame:
    """
    Compute annualized momentum using TICKER.get_momentum with period [10,20,50,100,200].

    Args:
        df: DataFrame with price data (normalized or raw), index should be datetime
        window_sizes: List of window sizes in days (kept for compatibility, not used for momentum)
        time_column: Name of the time column (not used if df index is datetime)

    Returns:
        DataFrame with symbols, single momentum (m), and all other stats/indicators.
        
        Key metrics:
        - m: Annualized momentum from TICKER.get_momentum with period [10,20,50,100,200]
        - sharpe: Robust annualized Sharpe ratio (excess returns over 4.5% risk-free rate, 
          using Opdyke 2007 formula with skewness and kurtosis adjustments, min 60 observations)
        - combined_score: Same as m (for backward compatibility), used for ranking
        - d0: Latest day-to-day percentage change
        - p: Current price
        - ema20: 20-day exponential moving average
        - ema50: 50-day exponential moving average
        - ema200: 200-day exponential moving average
        - rsi: Relative Strength Index with period 14
        - drawdown: Drop from maximum price observed in the data
    """
    symbols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Prepare price dataframe for TICKERS (ensure datetime index)
    if df.index.dtype != 'datetime64[ns]':
        if time_column in df.columns:
            df_prices = df.set_index(time_column)
        else:
            df_prices = df.copy()
            df_prices.index = pd.to_datetime(df_prices.index)
    else:
        df_prices = df.copy()
    
    # Use provided ticker_obj if available, otherwise create a new one
    # This avoids duplicate price fetches when ticker_obj is already available
    if ticker_obj is None:
        # Create TICKERS object to compute momentum
        # Calculate period from dataframe length (add some buffer for safety)
        days_needed = max(200, len(df_prices) + 10)  # Need at least 200 days for momentum calculation
        period_str = f"{days_needed}d"
        ticker_obj = TICKERS(symbols, period=period_str, normalize=True)
        # Replace prices with our dataframe (preserving index)
        ticker_obj.prices = df_prices[symbols].copy()
    else:
        # Reuse existing ticker_obj, but ensure prices match our dataframe
        ticker_obj.prices = df_prices[symbols].copy()
    # Update individual ticker dataframes to keep them in sync
    for ticker in symbols:
        if ticker in ticker_obj.prices.columns:
            ticker_df = ticker_obj.prices[[ticker]].copy()
            ticker_df.rename(columns={ticker: "P"}, inplace=True)
            setattr(ticker_obj, ticker, ticker_df)
    
    # Get momentum using TICKER.get_momentum with period [10,20,50,100,200]
    momentum_dict = ticker_obj.get_momentum(period=[10, 20, 50, 100, 200])
    
    # Initialize all other metrics
    d0_latest_change = {}
    drawdown = {}
    ema10_values = {}
    ema20_values = {}
    ema50_values = {}
    ema200_values = {}
    rsi_values = {}
    rsi1_values = {}
    rsi_delta_values = {}
    macd_values = {}
    macd1_values = {}
    macd_delta_values = {}
    current_prices = {}
    sharpe_ratios = {}
    
    for symbol in symbols:
        # Calculate d0 (latest day-to-day change)
        if len(df_prices[symbol]) >= 2:
            latest_change = (df_prices[symbol].iloc[-1] / df_prices[symbol].iloc[-2] - 1) 
            d0_latest_change[symbol] = latest_change
        else:
            d0_latest_change[symbol] = np.nan

        # Calculate current price (p)
        if len(df_prices[symbol]) > 0:
            current_price = df_prices[symbol].iloc[-1]
            current_prices[symbol] = np.round(current_price, 2)
        else:
            current_prices[symbol] = np.nan

        # Calculate EMA10
        if len(df_prices[symbol]) >= 10:
            ema10 = df_prices[symbol].ewm(span=10, adjust=False).mean()
            current_ema10 = ema10.iloc[-1]
            ema10_values[symbol] = np.round(current_ema10, 2)
        else:
            ema10_values[symbol] = np.nan

        # Calculate EMA20
        if len(df_prices[symbol]) >= 20:
            ema20 = df_prices[symbol].ewm(span=20, adjust=False).mean()
            current_ema20 = ema20.iloc[-1]
            ema20_values[symbol] = np.round(current_ema20, 2)
        else:
            ema20_values[symbol] = np.nan

        # Calculate EMA50
        if len(df_prices[symbol]) >= 50:
            ema50 = df_prices[symbol].ewm(span=50, adjust=False).mean()
            current_ema50 = ema50.iloc[-1]
            ema50_values[symbol] = np.round(current_ema50, 2)
        else:
            ema50_values[symbol] = np.nan

        # Calculate EMA200
        if len(df_prices[symbol]) >= 200:
            ema200 = df_prices[symbol].ewm(span=200, adjust=False).mean()
            current_ema200 = ema200.iloc[-1]
            ema200_values[symbol] = np.round(current_ema200, 2)
        else:
            ema200_values[symbol] = np.nan

        # Calculate RSI with period 14 using Wilder's smoothing method
        if len(df_prices[symbol]) >= 15:
            prices = df_prices[symbol]
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            period = 14
            # Initialize series for Wilder's smoothed averages
            avg_gain = pd.Series(index=prices.index, dtype=float)
            avg_loss = pd.Series(index=prices.index, dtype=float)
            
            # First average is SMA
            avg_gain.iloc[period] = gain.iloc[1:period+1].mean()
            avg_loss.iloc[period] = loss.iloc[1:period+1].mean()
            
            # Apply Wilder's smoothing for subsequent values
            for i in range(period + 1, len(prices)):
                avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
                avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            rsi_values[symbol] = np.round(rsi.iloc[-1], 2) if not pd.isna(rsi.iloc[-1]) else np.nan
            # Calculate RSI for previous day (rsi1) - used for rsi_delta calculation
            if len(rsi) >= 2:
                rsi1_val = rsi.iloc[-2] if not pd.isna(rsi.iloc[-2]) else np.nan
                rsi1_values[symbol] = np.round(rsi1_val, 2) if not pd.isna(rsi1_val) else np.nan
                
                # Calculate rsi_delta = rsi - rsi1
                rsi_val = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else np.nan
                if not pd.isna(rsi1_val) and not pd.isna(rsi_val):
                    rsi_delta_values[symbol] = np.round(rsi_val - rsi1_val, 2)
                else:
                    rsi_delta_values[symbol] = np.nan
            else:
                rsi1_values[symbol] = np.nan
                rsi_delta_values[symbol] = np.nan
        else:
            rsi_values[symbol] = np.nan
            rsi1_values[symbol] = np.nan
            rsi_delta_values[symbol] = np.nan

        # Calculate MACD (12, 26, 9)
        if len(df_prices[symbol]) >= 35:  # Need at least 35 values for MACD(12,26,9): 26 for slow EMA + 9 for signal line
            prices = df_prices[symbol]
            # Normalize prices so that average price is 100
            avg_price = prices.mean()
            if avg_price > 0:
                normalization_factor = 100.0 / avg_price
                normalized_prices = prices * normalization_factor
            else:
                normalized_prices = prices
            
            # Calculate MACD components on normalized prices
            exp1 = normalized_prices.ewm(span=12, adjust=False).mean()  # Fast EMA
            exp2 = normalized_prices.ewm(span=26, adjust=False).mean()  # Slow EMA
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line
            # Use the MACD histogram value (not line or signal line)
            macd_values[symbol] = np.round(histogram.iloc[-1], 2) if not pd.isna(histogram.iloc[-1]) else np.nan
            
            # Calculate MACD for previous day (macd1) - used for macd_delta calculation
            if len(histogram) >= 2:
                macd1_val = histogram.iloc[-2] if not pd.isna(histogram.iloc[-2]) else np.nan
                macd1_values[symbol] = np.round(macd1_val, 2) if not pd.isna(macd1_val) else np.nan
                
                # Calculate macd_delta = macd - macd1
                macd_val = histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else np.nan
                if not pd.isna(macd1_val) and not pd.isna(macd_val):
                    macd_delta_values[symbol] = np.round(macd_val - macd1_val, 2)
                else:
                    macd_delta_values[symbol] = np.nan
            else:
                macd1_values[symbol] = np.nan
                macd_delta_values[symbol] = np.nan
        else:
            macd_values[symbol] = np.nan
            macd1_values[symbol] = np.nan
            macd_delta_values[symbol] = np.nan

        # Calculate drawdown
        if len(df_prices[symbol]) > 0:
            max_price = df_prices[symbol].max()
            current_price = df_prices[symbol].iloc[-1]
            if max_price > 0:
                drawdown[symbol] = np.round(1 - (current_price / max_price), 2)
            else:
                drawdown[symbol] = np.nan
        else:
            drawdown[symbol] = np.nan
        
        # Calculate Sharpe ratio
        if len(df_prices[symbol]) >= 2:
            returns = df_prices[symbol].pct_change(fill_method=None).dropna()
            sharpe_ratio = ticker_obj.calculate_sharpe_ratio_robust(returns)
            sharpe_ratios[symbol] = sharpe_ratio
        else:
            sharpe_ratios[symbol] = np.nan
    # Create DataFrame
    result_data = []
    for symbol in symbols:
        row = {
            "Symbol": symbol,
            "m": np.round(momentum_dict.get(symbol, np.nan), 4),
            "sharpe": sharpe_ratios[symbol],
            "d0": d0_latest_change[symbol],
            "p": current_prices[symbol],
            "ema10": ema10_values[symbol],
            "ema20": ema20_values[symbol],
            "ema50": ema50_values[symbol],
            "ema200": ema200_values[symbol],
            "rsi": rsi_values[symbol],
            "rsi_delta": rsi_delta_values[symbol],
            "macd": macd_values[symbol],
            "macd_delta": macd_delta_values[symbol],
            "drawdown": drawdown[symbol],
        }
        result_data.append(row)

    result_df = pd.DataFrame(result_data)

    # Sort by m (momentum) in descending order and add rank
    result_df = result_df.sort_values("m", ascending=False).reset_index(drop=True)
    result_df["Rank"] = range(1, len(result_df) + 1)

    # Load metrics_order from config if not provided
    if metrics_order is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "conf", "main.yaml")
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                metrics_order = config.get("lenses", {}).get("metrics_order", ["m", "sharpe", "p", "d0", "ema10", "ema20", "ema50", "ema200", "rsi_delta", "rsi", "macd_delta", "macd", "drawdown"])
        except Exception:
            # Fallback to default order
            metrics_order = ["m", "sharpe", "p", "d0", "ema10", "ema20", "ema50", "ema200", "rsi_delta", "rsi", "macd_delta", "macd", "drawdown"]

    # Create column order: Rank and Symbol first, then metrics_order
    ordered_columns = ["Rank", "Symbol"] + [col for col in metrics_order if col in result_df.columns]

    result_df = result_df[ordered_columns]

    return result_df

