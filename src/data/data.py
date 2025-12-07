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


def calculate_sharpe_ratio_robust(returns: pd.Series, risk_free_rate: float = 0.045, 
                                  trading_days: int = 252, min_observations: int = 60) -> Tuple[float, float]:
    """
    Robust Sharpe ratio calculation with confidence intervals.

    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate (default 0.045 = 4.5%)
        trading_days: Number of trading days per year (default 252)
        min_observations: Minimum number of observations required (default 60)

    Returns:
        Tuple of (sharpe_ratio, se_sharpe) - both rounded to 2 decimal places, or (np.nan, np.nan) if insufficient data
    """
    if len(returns) < min_observations:
        return np.nan, np.nan
    
    # Calculate daily risk-free rate
    daily_rf = (1 + risk_free_rate) ** (1/trading_days) - 1
    
    # Excess returns
    excess_returns = returns - daily_rf
    
    # Remove NaN values
    excess_returns = excess_returns.dropna()
    
    if len(excess_returns) < min_observations:
        return np.nan, np.nan
    
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std(ddof=1)  # Sample standard deviation
    
    if std_excess <= 0:
        return np.nan, np.nan
    
    # Annualized Sharpe ratio
    sharpe_ratio = (mean_excess / std_excess) * np.sqrt(trading_days)
    
    # Standard error (Opdyke, 2007 robust formula)
    skewness = stats.skew(excess_returns)
    kurtosis = stats.kurtosis(excess_returns)
    
    sr_squared = sharpe_ratio ** 2
    se_sharpe = np.sqrt(
        (1 + sr_squared/2 - skewness * sharpe_ratio + 
         (kurtosis - 3) * sr_squared/4) / len(excess_returns)
    )
    
    return round(sharpe_ratio, 2), round(se_sharpe, 2)


def _compute_consecutive_streaks(df: pd.DataFrame, symbol: str) -> tuple[int, int]:
    """
    Compute consecutive positive/negative streaks for a symbol.
    Returns negative numbers for consecutive drops, positive for consecutive increases.

    Args:
        df: DataFrame with price data
        symbol: Symbol to compute streaks for

    Returns:
        Tuple of (last_streak_signed, previous_streak_signed)
    """
    if len(df[symbol]) < 2:
        return 0, 0

    # Calculate daily returns
    returns = df[symbol].pct_change(fill_method=None).dropna()

    if len(returns) < 2:
        return 0, 0

    # Determine if returns are positive or negative
    signs = (returns > 0).astype(int)

    # Find consecutive streaks
    streaks = []
    current_streak = 1
    current_sign = signs.iloc[0]

    for i in range(1, len(signs)):
        if signs.iloc[i] == current_sign:
            current_streak += 1
        else:
            streaks.append((current_streak, current_sign))
            current_streak = 1
            current_sign = signs.iloc[i]

    # Add the last streak
    streaks.append((current_streak, current_sign))

    if len(streaks) == 0:
        return 0, 0
    elif len(streaks) == 1:
        # Apply sign: positive for increases (sign=1), negative for decreases (sign=0)
        last_streak_signed = streaks[0][0] if streaks[0][1] == 1 else -streaks[0][0]
        return last_streak_signed, 0
    else:
        # Last streak and previous streak with signs
        last_streak_length, last_sign = streaks[-1]
        previous_streak_length, previous_sign = streaks[-2]

        last_streak_signed = last_streak_length if last_sign == 1 else -last_streak_length
        previous_streak_signed = previous_streak_length if previous_sign == 1 else -previous_streak_length

        return last_streak_signed, previous_streak_signed


def _compute_average_consecutive_movements(df: pd.DataFrame, symbol: str) -> tuple[float, float]:
    """
    Compute average consecutive up and down movements for a symbol, ignoring 0 movements.
    Uses the largest available window (all data) to calculate averages.
    
    Args:
        df: DataFrame with price data
        symbol: Symbol to compute streaks for
        
    Returns:
        Tuple of (avg_consecutive_up, avg_consecutive_down)
    """
    if len(df[symbol]) < 2:
        return 0.0, 0.0
    
    # Calculate daily returns
    returns = df[symbol].pct_change(fill_method=None).dropna()
    
    if len(returns) < 2:
        return 0.0, 0.0
    
    # Filter out zero movements (returns == 0)
    non_zero_returns = returns[returns != 0]
    
    if len(non_zero_returns) < 2:
        return 0.0, 0.0
    
    # Determine if returns are positive or negative
    signs = (non_zero_returns > 0).astype(int)
    
    # Find consecutive streaks
    streaks = []
    current_streak = 1
    current_sign = signs.iloc[0]
    
    for i in range(1, len(signs)):
        if signs.iloc[i] == current_sign:
            current_streak += 1
        else:
            streaks.append((current_streak, current_sign))
            current_streak = 1
            current_sign = signs.iloc[i]
    
    # Add the last streak
    streaks.append((current_streak, current_sign))
    
    if len(streaks) == 0:
        return 0.0, 0.0
    
    # Separate up and down streaks
    up_streaks = [length for length, sign in streaks if sign == 1]
    down_streaks = [length for length, sign in streaks if sign == 0]
    
    # Calculate exponentially weighted moving average (EWMA) for streak lengths using maximum possible span
    max_span_up = max(len(up_streaks), 1)
    avg_consecutive_up = (
        pd.Series(up_streaks).ewm(span=max_span_up, adjust=False).mean().iloc[-1]
        if up_streaks else 0.0
    )
    max_span_down = max(len(down_streaks), 1)
    avg_consecutive_down = (
        pd.Series(down_streaks).ewm(span=max_span_down, adjust=False).mean().iloc[-1]
        if down_streaks else 0.0
    )
    return avg_consecutive_up, avg_consecutive_down


def _compute_average_percentage_movements(df: pd.DataFrame, symbol: str) -> tuple[float, float]:
    """
    Compute average up and down percentage movements for a symbol, ignoring 0 movements.
    Uses the largest available window (all data) to calculate averages.
    
    Args:
        df: DataFrame with price data
        symbol: Symbol to compute percentage movements for
        
    Returns:
        Tuple of (avg_up_percentage, avg_down_percentage)
    """
    if len(df[symbol]) < 2:
        return 0.0, 0.0
    
    # Calculate daily returns
    returns = df[symbol].pct_change(fill_method=None).dropna()
    
    if len(returns) < 2:
        return 0.0, 0.0
    
    # Filter out zero movements (returns == 0)
    non_zero_returns = returns[returns != 0]
    
    if len(non_zero_returns) < 2:
        return 0.0, 0.0
    
    # Separate positive and negative returns
    up_returns = non_zero_returns[non_zero_returns > 0]
    down_returns = non_zero_returns[non_zero_returns < 0]
    
    # Calculate exponentially weighted averages (EMA) of movements (convert to percentage) using all available data
    max_span = max(len(up_returns), 1)
    avg_up_percentage = up_returns.ewm(span=max_span, adjust=False).mean().iloc[-1] * 100 if len(up_returns) > 0 else 0.0
    max_span_down = max(len(down_returns), 1)
    avg_down_percentage = down_returns.ewm(span=max_span_down, adjust=False).mean().iloc[-1] * 100 if len(down_returns) > 0 else 0.0
    return avg_up_percentage, avg_down_percentage


def get_pcr_m1(ticker_symbol: str) -> float:
    """
    Calculates the volume-based put-call ratio for a given stock ticker.

    Args:
        ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL').

    Returns:
        float: The calculated put-call ratio based on volume, or None if data is unavailable.
    """
    try:
        # Create a ticker object
        ticker = yf.Ticker(ticker_symbol)

        # Get the list of all available expiration dates
        expiration_dates = ticker.options
        # in next month expiration dates
        next_month = (datetime.today().replace(day=1) + timedelta(days=32)).replace(day=1)
        expiration_dates = [
            d
            for d in expiration_dates
            if datetime.strptime(d, "%Y-%m-%d").year == next_month.year and datetime.strptime(d, "%Y-%m-%d").month == next_month.month
        ]
        if not expiration_dates:
            logger.warning(f"No option expiration dates found for {ticker_symbol}.")
            return None

        total_put_volume = 0
        total_call_volume = 0

        # Loop through each expiration date to get the option chain
        for expiry in expiration_dates:
            opt_chain = ticker.option_chain(expiry)

            # Sum volume for all put options
            put_options = opt_chain.puts
            if not put_options.empty:
                total_put_volume += put_options["volume"].sum()

            # Sum volume for all call options
            call_options = opt_chain.calls
            if not call_options.empty:
                total_call_volume += call_options["volume"].sum()

        # Calculate the put-call ratio
        if total_call_volume > 0:
            pcr = total_put_volume / total_call_volume
            return pcr
        else:
            return float("inf")  # Handles division by zero

    except Exception as e:
        logger.error(f"An error occurred calculating put-call ratio for {ticker_symbol}: {e}")
        return None


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


def compute_annualized_momentum_sum(df: pd.DataFrame, window_sizes: List[int] = [7, 30, 90, 180, 360], time_column: str = "Date", metrics_order: List[str] = None, ticker_obj=None) -> pd.DataFrame:
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
        - stride: Compound growth factor based on average consecutive movements
        - s0, s1: Current and previous consecutive streaks
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
        ticker_obj = TICKERS(symbols, period=period_str, normalize=False)
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
    s0_streaks = {}
    s_minus_1_streaks = {}
    avg_s_plus = {}
    avg_s_minus = {}
    avg_plus_percent = {}
    avg_minus_percent = {}
    stride = {}
    d0_latest_change = {}
    drawdown = {}
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
        # Calculate consecutive streaks
        last_streak, previous_streak = _compute_consecutive_streaks(df_prices, symbol)
        s0_streaks[symbol] = last_streak
        s_minus_1_streaks[symbol] = previous_streak

        # Calculate average consecutive movements
        avg_up, avg_down = _compute_average_consecutive_movements(df_prices, symbol)
        avg_s_plus[symbol] = np.round(avg_up, 2)
        avg_s_minus[symbol] = np.round(avg_down, 2)

        # Calculate average percentage movements
        avg_up_pct, avg_down_pct = _compute_average_percentage_movements(df_prices, symbol)
        avg_plus_percent[symbol] = np.round(avg_up_pct, 2)
        avg_minus_percent[symbol] = np.round(avg_down_pct, 2)

        # Calculate stride: (1+avg%+/100)**(avg_s+) * (1-avg%-/100)**(avg_s-)
        avg_s_plus_val = avg_s_plus[symbol]
        avg_s_minus_val = avg_s_minus[symbol]
        sum_s = avg_s_plus_val + avg_s_minus_val
        if sum_s > 0:
            avg_plus_pct_val = avg_plus_percent[symbol]
            avg_minus_pct_val = avg_minus_percent[symbol]
            stride_value = ((1 + avg_plus_pct_val/100) ** avg_s_plus_val * (1 - abs(avg_minus_pct_val)/100) ** avg_s_minus_val) ** (252/sum_s)
            stride[symbol] = np.round(stride_value-1, 2)
        else:
            stride[symbol] = np.nan

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
            sharpe_ratio, se_sharpe = calculate_sharpe_ratio_robust(returns)
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
            "s0": s0_streaks[symbol],
            "s1": s_minus_1_streaks[symbol],
            "stride": stride[symbol],
            "d0": d0_latest_change[symbol],
            "p": current_prices[symbol],
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

    # Use m as combined_score for ranking
    result_df["combined_score"] = result_df["m"].fillna(0)
    
    # Sort by m (momentum) in descending order and add rank
    result_df = result_df.sort_values("m", ascending=False).reset_index(drop=True)
    result_df["Rank"] = range(1, len(result_df) + 1)

    # Load metrics_order from config if not provided
    if metrics_order is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "conf", "main.yaml")
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                metrics_order = config.get("lenses", {}).get("metrics_order", ["m", "sharpe", "p", "ema20", "ema50", "ema200", "rsi_delta", "rsi", "macd_delta", "macd", "drawdown", "combined_score", "stride", "s0", "s1", "d0"])
        except Exception:
            # Fallback to default order
            metrics_order = ["m", "sharpe", "p", "ema20", "ema50", "ema200", "rsi_delta", "rsi", "macd_delta", "macd", "drawdown", "combined_score", "stride", "s0", "s1", "d0"]

    # Create column order: Rank and Symbol first, then metrics_order
    ordered_columns = ["Rank", "Symbol"] + [col for col in metrics_order if col in result_df.columns]

    result_df = result_df[ordered_columns]

    return result_df

