from datetime import datetime, timedelta
from typing import List, Dict
import fear_and_greed
import pandas as pd
import yfinance as yf
from loguru import logger
from functools import lru_cache, cache
import streamlit as st
import numpy as np
import pytz


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
    
    # Calculate averages
    avg_consecutive_up = np.mean(up_streaks) if up_streaks else 0.0
    avg_consecutive_down = np.mean(down_streaks) if down_streaks else 0.0
    
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
    
    # Calculate averages (convert to percentage)
    avg_up_percentage = np.mean(up_returns) * 100 if len(up_returns) > 0 else 0.0
    avg_down_percentage = np.mean(down_returns) * 100 if len(down_returns) > 0 else 0.0
    
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


def compute_annualized_momentum_sum(df: pd.DataFrame, window_sizes: List[int] = [7, 30, 90, 180, 360], time_column: str = "Date") -> pd.DataFrame:
    """
    Compute the sum of annualized momentum across all window sizes for each symbol.

    Args:
        df: DataFrame with price data (normalized or raw)
        window_sizes: List of window sizes in days for momentum calculation
        time_column: Name of the time column

    Returns:
        DataFrame with symbols, individual window annualized momentum columns,
        weighted average momentum (m), acceleration (a), and rank.
        
        Key metrics:
        - m: Weighted average of annualized momentum across all window sizes
        - a: Weighted average acceleration (rate of change of momentum) across window sizes
        - combined_score: Sum of m + a, used for ranking
        - d0: Latest day-to-day percentage change
        - d6: 6-day percentage change (smallest window size)
        - p: Current price
        - ema20: 20-day exponential moving average
        - ema50: 50-day exponential moving average
        - rsi: Relative Strength Index with period 9
        - drawdown: Drop from maximum price observed in the data (1 - current/max, where 1.0 = 100% drop, 0.76 = 76% drop, 0.05 = 5% drop)
        - stride: Compound growth factor based on average consecutive movements
        - s0, s1: Current and previous consecutive streaks
        - avg_s+, avg_s-: Average consecutive up/down movements
        - avg%+, avg%-: Average percentage up/down movements
        - pcr_m1: Put-call ratio for next month expiration
    """
    symbols = df.select_dtypes(include=[np.number]).columns
    m = {}
    individual_momentum = {}
    individual_accelerations = {}
    s0_streaks = {}
    s_minus_1_streaks = {}
    avg_s_plus = {}
    avg_s_minus = {}
    avg_plus_percent = {}
    avg_minus_percent = {}
    stride = {}
    pct_change_smallest_window = {}
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
    a = {}
    # Calculate acceleration for all window sizes except the last one
    acceleration_windows = window_sizes[:-1]  # Exclude the last window
    momentum_windows = window_sizes[1:]
    for symbol in symbols:
        total_annualized_momentum = 0
        total_weight = 0
        symbol_momentum = {}
        symbol_accelerations = {}

        for window in momentum_windows:
            # Use the shared momentum calculation function
            momentum = _compute_momentum_for_symbol(df, symbol, window)

            if len(momentum) > 0:
                if len(momentum) == 1:
                    last_momentum = momentum.iloc[-1]
                else:
                    last_momentum = momentum.iloc[-6:].sum() / 6
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
                    ma_window = 3
                    acceleration = momentum_diffs.rolling(window=ma_window).mean().iloc[-1]
                    acceleration = (1 + acceleration) ** (252 / window) - 1
                    symbol_accelerations[f"a{window}"] = np.round(acceleration, 2)
                else:
                    symbol_accelerations[f"a{window}"] = np.nan

        # Calculate consecutive streaks
        last_streak, previous_streak = _compute_consecutive_streaks(df, symbol)
        s0_streaks[symbol] = last_streak
        s_minus_1_streaks[symbol] = previous_streak

        # Calculate average consecutive movements
        avg_up, avg_down = _compute_average_consecutive_movements(df, symbol)
        avg_s_plus[symbol] = np.round(avg_up, 2)
        avg_s_minus[symbol] = np.round(avg_down, 2)

        # Calculate average percentage movements
        avg_up_pct, avg_down_pct = _compute_average_percentage_movements(df, symbol)
        avg_plus_percent[symbol] = np.round(avg_up_pct, 2)
        avg_minus_percent[symbol] = np.round(avg_down_pct, 2)

        # Calculate stride: (1+avg%+/100)**(avg_s+) * (1-avg%-/100)**(avg_s-)
        avg_s_plus_val = avg_s_plus[symbol]
        avg_s_minus_val = avg_s_minus[symbol]
        sum_s = avg_s_plus_val + avg_s_minus_val
        avg_plus_pct_val = avg_plus_percent[symbol]
        avg_minus_pct_val = avg_minus_percent[symbol]
        
        stride_value = ((1 + avg_plus_pct_val/100) ** avg_s_plus_val * (1 - abs(avg_minus_pct_val)/100) ** avg_s_minus_val) ** (252/sum_s)
        
        stride[symbol] = np.round(stride_value-1, 2)

        # Calculate %change for the smallest window size
        smallest_window = min(window_sizes)
        smallest_window_momentum = _compute_momentum_for_symbol(df, symbol, smallest_window)
        if len(smallest_window_momentum) > 0:
            pct_change_smallest_window[symbol] = np.round(smallest_window_momentum.iloc[-1], 2)
        else:
            pct_change_smallest_window[symbol] = np.nan

        # Calculate d0 (latest day-to-day change)
        if len(df[symbol]) >= 2:
            latest_change = (df[symbol].iloc[-1] / df[symbol].iloc[-2] - 1) 
            d0_latest_change[symbol] = np.round(latest_change, 2)
        else:
            d0_latest_change[symbol] = np.nan

        # Calculate current price (p)
        if len(df[symbol]) > 0:
            current_price = df[symbol].iloc[-1]
            current_prices[symbol] = np.round(current_price, 2)
        else:
            current_prices[symbol] = np.nan

        # Calculate EMA20
        if len(df[symbol]) >= 20:
            ema20 = df[symbol].ewm(span=20, adjust=False).mean()
            current_ema20 = ema20.iloc[-1]
            ema20_values[symbol] = np.round(current_ema20, 2)
        else:
            ema20_values[symbol] = np.nan

        # Calculate EMA50
        if len(df[symbol]) >= 50:
            ema50 = df[symbol].ewm(span=50, adjust=False).mean()
            current_ema50 = ema50.iloc[-1]
            ema50_values[symbol] = np.round(current_ema50, 2)
        else:
            ema50_values[symbol] = np.nan

        # Calculate EMA200
        if len(df[symbol]) >= 200:
            ema200 = df[symbol].ewm(span=200, adjust=False).mean()
            current_ema200 = ema200.iloc[-1]
            ema200_values[symbol] = np.round(current_ema200, 2)
        else:
            ema200_values[symbol] = np.nan

        # Calculate RSI with period 9 using Wilder's smoothing method
        if len(df[symbol]) >= 15:  # Need at least 15 values for RSI(14)
            prices = df[symbol]
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
        if len(df[symbol]) >= 35:  # Need at least 35 values for MACD(12,26,9): 26 for slow EMA + 9 for signal line
            prices = df[symbol]
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

        # Calculate drawdown (drop from maximum price observed in the data)
        if len(df[symbol]) > 0:
            max_price = df[symbol].max()
            current_price = df[symbol].iloc[-1]
            if max_price > 0:
                drawdown[symbol] = np.round(1 - (current_price / max_price), 2)
            else:
                drawdown[symbol] = np.nan
        else:
            drawdown[symbol] = np.nan

        # Calculate put-call ratio
        # pcr = get_pcr_m1(symbol)
        # pcr_m1s[symbol] = pcr if pcr is not None else np.nan

        # Weighted average of annualized momentum by window size
        m[symbol] = total_annualized_momentum / total_weight if total_weight != 0 else 0
        individual_momentum[symbol] = symbol_momentum
        individual_accelerations[symbol] = symbol_accelerations
        weights = [1 / np.log(window) for window in acceleration_windows if f"a{window}" in symbol_accelerations]
        acceleration = sum([symbol_accelerations[f"{window}"] * weights[i] for i, window in enumerate(symbol_accelerations.keys())]) / sum(weights)
        a[symbol] = acceleration
    # Create DataFrame with individual momentum and acceleration columns
    result_data = []
    for symbol in symbols:
        row = {
            "Symbol": symbol,
            "m": np.round(m[symbol], 4),
            "a": np.round(a[symbol], 4),
            "s0": s0_streaks[symbol],
            "s1": s_minus_1_streaks[symbol],
            "avg_s+": avg_s_plus[symbol],
            "avg_s-": avg_s_minus[symbol],
            "avg%+": avg_plus_percent[symbol],
            "avg%-": avg_minus_percent[symbol],
            "stride": stride[symbol],
            # "pcr_m1": pcr_m1s[symbol],
            "d0": d0_latest_change[symbol],
            f"d{min(window_sizes)}": pct_change_smallest_window[symbol],
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
        row.update(individual_momentum[symbol])
        row.update(individual_accelerations[symbol])
        result_data.append(row)

    result_df = pd.DataFrame(result_data)

    # Calculate combined score (m + a) for ranking
    result_df["combined_score"] = result_df["m"] + result_df.get("a", 0).fillna(0)
    
    # Sort by combined score in descending order and add rank
    result_df = result_df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    result_df["Rank"] = range(1, len(result_df) + 1)

    # Create column order: Rank, Symbol, then momentum-acceleration pairs, then weighted average, then streaks
    ordered_columns = ["Rank", "Symbol"]

    # Add momentum-acceleration pairs for each window (except last)
    for window in momentum_windows:
        ordered_columns.append(f"m{window}")
        if window in acceleration_windows:
            ordered_columns.append(f"a{window}")

    # Add weighted average, acceleration, combined score, stride, streaks, average consecutive movements, average percentage movements, p, ema20, ema50, ema200, rsi, rsi_delta, macd, macd_delta, drawdown, and put-call ratio at the end
    ordered_columns.extend(["m", "a", "combined_score", "stride", "s0", "s1", "avg_s+", "avg_s-", "avg%+", "avg%-", "p", "ema20", "ema50", "ema200", "rsi", "rsi_delta", "macd", "macd_delta", "drawdown"]) #, "pcr_m1"])

    result_df = result_df[ordered_columns]

    return result_df

