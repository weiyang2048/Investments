"""
Moving Average (MA) based trading strategies.

This module contains EMA crossover, EMA_x, MACD, RSI, and combined strategies
for trading strategy backtesting.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class StrategyResult:
    """Container for strategy signals and indicators"""

    signals: pd.DataFrame  # Contains 'signal' and 'positions' columns
    indicators: pd.DataFrame  # Contains all calculated indicators

    def get_buy_signals(self) -> pd.Series:
        """Get dates where buy signals occur (position changes from 0 to 1)"""
        return self.signals[self.signals["positions"] == 1.0].index

    def get_sell_signals(self) -> pd.Series:
        """Get dates where sell signals occur (position changes from 1 to 0)"""
        return self.signals[self.signals["positions"] == -1.0].index


def ema_cross_strategy(data: pd.Series, short_window: int = 50, long_window: int = 200) -> StrategyResult:
    """
    EMA Crossover Strategy: Buy when short EMA crosses above long EMA, sell when it crosses below.

    Args:
        data: Price series (typically adjusted close)
        short_window: Period for short EMA (default: 50)
        long_window: Period for long EMA (default: 200)

    Returns:
        StrategyResult with signals and indicators
    """
    signals = pd.DataFrame(index=data.index)
    signals["price"] = data

    # Calculate EMAs
    signals["short_ema"] = data.ewm(span=short_window, adjust=False).mean()
    signals["long_ema"] = data.ewm(span=long_window, adjust=False).mean()

    # Drop rows where we don't have enough data for the longest window (long_window)
    # This ensures all EMAs are fully formed
    signals = signals.iloc[long_window:].copy()

    # Generate signals: 1 when short EMA > long EMA, 0 otherwise
    signals["signal"] = (signals["short_ema"] > signals["long_ema"]).astype(float)

    # Calculate position changes (1 = buy, -1 = sell, 0 = hold)
    signals["positions"] = signals["signal"].diff()

    # Handle initial state: if short_ema > long_ema at start, hold; otherwise wait for next signal
    if len(signals) > 0:
        if signals.iloc[0]["signal"] == 1.0:
            # Start with holding position if short_ema > long_ema
            signals.iloc[0, signals.columns.get_loc("positions")] = 1.0
        else:
            # Don't hold initially, wait for next signal to enter
            signals.iloc[0, signals.columns.get_loc("positions")] = 0.0

    indicators = signals[["short_ema", "long_ema"]].copy()

    return StrategyResult(signals=signals[["signal", "positions"]], indicators=indicators)


def macd_strategy(data: pd.Series, fast_window: int = 12, slow_window: int = 26, signal_window: int = 9) -> StrategyResult:
    """
    MACD Strategy: Buy when MACD line crosses above signal line, sell when it crosses below.

    Args:
        data: Price series (typically adjusted close)
        fast_window: Period for fast EMA (default: 12)
        slow_window: Period for slow EMA (default: 26)
        signal_window: Period for signal line EMA (default: 9)

    Returns:
        StrategyResult with signals and indicators
    """
    signals = pd.DataFrame(index=data.index)
    signals["price"] = data

    # Calculate MACD components
    exp1 = data.ewm(span=fast_window, adjust=False).mean()
    exp2 = data.ewm(span=slow_window, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    histogram = macd_line - signal_line

    signals["macd"] = macd_line
    signals["signal_line"] = signal_line
    signals["histogram"] = histogram

    # Drop rows where we don't have enough data for the signal line
    # Need slow_window for MACD calculation, then signal_window for signal line
    min_period = slow_window + signal_window
    signals = signals.iloc[min_period:].copy()

    # Generate signals: 1 when MACD > signal line, 0 otherwise
    signals["signal"] = (signals["macd"] > signals["signal_line"]).astype(float)

    # Calculate position changes (1 = buy, -1 = sell, 0 = hold)
    signals["positions"] = signals["signal"].diff()
    
    # Handle initial state: if MACD > signal line at start, hold; otherwise wait for next signal
    if len(signals) > 0:
        if signals.iloc[0]["signal"] == 1.0:
            # Start with holding position if MACD > signal line
            signals.iloc[0, signals.columns.get_loc("positions")] = 1.0
        else:
            # Don't hold initially, wait for next signal to enter
            signals.iloc[0, signals.columns.get_loc("positions")] = 0.0

    indicators = signals[["macd", "signal_line", "histogram"]].copy()

    return StrategyResult(signals=signals[["signal", "positions"]], indicators=indicators)


def simple_ema_crossing_strategy(data: pd.Series, ema_period: int = 50) -> StrategyResult:
    """
    EMA_x Strategy: Buy when price crosses above EMA, sell when price crosses below EMA.

    Args:
        data: Price series (typically adjusted close)
        ema_period: Period for EMA (default: 50)

    Returns:
        StrategyResult with signals and indicators
    """
    signals = pd.DataFrame(index=data.index)
    signals["price"] = data

    # Calculate EMA
    signals["ema"] = data.ewm(span=ema_period, adjust=False).mean()

    # Drop rows where we don't have enough data for the EMA
    # This ensures the EMA is fully formed
    signals = signals.iloc[ema_period:].copy()

    # Generate signals: 1 when price > EMA, 0 otherwise
    signals["signal"] = (signals["price"] > signals["ema"]).astype(float)

    # Calculate position changes (1 = buy, -1 = sell, 0 = hold)
    signals["positions"] = signals["signal"].diff()

    # Handle initial state: if price > EMA at start, hold; otherwise wait for next signal
    if len(signals) > 0:
        if signals.iloc[0]["signal"] == 1.0:
            # Start with holding position if price > EMA
            signals.iloc[0, signals.columns.get_loc("positions")] = 1.0
        else:
            # Don't hold initially, wait for next signal to enter
            signals.iloc[0, signals.columns.get_loc("positions")] = 0.0

    indicators = signals[["ema"]].copy()

    return StrategyResult(signals=signals[["signal", "positions"]], indicators=indicators)


def ema50_macd_strategy(
    data: pd.Series,
    ema_period: int = 50,
    macd_fast_window: int = 12,
    macd_slow_window: int = 26,
    macd_signal_window: int = 9,
) -> StrategyResult:
    """
    EMA_x + MACD Combined Strategy:
    - When not holding: buy when price > EMA_x AND MACD > 0
    - When holding: sell when price < EMA_x AND MACD < 0

    Args:
        data: Price series (typically adjusted close)
        ema_period: Period for EMA (default: 50)
        macd_fast_window: Period for fast EMA in MACD (default: 12)
        macd_slow_window: Period for slow EMA in MACD (default: 26)
        macd_signal_window: Period for signal line EMA in MACD (default: 9)

    Returns:
        StrategyResult with signals and indicators
    """
    signals = pd.DataFrame(index=data.index)
    signals["price"] = data

    # Calculate EMA50
    signals["ema50"] = data.ewm(span=ema_period, adjust=False).mean()

    # Calculate MACD components
    exp1 = data.ewm(span=macd_fast_window, adjust=False).mean()
    exp2 = data.ewm(span=macd_slow_window, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=macd_signal_window, adjust=False).mean()
    histogram = macd_line - signal_line

    signals["macd"] = macd_line
    signals["signal_line"] = signal_line
    signals["histogram"] = histogram

    # Drop rows where we don't have enough data
    # Need max(ema_period, macd_slow_window + macd_signal_window)
    min_period = max(ema_period, macd_slow_window + macd_signal_window)
    signals = signals.iloc[min_period:].copy()

    # Generate signals based on combined conditions
    # Buy condition: price > EMA_x AND MACD > 0
    buy_condition = (signals["price"] > signals["ema50"]) & (signals["macd"] > 0)
    # Sell condition: price < EMA_x AND MACD < 0
    sell_condition = (signals["price"] < signals["ema50"]) & (signals["macd"] < 0)

    # Initialize signal column
    signals["signal"] = 0.0

    # State-based signal generation
    # Start with no position
    current_signal = 0.0

    for i in range(len(signals)):
        if buy_condition.iloc[i] and current_signal == 0.0:
            # Buy: when not holding and buy condition is met
            current_signal = 1.0
        elif sell_condition.iloc[i] and current_signal == 1.0:
            # Sell: when holding and sell condition is met
            current_signal = 0.0
        # Otherwise, maintain current position

        signals.iloc[i, signals.columns.get_loc("signal")] = current_signal

    # Calculate position changes (1 = buy, -1 = sell, 0 = hold)
    signals["positions"] = signals["signal"].diff()

    # Handle initial state
    if len(signals) > 0:
        if signals.iloc[0]["signal"] == 1.0:
            # Start with holding position if buy condition is met at start
            signals.iloc[0, signals.columns.get_loc("positions")] = 1.0
        else:
            # Don't hold initially
            signals.iloc[0, signals.columns.get_loc("positions")] = 0.0

    indicators = signals[["ema50", "macd", "signal_line", "histogram"]].copy()

    return StrategyResult(signals=signals[["signal", "positions"]], indicators=indicators)


def rsi_strategy(data: pd.Series, rsi_period: int = 14) -> StrategyResult:
    """
    RSI Strategy: Buy when RSI > 50, sell when RSI < 50.

    Args:
        data: Price series (typically adjusted close)
        rsi_period: Period for RSI calculation (default: 14)

    Returns:
        StrategyResult with signals and indicators
    """
    signals = pd.DataFrame(index=data.index)
    signals["price"] = data

    # Calculate RSI using Wilder's smoothing method
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Initialize series for Wilder's smoothed averages
    avg_gain = pd.Series(index=data.index, dtype=float)
    avg_loss = pd.Series(index=data.index, dtype=float)

    # First average is SMA
    if len(data) > rsi_period:
        avg_gain.iloc[rsi_period] = gain.iloc[1:rsi_period + 1].mean()
        avg_loss.iloc[rsi_period] = loss.iloc[1:rsi_period + 1].mean()

        # Apply Wilder's smoothing for subsequent values
        for i in range(rsi_period + 1, len(data)):
            avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (rsi_period - 1) + gain.iloc[i]) / rsi_period
            avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (rsi_period - 1) + loss.iloc[i]) / rsi_period

        # Calculate RS and RSI
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = pd.Series(index=data.index, dtype=float)

    signals["rsi"] = rsi

    # Drop rows where we don't have enough data for RSI
    # Need at least rsi_period + 1 values
    signals = signals.iloc[rsi_period:].copy()

    # Generate signals: 1 when RSI > 50, 0 otherwise
    signals["signal"] = (signals["rsi"] > 50).astype(float)

    # Calculate position changes (1 = buy, -1 = sell, 0 = hold)
    signals["positions"] = signals["signal"].diff()

    # Handle initial state: if RSI > 50 at start, hold; otherwise wait for next signal
    if len(signals) > 0:
        if signals.iloc[0]["signal"] == 1.0:
            # Start with holding position if RSI > 50
            signals.iloc[0, signals.columns.get_loc("positions")] = 1.0
        else:
            # Don't hold initially, wait for next signal to enter
            signals.iloc[0, signals.columns.get_loc("positions")] = 0.0

    indicators = signals[["rsi"]].copy()

    return StrategyResult(signals=signals[["signal", "positions"]], indicators=indicators)


def ema_x_rsi_strategy(
    data: pd.Series,
    ema_period: int = 50,
    rsi_period: int = 14,
) -> StrategyResult:
    """
    EMA_x + RSI Combined Strategy:
    - When not holding: buy when price > EMA_x AND RSI > 50
    - When holding: sell when price < EMA_x AND RSI < 50

    Args:
        data: Price series (typically adjusted close)
        ema_period: Period for EMA (default: 50)
        rsi_period: Period for RSI calculation (default: 14)

    Returns:
        StrategyResult with signals and indicators
    """
    signals = pd.DataFrame(index=data.index)
    signals["price"] = data

    # Calculate EMA_x
    signals["ema"] = data.ewm(span=ema_period, adjust=False).mean()

    # Calculate RSI using Wilder's smoothing method
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Initialize series for Wilder's smoothed averages
    avg_gain = pd.Series(index=data.index, dtype=float)
    avg_loss = pd.Series(index=data.index, dtype=float)

    # First average is SMA
    if len(data) > rsi_period:
        avg_gain.iloc[rsi_period] = gain.iloc[1:rsi_period + 1].mean()
        avg_loss.iloc[rsi_period] = loss.iloc[1:rsi_period + 1].mean()

        # Apply Wilder's smoothing for subsequent values
        for i in range(rsi_period + 1, len(data)):
            avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (rsi_period - 1) + gain.iloc[i]) / rsi_period
            avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (rsi_period - 1) + loss.iloc[i]) / rsi_period

        # Calculate RS and RSI
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = pd.Series(index=data.index, dtype=float)

    signals["rsi"] = rsi

    # Drop rows where we don't have enough data
    # Need max(ema_period, rsi_period)
    min_period = max(ema_period, rsi_period)
    signals = signals.iloc[min_period:].copy()

    # Generate signals based on combined conditions
    # Buy condition: price > EMA_x AND RSI > 50
    buy_condition = (signals["price"] > signals["ema"]) & (signals["rsi"] > 50)
    # Sell condition: price < EMA_x AND RSI < 50
    sell_condition = (signals["price"] < signals["ema"]) & (signals["rsi"] < 50)

    # Initialize signal column
    signals["signal"] = 0.0

    # State-based signal generation
    # Start with no position
    current_signal = 0.0

    for i in range(len(signals)):
        if buy_condition.iloc[i] and current_signal == 0.0:
            # Buy: when not holding and buy condition is met
            current_signal = 1.0
        elif sell_condition.iloc[i] and current_signal == 1.0:
            # Sell: when holding and sell condition is met
            current_signal = 0.0
        # Otherwise, maintain current position

        signals.iloc[i, signals.columns.get_loc("signal")] = current_signal

    # Calculate position changes (1 = buy, -1 = sell, 0 = hold)
    signals["positions"] = signals["signal"].diff()

    # Handle initial state
    if len(signals) > 0:
        if signals.iloc[0]["signal"] == 1.0:
            # Start with holding position if buy condition is met at start
            signals.iloc[0, signals.columns.get_loc("positions")] = 1.0
        else:
            # Don't hold initially
            signals.iloc[0, signals.columns.get_loc("positions")] = 0.0

    indicators = signals[["ema", "rsi"]].copy()

    return StrategyResult(signals=signals[["signal", "positions"]], indicators=indicators)


def ema_x_macd_rsi_strategy(
    data: pd.Series,
    ema_period: int = 50,
    macd_fast_window: int = 12,
    macd_slow_window: int = 26,
    macd_signal_window: int = 9,
    rsi_period: int = 14,
) -> StrategyResult:
    """
    EMA_x + MACD + RSI Combined Strategy:
    - When not holding: buy when price > EMA_x AND MACD > 0 AND RSI > 50
    - When holding: sell when price < EMA_x AND MACD < 0 AND RSI < 50

    Args:
        data: Price series (typically adjusted close)
        ema_period: Period for EMA (default: 50)
        macd_fast_window: Period for fast EMA in MACD (default: 12)
        macd_slow_window: Period for slow EMA in MACD (default: 26)
        macd_signal_window: Period for signal line EMA in MACD (default: 9)
        rsi_period: Period for RSI calculation (default: 14)

    Returns:
        StrategyResult with signals and indicators
    """
    signals = pd.DataFrame(index=data.index)
    signals["price"] = data

    # Calculate EMA_x
    signals["ema"] = data.ewm(span=ema_period, adjust=False).mean()

    # Calculate MACD components
    exp1 = data.ewm(span=macd_fast_window, adjust=False).mean()
    exp2 = data.ewm(span=macd_slow_window, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=macd_signal_window, adjust=False).mean()
    histogram = macd_line - signal_line

    signals["macd"] = macd_line
    signals["signal_line"] = signal_line
    signals["histogram"] = histogram

    # Calculate RSI using Wilder's smoothing method
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Initialize series for Wilder's smoothed averages
    avg_gain = pd.Series(index=data.index, dtype=float)
    avg_loss = pd.Series(index=data.index, dtype=float)

    # First average is SMA
    if len(data) > rsi_period:
        avg_gain.iloc[rsi_period] = gain.iloc[1:rsi_period + 1].mean()
        avg_loss.iloc[rsi_period] = loss.iloc[1:rsi_period + 1].mean()

        # Apply Wilder's smoothing for subsequent values
        for i in range(rsi_period + 1, len(data)):
            avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (rsi_period - 1) + gain.iloc[i]) / rsi_period
            avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (rsi_period - 1) + loss.iloc[i]) / rsi_period

        # Calculate RS and RSI
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = pd.Series(index=data.index, dtype=float)

    signals["rsi"] = rsi

    # Drop rows where we don't have enough data
    # Need max(ema_period, macd_slow_window + macd_signal_window, rsi_period)
    min_period = max(ema_period, macd_slow_window + macd_signal_window, rsi_period)
    signals = signals.iloc[min_period:].copy()

    # Generate signals based on combined conditions
    # Buy condition: price > EMA_x AND MACD > 0 AND RSI > 50
    buy_condition = (signals["price"] > signals["ema"]) & (signals["macd"] > 0) & (signals["rsi"] > 50)
    # Sell condition: price < EMA_x AND MACD < 0 AND RSI < 50
    sell_condition = (signals["price"] < signals["ema"]) & (signals["macd"] < 0) & (signals["rsi"] < 50)

    # Initialize signal column
    signals["signal"] = 0.0

    # State-based signal generation
    # Start with no position
    current_signal = 0.0

    for i in range(len(signals)):
        if buy_condition.iloc[i] and current_signal == 0.0:
            # Buy: when not holding and buy condition is met
            current_signal = 1.0
        elif sell_condition.iloc[i] and current_signal == 1.0:
            # Sell: when holding and sell condition is met
            current_signal = 0.0
        # Otherwise, maintain current position

        signals.iloc[i, signals.columns.get_loc("signal")] = current_signal

    # Calculate position changes (1 = buy, -1 = sell, 0 = hold)
    signals["positions"] = signals["signal"].diff()

    # Handle initial state
    if len(signals) > 0:
        if signals.iloc[0]["signal"] == 1.0:
            # Start with holding position if buy condition is met at start
            signals.iloc[0, signals.columns.get_loc("positions")] = 1.0
        else:
            # Don't hold initially
            signals.iloc[0, signals.columns.get_loc("positions")] = 0.0

    indicators = signals[["ema", "macd", "signal_line", "histogram", "rsi"]].copy()

    return StrategyResult(signals=signals[["signal", "positions"]], indicators=indicators)

