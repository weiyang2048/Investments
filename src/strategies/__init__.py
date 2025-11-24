"""
Trading strategies module.

This package contains various trading strategy implementations.
"""

from src.strategies.ma_strat import (
    StrategyResult,
    ema_cross_strategy,
    macd_strategy,
    simple_ema_crossing_strategy,
    ema50_macd_strategy,
    rsi_strategy,
    ema_x_rsi_strategy,
    ema_x_macd_rsi_strategy,
)

__all__ = [
    "StrategyResult",
    "ema_cross_strategy",
    "macd_strategy",
    "simple_ema_crossing_strategy",
    "ema50_macd_strategy",
    "rsi_strategy",
    "ema_x_rsi_strategy",
    "ema_x_macd_rsi_strategy",
]

