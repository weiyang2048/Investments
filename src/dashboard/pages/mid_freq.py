"""
Streamlit page for mid-frequency trading strategy backtesting.

This page allows users to backtest EMA 50/200 crossover, simple EMA crossing, and MACD strategies
with customizable parameters and interactive visualizations.
"""

from datetime import date, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
import hydra
import yfinance as yf
from src.configurations.yaml import register_resolvers
from src.dashboard.create_page import setup_page_and_sidebar


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


def fetch_data(symbol: str, period: str, end_date: Optional[str] = None) -> pd.Series:
    """
    Fetch historical price data for a symbol using yfinance.
    Follows the pattern from src.data.price.

    Args:
        symbol: Stock ticker symbol
        period: Lookback period (e.g., '1y', '5y', '10y', 'max')
        end_date: End date in 'YYYY-MM-DD' format (defaults to today)

    Returns:
        Series with close prices
    """
    from datetime import date, timedelta

    symbol = symbol.replace(".", "-")
    ticker = yf.Ticker(symbol)

    # Default end_date to today if not provided
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    # Calculate start_date from period and end_date
    end_dt = pd.to_datetime(end_date).date()

    if period == "max":
        # For max, fetch all available data up to end_date
        data = ticker.history(period="max", end=end_date)
    else:
        # Parse period and calculate start_date
        if period.endswith("y"):
            years = int(period[:-1])
            start_dt = end_dt - timedelta(days=years * 365)
        elif period.endswith("mo"):
            months = int(period[:-2])
            start_dt = end_dt - timedelta(days=months * 30)
        elif period.endswith("d"):
            days = int(period[:-1])
            start_dt = end_dt - timedelta(days=days)
        else:
            # Try to use period directly
            data = ticker.history(period=period, end=end_date)
            if data.empty:
                raise ValueError(f"No data retrieved for {symbol}")
            logger.info(f"Fetched {len(data)} days of data for {symbol} (period: {period}, end: {end_date})")
            return data["Close"]

        start_date = start_dt.strftime("%Y-%m-%d")
        data = ticker.history(start=start_date, end=end_date)

    if data.empty:
        raise ValueError(f"No data retrieved for {symbol}")

    logger.info(f"Fetched {len(data)} days of data for {symbol} (period: {period}, end: {end_date})")
    return data["Close"]


def ema_2crossings_strategy(data: pd.Series, short_window: int = 50, long_window: int = 200) -> StrategyResult:
    """
    EMA 2 Crossings Strategy: Buy when short EMA > long EMA AND price > short EMA, sell when either condition fails.

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

    # Generate signals: 1 when short EMA > long EMA AND price > short EMA, 0 otherwise
    signals["signal"] = ((signals["short_ema"] > signals["long_ema"]) & (signals["price"] > signals["short_ema"])).astype(float)

    # Calculate position changes (1 = buy, -1 = sell, 0 = hold)
    signals["positions"] = signals["signal"].diff()

    # Handle initial state: if short_ema > long_ema AND price > short_ema at start, hold; otherwise wait for next signal
    if len(signals) > 0:
        if signals.iloc[0]["signal"] == 1.0:
            # Start with holding position if short_ema > long_ema AND price > short_ema
            signals.iloc[0, signals.columns.get_loc("positions")] = 1.0
        else:
            # Don't hold initially, wait for next signal to enter
            signals.iloc[0, signals.columns.get_loc("positions")] = 0.0

    indicators = signals[["short_ema", "long_ema"]].copy()

    return StrategyResult(signals=signals[["signal", "positions"]], indicators=indicators)


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

    # Calculate position changes
    signals["positions"] = signals["signal"].diff()
    
    # MACD always starts with no position (not holding)
    if len(signals) > 0:
        signals.iloc[0, signals.columns.get_loc("positions")] = 0.0

    indicators = signals[["macd", "signal_line", "histogram"]].copy()

    return StrategyResult(signals=signals[["signal", "positions"]], indicators=indicators)


def simple_ema_crossing_strategy(data: pd.Series, ema_period: int = 50) -> StrategyResult:
    """
    Simple EMA Crossing Strategy: Buy when price crosses above EMA, sell when price crosses below EMA.

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


def backtest_strategy(data: pd.Series, strategy_result: StrategyResult, initial_capital: float = 100000.0) -> pd.DataFrame:
    """
    Backtest a trading strategy and calculate portfolio performance.

    Args:
        data: Price series
        strategy_result: StrategyResult containing signals (may have fewer rows due to dropped insufficient data)
        initial_capital: Starting capital

    Returns:
        DataFrame with portfolio metrics
    """
    # Use signals index (which may have dropped insufficient data points)
    signals_index = strategy_result.signals.index
    portfolio = pd.DataFrame(index=signals_index)
    portfolio["price"] = data.loc[signals_index]
    portfolio["signal"] = strategy_result.signals["signal"]
    portfolio["positions"] = strategy_result.signals["positions"]

    # Initialize portfolio
    portfolio["holdings"] = 0.0
    portfolio["cash"] = initial_capital
    portfolio["total"] = initial_capital

    # Track shares owned and cash
    shares = 0.0
    cash = initial_capital

    for i in range(len(portfolio)):
        current_price = portfolio.iloc[i]["price"]
        position_change = portfolio.iloc[i]["positions"]

        if position_change == 1.0:  # Buy signal
            # Buy with all available cash
            shares = cash / current_price
            cash = 0.0
            portfolio.iloc[i, portfolio.columns.get_loc("cash")] = cash
            portfolio.iloc[i, portfolio.columns.get_loc("holdings")] = shares * current_price

        elif position_change == -1.0:  # Sell signal
            # Sell all shares
            cash = shares * current_price
            shares = 0.0
            portfolio.iloc[i, portfolio.columns.get_loc("cash")] = cash
            portfolio.iloc[i, portfolio.columns.get_loc("holdings")] = 0.0

        else:  # Hold
            # Update holdings value with current price, cash remains the same
            portfolio.iloc[i, portfolio.columns.get_loc("cash")] = cash
            if shares > 0:
                portfolio.iloc[i, portfolio.columns.get_loc("holdings")] = shares * current_price
            else:
                portfolio.iloc[i, portfolio.columns.get_loc("holdings")] = 0.0

        portfolio.iloc[i, portfolio.columns.get_loc("total")] = portfolio.iloc[i]["cash"] + portfolio.iloc[i]["holdings"]

    # Forward fill to handle edge cases
    portfolio["cash"] = portfolio["cash"].ffill()
    portfolio["holdings"] = portfolio["holdings"].ffill()
    portfolio["total"] = portfolio["total"].ffill()

    # Calculate returns
    portfolio["returns"] = portfolio["total"].pct_change()

    return portfolio


def calculate_performance_metrics(portfolio: pd.DataFrame, strategy_name: str) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics for a strategy.

    Args:
        portfolio: Portfolio DataFrame from backtest_strategy
        strategy_name: Name of the strategy

    Returns:
        Dictionary with performance metrics
    """
    returns = portfolio["returns"].dropna()
    total_return = (portfolio["total"].iloc[-1] / portfolio["total"].iloc[0] - 1) * 100

    # Annualized return
    days = (portfolio.index[-1] - portfolio.index[0]).days
    years = days / 365.25
    annualized_return = ((portfolio["total"].iloc[-1] / portfolio["total"].iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(252) * 100

    # Sharpe ratio (assuming risk-free rate of 0)
    sharpe_ratio = (annualized_return / volatility) if volatility > 0 else 0

    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    max_drawdown = drawdown.min()

    # Win rate (for individual trades)
    trade_returns = []
    in_position = False
    entry_price = 0

    for i in range(len(portfolio)):
        if portfolio.iloc[i]["positions"] == 1.0 and not in_position:
            in_position = True
            entry_price = portfolio.iloc[i]["price"]
        elif portfolio.iloc[i]["positions"] == -1.0 and in_position:
            in_position = False
            trade_return = (portfolio.iloc[i]["price"] / entry_price - 1) * 100
            trade_returns.append(trade_return)

    win_rate = (np.array(trade_returns) > 0).mean() * 100 if trade_returns else 0
    num_trades = len(trade_returns)
    avg_trade_return = np.mean(trade_returns) if trade_returns else 0

    metrics = {
        "Strategy": strategy_name,
        "Total Return (%)": total_return,
        "Annualized Return (%)": annualized_return,
        "Volatility (%)": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown,
        "Win Rate (%)": win_rate,
        "Number of Trades": num_trades,
        "Avg Trade Return (%)": avg_trade_return,
    }

    return metrics


def optimize_simple_ema_crossing(
    data: pd.Series,
    ema_period_range: Tuple[int, int] = (5, 200),
    step: int = 5,
    initial_capital: float = 100000.0,
    optimization_metric: str = "sharpe_ratio",
) -> Tuple[int, Dict[str, float], pd.DataFrame]:
    """
    Optimize Simple EMA Crossing strategy by testing different EMA periods.

    Args:
        data: Price series
        ema_period_range: Tuple of (min_period, max_period) to test
        step: Step size for EMA period range
        initial_capital: Starting capital for backtest
        optimization_metric: Metric to optimize for ('sharpe_ratio', 'total_return', 'annualized_return')

    Returns:
        Tuple of (best_ema_period, best_metrics, optimization_results_df)
    """
    min_period, max_period = ema_period_range
    ema_periods = range(min_period, max_period + 1, step)

    optimization_results = []

    for ema_period in ema_periods:
        try:
            # Run strategy
            strategy_result = simple_ema_crossing_strategy(data, ema_period=ema_period)
            portfolio = backtest_strategy(data, strategy_result, initial_capital=initial_capital)

            # Calculate metrics
            metrics = calculate_performance_metrics(portfolio, f"Simple EMA Crossing {ema_period}")

            # Add EMA period to results
            result = {"EMA Period": ema_period}
            result.update(metrics)
            optimization_results.append(result)

        except Exception as e:
            logger.warning(f"Failed to test EMA period {ema_period}: {e}")
            continue

    if not optimization_results:
        raise ValueError("No valid optimization results found")

    # Convert to DataFrame
    results_df = pd.DataFrame(optimization_results)

    # Find best period based on optimization metric
    metric_map = {
        "sharpe_ratio": "Sharpe Ratio",
        "total_return": "Total Return (%)",
        "annualized_return": "Annualized Return (%)",
    }

    if optimization_metric not in metric_map:
        optimization_metric = "sharpe_ratio"

    metric_column = metric_map[optimization_metric]

    # Find best period (highest value for the metric)
    best_idx = results_df[metric_column].idxmax()
    best_ema_period = int(results_df.loc[best_idx, "EMA Period"])
    best_metrics = results_df.loc[best_idx].to_dict()

    return best_ema_period, best_metrics, results_df


def create_strategy_comparison_plot(data: pd.Series, strategies: Dict[str, Tuple[StrategyResult, pd.DataFrame]], symbol: str):
    """
    Create interactive Plotly visualization comparing multiple strategies.

    Args:
        data: Price series
        strategies: Dictionary mapping strategy names to (StrategyResult, portfolio) tuples
        symbol: Stock symbol
    """
    # Create subplots - 4 rows: Price/EMA, MACD, Portfolio, Drawdown
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(f"{symbol} - Price and EMA Indicators", "MACD Indicators", "Portfolio Performance Comparison", "Drawdown Analysis"),
        row_heights=[0.3, 0.2, 0.25, 0.25],
    )

    # Plot 1: Price and indicators
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.values,
            name="Price",
            line=dict(width=3, color="#1f77b4"),  # Blue color for better visibility
            opacity=1.0,
            hovertemplate="Price: $%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add buy/sell signals for first strategy
    first_strategy = list(strategies.values())[0][0]
    buy_signals = first_strategy.get_buy_signals()
    sell_signals = first_strategy.get_sell_signals()

    if len(buy_signals) > 0:
        # Only show signals that are in the data index
        buy_signals_in_data = buy_signals.intersection(data.index)
        if len(buy_signals_in_data) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals_in_data,
                    y=data.loc[buy_signals_in_data],
                    mode="markers",
                    name="Buy Signal",
                    marker=dict(symbol="triangle-up", size=10, color="green"),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
    if len(sell_signals) > 0:
        # Only show signals that are in the data index
        sell_signals_in_data = sell_signals.intersection(data.index)
        if len(sell_signals_in_data) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals_in_data,
                    y=data.loc[sell_signals_in_data],
                    mode="markers",
                    name="Sell Signal",
                    marker=dict(symbol="triangle-down", size=10, color="red"),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

    # Add EMA indicators (only on row 1)
    for strategy_name, (strategy_result, portfolio) in strategies.items():
        indicators = strategy_result.indicators
        if "short_ema" in indicators.columns and "long_ema" in indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=indicators.index,
                    y=indicators["short_ema"],
                    name=f"{strategy_name} - Short EMA",
                    line=dict(dash="dash", width=1.5, color="lightgreen"),
                    opacity=0.6,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=indicators.index,
                    y=indicators["long_ema"],
                    name=f"{strategy_name} - Long EMA",
                    line=dict(dash="dash", width=1.5, color="coral"),
                    opacity=0.6,
                ),
                row=1,
                col=1,
            )
        elif "ema" in indicators.columns:
            # Simple EMA crossing strategy
            fig.add_trace(
                go.Scatter(
                    x=indicators.index,
                    y=indicators["ema"],
                    name=f"{strategy_name} - EMA",
                    line=dict(dash="dash", width=1.5, color="orange"),
                    opacity=0.6,
                ),
                row=1,
                col=1,
            )

    # Plot 2: MACD indicators (separate plot)
    for strategy_name, (strategy_result, portfolio) in strategies.items():
        indicators = strategy_result.indicators
        if "macd" in indicators.columns and "signal_line" in indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=indicators.index,
                    y=indicators["macd"],
                    name=f"{strategy_name} - MACD",
                    line=dict(width=1.5, color="green"),
                    opacity=0.7,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=indicators.index,
                    y=indicators["signal_line"],
                    name=f"{strategy_name} - Signal Line",
                    line=dict(width=1.5, color="red"),
                    opacity=0.7,
                ),
                row=2,
                col=1,
            )
            # Add histogram if available
            if "histogram" in indicators.columns:
                fig.add_trace(
                    go.Bar(
                        x=indicators.index,
                        y=indicators["histogram"],
                        name=f"{strategy_name} - Histogram",
                        opacity=0.3,
                        marker_color="blue",
                        legendgroup="row2",
                    ),
                    row=2,
                    col=1,
                )

    # Plot 3: Portfolio values
    for strategy_name, (_, portfolio) in strategies.items():
        fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio["total"], name=strategy_name, line=dict(width=2)), row=3, col=1)

    # Buy and hold baseline
    first_portfolio = list(strategies.values())[0][1]
    initial_value = first_portfolio["total"].iloc[0]
    buy_hold = initial_value * (data / data.iloc[0])
    fig.add_trace(go.Scatter(x=data.index, y=buy_hold, name="Buy & Hold", line=dict(dash="dash", width=2), opacity=0.7), row=3, col=1)

    # Plot 4: Drawdown
    for strategy_name, (_, portfolio) in strategies.items():
        returns = portfolio["returns"].dropna()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100

        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values, name=strategy_name, fill="tozeroy", line=dict(width=1.5), opacity=0.3), row=4, col=1
        )

    # Update layout - configure legend to group by subplot with clear separation
    # Note: Plotly doesn't support truly separate legends per subplot, but we can
    # create clear visual separation using legend groups with large gaps
    fig.update_layout(
        height=1200,
        hovermode="x unified",  # Shows vertical line across all subplots on hover
        showlegend=True,
        legend=dict(
            tracegroupgap=60,  # Very large gap between legend groups to visually separate each subplot
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
    )

    # Update x-axes to ensure they're all shared and show date
    fig.update_xaxes(title_text="Date", row=4, col=1)
    # Ensure all x-axes are synchronized
    for row in range(1, 5):
        fig.update_xaxes(matches="x", row=row, col=1)
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=3, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=4, col=1)

    return fig


def page_controls():
    """Create page controls for backtest parameters."""
    with st.expander("⚙️ Backtest Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Symbol", value="QQQ", help="Stock or ETF ticker symbol")
            initial_capital = st.number_input(
                "Initial Capital ($)", min_value=1000.0, max_value=10000000.0, value=100000.0, step=10000.0, format="%.0f"
            )
        with col2:
            lookback_options = {"1 Year": "1y", "2 Years": "2y", "3 Years": "3y", "5 Years": "5y", "10 Years": "10y", "Max": "max"}
            lookback_label = st.selectbox(
                "Lookback Period", options=list(lookback_options.keys()), index=3, help="How far back to fetch historical data"  # Default to 5 Years
            )
            period = lookback_options[lookback_label]

            end_date = st.date_input("End Date", value=date.today() + timedelta(days=1), help="End date for backtest (defaults to today)")

        st.divider()

        # Strategy enable/disable checkboxes
        st.subheader("🎛️ Strategy Selection")
        col_strategy = st.columns(4)
        with col_strategy[0]:
            enable_ema_2crossings = st.checkbox("EMA 2 Crossings", value=True, key="enable_ema_2crossings", help="Buy when short EMA > long EMA AND price > short EMA")
        with col_strategy[1]:
            enable_ema_cross = st.checkbox("EMA Cross", value=True, key="enable_ema_cross", help="Buy when short EMA crosses above long EMA")
        with col_strategy[2]:
            enable_macd = st.checkbox("MACD", value=True, key="enable_macd", help="Buy when MACD line crosses above signal line")
        with col_strategy[3]:
            enable_simple_ema = st.checkbox("Simple EMA Crossing", value=True, key="enable_simple_ema", help="Buy when price crosses above EMA")

        st.divider()

        col3, col4, col5 = st.columns(3)
        with col3:
            st.subheader("📈 EMA Strategy Parameters")
            ema_short = st.slider("Short EMA Period", min_value=5, max_value=100, value=50, step=5, key="ema_short", disabled=not (enable_ema_2crossings or enable_ema_cross))
            ema_long = st.slider("Long EMA Period", min_value=50, max_value=300, value=200, step=10, key="ema_long", disabled=not (enable_ema_2crossings or enable_ema_cross))

        with col4:
            st.subheader("📉 MACD Strategy Parameters")
            macd_fast = st.slider("Fast EMA Period", min_value=5, max_value=30, value=12, step=1, key="macd_fast", disabled=not enable_macd)
            macd_slow = st.slider("Slow EMA Period", min_value=15, max_value=50, value=26, step=1, key="macd_slow", disabled=not enable_macd)
            macd_signal = st.slider("Signal Line Period", min_value=5, max_value=20, value=9, step=1, key="macd_signal", disabled=not enable_macd)

        with col5:
            st.subheader("🎯 Simple EMA Crossing Parameters")
            optimize_simple_ema = st.checkbox(
                "🔍 Optimize Parameters",
                value=False,
                key="optimize_simple_ema",
                help="Automatically find the best EMA period based on performance metrics",
                disabled=not enable_simple_ema,
            )
            if optimize_simple_ema and enable_simple_ema:
                optimization_metric = st.selectbox(
                    "Optimization Metric",
                    options=["sharpe_ratio", "total_return", "annualized_return"],
                    index=0,
                    key="optimization_metric",
                    help="Metric to optimize for: Sharpe Ratio (risk-adjusted return), Total Return, or Annualized Return",
                )
                ema_min = st.slider("Min EMA Period", min_value=5, max_value=100, value=10, step=5, key="ema_min")
                ema_max = st.slider("Max EMA Period", min_value=50, max_value=200, value=200, step=5, key="ema_max")
                ema_step = st.slider("Step Size", min_value=5, max_value=20, value=5, step=5, key="ema_step")
                simple_ema_period = None  # Will be determined by optimization
            elif enable_simple_ema:
                simple_ema_period = st.slider("EMA Period", min_value=5, max_value=200, value=50, step=5, key="simple_ema_period")
                optimization_metric = None
                ema_min = None
                ema_max = None
                ema_step = None
            else:
                simple_ema_period = None
                optimization_metric = None
                ema_min = None
                ema_max = None
                ema_step = None

    return (
        symbol,
        period,
        end_date,
        initial_capital,
        ema_short,
        ema_long,
        macd_fast,
        macd_slow,
        macd_signal,
        simple_ema_period,
        optimize_simple_ema,
        optimization_metric,
        ema_min,
        ema_max,
        ema_step,
        enable_ema_2crossings,
        enable_ema_cross,
        enable_macd,
        enable_simple_ema,
    )


def main():
    """Main function for mid-frequency backtest page."""
    # Setup page
    if hydra.core.global_hydra.GlobalHydra().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_resolvers()

    with hydra.initialize(version_base=None, config_path="../../../conf"):
        config = hydra.compose(config_name="main")

    setup_page_and_sidebar(config["style_conf"])

    st.title("📊 Mid-Frequency Strategy Backtest")
    st.markdown(
        """
    Backtest and compare EMA crossover, simple EMA crossing, and MACD trading strategies with customizable parameters.
    This tool allows you to analyze strategy performance across different time periods and parameter settings.
    """
    )

    # Page controls
    (
        symbol,
        period,
        end_date,
        initial_capital,
        ema_short,
        ema_long,
        macd_fast,
        macd_slow,
        macd_signal,
        simple_ema_period,
        optimize_simple_ema,
        optimization_metric,
        ema_min,
        ema_max,
        ema_step,
        enable_ema_2crossings,
        enable_ema_cross,
        enable_macd,
        enable_simple_ema,
    ) = page_controls()

    # Validate at least one strategy is enabled
    if not (enable_ema_2crossings or enable_ema_cross or enable_macd or enable_simple_ema):
        st.warning("⚠️ Please enable at least one strategy to run the backtest.")

    # Run backtest button
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        with st.spinner(f"Fetching data for {symbol}..."):
            try:
                # Fetch data
                data = fetch_data(symbol, period, str(end_date))

                # Initialize strategy results and portfolios
                ema_2crossings_result = None
                ema_2crossings_portfolio = None
                ema_cross_result = None
                ema_cross_portfolio = None
                macd_result = None
                macd_portfolio = None
                simple_ema_result = None
                simple_ema_portfolio = None

                # Run enabled strategies
                if enable_ema_2crossings or enable_ema_cross:
                    with st.spinner("Running EMA strategies..."):
                        if enable_ema_2crossings:
                            ema_2crossings_result = ema_2crossings_strategy(data, short_window=ema_short, long_window=ema_long)
                            ema_2crossings_portfolio = backtest_strategy(data, ema_2crossings_result, initial_capital=initial_capital)
                        
                        if enable_ema_cross:
                            ema_cross_result = ema_cross_strategy(data, short_window=ema_short, long_window=ema_long)
                            ema_cross_portfolio = backtest_strategy(data, ema_cross_result, initial_capital=initial_capital)

                if enable_macd:
                    with st.spinner("Running MACD strategy..."):
                        macd_result = macd_strategy(data, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
                        macd_portfolio = backtest_strategy(data, macd_result, initial_capital=initial_capital)

                # Handle Simple EMA Crossing strategy with optional optimization
                if enable_simple_ema:
                    if optimize_simple_ema:
                        with st.spinner("Optimizing Simple EMA Crossing strategy..."):
                            best_ema_period, best_metrics, optimization_results_df = optimize_simple_ema_crossing(
                                data,
                                ema_period_range=(ema_min, ema_max),
                                step=ema_step,
                                initial_capital=initial_capital,
                                optimization_metric=optimization_metric,
                            )
                            simple_ema_period = best_ema_period
                            
                            # Store optimization results in session state
                            st.session_state["simple_ema_optimization_results"] = optimization_results_df
                            st.session_state["simple_ema_best_metrics"] = best_metrics
                            st.session_state["simple_ema_best_period"] = best_ema_period
                            
                            st.info(f"✅ Optimization complete! Best EMA Period: {best_ema_period} (based on {optimization_metric})")

                    with st.spinner("Running Simple EMA Crossing strategy..."):
                        simple_ema_result = simple_ema_crossing_strategy(data, ema_period=simple_ema_period)
                        simple_ema_portfolio = backtest_strategy(data, simple_ema_result, initial_capital=initial_capital)

                # Find common start time (maximum of dropped periods) for enabled strategies only
                # EMA strategies drop first long_window rows, MACD drops first (slow_window + signal_window) rows
                # Simple EMA drops first ema_period rows
                all_start_indices = []
                if enable_ema_2crossings and ema_2crossings_portfolio is not None:
                    all_start_indices.append(ema_2crossings_portfolio.index[0])
                if enable_ema_cross and ema_cross_portfolio is not None:
                    all_start_indices.append(ema_cross_portfolio.index[0])
                if enable_macd and macd_portfolio is not None:
                    all_start_indices.append(macd_portfolio.index[0])
                if enable_simple_ema and simple_ema_portfolio is not None:
                    all_start_indices.append(simple_ema_portfolio.index[0])
                
                if not all_start_indices:
                    raise ValueError("No strategies were successfully run. Please check your strategy selections.")
                
                common_start_idx = max(all_start_indices)
                
                # Trim enabled portfolios and data to start at the same time
                ema_2crossings_portfolio_aligned = None
                ema_cross_portfolio_aligned = None
                macd_portfolio_aligned = None
                simple_ema_portfolio_aligned = None
                
                ema_2crossings_signals_aligned = None
                ema_cross_signals_aligned = None
                macd_signals_aligned = None
                simple_ema_signals_aligned = None
                
                ema_2crossings_indicators_aligned = None
                ema_cross_indicators_aligned = None
                macd_indicators_aligned = None
                simple_ema_indicators_aligned = None
                
                if enable_ema_2crossings and ema_2crossings_portfolio is not None:
                    ema_2crossings_portfolio_aligned = ema_2crossings_portfolio.loc[common_start_idx:].copy()
                    ema_2crossings_signals_aligned = ema_2crossings_result.signals.loc[common_start_idx:].copy()
                    ema_2crossings_indicators_aligned = ema_2crossings_result.indicators.loc[common_start_idx:].copy()
                
                if enable_ema_cross and ema_cross_portfolio is not None:
                    ema_cross_portfolio_aligned = ema_cross_portfolio.loc[common_start_idx:].copy()
                    ema_cross_signals_aligned = ema_cross_result.signals.loc[common_start_idx:].copy()
                    ema_cross_indicators_aligned = ema_cross_result.indicators.loc[common_start_idx:].copy()
                
                if enable_macd and macd_portfolio is not None:
                    macd_portfolio_aligned = macd_portfolio.loc[common_start_idx:].copy()
                    macd_signals_aligned = macd_result.signals.loc[common_start_idx:].copy()
                    macd_indicators_aligned = macd_result.indicators.loc[common_start_idx:].copy()
                
                if enable_simple_ema and simple_ema_portfolio is not None:
                    simple_ema_portfolio_aligned = simple_ema_portfolio.loc[common_start_idx:].copy()
                    simple_ema_signals_aligned = simple_ema_result.signals.loc[common_start_idx:].copy()
                    simple_ema_indicators_aligned = simple_ema_result.indicators.loc[common_start_idx:].copy()
                
                # Get trimmed data that matches the common start time
                data_display = data.loc[common_start_idx:]
                
                # Ensure all signal indices align with data_display index
                # Use intersection to only keep dates that exist in both signals and data
                data_display_index = data_display.index
                common_index = data_display_index.copy()
                
                # Intersect with enabled strategies only
                if enable_ema_2crossings and ema_2crossings_signals_aligned is not None:
                    common_index = common_index.intersection(ema_2crossings_signals_aligned.index)
                if enable_ema_cross and ema_cross_signals_aligned is not None:
                    common_index = common_index.intersection(ema_cross_signals_aligned.index)
                if enable_macd and macd_signals_aligned is not None:
                    common_index = common_index.intersection(macd_signals_aligned.index)
                if enable_simple_ema and simple_ema_signals_aligned is not None:
                    common_index = common_index.intersection(simple_ema_signals_aligned.index)
                
                # Align enabled strategies to the common index
                if enable_ema_2crossings and ema_2crossings_signals_aligned is not None:
                    ema_2crossings_signals_aligned = ema_2crossings_signals_aligned.loc[common_index]
                    ema_2crossings_indicators_aligned = ema_2crossings_indicators_aligned.loc[common_index]
                
                if enable_ema_cross and ema_cross_signals_aligned is not None:
                    ema_cross_signals_aligned = ema_cross_signals_aligned.loc[common_index]
                    ema_cross_indicators_aligned = ema_cross_indicators_aligned.loc[common_index]
                
                if enable_macd and macd_signals_aligned is not None:
                    macd_signals_aligned = macd_signals_aligned.loc[common_index]
                    macd_indicators_aligned = macd_indicators_aligned.loc[common_index]
                
                if enable_simple_ema and simple_ema_signals_aligned is not None:
                    simple_ema_signals_aligned = simple_ema_signals_aligned.loc[common_index]
                    simple_ema_indicators_aligned = simple_ema_indicators_aligned.loc[common_index]
                
                # Update data_display to match the common index
                data_display = data_display.loc[common_index]
                
                # Ensure MACD always starts with no position at the common start time (after alignment)
                if enable_macd and macd_signals_aligned is not None and len(macd_signals_aligned) > 0:
                    macd_signals_aligned.iloc[0, macd_signals_aligned.columns.get_loc("positions")] = 0.0
                
                # Ensure Simple EMA initial position is set correctly after alignment
                if enable_simple_ema and simple_ema_signals_aligned is not None and len(simple_ema_signals_aligned) > 0:
                    if simple_ema_signals_aligned.iloc[0]["signal"] == 1.0:
                        # Start with holding position if price > EMA
                        simple_ema_signals_aligned.iloc[0, simple_ema_signals_aligned.columns.get_loc("positions")] = 1.0
                    else:
                        # Don't hold initially, wait for next signal to enter
                        simple_ema_signals_aligned.iloc[0, simple_ema_signals_aligned.columns.get_loc("positions")] = 0.0
                
                # Create aligned StrategyResult objects for enabled strategies
                ema_2crossings_result_aligned = None
                ema_cross_result_aligned = None
                macd_result_aligned = None
                simple_ema_result_aligned = None
                
                if enable_ema_2crossings and ema_2crossings_signals_aligned is not None:
                    ema_2crossings_result_aligned = StrategyResult(
                        signals=ema_2crossings_signals_aligned,
                        indicators=ema_2crossings_indicators_aligned
                    )
                
                if enable_ema_cross and ema_cross_signals_aligned is not None:
                    ema_cross_result_aligned = StrategyResult(
                        signals=ema_cross_signals_aligned,
                        indicators=ema_cross_indicators_aligned
                    )
                
                if enable_macd and macd_signals_aligned is not None:
                    macd_result_aligned = StrategyResult(
                        signals=macd_signals_aligned,
                        indicators=macd_indicators_aligned
                    )
                
                if enable_simple_ema and simple_ema_signals_aligned is not None:
                    simple_ema_result_aligned = StrategyResult(
                        signals=simple_ema_signals_aligned,
                        indicators=simple_ema_indicators_aligned
                    )
                
                # Recalculate portfolios with aligned signals to ensure they start correctly
                if enable_ema_2crossings and ema_2crossings_result_aligned is not None:
                    ema_2crossings_portfolio_aligned = backtest_strategy(data_display, ema_2crossings_result_aligned, initial_capital=initial_capital)
                
                if enable_ema_cross and ema_cross_result_aligned is not None:
                    ema_cross_portfolio_aligned = backtest_strategy(data_display, ema_cross_result_aligned, initial_capital=initial_capital)
                
                if enable_macd and macd_result_aligned is not None:
                    macd_portfolio_aligned = backtest_strategy(data_display, macd_result_aligned, initial_capital=initial_capital)
                
                if enable_simple_ema and simple_ema_result_aligned is not None:
                    simple_ema_portfolio_aligned = backtest_strategy(data_display, simple_ema_result_aligned, initial_capital=initial_capital)
                
                # Calculate metrics on aligned portfolio data for enabled strategies
                ema_2crossings_metrics = None
                ema_cross_metrics = None
                macd_metrics = None
                simple_ema_metrics = None
                
                if enable_ema_2crossings and ema_2crossings_portfolio_aligned is not None:
                    ema_2crossings_metrics = calculate_performance_metrics(ema_2crossings_portfolio_aligned, f"EMA 2 Crossings {ema_short}/{ema_long}")
                
                if enable_ema_cross and ema_cross_portfolio_aligned is not None:
                    ema_cross_metrics = calculate_performance_metrics(ema_cross_portfolio_aligned, f"EMA Cross {ema_short}/{ema_long}")
                
                if enable_macd and macd_portfolio_aligned is not None:
                    macd_metrics = calculate_performance_metrics(macd_portfolio_aligned, "MACD")
                
                if enable_simple_ema and simple_ema_portfolio_aligned is not None:
                    simple_ema_metrics = calculate_performance_metrics(simple_ema_portfolio_aligned, f"Simple EMA Crossing {simple_ema_period}")

                # Display data info (using trimmed data)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Data Range", f"{data_display.index[0].date()} to {data_display.index[-1].date()}")
                with col2:
                    st.metric("Total Days", len(data_display))
                with col3:
                    st.metric("Initial Price", f"${data_display.iloc[0]:.2f}")
                with col4:
                    buy_hold_return = (data_display.iloc[-1] / data_display.iloc[0] - 1) * 100
                    st.metric("Buy & Hold Return", f"{buy_hold_return:.2f}%")

                # Store aligned data in session state for display (only enabled strategies)
                st.session_state["data"] = data_display
                strategies_dict = {}
                metrics_dict = {}
                
                if enable_ema_2crossings and ema_2crossings_result_aligned is not None:
                    strategies_dict[f"EMA 2 Crossings {ema_short}/{ema_long}"] = (ema_2crossings_result_aligned, ema_2crossings_portfolio_aligned)
                    if ema_2crossings_metrics is not None:
                        metrics_dict[f"EMA 2 Crossings {ema_short}/{ema_long}"] = ema_2crossings_metrics
                
                if enable_ema_cross and ema_cross_result_aligned is not None:
                    strategies_dict[f"EMA Cross {ema_short}/{ema_long}"] = (ema_cross_result_aligned, ema_cross_portfolio_aligned)
                    if ema_cross_metrics is not None:
                        metrics_dict[f"EMA Cross {ema_short}/{ema_long}"] = ema_cross_metrics
                
                if enable_macd and macd_result_aligned is not None:
                    strategies_dict["MACD"] = (macd_result_aligned, macd_portfolio_aligned)
                    if macd_metrics is not None:
                        metrics_dict["MACD"] = macd_metrics
                
                if enable_simple_ema and simple_ema_result_aligned is not None:
                    strategies_dict[f"Simple EMA Crossing {simple_ema_period}"] = (simple_ema_result_aligned, simple_ema_portfolio_aligned)
                    if simple_ema_metrics is not None:
                        metrics_dict[f"Simple EMA Crossing {simple_ema_period}"] = simple_ema_metrics
                
                st.session_state["strategies"] = strategies_dict
                st.session_state["metrics"] = metrics_dict
                st.session_state["symbol"] = symbol

                st.success("✅ Backtest completed successfully!")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Backtest error: {e}")

    # Display results if available
    if "strategies" in st.session_state and "metrics" in st.session_state:
        st.divider()

        # Performance metrics comparison
        st.header("📈 Performance Metrics")
        metrics_df = pd.DataFrame([st.session_state["metrics"][k] for k in st.session_state["metrics"].keys()])
        metrics_df = metrics_df.set_index("Strategy")

        # Format the dataframe for display
        display_df = metrics_df.copy()
        for col in display_df.columns:
            if col != "Number of Trades":
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

        st.dataframe(display_df, use_container_width=True)

        # Display optimization results if available
        if "simple_ema_optimization_results" in st.session_state:
            st.divider()
            st.header("🔍 Simple EMA Crossing Optimization Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best EMA Period", int(st.session_state["simple_ema_best_period"]))
            with col2:
                best_metrics = st.session_state["simple_ema_best_metrics"]
                st.metric("Best Sharpe Ratio", f"{best_metrics.get('Sharpe Ratio', 0):.2f}")
            with col3:
                st.metric("Best Total Return", f"{best_metrics.get('Total Return (%)', 0):.2f}%")
            
            # Display optimization results table
            opt_results = st.session_state["simple_ema_optimization_results"].copy()
            # Format numeric columns
            for col in opt_results.columns:
                if col != "EMA Period" and col != "Strategy" and col != "Number of Trades":
                    opt_results[col] = opt_results[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
            
            # Highlight the best period
            st.dataframe(opt_results, use_container_width=True)
            
            # Create a visualization of optimization results
            st.subheader("Optimization Parameter Analysis")
            opt_results_numeric = st.session_state["simple_ema_optimization_results"].copy()
            
            fig_opt = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=("Sharpe Ratio vs EMA Period", "Total Return vs EMA Period", "Annualized Return vs EMA Period", "Max Drawdown vs EMA Period"),
                vertical_spacing=0.12,
            )
            
            # Sharpe Ratio
            fig_opt.add_trace(
                go.Scatter(
                    x=opt_results_numeric["EMA Period"],
                    y=opt_results_numeric["Sharpe Ratio"],
                    mode="lines+markers",
                    name="Sharpe Ratio",
                    line=dict(color="green", width=2),
                    marker=dict(size=6),
                ),
                row=1,
                col=1,
            )
            
            # Total Return
            fig_opt.add_trace(
                go.Scatter(
                    x=opt_results_numeric["EMA Period"],
                    y=opt_results_numeric["Total Return (%)"],
                    mode="lines+markers",
                    name="Total Return (%)",
                    line=dict(color="blue", width=2),
                    marker=dict(size=6),
                ),
                row=1,
                col=2,
            )
            
            # Annualized Return
            fig_opt.add_trace(
                go.Scatter(
                    x=opt_results_numeric["EMA Period"],
                    y=opt_results_numeric["Annualized Return (%)"],
                    mode="lines+markers",
                    name="Annualized Return (%)",
                    line=dict(color="purple", width=2),
                    marker=dict(size=6),
                ),
                row=2,
                col=1,
            )
            
            # Max Drawdown
            fig_opt.add_trace(
                go.Scatter(
                    x=opt_results_numeric["EMA Period"],
                    y=opt_results_numeric["Max Drawdown (%)"],
                    mode="lines+markers",
                    name="Max Drawdown (%)",
                    line=dict(color="red", width=2),
                    marker=dict(size=6),
                ),
                row=2,
                col=2,
            )
            
            # Highlight best period
            best_period = st.session_state["simple_ema_best_period"]
            for row in [1, 2]:
                for col in [1, 2]:
                    fig_opt.add_vline(
                        x=best_period,
                        line_dash="dash",
                        line_color="orange",
                        annotation_text=f"Best: {best_period}",
                        row=row,
                        col=col,
                    )
            
            fig_opt.update_layout(height=600, showlegend=False, hovermode="x unified")
            fig_opt.update_xaxes(title_text="EMA Period", row=2, col=1)
            fig_opt.update_xaxes(title_text="EMA Period", row=2, col=2)
            fig_opt.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
            fig_opt.update_yaxes(title_text="Total Return (%)", row=1, col=2)
            fig_opt.update_yaxes(title_text="Annualized Return (%)", row=2, col=1)
            fig_opt.update_yaxes(title_text="Max Drawdown (%)", row=2, col=2)
            
            st.plotly_chart(fig_opt, use_container_width=True)

        # Visualization
        st.header("📊 Strategy Comparison")
        fig = create_strategy_comparison_plot(st.session_state["data"], st.session_state["strategies"], st.session_state["symbol"])
        st.plotly_chart(fig, use_container_width=True)

        # Detailed metrics
        st.header("📋 Detailed Metrics")
        # Use 2 columns for 2 strategies
        col1, col2 = st.columns(2)

        for idx, (strategy_name, metrics) in enumerate(st.session_state["metrics"].items()):
            with col1 if idx % 2 == 0 else col2:
                st.subheader(strategy_name)
                for key, value in metrics.items():
                    if key != "Strategy":
                        if isinstance(value, float):
                            st.metric(key, f"{value:.2f}")
                        else:
                            st.metric(key, value)


if __name__ == "__main__":
    main()
