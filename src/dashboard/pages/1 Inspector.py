"""
Combined Inspector page for fund inspection and strategy backtesting.

This page combines:
1. Fund Inspector - Analyze fund/ETF details, country exposure, and performance
2. Strategy Backtest - Backtest EMA, MACD, RSI, and combined trading strategies
"""

import os
from datetime import date, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import yaml
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, Tuple, Optional
from loguru import logger
import hydra
import yfinance as yf
from src.configurations.yaml import register_resolvers
from src.dashboard.create_page import setup_page_and_sidebar
from src.data import get_fund_snap
from src.data.mstar import get_number_of_holdings
from src.viz import create_plotly_bar_chart, create_plotly_choropleth
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
from src.indicators.rsi import compute_rsi

# Load Morningstar config
with open("conf/mstar.yaml", "r") as f:
    mstar_config = yaml.safe_load(f)


# ============================================================================
# FUND INSPECTOR FUNCTIONS
# ============================================================================

def create_shared_symbol_selector(config) -> str:
    """
    Create a shared symbol selector for both fund inspection and strategy backtesting.
    Allows users to:
    1. Select from predefined popular funds/ETFs
    2. Enter custom symbols manually
    3. Randomly select a category and fund

    Returns:
        Selected symbol string
    """
    # Predefined popular funds/ETFs
    # Access lenses.pages structure (main, stocks, etfs) and flatten all lens options
    portfolios = dict()
    pages = config["lenses"].get("pages", {})
    for page_name, page_lenses in pages.items():
        if page_lenses:  # Only process if page_lenses is not empty/None
            for category, symbols in page_lenses.items():
                portfolios[category] = symbols
    popular_funds = {portfolio: [fund for fund in portfolios[portfolio]] for portfolio in portfolios}
    # % Option 1: Select from popular categories
    fund_list = list(popular_funds.keys())

    # Initialize random selection on first load
    if "shared_random_category_index" not in st.session_state:
        random_category_index = random.choice(range(len(fund_list)))
        random_fund_index = random.choice(range(len(popular_funds[fund_list[random_category_index]])))
        st.session_state.shared_random_category_index = random_category_index
        st.session_state.shared_random_fund_index = random_fund_index

    st.sidebar.markdown("### Symbol Selection")
    selected_category = st.sidebar.selectbox(
        "Select Category:",
        fund_list,
        index=st.session_state.get("shared_random_category_index", 0),
        help="Choose from predefined fund categories",
        placeholder="Select Category",
        key="shared_category_selectbox",
    )

    selected_fund = st.sidebar.selectbox(
        "Select Fund:",
        popular_funds[selected_category],
        index=st.session_state.get("shared_random_fund_index", 0),
        help="Choose a specific fund from the selected category",
        placeholder="Select Fund",
        key="shared_fund_selectbox",
    )

    # % Option 2: Custom symbol input
    st.sidebar.subheader("Or")
    custom_symbol = st.sidebar.text_input(
        "Enter custom symbol:",
        value=selected_fund,
        placeholder="e.g., VT, VTI, AAPL",
        help="Enter any fund/ETF symbol you want to analyze",
        key="shared_custom_symbol_input",
    )

    if custom_symbol.strip():
        final_symbol = custom_symbol.strip().upper()
    else:
        final_symbol = selected_fund

    def random_button_click():
        random_category_index = random.choice(range(len(fund_list)))
        random_category = fund_list[random_category_index]
        random_fund_index = random.choice(range(len(popular_funds[random_category])))
        st.session_state.shared_random_category_index = random_category_index
        st.session_state.shared_random_fund_index = random_fund_index

    # Add random selection button
    st.sidebar.button("🎲", on_click=random_button_click, key="shared_random_button", help="Randomly select a category and fund")

    return final_symbol


def get_country_exposure(snap: dict) -> Dict[str, float]:
    """
    Get country exposure data for a given symbol using Morningstar API.

    Args:
        snap: Fund snapshot dictionary

    Returns:
        Dictionary mapping country names to exposure percentages
    """
    try:
        # Get fund data from Morningstar
        if not snap or "Portfolios" not in snap:
            return {}

        portfolios = snap.get("Portfolios", [])
        if not portfolios:
            return {}

        country_exposure = portfolios[0].get("CountryExposure", [])
        if not country_exposure:
            return {}

        breakdown_values = country_exposure[0].get("BreakdownValues", [])

        # Convert to dictionary mapping country names to percentages
        exposure_dict = {}
        for item in breakdown_values:
            country_code = item.get("Type", "")
            percentage = item.get("Value", 0.0)

            if country_code and percentage is not None:
                # Convert country code to full name
                country_name = mstar_config.get("country_code_to_name", {}).get(country_code, country_code)
                exposure_dict[country_name] = percentage

        return exposure_dict

    except Exception as e:
        print(f"Error getting country exposure for {snap.get('Symbol')}: {e}")
        return {}


def create_country_exposure_table(exposure_df: pd.DataFrame):
    """
    Create a table showing country exposure for a given symbol.
    """
    fig = create_plotly_bar_chart(
        exposure_df.sort_values("Exposure %", ascending=True),
        x_col="Exposure %",
        y_col="Country",
        text="Exposure %",
        hover_data={"Country": True, "Exposure %": ":.2f"},
    )

    return fig


def display_country_exposure_map_streamlit(snap: dict):
    """
    Display the country exposure map directly in Streamlit.

    Args:
        snap: Fund snapshot dictionary
    """
    st.markdown(f"### Country Exposure for <span style='color:gold; font-weight:bold;'>{snap.get('Symbol')}</span>", unsafe_allow_html=True)

    with st.spinner(f"Loading country exposure data for {snap.get('Symbol')}..."):
        exposure_data = get_country_exposure(snap)

        if not exposure_data:
            st.warning(f"No country exposure data found for {snap.get('Symbol')}")
            return

        exposure_df = pd.DataFrame(list(exposure_data.items()), columns=["Country", "Exposure %"])
        exposure_df = exposure_df.sort_values("Exposure %", ascending=False)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(create_country_exposure_table(exposure_df))

        with col2:
            world_map = create_plotly_choropleth(
                exposure_df,
                "Country",
                "Exposure %",
                hover_name_col="Country",
                color_scale="YlOrRd",
            )
            st.plotly_chart(world_map)


def main_fund_inspect_page(selections: str, config):
    """
    Main function for the fund inspection page with hybrid symbol selection.
    """
    st.title("Fund Inspector")

    # % Sidebar: Symbol Selector
    selected_symbol = selections

    # Display the country exposure map
    snap = get_fund_snap(selected_symbol)
    if not snap:
        st.error(f"No data found for {selected_symbol}")
        return
    # Display the selected fund's symbol and name
    col1, col2, col3 = st.columns(3)
    with col1:
        basic_info = (
            f"<span class='field'>Symbol : </span> <span class='data' style='color:gold; font-weight:bold;'>{snap.get('Symbol')}</span>"
            + f"<br><span class='field'>Fund Name : </span> <span class='data'>{snap.get('Name')}</span>"
            + f"<br><span class='field'>Number of Holdings : </span> <span class='data'>{get_number_of_holdings(snap)}</span>"
            + f"<br><span class='field'>Total Expense : </span> <span class='data'>{snap.get('TotalExpenseRatio')}%</span>"
            + f"<br><span class='field'>Yield : </span> <span class='data'>{snap.get('YieldHistory').get('Value')}% [{snap.get('YieldHistory').get('Type')}]</span>"
            + (
                f"<br><span class='field'>Sector : </span> <span class='data'>{snap.get('Sector').get('SectorName')} [{snap.get('Sector').get('SectorCode')}]</span>"
                if snap.get("Sector")
                else ""
            )
            + (
                f"<br><span class='field'>Industry : </span> <span class='data'>{snap.get('Industry').get('IndustryName')} [{snap.get('Industry').get('IndustryCode')}]</span>"
                if snap.get("Industry")
                else ""
            )
            + f"<br><span class='field'>Company : </span> <span class='data'>{snap.get('CompanyName', snap.get('BrandingCompanyName'))}</span>"
        )
        st.markdown(basic_info, unsafe_allow_html=True)

    with col2:
        style_box_breakdown = snap.get("Portfolios")[0].get("StyleBoxBreakdown")[0].get("BreakdownValues")
        # Prepare matrix
        matrix = [[None for _ in range(3)] for _ in range(3)]  # rows: Large/Mid/Small, cols: Value/Blend/Growth
        labels_row = ["Large", "Mid", "Small"]
        labels_col = ["Value", "Blend", "Growth"]
        for i, cell in enumerate(style_box_breakdown):
            row = i // 3
            col = i % 3
            matrix[row][col] = cell["Value"]
        style_box_df = pd.DataFrame(matrix, index=labels_row, columns=labels_col)
        st.markdown("<span class='field'>Style Box Breakdown </span>", unsafe_allow_html=True)
        st.dataframe(
            style_box_df.style.format("{:.2f}").background_gradient(cmap="Reds", axis=None),
            width=250,
        )

    with col3:
        investment_strategy = snap.get("InvestmentStrategy") or snap.get("Investment Strategy")
        investment_strategy = "<span style='color:lightblue'>" + investment_strategy + "</span>"
        if investment_strategy:
            st.markdown("<span class='field'>Investment Strategy </span>", unsafe_allow_html=True)
            st.markdown(investment_strategy, unsafe_allow_html=True)
        else:
            st.markdown("**No investment strategy information available.**")
    display_country_exposure_map_streamlit(snap)

    # display pandas df from records snap.get("GrowthOf10K")
    if snap.get("GrowthOf10K"):
        st.markdown("### Performances")
        growth_df = pd.DataFrame(snap.get("GrowthOf10K")[0]["HistoryDetails"])
        initial_value = growth_df.iloc[0]["Value"]
        end_value = growth_df.iloc[-1]["Value"]
        annualized_return = ((end_value / initial_value) ** (12 / len(growth_df))) - 1
        fig = px.line(
            growth_df,
            x="EndDate",
            y="Value",
            markers=True,
            title=f"Growth of $10K Over Time, Annualized Return: {annualized_return:.2%}",
            labels={"EndDate": "Date", "Value": "Value"},
            hover_data={"EndDate": True, "Value": ":.2f"},
        )

        st.plotly_chart(fig)


# ============================================================================
# STRATEGY BACKTEST FUNCTIONS
# ============================================================================

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
    annualized_return = ((portfolio["total"].iloc[-1]  / portfolio["total"].iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else 0

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
    Optimize EMA_x strategy by testing different EMA periods.

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
            metrics = calculate_performance_metrics(portfolio, f"EMA_x {ema_period}")

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


def optimize_rsi(
    data: pd.Series,
    rsi_period_range: Tuple[int, int] = (5, 30),
    step: int = 1,
    initial_capital: float = 100000.0,
    optimization_metric: str = "sharpe_ratio",
) -> Tuple[int, Dict[str, float], pd.DataFrame]:
    """
    Optimize RSI strategy by testing different RSI periods.

    Args:
        data: Price series
        rsi_period_range: Tuple of (min_period, max_period) to test
        step: Step size for RSI period range
        initial_capital: Starting capital for backtest
        optimization_metric: Metric to optimize for ('sharpe_ratio', 'total_return', 'annualized_return')

    Returns:
        Tuple of (best_rsi_period, best_metrics, optimization_results_df)
    """
    min_period, max_period = rsi_period_range
    rsi_periods = range(min_period, max_period + 1, step)

    optimization_results = []

    for rsi_period in rsi_periods:
        try:
            # Run strategy
            strategy_result = rsi_strategy(data, rsi_period=rsi_period)
            portfolio = backtest_strategy(data, strategy_result, initial_capital=initial_capital)

            # Calculate metrics
            metrics = calculate_performance_metrics(portfolio, f"RSI {rsi_period}")

            # Add RSI period to results
            result = {"RSI Period": rsi_period}
            result.update(metrics)
            optimization_results.append(result)

        except Exception as e:
            logger.warning(f"Failed to test RSI period {rsi_period}: {e}")
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
    best_rsi_period = int(results_df.loc[best_idx, "RSI Period"])
    best_metrics = results_df.loc[best_idx].to_dict()

    return best_rsi_period, best_metrics, results_df


def create_strategy_comparison_plot(
    data: pd.Series, 
    strategies: Dict[str, Tuple[StrategyResult, pd.DataFrame]], 
    symbol: str,
    rsi_indicator: Optional[pd.Series] = None,
    macd_indicator: Optional[pd.Series] = None,
    macd_signal: Optional[pd.Series] = None,
    macd_histogram: Optional[pd.Series] = None,
):
    """
    Create interactive Plotly visualization comparing multiple strategies.

    Args:
        data: Price series
        strategies: Dictionary mapping strategy names to (StrategyResult, portfolio) tuples
        symbol: Stock symbol
        rsi_indicator: Optional RSI indicator series to always display
        macd_indicator: Optional MACD indicator series to always display
        macd_signal: Optional MACD signal line series to always display
        macd_histogram: Optional MACD histogram series to always display
    """
    # Create subplots - 5 rows: Price/EMA, MACD, RSI, Portfolio, Drawdown
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f"{symbol} - Price and EMA Indicators",
            f"{symbol} - MACD Indicators",
            f"{symbol} - RSI Indicators",
            f"{symbol} - Strategy Performance Comparison",
            f"{symbol} - Drawdown Analysis",
        ),
        row_heights=[0.25, 0.15, 0.15, 0.25, 0.20],
    )

    # Plot 1: Price and indicators
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.values,
            name="Price",
            line=dict(width=3, color="#1f77b4"),
            opacity=1.0,
            hovertemplate="Price: $%{y:.2f}<extra></extra>",
            legendgroup="row1",
        ),
        row=1,
        col=1,
    )

    # Add buy/sell signals for first strategy (if strategies exist)
    if len(strategies) > 0:
        first_strategy = list(strategies.values())[0][0]
        buy_signals = first_strategy.get_buy_signals()
        sell_signals = first_strategy.get_sell_signals()

        if len(buy_signals) > 0:
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
                        legendgroup="row1",
                    ),
                    row=1,
                    col=1,
                )
        if len(sell_signals) > 0:
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
                        legendgroup="row1",
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
                    legendgroup="row1",
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
                    legendgroup="row1",
                ),
                row=1,
                col=1,
            )
        elif "ema50" in indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=indicators.index,
                    y=indicators["ema50"],
                    name=f"{strategy_name} - EMA_x",
                    line=dict(dash="dash", width=1.5, color="purple"),
                    opacity=0.6,
                    legendgroup="row1",
                ),
                row=1,
                col=1,
            )
        elif "ema" in indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=indicators.index,
                    y=indicators["ema"],
                    name=f"{strategy_name} - EMA",
                    line=dict(dash="dash", width=1.5, color="orange"),
                    opacity=0.6,
                    legendgroup="row1",
                ),
                row=1,
                col=1,
            )

    # Plot 2: MACD indicators
    # Always plot MACD if provided, even if no strategies are selected
    if macd_indicator is not None and macd_signal is not None:
        fig.add_trace(
            go.Scatter(
                x=macd_indicator.index,
                y=macd_indicator.values,
                name="MACD",
                line=dict(width=1.5, color="green"),
                opacity=0.7,
                legendgroup="row2",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=macd_signal.index,
                y=macd_signal.values,
                name="Signal Line",
                line=dict(width=1.5, color="red"),
                opacity=0.7,
                legendgroup="row2",
            ),
            row=2,
            col=1,
        )
        if macd_histogram is not None:
            fig.add_trace(
                go.Bar(
                    x=macd_histogram.index,
                    y=macd_histogram.values,
                    name="Histogram",
                    opacity=0.3,
                    marker_color="blue",
                    legendgroup="row2",
                ),
                row=2,
                col=1,
            )
    
    # Also plot MACD from strategies if available
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
                    legendgroup="row2",
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
                    legendgroup="row2",
                ),
                row=2,
                col=1,
            )
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

    # Plot 3: RSI indicators
    # Always plot RSI if provided, even if no strategies are selected
    if rsi_indicator is not None:
        fig.add_trace(
            go.Scatter(
                x=rsi_indicator.index,
                y=rsi_indicator.values,
                name="RSI",
                line=dict(width=1.5, color="purple"),
                opacity=0.7,
                legendgroup="row3",
            ),
            row=3,
            col=1,
        )
    
    # Also plot RSI from strategies if available
    for strategy_name, (strategy_result, portfolio) in strategies.items():
        indicators = strategy_result.indicators
        if "rsi" in indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=indicators.index,
                    y=indicators["rsi"],
                    name=f"{strategy_name} - RSI",
                    line=dict(width=1.5, color="purple"),
                    opacity=0.7,
                    legendgroup="row3",
                ),
                row=3,
                col=1,
            )

    # Add horizontal line at 50 (RSI threshold) - always show if RSI is available
    has_rsi = (rsi_indicator is not None) or any("rsi" in strategy_result.indicators.columns for strategy_result, _ in strategies.values())
    if has_rsi:
        fig.add_hline(
            y=50,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text="RSI 50",
            row=3,
            col=1,
        )

    # Plot 4: Portfolio values
    for strategy_name, (_, portfolio) in strategies.items():
        fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio["total"], name=strategy_name, line=dict(width=2), legendgroup="row4"), row=4, col=1)

    # Buy and hold baseline (always show)
    if len(strategies) > 0:
        first_portfolio = list(strategies.values())[0][1]
        initial_value = first_portfolio["total"].iloc[0]
    else:
        # If no strategies, use a default initial value
        initial_value = 100000.0
    buy_hold = initial_value * (data / data.iloc[0])
    fig.add_trace(go.Scatter(x=data.index, y=buy_hold, name="Buy & Hold", line=dict(dash="dash", width=2), opacity=0.7, legendgroup="row4"), row=4, col=1)

    # Plot 5: Drawdown
    for strategy_name, (_, portfolio) in strategies.items():
        returns = portfolio["returns"].dropna()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100

        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values, name=strategy_name, fill="tozeroy", line=dict(width=1.5), opacity=0.3, legendgroup="row5"), row=5, col=1
        )

    # Update layout
    fig.update_layout(
        height=1400,
        hovermode="x unified",
        showlegend=True,
    )

    # Update x-axes
    fig.update_xaxes(title_text="Date", row=5, col=1)
    for row in range(1, 6):
        fig.update_xaxes(matches="x", row=row, col=1)

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=4, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=5, col=1)

    return fig


def page_controls(shared_symbol: str):
    """Create page controls for backtest parameters."""
    with st.expander("⚙️ Backtest Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            # Use shared symbol instead of separate input
            st.text_input("Symbol", value=shared_symbol, help="Stock or ETF ticker symbol (shared with Fund Inspector)", key="backtest_symbol_display", disabled=True)
            initial_capital = st.number_input(
                "Initial Capital ($)", min_value=1000.0, max_value=10000000.0, value=100000.0, step=10000.0, format="%.0f"
            )
        with col2:
            lookback_options = {"1 Year": "1y", "2 Years": "2y", "3 Years": "3y", "5 Years": "5y", "10 Years": "10y", "Max": "max"}
            lookback_label = st.selectbox(
                "Lookback Period", options=list(lookback_options.keys()), index=3, help="How far back to fetch historical data"
            )
            period = lookback_options[lookback_label]

            end_date = st.date_input("End Date", value=date.today() + timedelta(days=1), help="End date for backtest (defaults to today)")

        st.divider()

        # Strategy enable/disable checkboxes
        st.subheader("🎛️ Strategy Selection")
        col_strategy_row1 = st.columns(4)
        with col_strategy_row1[0]:
            enable_simple_ema = st.checkbox("EMA_x", value=True, key="enable_simple_ema", help="Buy when price crosses above EMA")
        with col_strategy_row1[1]:
            enable_ema_cross = st.checkbox("EMA Cross", value=False, key="enable_ema_cross", help="Buy when short EMA crosses above long EMA")
        with col_strategy_row1[2]:
            enable_macd = st.checkbox("MACD", value=False, key="enable_macd", help="Buy when MACD line crosses above signal line")
        with col_strategy_row1[3]:
            enable_ema50_macd = st.checkbox(
                "EMA_x + MACD", value=False, key="enable_ema50_macd", help="Buy when price > EMA_x AND MACD > 0, sell when price < EMA_x AND MACD < 0"
            )

        col_strategy_row2 = st.columns(3)
        with col_strategy_row2[0]:
            enable_rsi = st.checkbox("RSI", value=False, key="enable_rsi", help="Buy when RSI > 50, sell when RSI < 50")
        with col_strategy_row2[1]:
            enable_ema_x_rsi = st.checkbox(
                "EMA_x + RSI", value=False, key="enable_ema_x_rsi", help="Buy when price > EMA_x AND RSI > 50, sell when price < EMA_x AND RSI < 50"
            )
        with col_strategy_row2[2]:
            enable_ema_x_macd_rsi = st.checkbox(
                "EMA_x + MACD + RSI",
                value=False,
                key="enable_ema_x_macd_rsi",
                help="Buy when price > EMA_x AND MACD > 0 AND RSI > 50, sell when price < EMA_x AND MACD < 0 AND RSI < 50",
            )

        st.divider()

        # Create columns dynamically based on enabled strategies
        enabled_count = sum(
            [enable_simple_ema, enable_ema_cross, enable_macd, enable_ema50_macd, enable_rsi, enable_ema_x_rsi, enable_ema_x_macd_rsi]
        )
        if enabled_count > 0:
            cols = st.columns(enabled_count)
            col_idx = 0

            # EMA_x Parameters
            if enable_simple_ema:
                with cols[col_idx]:
                    st.subheader("🎯 EMA_x Parameters")
                    optimize_simple_ema = st.checkbox(
                        "🔍 Optimize Parameters",
                        value=False,
                        key="optimize_simple_ema",
                        help="Automatically find the best EMA period based on performance metrics",
                    )
                    if optimize_simple_ema:
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
                        simple_ema_period = None
                    else:
                        simple_ema_period = st.slider("EMA Period", min_value=5, max_value=200, value=50, step=5, key="simple_ema_period")
                        optimization_metric = None
                        ema_min = None
                        ema_max = None
                        ema_step = None
                col_idx += 1
            else:
                simple_ema_period = None
                optimize_simple_ema = False
                optimization_metric = None
                ema_min = None
                ema_max = None
                ema_step = None

            # EMA Strategy Parameters
            if enable_ema_cross:
                with cols[col_idx]:
                    st.subheader("📈 EMA Strategy Parameters")
                    ema_short = st.slider("Short EMA Period", min_value=5, max_value=100, value=50, step=5, key="ema_short")
                    ema_long = st.slider("Long EMA Period", min_value=50, max_value=300, value=200, step=10, key="ema_long")
                col_idx += 1
            else:
                ema_short = 50
                ema_long = 200

            # MACD Strategy Parameters
            if enable_macd:
                with cols[col_idx]:
                    st.subheader("📉 MACD Strategy Parameters")
                    macd_fast = st.slider("Fast EMA Period", min_value=5, max_value=30, value=12, step=1, key="macd_fast")
                    macd_slow = st.slider("Slow EMA Period", min_value=15, max_value=50, value=26, step=1, key="macd_slow")
                    macd_signal = st.slider("Signal Line Period", min_value=5, max_value=20, value=9, step=1, key="macd_signal")
                col_idx += 1
            else:
                macd_fast = 12
                macd_slow = 26
                macd_signal = 9

            # EMA_x + MACD Strategy Parameters
            if enable_ema50_macd:
                with cols[col_idx]:
                    st.subheader("🎯 EMA_x + MACD Parameters")
                    ema50_macd_ema = st.slider("EMA Period", min_value=5, max_value=200, value=50, step=5, key="ema50_macd_ema")
                    ema50_macd_fast = st.slider("MACD Fast Period", min_value=5, max_value=30, value=12, step=1, key="ema50_macd_fast")
                    ema50_macd_slow = st.slider("MACD Slow Period", min_value=15, max_value=50, value=26, step=1, key="ema50_macd_slow")
                    ema50_macd_signal = st.slider("MACD Signal Period", min_value=5, max_value=20, value=9, step=1, key="ema50_macd_signal")
                col_idx += 1
            else:
                ema50_macd_ema = 50
                ema50_macd_fast = 12
                ema50_macd_slow = 26
                ema50_macd_signal = 9

            # RSI Strategy Parameters
            if enable_rsi:
                with cols[col_idx]:
                    st.subheader("📊 RSI Parameters")
                    optimize_rsi_strategy = st.checkbox(
                        "🔍 Optimize Parameters",
                        value=False,
                        key="optimize_rsi_strategy",
                        help="Automatically find the best RSI period based on performance metrics",
                    )
                    if optimize_rsi_strategy:
                        rsi_optimization_metric = st.selectbox(
                            "Optimization Metric",
                            options=["sharpe_ratio", "total_return", "annualized_return"],
                            index=0,
                            key="rsi_optimization_metric",
                            help="Metric to optimize for: Sharpe Ratio (risk-adjusted return), Total Return, or Annualized Return",
                        )
                        rsi_min = st.slider("Min RSI Period", min_value=5, max_value=20, value=5, step=1, key="rsi_min")
                        rsi_max = st.slider("Max RSI Period", min_value=10, max_value=30, value=30, step=1, key="rsi_max")
                        rsi_step = st.slider("Step Size", min_value=1, max_value=5, value=1, step=1, key="rsi_step")
                        rsi_period = None
                    else:
                        rsi_period = st.slider("RSI Period", min_value=5, max_value=30, value=14, step=1, key="rsi_period")
                        rsi_optimization_metric = None
                        rsi_min = None
                        rsi_max = None
                        rsi_step = None
                col_idx += 1
            else:
                rsi_period = 14
                optimize_rsi_strategy = False
                rsi_optimization_metric = None
                rsi_min = None
                rsi_max = None
                rsi_step = None

            # EMA_x + RSI Strategy Parameters
            if enable_ema_x_rsi:
                with cols[col_idx]:
                    st.subheader("🎯 EMA_x + RSI Parameters")
                    ema_x_rsi_ema = st.slider("EMA Period", min_value=5, max_value=200, value=50, step=5, key="ema_x_rsi_ema")
                    ema_x_rsi_rsi_period = st.slider("RSI Period", min_value=5, max_value=30, value=14, step=1, key="ema_x_rsi_rsi_period")
                col_idx += 1
            else:
                ema_x_rsi_ema = 50
                ema_x_rsi_rsi_period = 14

            # EMA_x + MACD + RSI Strategy Parameters
            if enable_ema_x_macd_rsi:
                with cols[col_idx]:
                    st.subheader("🎯 EMA_x + MACD + RSI Parameters")
                    ema_x_macd_rsi_ema = st.slider("EMA Period", min_value=5, max_value=200, value=50, step=5, key="ema_x_macd_rsi_ema")
                    ema_x_macd_rsi_fast = st.slider("MACD Fast Period", min_value=5, max_value=30, value=12, step=1, key="ema_x_macd_rsi_fast")
                    ema_x_macd_rsi_slow = st.slider("MACD Slow Period", min_value=15, max_value=50, value=26, step=1, key="ema_x_macd_rsi_slow")
                    ema_x_macd_rsi_signal = st.slider("MACD Signal Period", min_value=5, max_value=20, value=9, step=1, key="ema_x_macd_rsi_signal")
                    ema_x_macd_rsi_rsi_period = st.slider("RSI Period", min_value=5, max_value=30, value=14, step=1, key="ema_x_macd_rsi_rsi_period")
            else:
                ema_x_macd_rsi_ema = 50
                ema_x_macd_rsi_fast = 12
                ema_x_macd_rsi_slow = 26
                ema_x_macd_rsi_signal = 9
                ema_x_macd_rsi_rsi_period = 14

    return (
        shared_symbol,  # Use shared symbol instead of separate input
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
        ema50_macd_ema,
        ema50_macd_fast,
        ema50_macd_slow,
        ema50_macd_signal,
        rsi_period,
        optimize_rsi_strategy,
        rsi_optimization_metric,
        rsi_min,
        rsi_max,
        rsi_step,
        ema_x_rsi_ema,
        ema_x_rsi_rsi_period,
        ema_x_macd_rsi_ema,
        ema_x_macd_rsi_fast,
        ema_x_macd_rsi_slow,
        ema_x_macd_rsi_signal,
        ema_x_macd_rsi_rsi_period,
        enable_ema_cross,
        enable_macd,
        enable_simple_ema,
        enable_ema50_macd,
        enable_rsi,
        enable_ema_x_rsi,
        enable_ema_x_macd_rsi,
    )


def run_backtest_workflow(
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
    ema50_macd_ema,
    ema50_macd_fast,
    ema50_macd_slow,
    ema50_macd_signal,
    rsi_period,
    optimize_rsi_strategy,
    rsi_optimization_metric,
    rsi_min,
    rsi_max,
    rsi_step,
    ema_x_rsi_ema,
    ema_x_rsi_rsi_period,
    ema_x_macd_rsi_ema,
    ema_x_macd_rsi_fast,
    ema_x_macd_rsi_slow,
    ema_x_macd_rsi_signal,
    ema_x_macd_rsi_rsi_period,
    enable_ema_cross,
    enable_macd,
    enable_simple_ema,
    enable_ema50_macd,
    enable_rsi,
    enable_ema_x_rsi,
    enable_ema_x_macd_rsi,
):
    """Run the complete backtest workflow."""
    with st.spinner(f"Fetching data for {symbol}..."):
        try:
            # Fetch data
            data = fetch_data(symbol, period, str(end_date))

            # Always compute RSI and MACD indicators regardless of strategy selection
            with st.spinner("Computing RSI and MACD indicators..."):
                # Compute RSI with default period (14)
                rsi_indicator = compute_rsi(data, window=14)
                
                # Compute MACD with default parameters (12, 26, 9)
                exp1 = data.ewm(span=12, adjust=False).mean()
                exp2 = data.ewm(span=26, adjust=False).mean()
                macd_indicator = exp1 - exp2
                macd_signal_line = macd_indicator.ewm(span=9, adjust=False).mean()
                macd_histogram = macd_indicator - macd_signal_line

            # Initialize strategy results and portfolios
            ema_cross_result = None
            ema_cross_portfolio = None
            macd_result = None
            macd_portfolio = None
            simple_ema_result = None
            simple_ema_portfolio = None
            ema50_macd_result = None
            ema50_macd_portfolio = None
            rsi_result = None
            rsi_portfolio = None
            ema_x_rsi_result = None
            ema_x_rsi_portfolio = None
            ema_x_macd_rsi_result = None
            ema_x_macd_rsi_portfolio = None

            # Run enabled strategies
            if enable_ema_cross:
                with st.spinner("Running EMA strategies..."):
                    ema_cross_result = ema_cross_strategy(data, short_window=ema_short, long_window=ema_long)
                    ema_cross_portfolio = backtest_strategy(data, ema_cross_result, initial_capital=initial_capital)

            if enable_macd:
                with st.spinner("Running MACD strategy..."):
                    macd_result = macd_strategy(data, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
                    macd_portfolio = backtest_strategy(data, macd_result, initial_capital=initial_capital)

            # Handle EMA_x strategy with optional optimization
            if enable_simple_ema:
                if optimize_simple_ema:
                    with st.spinner("Optimizing EMA_x strategy..."):
                        best_ema_period, best_metrics, optimization_results_df = optimize_simple_ema_crossing(
                            data,
                            ema_period_range=(ema_min, ema_max),
                            step=ema_step,
                            initial_capital=initial_capital,
                            optimization_metric=optimization_metric,
                        )
                        simple_ema_period = best_ema_period

                        st.session_state["simple_ema_optimization_results"] = optimization_results_df
                        st.session_state["simple_ema_best_metrics"] = best_metrics
                        st.session_state["simple_ema_best_period"] = best_ema_period

                        st.info(f"✅ Optimization complete! Best EMA Period: {simple_ema_period} (based on {optimization_metric})")

                with st.spinner("Running EMA_x strategy..."):
                    simple_ema_result = simple_ema_crossing_strategy(data, ema_period=simple_ema_period)
                    simple_ema_portfolio = backtest_strategy(data, simple_ema_result, initial_capital=initial_capital)

            if enable_ema50_macd:
                with st.spinner("Running EMA_x + MACD strategy..."):
                    ema50_macd_result = ema50_macd_strategy(
                        data,
                        ema_period=ema50_macd_ema,
                        macd_fast_window=ema50_macd_fast,
                        macd_slow_window=ema50_macd_slow,
                        macd_signal_window=ema50_macd_signal,
                    )
                    ema50_macd_portfolio = backtest_strategy(data, ema50_macd_result, initial_capital=initial_capital)

            # Handle RSI strategy with optional optimization
            if enable_rsi:
                if optimize_rsi_strategy:
                    with st.spinner("Optimizing RSI strategy..."):
                        best_rsi_period, best_metrics, optimization_results_df = optimize_rsi(
                            data,
                            rsi_period_range=(rsi_min, rsi_max),
                            step=rsi_step,
                            initial_capital=initial_capital,
                            optimization_metric=rsi_optimization_metric,
                        )
                        rsi_period = best_rsi_period

                        st.session_state["rsi_optimization_results"] = optimization_results_df
                        st.session_state["rsi_best_metrics"] = best_metrics
                        st.session_state["rsi_best_period"] = best_rsi_period

                        st.info(f"✅ Optimization complete! Best RSI Period: {rsi_period} (based on {rsi_optimization_metric})")

                with st.spinner("Running RSI strategy..."):
                    rsi_result = rsi_strategy(data, rsi_period=rsi_period)
                    rsi_portfolio = backtest_strategy(data, rsi_result, initial_capital=initial_capital)

            if enable_ema_x_rsi:
                with st.spinner("Running EMA_x + RSI strategy..."):
                    ema_x_rsi_result = ema_x_rsi_strategy(data, ema_period=ema_x_rsi_ema, rsi_period=ema_x_rsi_rsi_period)
                    ema_x_rsi_portfolio = backtest_strategy(data, ema_x_rsi_result, initial_capital=initial_capital)

            if enable_ema_x_macd_rsi:
                with st.spinner("Running EMA_x + MACD + RSI strategy..."):
                    ema_x_macd_rsi_result = ema_x_macd_rsi_strategy(
                        data,
                        ema_period=ema_x_macd_rsi_ema,
                        macd_fast_window=ema_x_macd_rsi_fast,
                        macd_slow_window=ema_x_macd_rsi_slow,
                        macd_signal_window=ema_x_macd_rsi_signal,
                        rsi_period=ema_x_macd_rsi_rsi_period,
                    )
                    ema_x_macd_rsi_portfolio = backtest_strategy(data, ema_x_macd_rsi_result, initial_capital=initial_capital)

            # Find common start time for enabled strategies
            all_start_indices = []
            if enable_ema_cross and ema_cross_portfolio is not None:
                all_start_indices.append(ema_cross_portfolio.index[0])
            if enable_macd and macd_portfolio is not None:
                all_start_indices.append(macd_portfolio.index[0])
            if enable_simple_ema and simple_ema_portfolio is not None:
                all_start_indices.append(simple_ema_portfolio.index[0])
            if enable_ema50_macd and ema50_macd_portfolio is not None:
                all_start_indices.append(ema50_macd_portfolio.index[0])
            if enable_rsi and rsi_portfolio is not None:
                all_start_indices.append(rsi_portfolio.index[0])
            if enable_ema_x_rsi and ema_x_rsi_portfolio is not None:
                all_start_indices.append(ema_x_rsi_portfolio.index[0])
            if enable_ema_x_macd_rsi and ema_x_macd_rsi_portfolio is not None:
                all_start_indices.append(ema_x_macd_rsi_portfolio.index[0])

            # If no strategies selected, use data start index
            if not all_start_indices:
                common_start_idx = data.index[0]
            else:
                common_start_idx = max(all_start_indices)

            # Trim portfolios and data to start at the same time
            data_display = data.loc[common_start_idx:]

            # Align all strategies to common index
            strategies_dict = {}
            metrics_dict = {}

            if enable_ema_cross and ema_cross_portfolio is not None:
                portfolio_aligned = ema_cross_portfolio.loc[common_start_idx:].copy()
                signals_aligned = ema_cross_result.signals.loc[common_start_idx:].copy()
                indicators_aligned = ema_cross_result.indicators.loc[common_start_idx:].copy()
                result_aligned = StrategyResult(signals=signals_aligned, indicators=indicators_aligned)
                portfolio_aligned = backtest_strategy(data_display, result_aligned, initial_capital=initial_capital)
                strategies_dict[f"EMA Cross {ema_short}/{ema_long}"] = (result_aligned, portfolio_aligned)
                metrics_dict[f"EMA Cross {ema_short}/{ema_long}"] = calculate_performance_metrics(portfolio_aligned, f"EMA Cross {ema_short}/{ema_long}")

            if enable_macd and macd_portfolio is not None:
                portfolio_aligned = macd_portfolio.loc[common_start_idx:].copy()
                signals_aligned = macd_result.signals.loc[common_start_idx:].copy()
                indicators_aligned = macd_result.indicators.loc[common_start_idx:].copy()
                if len(signals_aligned) > 0:
                    if signals_aligned.iloc[0]["signal"] == 1.0:
                        signals_aligned.iloc[0, signals_aligned.columns.get_loc("positions")] = 1.0
                    else:
                        signals_aligned.iloc[0, signals_aligned.columns.get_loc("positions")] = 0.0
                result_aligned = StrategyResult(signals=signals_aligned, indicators=indicators_aligned)
                portfolio_aligned = backtest_strategy(data_display, result_aligned, initial_capital=initial_capital)
                strategies_dict["MACD"] = (result_aligned, portfolio_aligned)
                metrics_dict["MACD"] = calculate_performance_metrics(portfolio_aligned, "MACD")

            if enable_simple_ema and simple_ema_portfolio is not None:
                portfolio_aligned = simple_ema_portfolio.loc[common_start_idx:].copy()
                signals_aligned = simple_ema_result.signals.loc[common_start_idx:].copy()
                indicators_aligned = simple_ema_result.indicators.loc[common_start_idx:].copy()
                if len(signals_aligned) > 0:
                    if signals_aligned.iloc[0]["signal"] == 1.0:
                        signals_aligned.iloc[0, signals_aligned.columns.get_loc("positions")] = 1.0
                    else:
                        signals_aligned.iloc[0, signals_aligned.columns.get_loc("positions")] = 0.0
                result_aligned = StrategyResult(signals=signals_aligned, indicators=indicators_aligned)
                portfolio_aligned = backtest_strategy(data_display, result_aligned, initial_capital=initial_capital)
                strategies_dict[f"EMA_x {simple_ema_period}"] = (result_aligned, portfolio_aligned)
                metrics_dict[f"EMA_x {simple_ema_period}"] = calculate_performance_metrics(portfolio_aligned, f"EMA_x {simple_ema_period}")

            if enable_ema50_macd and ema50_macd_portfolio is not None:
                portfolio_aligned = ema50_macd_portfolio.loc[common_start_idx:].copy()
                signals_aligned = ema50_macd_result.signals.loc[common_start_idx:].copy()
                indicators_aligned = ema50_macd_result.indicators.loc[common_start_idx:].copy()
                if len(signals_aligned) > 0:
                    if signals_aligned.iloc[0]["signal"] == 1.0:
                        signals_aligned.iloc[0, signals_aligned.columns.get_loc("positions")] = 1.0
                    else:
                        signals_aligned.iloc[0, signals_aligned.columns.get_loc("positions")] = 0.0
                result_aligned = StrategyResult(signals=signals_aligned, indicators=indicators_aligned)
                portfolio_aligned = backtest_strategy(data_display, result_aligned, initial_capital=initial_capital)
                strategies_dict["EMA_x + MACD"] = (result_aligned, portfolio_aligned)
                metrics_dict["EMA_x + MACD"] = calculate_performance_metrics(portfolio_aligned, "EMA_x + MACD")

            if enable_rsi and rsi_portfolio is not None:
                portfolio_aligned = rsi_portfolio.loc[common_start_idx:].copy()
                signals_aligned = rsi_result.signals.loc[common_start_idx:].copy()
                indicators_aligned = rsi_result.indicators.loc[common_start_idx:].copy()
                if len(signals_aligned) > 0:
                    if signals_aligned.iloc[0]["signal"] == 1.0:
                        signals_aligned.iloc[0, signals_aligned.columns.get_loc("positions")] = 1.0
                    else:
                        signals_aligned.iloc[0, signals_aligned.columns.get_loc("positions")] = 0.0
                result_aligned = StrategyResult(signals=signals_aligned, indicators=indicators_aligned)
                portfolio_aligned = backtest_strategy(data_display, result_aligned, initial_capital=initial_capital)
                strategies_dict[f"RSI {rsi_period}"] = (result_aligned, portfolio_aligned)
                metrics_dict[f"RSI {rsi_period}"] = calculate_performance_metrics(portfolio_aligned, f"RSI {rsi_period}")

            if enable_ema_x_rsi and ema_x_rsi_portfolio is not None:
                portfolio_aligned = ema_x_rsi_portfolio.loc[common_start_idx:].copy()
                signals_aligned = ema_x_rsi_result.signals.loc[common_start_idx:].copy()
                indicators_aligned = ema_x_rsi_result.indicators.loc[common_start_idx:].copy()
                if len(signals_aligned) > 0:
                    if signals_aligned.iloc[0]["signal"] == 1.0:
                        signals_aligned.iloc[0, signals_aligned.columns.get_loc("positions")] = 1.0
                    else:
                        signals_aligned.iloc[0, signals_aligned.columns.get_loc("positions")] = 0.0
                result_aligned = StrategyResult(signals=signals_aligned, indicators=indicators_aligned)
                portfolio_aligned = backtest_strategy(data_display, result_aligned, initial_capital=initial_capital)
                strategies_dict["EMA_x + RSI"] = (result_aligned, portfolio_aligned)
                metrics_dict["EMA_x + RSI"] = calculate_performance_metrics(portfolio_aligned, "EMA_x + RSI")

            if enable_ema_x_macd_rsi and ema_x_macd_rsi_portfolio is not None:
                portfolio_aligned = ema_x_macd_rsi_portfolio.loc[common_start_idx:].copy()
                signals_aligned = ema_x_macd_rsi_result.signals.loc[common_start_idx:].copy()
                indicators_aligned = ema_x_macd_rsi_result.indicators.loc[common_start_idx:].copy()
                if len(signals_aligned) > 0:
                    if signals_aligned.iloc[0]["signal"] == 1.0:
                        signals_aligned.iloc[0, signals_aligned.columns.get_loc("positions")] = 1.0
                    else:
                        signals_aligned.iloc[0, signals_aligned.columns.get_loc("positions")] = 0.0
                result_aligned = StrategyResult(signals=signals_aligned, indicators=indicators_aligned)
                portfolio_aligned = backtest_strategy(data_display, result_aligned, initial_capital=initial_capital)
                strategies_dict["EMA_x + MACD + RSI"] = (result_aligned, portfolio_aligned)
                metrics_dict["EMA_x + MACD + RSI"] = calculate_performance_metrics(portfolio_aligned, "EMA_x + MACD + RSI")

            # Display data info
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

            # Align RSI and MACD indicators to common start index
            rsi_aligned = rsi_indicator.loc[common_start_idx:] if common_start_idx in rsi_indicator.index else rsi_indicator
            macd_aligned = macd_indicator.loc[common_start_idx:] if common_start_idx in macd_indicator.index else macd_indicator
            macd_signal_aligned = macd_signal_line.loc[common_start_idx:] if common_start_idx in macd_signal_line.index else macd_signal_line
            macd_hist_aligned = macd_histogram.loc[common_start_idx:] if common_start_idx in macd_histogram.index else macd_histogram

            # Store in session state
            st.session_state["data"] = data_display
            st.session_state["strategies"] = strategies_dict
            st.session_state["metrics"] = metrics_dict
            st.session_state["symbol"] = symbol
            st.session_state["rsi_indicator"] = rsi_aligned
            st.session_state["macd_indicator"] = macd_aligned
            st.session_state["macd_signal"] = macd_signal_aligned
            st.session_state["macd_histogram"] = macd_hist_aligned

            st.success("✅ Backtest completed successfully!")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.error(f"Backtest error: {e}")


def display_backtest_results():
    """Display backtest results if available."""
    if "data" not in st.session_state:
        return

    st.divider()

    # Performance metrics comparison (only if strategies were run)
    if "strategies" in st.session_state and "metrics" in st.session_state and len(st.session_state["metrics"]) > 0:
        st.header("📈 Performance Metrics")
        metrics_df = pd.DataFrame([st.session_state["metrics"][k] for k in st.session_state["metrics"].keys()])
        metrics_df = metrics_df.set_index("Strategy")

        # Format the dataframe for display
        display_df = metrics_df.copy()
        for col in display_df.columns:
            if col != "Number of Trades":
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

        st.dataframe(display_df, width="stretch")

    # Display optimization results if available
    if "simple_ema_optimization_results" in st.session_state:
        st.divider()
        st.header("🔍 EMA_x Optimization Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best EMA Period", int(st.session_state["simple_ema_best_period"]))
        with col2:
            best_metrics = st.session_state["simple_ema_best_metrics"]
            st.metric("Best Sharpe Ratio", f"{best_metrics.get('Sharpe Ratio', 0):.2f}")
        with col3:
            st.metric("Best Total Return", f"{best_metrics.get('Total Return (%)', 0):.2f}%")

        opt_results = st.session_state["simple_ema_optimization_results"].copy()
        for col in opt_results.columns:
            if col not in ["EMA Period", "Strategy", "Number of Trades"]:
                opt_results[col] = opt_results[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

        st.dataframe(opt_results, width="stretch")

        # Visualization
        st.subheader("Optimization Parameter Analysis")
        opt_results_numeric = st.session_state["simple_ema_optimization_results"].copy()
        symbol = st.session_state.get("symbol", "")
        fig_opt = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                f"{symbol} - Sharpe Ratio vs EMA Period",
                f"{symbol} - Total Return vs EMA Period",
                f"{symbol} - Annualized Return vs EMA Period",
                f"{symbol} - Max Drawdown vs EMA Period",
            ),
            vertical_spacing=0.12,
        )

        fig_opt.add_trace(
            go.Scatter(x=opt_results_numeric["EMA Period"], y=opt_results_numeric["Sharpe Ratio"], mode="lines+markers", name="Sharpe Ratio", line=dict(color="green", width=2), marker=dict(size=6)),
            row=1,
            col=1,
        )
        fig_opt.add_trace(
            go.Scatter(x=opt_results_numeric["EMA Period"], y=opt_results_numeric["Total Return (%)"], mode="lines+markers", name="Total Return (%)", line=dict(color="blue", width=2), marker=dict(size=6)),
            row=1,
            col=2,
        )
        fig_opt.add_trace(
            go.Scatter(x=opt_results_numeric["EMA Period"], y=opt_results_numeric["Annualized Return (%)"], mode="lines+markers", name="Annualized Return (%)", line=dict(color="purple", width=2), marker=dict(size=6)),
            row=2,
            col=1,
        )
        fig_opt.add_trace(
            go.Scatter(x=opt_results_numeric["EMA Period"], y=opt_results_numeric["Max Drawdown (%)"], mode="lines+markers", name="Max Drawdown (%)", line=dict(color="red", width=2), marker=dict(size=6)),
            row=2,
            col=2,
        )

        best_period = st.session_state["simple_ema_best_period"]
        for row in [1, 2]:
            for col in [1, 2]:
                fig_opt.add_vline(x=best_period, line_dash="dash", line_color="orange", annotation_text=f"Best: {best_period}", row=row, col=col)

        fig_opt.update_layout(height=600, showlegend=False, hovermode="x unified")
        fig_opt.update_xaxes(title_text="EMA Period", row=2, col=1)
        fig_opt.update_xaxes(title_text="EMA Period", row=2, col=2)
        fig_opt.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig_opt.update_yaxes(title_text="Total Return (%)", row=1, col=2)
        fig_opt.update_yaxes(title_text="Annualized Return (%)", row=2, col=1)
        fig_opt.update_yaxes(title_text="Max Drawdown (%)", row=2, col=2)

        st.plotly_chart(fig_opt, width="stretch")

    # Display RSI optimization results if available
    if "rsi_optimization_results" in st.session_state:
        st.divider()
        st.header("🔍 RSI Optimization Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best RSI Period", int(st.session_state["rsi_best_period"]))
        with col2:
            best_metrics = st.session_state["rsi_best_metrics"]
            st.metric("Best Sharpe Ratio", f"{best_metrics.get('Sharpe Ratio', 0):.2f}")
        with col3:
            st.metric("Best Total Return", f"{best_metrics.get('Total Return (%)', 0):.2f}%")

        opt_results = st.session_state["rsi_optimization_results"].copy()
        for col in opt_results.columns:
            if col not in ["RSI Period", "Strategy", "Number of Trades"]:
                opt_results[col] = opt_results[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

        st.dataframe(opt_results, width="stretch")

        # Visualization
        st.subheader("Optimization Parameter Analysis")
        opt_results_numeric = st.session_state["rsi_optimization_results"].copy()
        symbol = st.session_state.get("symbol", "")
        fig_opt = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                f"{symbol} - Sharpe Ratio vs RSI Period",
                f"{symbol} - Total Return vs RSI Period",
                f"{symbol} - Annualized Return vs RSI Period",
                f"{symbol} - Max Drawdown vs RSI Period",
            ),
            vertical_spacing=0.12,
        )

        fig_opt.add_trace(
            go.Scatter(x=opt_results_numeric["RSI Period"], y=opt_results_numeric["Sharpe Ratio"], mode="lines+markers", name="Sharpe Ratio", line=dict(color="green", width=2), marker=dict(size=6)),
            row=1,
            col=1,
        )
        fig_opt.add_trace(
            go.Scatter(x=opt_results_numeric["RSI Period"], y=opt_results_numeric["Total Return (%)"], mode="lines+markers", name="Total Return (%)", line=dict(color="blue", width=2), marker=dict(size=6)),
            row=1,
            col=2,
        )
        fig_opt.add_trace(
            go.Scatter(x=opt_results_numeric["RSI Period"], y=opt_results_numeric["Annualized Return (%)"], mode="lines+markers", name="Annualized Return (%)", line=dict(color="purple", width=2), marker=dict(size=6)),
            row=2,
            col=1,
        )
        fig_opt.add_trace(
            go.Scatter(x=opt_results_numeric["RSI Period"], y=opt_results_numeric["Max Drawdown (%)"], mode="lines+markers", name="Max Drawdown (%)", line=dict(color="red", width=2), marker=dict(size=6)),
            row=2,
            col=2,
        )

        best_period = st.session_state["rsi_best_period"]
        for row in [1, 2]:
            for col in [1, 2]:
                fig_opt.add_vline(x=best_period, line_dash="dash", line_color="orange", annotation_text=f"Best: {best_period}", row=row, col=col)

        fig_opt.update_layout(height=600, showlegend=False, hovermode="x unified")
        fig_opt.update_xaxes(title_text="RSI Period", row=2, col=1)
        fig_opt.update_xaxes(title_text="RSI Period", row=2, col=2)
        fig_opt.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig_opt.update_yaxes(title_text="Total Return (%)", row=1, col=2)
        fig_opt.update_yaxes(title_text="Annualized Return (%)", row=2, col=1)
        fig_opt.update_yaxes(title_text="Max Drawdown (%)", row=2, col=2)

        st.plotly_chart(fig_opt, width="stretch")

    # Visualization
    st.header("📊 Strategy Comparison")
    rsi_ind = st.session_state.get("rsi_indicator", None)
    macd_ind = st.session_state.get("macd_indicator", None)
    macd_sig = st.session_state.get("macd_signal", None)
    macd_hist = st.session_state.get("macd_histogram", None)
    strategies_dict = st.session_state.get("strategies", {})
    fig = create_strategy_comparison_plot(
        st.session_state["data"], 
        strategies_dict, 
        st.session_state["symbol"],
        rsi_indicator=rsi_ind,
        macd_indicator=macd_ind,
        macd_signal=macd_sig,
        macd_histogram=macd_hist,
    )
    st.plotly_chart(fig, width="stretch")

    # Detailed metrics (only if strategies were run)
    if "metrics" in st.session_state and len(st.session_state["metrics"]) > 0:
        st.header("📋 Detailed Metrics")
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


def main_strategy_backtest_page(shared_symbol: str):
    """Main function for mid-frequency backtest page."""
    st.title("📊 Mid-Frequency Strategy Backtest")
    st.markdown(
        """
    Backtest and compare EMA crossover, EMA_x, MACD, RSI, and combined strategies with customizable parameters.
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
        ema50_macd_ema,
        ema50_macd_fast,
        ema50_macd_slow,
        ema50_macd_signal,
        rsi_period,
        optimize_rsi_strategy,
        rsi_optimization_metric,
        rsi_min,
        rsi_max,
        rsi_step,
        ema_x_rsi_ema,
        ema_x_rsi_rsi_period,
        ema_x_macd_rsi_ema,
        ema_x_macd_rsi_fast,
        ema_x_macd_rsi_slow,
        ema_x_macd_rsi_signal,
        ema_x_macd_rsi_rsi_period,
        enable_ema_cross,
        enable_macd,
        enable_simple_ema,
        enable_ema50_macd,
        enable_rsi,
        enable_ema_x_rsi,
        enable_ema_x_macd_rsi,
    ) = page_controls(shared_symbol)

    # Note: RSI and MACD indicators will always be computed and displayed even if no strategies are selected

    # Run backtest button
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        run_backtest_workflow(
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
            ema50_macd_ema,
            ema50_macd_fast,
            ema50_macd_slow,
            ema50_macd_signal,
            rsi_period,
            optimize_rsi_strategy,
            rsi_optimization_metric,
            rsi_min,
            rsi_max,
            rsi_step,
            ema_x_rsi_ema,
            ema_x_rsi_rsi_period,
            ema_x_macd_rsi_ema,
            ema_x_macd_rsi_fast,
            ema_x_macd_rsi_slow,
            ema_x_macd_rsi_signal,
            ema_x_macd_rsi_rsi_period,
            enable_ema_cross,
            enable_macd,
            enable_simple_ema,
            enable_ema50_macd,
            enable_rsi,
            enable_ema_x_rsi,
            enable_ema_x_macd_rsi,
        )

    # Display results if available
    display_backtest_results()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function that displays both Fund Inspector and Strategy Backtest on the same page."""
    # Setup page
    if hydra.core.global_hydra.GlobalHydra().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_resolvers()

    with hydra.initialize(version_base=None, config_path="../../../conf"):
        config = hydra.compose(config_name="main")

    # Setup page and get shared symbol selector
    shared_symbol = setup_page_and_sidebar(config["style_conf"], lambda: create_shared_symbol_selector(config))

    # Display Fund Inspector section
    st.header("🔍 Fund Inspector")
    main_fund_inspect_page(shared_symbol, config)

    # Add divider
    st.divider()

    # Display Strategy Backtest section
    st.header("📊 Strategy Backtest")
    main_strategy_backtest_page(shared_symbol)


if __name__ == "__main__":
    main()

