"""
Streamlit macro page for visualizing the Global Liquidity Index.

This page uses the `liquidity` package's GlobalLiquidity model and
displays the index from a selected beginning date (minus 15 weeks) to
now. It also overlays VTI, VXUS ETF, BTC-USD, and GC=F (Gold Futures)
prices normalized to 1 at the start of the range.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import hydra
from omegaconf import OmegaConf

from src.configurations.yaml import register_resolvers
from src.dashboard.create_page import setup_page_and_sidebar

from src.data.GLI import st_load_global_liquidity
from src.data.P import st_get_tickers_close_prices
from src.data.FearGreed import FearGreed
from src.data.TICKER import TICKERS


def parse_custom_symbols(symbols_text: str) -> list:
    """Parse comma-separated symbols and clean them."""
    if not symbols_text or symbols_text.strip() == "":
        return []

    # Split by comma and clean each symbol
    symbols = [symbol.strip().upper() for symbol in symbols_text.split(",")]
    # Remove empty strings
    symbols = [symbol for symbol in symbols if symbol]
    return symbols


def get_asset_config_with_custom(default_config: dict, custom_symbols: list, colors: list) -> dict:
    """Merge default asset config with custom symbols, assigning colors from config."""
    config = default_config.copy()

    # Add custom symbols with colors from config
    for idx, symbol in enumerate(custom_symbols):
        if symbol not in config:
            color = colors[idx % len(colors)]
            config[symbol] = {"name": symbol, "color": color, "opacity": 0.7}  # Default opacity for custom symbols

    return config


def _add_macro_sidebar(macro_config: dict):
    """Sidebar controls for the macro / liquidity view."""
    st.sidebar.header("Macro Settings")

    lookback_weeks = macro_config.get("lookback_weeks", 15)

    # Period options mapping: display name -> timedelta
    period_options = {
        "3 months": pd.Timedelta(days=90),
        "6 months": pd.Timedelta(days=180),
        "12 months": pd.Timedelta(days=365),
        "3 years": pd.Timedelta(days=365 * 3),
        "5 years": pd.Timedelta(days=365 * 5),
        "10 years": pd.Timedelta(days=365 * 10),
        "20 years": pd.Timedelta(days=365 * 20),
    }

    selected_period = st.sidebar.selectbox(
        "Time Period",
        options=list(period_options.keys()),
        index=0,  # Default to 3 months
        help="Select how far back from today to start the visualization. Data will be shown from 15 weeks before this date to now.",
    )

    # Calculate beginning_date based on selected period
    beginning_date = pd.Timestamp.today() - period_options[selected_period]

    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("**Lag / Look-ahead**")
    show_lag = st.sidebar.checkbox(
        "Show Lag",
        value=True,
        help="Display a lagged version of the Global Liquidity Index to visualize lead/lag effects.",
    )
    lag_weeks = st.sidebar.slider(
        "Lag / Look-ahead (weeks)",
        min_value=0,
        max_value=52,
        value=lookback_weeks,
        step=1,
        disabled=not show_lag,
        help="Shift the index forward in time by this many weeks to visualize a look-ahead effect.",
    )

    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("**EMA / RSI Smoothing**")
    # Single global smoothing control, used for GLI EMA and RSI / overlays
    smoothing_span = st.sidebar.slider(
        "Smoothing window (EMA span)",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        help="Apply an EMA smoothing window to the Global Liquidity EMA and RSI / overlay series (1 = no smoothing, higher = smoother).",
        key="smoothing_span",
    )
    # Always show EMA; reuse smoothing_span as EMA period
    show_ema = True
    ema_period = int(smoothing_span)

    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("**Custom Symbols**")
    custom_symbols_text = st.sidebar.text_input(
        "Add Custom Symbols",
        placeholder="Enter symbols separated by commas (e.g., AAPL, MSFT, GOOGL)",
        help="Enter custom ticker symbols separated by commas to overlay on the chart. Colors will be assigned automatically.",
        key="custom_symbols_input",
    )
    custom_symbols = parse_custom_symbols(custom_symbols_text)

    return beginning_date, int(lag_weeks), custom_symbols, show_ema, int(ema_period), show_lag


def _prepare_liquidity_data(
    df: pd.DataFrame,
    beginning_date: pd.Timestamp,
    lookback_weeks: int,
    lag_weeks: int,
    show_ema: bool,
    ema_period: int,
    show_lag: bool,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Prepare liquidity data: calculate date range, reindex, interpolate, and add EMA/lag columns."""
    lookback_days = lookback_weeks * 7
    start_date = beginning_date - pd.Timedelta(days=lookback_days + 7)
    end_date = pd.Timestamp.today()
    lag_days = lag_weeks * 7

    date_range_full = pd.date_range(
        start=start_date.normalize() if hasattr(start_date, "normalize") else pd.Timestamp(start_date).normalize(),
        end=end_date.normalize() if hasattr(end_date, "normalize") else pd.Timestamp(end_date).normalize(),
        freq="D",
    )

    df_extended = df.reindex(date_range_full, method=None)

    if "Liquidity Index" in df_extended.columns:
        df_extended["Liquidity Index"] = df_extended["Liquidity Index"].interpolate(method="time")
        if show_ema:
            df_extended[f"Liquidity Index (EMA {ema_period})"] = df_extended["Liquidity Index"].ewm(span=ema_period, adjust=False).mean()

    if show_lag and lag_days > 0:
        df_extended["Liquidity Index (Lag)"] = df_extended["Liquidity Index"].shift(lag_days)
        if show_ema:
            df_extended[f"Liquidity Index (Lag EMA {ema_period})"] = df_extended["Liquidity Index (Lag)"].ewm(span=ema_period, adjust=False).mean()

    df_plot = df_extended.loc[df_extended.index <= end_date]
    return df_plot, start_date, end_date


def _add_liquidity_traces(
    fig: go.Figure,
    df_plot: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    lag_weeks: int,
    show_ema: bool,
    ema_period: int,
    show_lag: bool,
):
    """Add all liquidity index traces to the figure."""
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot["Liquidity Index"],
            name=f"Global Liquidity Index ({start_date.date()} to {end_date.date()})",
            line=dict(color="royalblue", width=2),
            opacity=0.75,
        ),
        secondary_y=False,
    )

    if show_ema and f"Liquidity Index (EMA {ema_period})" in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot[f"Liquidity Index (EMA {ema_period})"],
                name=f"Liquidity Index (EMA {ema_period})",
                line=dict(color="skyblue", width=2, dash="dot"),
                opacity=0.9,
            ),
            secondary_y=False,
        )

    if show_lag and "Liquidity Index (Lag)" in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot["Liquidity Index (Lag)"],
                name=f"Liquidity Index (Lag {lag_weeks}w)",
                line=dict(color="green", width=2, dash="dash"),
                opacity=1.0,
            ),
            secondary_y=False,
        )

        if show_ema and f"Liquidity Index (Lag EMA {ema_period})" in df_plot.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_plot.index,
                    y=df_plot[f"Liquidity Index (Lag EMA {ema_period})"],
                    name=f"Liquidity Index (Lag {lag_weeks}w EMA {ema_period})",
                    line=dict(color="lightgreen", width=2, dash="dot"),
                    opacity=0.9,
                ),
                secondary_y=False,
            )


def _add_asset_overlays(
    fig: go.Figure,
    merged_asset_config: dict,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
):
    """Load, normalize, and plot asset overlays."""
    tickers = list(merged_asset_config.keys())
    assets_hist = st_get_tickers_close_prices(tickers, period="max")

    if assets_hist.empty:
        st.info("Asset price data could not be loaded (yfinance returned no data).")
        return

    assets_mask = (assets_hist.index >= start_date) & (assets_hist.index <= end_date)
    assets_plot = assets_hist.loc[assets_mask]
    smooth_span = st.session_state.get("smoothing_span", 1)

    for ticker in tickers:
        if ticker not in assets_plot.columns:
            continue

        asset_data = assets_plot[ticker].dropna()
        if asset_data.empty:
            continue

        if smooth_span > 1:
            asset_data = asset_data.ewm(span=smooth_span, adjust=False).mean()

        asset_norm = asset_data / asset_data.iloc[0]
        config = merged_asset_config[ticker]
        asset_opacity = config.get("opacity", 1.0)

        fig.add_trace(
            go.Scatter(
                x=asset_data.index,
                y=asset_norm,
                name=f"{config['name']} (price / first price)",
                line=dict(color=config["color"], width=2),
                opacity=asset_opacity,
            ),
            secondary_y=True,
        )


def _add_lag_reference_line(fig: go.Figure, end_date: pd.Timestamp, lag_weeks: int, lag_days: int):
    """Add vertical reference line showing where lagged series aligns with today."""
    lag_reference_date = end_date - pd.Timedelta(days=lag_days)
    lag_reference_datetime = pd.Timestamp(lag_reference_date).to_pydatetime()

    fig.add_shape(
        type="line",
        x0=lag_reference_datetime,
        x1=lag_reference_datetime,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="gray", width=2, dash="dot"),
        opacity=0.7,
    )

    fig.add_annotation(
        x=lag_reference_datetime,
        y=1,
        yref="paper",
        text=f"Today - {lag_weeks}w",
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        font=dict(color="black"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
    )


def _configure_liquidity_layout(fig: go.Figure, merged_asset_config: dict):
    """Configure axes labels and layout."""
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Global Liquidity Index", secondary_y=False)

    asset_names = [merged_asset_config[t]["name"] for t in merged_asset_config.keys()]
    asset_label = ", ".join(asset_names[:5])
    if len(asset_names) > 5:
        asset_label += f", ... ({len(asset_names)} total)"
    fig.update_yaxes(title_text=f"{asset_label} (normalized price)", secondary_y=True)

    fig.update_layout(
        height=500,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )


def _plot_global_liquidity(
    df: pd.DataFrame,
    beginning_date: pd.Timestamp,
    lag_weeks: int,
    asset_config: dict,
    custom_symbols: list = None,
    lookback_weeks: int = 15,
    colors: list = None,
    show_ema: bool = True,
    ema_period: int = 25,
    show_lag: bool = True,
):
    """Create and render the Global Liquidity plot with asset overlays (normalized)."""
    # Prepare data
    df_plot, start_date, end_date = _prepare_liquidity_data(
        df, beginning_date, lookback_weeks, lag_weeks, show_ema, ema_period, show_lag
    )

    # Create figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add liquidity traces
    _add_liquidity_traces(fig, df_plot, start_date, end_date, lag_weeks, show_ema, ema_period, show_lag)

    # Merge asset configs
    custom_symbols = custom_symbols or []
    colors = colors or []
    merged_asset_config = get_asset_config_with_custom(asset_config, custom_symbols, colors)

    # Add asset overlays
    _add_asset_overlays(fig, merged_asset_config, start_date, end_date)

    # Add lag reference line
    if show_lag and lag_weeks > 0:
        lag_days = lag_weeks * 7
        _add_lag_reference_line(fig, end_date, lag_weeks, lag_days)

    # Configure layout
    _configure_liquidity_layout(fig, merged_asset_config)

    st.plotly_chart(fig, config={"displayModeBar": False})


def _add_dominance_to_subplot(
    fig,
    row: int,
    col: int,
    beginning_date: pd.Timestamp,
    ticker1: str,
    ticker2: str,
    dominance_name: str,
    yaxis_title: str,
    ticker1_label: str,
    ticker2_label: str,
    ticker1_color: str = "lightblue",
    ticker2_color: str = "lightgreen",
    fill_color_above: str = "rgba(11, 11, 255, 1)",
    fill_color_below: str = "rgba(144, 238, 144, 0.5)",
    lookback_weeks: int = 15,
):
    """Add a dominance plot to a subplot figure."""
    lookback_days = lookback_weeks * 7
    start_date = beginning_date - pd.Timedelta(days=lookback_days)
    end_date = pd.Timestamp.today()

    # Load ETF data
    tickers = [ticker1, ticker2]
    etf_data = st_get_tickers_close_prices(tickers, period="max")

    if etf_data.empty or ticker1 not in etf_data.columns or ticker2 not in etf_data.columns:
        return

    # Filter to the date range (need extra days for the 5-day lookback)
    mask = (etf_data.index >= start_date - pd.Timedelta(days=5)) & (etf_data.index <= end_date)
    etf_filtered = etf_data.loc[mask].copy()

    if etf_filtered.empty:
        return

    # Calculate 5-day ratio: price[t] / price[t-5] for each ETF
    ticker1_5day_ratio = etf_filtered[ticker1] / etf_filtered[ticker1].shift(5)
    ticker2_5day_ratio = etf_filtered[ticker2] / etf_filtered[ticker2].shift(5)

    # Calculate Dominance: (ticker1 5-day ratio) / (ticker2 5-day ratio)
    dominance_raw = ticker1_5day_ratio / ticker2_5day_ratio

    # Replace inf/nan values with NaN for cleaner processing
    dominance_raw = dominance_raw.replace([float("inf"), float("-inf")], pd.NA)

    # Apply 14-day exponential smoothing (EMA)
    dominance = dominance_raw.ewm(span=10, adjust=False).mean()

    # Filter to the actual plotting range (after we have enough data for calculations)
    dominance_plot = dominance.loc[dominance.index >= start_date].dropna()

    # Prepare normalized price data for secondary axis
    etf_plot = etf_filtered.loc[etf_filtered.index >= start_date]

    # Get first non-null values for normalization
    ticker1_first = etf_plot[ticker1].dropna().iloc[0] if not etf_plot[ticker1].dropna().empty else None
    ticker2_first = etf_plot[ticker2].dropna().iloc[0] if not etf_plot[ticker2].dropna().empty else None

    if ticker1_first is None or ticker2_first is None or ticker1_first == 0 or ticker2_first == 0:
        return

    ticker1_normalized = etf_plot[ticker1] / ticker1_first
    ticker2_normalized = etf_plot[ticker2] / ticker2_first

    # Drop NaN values for cleaner plotting
    ticker1_normalized = ticker1_normalized.dropna()
    ticker2_normalized = ticker2_normalized.dropna()

    # Add reference line at y=1 (needed for fill) - primary axis
    reference_line = pd.Series(1.0, index=dominance_plot.index)
    fig.add_trace(
        go.Scatter(
            x=reference_line.index,
            y=reference_line.values,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
        secondary_y=False,
    )

    # Add shading
    above_one = dominance_plot.copy()
    above_one[above_one < 1] = 1
    below_one = dominance_plot.copy()
    below_one[below_one > 1] = 1

    # Add fill for area above 1
    fig.add_trace(
        go.Scatter(
            x=dominance_plot.index,
            y=above_one,
            fill="tonexty",
            fillcolor=fill_color_above,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
        secondary_y=False,
    )

    # Add fill for area below 1
    fig.add_trace(
        go.Scatter(
            x=dominance_plot.index,
            y=below_one,
            fill="tonexty",
            fillcolor=fill_color_below,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
        secondary_y=False,
    )

    # Add the main line trace - primary axis
    fig.add_trace(
        go.Scatter(
            x=dominance_plot.index,
            y=dominance_plot,
            name=f"{dominance_name}",
            line=dict(color="black", width=2),
            mode="lines",
            showlegend=(row == 1 and col == 1),  # Only show legend for first plot
        ),
        row=row,
        col=col,
        secondary_y=False,
    )

    # Add normalized ticker1 price curve - secondary axis
    if not ticker1_normalized.empty:
        fig.add_trace(
            go.Scatter(
                x=ticker1_normalized.index,
                y=ticker1_normalized,
                name=f"{ticker1_label} (normalized)",
                line=dict(color=ticker1_color, width=1),
                mode="lines",
                showlegend=(row == 1 and col == 1),
            ),
            row=row,
            col=col,
            secondary_y=True,
        )

    # Add normalized ticker2 price curve - secondary axis
    if not ticker2_normalized.empty:
        fig.add_trace(
            go.Scatter(
                x=ticker2_normalized.index,
                y=ticker2_normalized,
                name=f"{ticker2_label} (normalized)",
                line=dict(color=ticker2_color, width=1),
                mode="lines",
                showlegend=(row == 1 and col == 1),
            ),
            row=row,
            col=col,
            secondary_y=True,
        )

    # Add horizontal line at y=1
    fig.add_hline(
        y=1.0,
        line_color="gray",
        opacity=0.5,
        row=row,
        col=col,
    )

    # Update axes for this subplot
    fig.update_xaxes(title_text="Date", row=row, col=col)
    fig.update_yaxes(title_text=yaxis_title, row=row, col=col, secondary_y=False)
    fig.update_yaxes(title_text="Normalized Price", row=row, col=col, secondary_y=True)


def _calculate_price_pct_change_for_period(ticker: str, days_back: int) -> float:
    """Calculate percent change in price over a specific number of days."""
    end_date = pd.Timestamp.today()

    # Load price data with max period to ensure we have enough history (at least 1 year)
    price_data = st_get_tickers_close_prices([ticker], period="max")
    if price_data.empty or ticker not in price_data.columns:
        return 0.0

    # Filter to get data up to end_date
    mask = price_data.index <= end_date
    price_filtered = price_data.loc[mask].copy()

    if price_filtered.empty:
        return 0.0

    # For 1 day change, use last 2 prices
    if days_back <= 1:
        if len(price_filtered) < 2:
            return 0.0

        latest = price_filtered[ticker].iloc[-1]
        previous = price_filtered[ticker].iloc[-2]

        if pd.isna(latest) or pd.isna(previous) or previous == 0:
            return 0.0

        pct_change = ((latest - previous) / previous) * 100
        return pct_change

    # For longer periods, find price from days_back ago
    if len(price_filtered) < days_back + 1:
        return 0.0

    latest = price_filtered[ticker].iloc[-1]
    target_date = end_date - pd.Timedelta(days=days_back)
    # Sort the index first to ensure proper chronological order
    sorted_prices = price_filtered.sort_index()
    past_prices = sorted_prices[sorted_prices.index <= target_date]

    if past_prices.empty:
        return 0.0

    previous = past_prices[ticker].iloc[-1]

    if pd.isna(latest) or pd.isna(previous) or previous == 0:
        return 0.0

    pct_change = ((latest - previous) / previous) * 100
    return pct_change


def _plot_dominance_pie_charts(dominance_plots: list):
    """Plot multiple pie charts showing price percent changes for different time periods."""
    time_periods = [
        ("1 Day", 1),
        ("1 Week", 7),
        ("1 Month", 30),
        ("3 Months", 90),
        ("1 Year", 365),
    ]

    # Define the assets to track with their display names and colors
    assets = {
        "VTI": {"name": "US", "color": "lightblue"},
        "CNYA": {"name": "China", "color": "coral"},
        "SPEU": {"name": "EU", "color": "royalblue"},
        "BTC-USD": {"name": "Bitcoin", "color": "purple"},
        "GC=F": {"name": "Gold", "color": "gold"},
    }

    # Create Streamlit columns for the pie charts
    cols = st.columns(5)

    for col_idx, (period_name, days_back) in enumerate(time_periods):
        with cols[col_idx]:
            st.markdown(f"**{period_name} Change**")

            labels = []
            values = []
            colors_list = []

            for ticker, info in assets.items():
                pct_change = _calculate_price_pct_change_for_period(ticker, days_back)

                # Include all non-zero values
                if pct_change > 0.001 and not pd.isna(pct_change):
                    labels.append(info["name"])
                    values.append(abs(pct_change))
                    colors_list.append(info["color"])

            # Create and display pie chart if we have data
            if values and sum(values) > 0:
                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=labels,
                            values=values,
                            marker=dict(colors=colors_list),
                            textinfo="label+percent",
                            hole=0.3,  # Donut chart
                            showlegend=False,
                            textposition="outside",
                        )
                    ]
                )

                fig.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                )

                st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)
            else:
                st.info("No data available")


def _plot_all_dominance_subplots(beginning_date: pd.Timestamp, lookback_weeks: int = 15, dominance_plots: list = None):
    """Create a 3x2 subplot grid with all dominance plots."""
    if dominance_plots is None:
        dominance_plots = []

    # Build subplot titles in row-major order (row 1 col 1, row 1 col 2, row 2 col 1, etc.)
    # Sort plots by row first, then by col
    sorted_plots = sorted(dominance_plots, key=lambda p: (p["row"], p["col"]))
    subplot_titles = [plot["title"] for plot in sorted_plots]

    # Create subplots: 3 rows, 2 columns, each with secondary y-axis
    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": True}, {"secondary_y": True}],
        ],
        subplot_titles=tuple(subplot_titles),
        vertical_spacing=0.08,
        horizontal_spacing=0.1,
    )

    # Add all dominance plots
    for plot in dominance_plots:
        _add_dominance_to_subplot(
            fig,
            row=plot["row"],
            col=plot["col"],
            beginning_date=beginning_date,
            ticker1=plot["ticker1"],
            ticker2=plot["ticker2"],
            dominance_name=plot["dominance_name"],
            yaxis_title="Ratio",
            ticker1_label=plot["ticker1_label"],
            ticker2_label=plot["ticker2_label"],
            ticker1_color=plot["ticker1_color"],
            ticker2_color=plot["ticker2_color"],
            fill_color_above=plot.get("fill_color_above", "rgba(11, 11, 255, 1)"),
            fill_color_below=plot.get("fill_color_below", "rgba(144, 238, 144, 0.5)"),
            lookback_weeks=lookback_weeks,
        )

    # Update layout
    fig.update_layout(
        height=1200,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5),
    )

    st.plotly_chart(fig, config={"displayModeBar": False})


def main():
    """Main entry point for the macro / Global Liquidity page."""
    register_resolvers()

    # Load Hydra config so we can reuse global dashboard styling
    with hydra.initialize(version_base=None, config_path="../../conf"):
        config = hydra.compose(config_name="main")
    beginning_date, lag_weeks, custom_symbols, show_ema, ema_period, show_lag = setup_page_and_sidebar(
        config["style_conf"],
        add_to_sidebar=lambda: _add_macro_sidebar(config["macro"]),
    )

    # Load asset configuration and lookback_weeks from YAML
    macro_config = config.get("macro", {})

    # Convert OmegaConf DictConfig to regular Python dict to avoid struct mode issues
    assets_raw = macro_config.get("assets", {})
    asset_config = OmegaConf.to_container(assets_raw, resolve=True) if assets_raw else {}
    lookback_weeks = macro_config.get("lookback_weeks", 15)

    # Load colors from style_conf
    style_conf = config.get("style_conf", {})
    colors = list(style_conf.get("colors", []))

    # Get Fear and Greed data
    fear_greed = FearGreed()
    traditional_data = fear_greed.get_fear_and_greed()
    crypto_data = fear_greed.get_crypto_fear_and_greed()

    stock_value = traditional_data["value"]
    stock_desc = traditional_data["description"]
    crypto_value = crypto_data["value"] if crypto_data["value"] is not None else "N/A"
    crypto_desc = crypto_data["description"] if crypto_data["value"] is not None else "Error"

    # Calculate colors for both metrics
    normalized_stock = stock_value / 100.0
    red_stock = int(255 * (1 - normalized_stock))
    green_stock = int(255 * normalized_stock)
    color_hex_stock = f"#{red_stock:02x}{green_stock:02x}{50:02x}"

    if crypto_data["value"] is not None:
        normalized_crypto = crypto_data["value"] / 100.0
        red_crypto = int(255 * (1 - normalized_crypto))
        green_crypto = int(255 * normalized_crypto)
        color_hex_crypto = f"#{red_crypto:02x}{green_crypto:02x}{50:02x}"
    else:
        color_hex_crypto = "#808080"

    # Add TradingView Economic Calendar and Fear & Greed Index on the same line
    st.markdown(
        f"""
        <div style='text-align: center; margin: 10px 0; padding: 10px; background-color: #f0f2f6; border-radius: 5px; display: flex; justify-content: center; align-items: center; gap: 20px; flex-wrap: wrap;'>
            <a href='https://www.tradingview.com/economic-calendar/' target='_blank' style='text-decoration: none; color: #1f77b4; font-weight: bold; font-size: 16px;'>
                📅  Economic Calendar
            </a>
            <span style='color: #666;'>|</span>
            <a href="https://www.cnn.com/markets/fear-and-greed" target="_blank" style="color:{color_hex_stock};text-decoration:none; font-weight: bold; font-size: 16px;">{stock_value} {stock_desc} (S&P 500)</a>
            <span style='color: #666;'>|</span>
            <a href="https://alternative.me/crypto/fear-and-greed-index/" target="_blank" style="color:{color_hex_crypto};text-decoration:none; font-weight: bold; font-size: 16px;">{crypto_value} {crypto_desc} (Crypto)</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Market Strength Indicator (US Market RSI)
    st.title("Market Strength Indicator")
    st.markdown("RSI (Relative Strength Index) for US and International markets")

    # Load RSI configuration from YAML
    rsi_config = config.get("macro", {}).get("rsi", {})

    # Use global smoothing span from sidebar (fallback to default if missing)
    smoothing_span = st.session_state.get("smoothing_span", 5)

    with st.spinner("Loading market RSI data..."):
        # Extract ticker configuration (support both dict and list formats for backwards compatibility)
        tickers_config = rsi_config.get("tickers", {})
        if isinstance(tickers_config, dict):
            ticker_list = list(tickers_config.keys())
        else:
            # Fallback to list format for backwards compatibility
            ticker_list = tickers_config if isinstance(tickers_config, list) else ["VTI", "VXUS", "IBIT", "IAUM"]
        
        # Ensure TLT and XLU are included for the plots
        all_tickers = list(set(ticker_list + ["VTI", "TLT", "IAUM", "IBIT", "XLU"]))
        
        ticker = TICKERS(
            all_tickers,
            period=rsi_config.get("period", "360d"),
            normalize=rsi_config.get("normalize", True)
        )
        ticker.get_rsi()
        rsi_df = ticker.rsis[rsi_config.get("rsi_period", 14)]
        rsi_df_original = rsi_df.copy()
        # Apply optional EMA smoothing to RSI
        if smoothing_span > 1:
            rsi_df = rsi_df.ewm(span=smoothing_span, adjust=False).mean()

        # Create subplot figure: 3 rows, 2 columns
        # Row 1: All RSI lines (spans both columns)
        # Row 2, Col 1: VTI and TLT
        # Row 2, Col 2: IAUM and IBIT
        # Row 3, Col 1: VTI vs IAUM
        # Row 3, Col 2: VTI vs XLU
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                rsi_config.get("title", "Market Average RSI"),
                "VTI vs TLT RSI",
                "IAUM vs IBIT RSI",
                "VTI vs IAUM RSI",
                "VTI vs XLU RSI"
            ),
            specs=[
                [{"colspan": 2}, None],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
        )

        # Plot individual RSIs in first row (all tickers)
        # Store color and line_style for each ticker to reuse in second row
        ticker_colors = {}
        ticker_line_styles = {}
        default_visible_tickers = ["VTI", "VXUS"]  # Only VTI and VXUS visible by default
        default_colors = ["#1f77b4", "#2ca02c", "purple", "goldenrod"]
        default_line_styles = ["solid", "dash", "dot", "dashdot"]
        
        for idx, col in enumerate(rsi_df.columns):
            is_visible = col in default_visible_tickers
            
            # Get color and line_style from ticker config, with fallbacks
            if isinstance(tickers_config, dict) and col in tickers_config:
                ticker_config = tickers_config[col]
                color = ticker_config.get("color", default_colors[idx % len(default_colors)])
                line_style = ticker_config.get("line_style", default_line_styles[idx % len(default_line_styles)])
            else:
                # Fallback for backwards compatibility
                color = default_colors[idx % len(default_colors)]
                line_style = default_line_styles[idx % len(default_line_styles)]
            
            # Store color and line_style for reuse
            ticker_colors[col] = color
            ticker_line_styles[col] = line_style
            
            fig.add_trace(
                go.Scatter(
                    x=rsi_df.index,
                    y=rsi_df[col],
                    mode="lines",
                    name=f"{col} RSI",
                    line=dict(
                        color=color, 
                        width=rsi_config.get("line_width", 2), 
                        dash=line_style
                    ),
                    opacity=rsi_config.get("opacity", 0.7),
                    visible="legendonly" if not is_visible else True,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        # Add VTI and TLT to second row, first column
        for ticker_symbol in ["VTI", "TLT"]:
            if ticker_symbol in rsi_df.columns:
                # Use the same color and line_style as in first row
                color = ticker_colors.get(ticker_symbol, default_colors[0])
                line_style = ticker_line_styles.get(ticker_symbol, default_line_styles[0])
                fig.add_trace(
                    go.Scatter(
                        x=rsi_df.index,
                        y=rsi_df[ticker_symbol],
                        mode="lines",
                        name=f"{ticker_symbol} RSI",
                        line=dict(
                            color=color,
                            width=rsi_config.get("line_width", 2),
                            dash=line_style,
                        ),
                        opacity=rsi_config.get("opacity", 0.7),
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

        # Add IAUM and IBIT to second row, second column
        for ticker_symbol in ["IAUM", "IBIT"]:
            if ticker_symbol in rsi_df.columns:
                # Use the same color and line_style as in first row
                color = ticker_colors.get(ticker_symbol, default_colors[0])
                line_style = ticker_line_styles.get(ticker_symbol, default_line_styles[0])
                fig.add_trace(
                    go.Scatter(
                        x=rsi_df.index,
                        y=rsi_df[ticker_symbol],
                        mode="lines",
                        name=f"{ticker_symbol} RSI",
                        line=dict(
                            color=color,
                            width=rsi_config.get("line_width", 2),
                            dash=line_style,
                        ),
                        opacity=rsi_config.get("opacity", 0.7),
                        showlegend=False,
                    ),
                    row=2,
                    col=2,
                )

        # Add VTI vs IAUM to third row, first column
        for ticker_symbol in ["VTI", "IAUM"]:
            if ticker_symbol in rsi_df.columns:
                # Use the same color and line_style as in first row
                color = ticker_colors.get(ticker_symbol, default_colors[0])
                line_style = ticker_line_styles.get(ticker_symbol, default_line_styles[0])
                fig.add_trace(
                    go.Scatter(
                        x=rsi_df.index,
                        y=rsi_df[ticker_symbol],
                        mode="lines",
                        name=f"{ticker_symbol} RSI",
                        line=dict(
                            color=color,
                            width=rsi_config.get("line_width", 2),
                            dash=line_style,
                        ),
                        opacity=rsi_config.get("opacity", 0.7),
                        showlegend=False,
                    ),
                    row=3,
                    col=1,
                )

        # Add VTI vs XLU to third row, second column
        for ticker_symbol in ["VTI", "XLU"]:
            if ticker_symbol in rsi_df.columns:
                # Use the same color and line_style as in first row
                color = ticker_colors.get(ticker_symbol, default_colors[0])
                line_style = ticker_line_styles.get(ticker_symbol, default_line_styles[0])
                fig.add_trace(
                    go.Scatter(
                        x=rsi_df.index,
                        y=rsi_df[ticker_symbol],
                        mode="lines",
                        name=f"{ticker_symbol} RSI",
                        line=dict(
                            color=color,
                            width=rsi_config.get("line_width", 2),
                            dash=line_style,
                        ),
                        opacity=rsi_config.get("opacity", 0.7),
                        showlegend=False,
                    ),
                    row=3,
                    col=2,
                )

        # Add reference lines from config to all subplots
        for ref_line in rsi_config.get("reference_lines", []):
            # Add to first row
            fig.add_hline(
                y=ref_line.get("y"),
                line_dash=ref_line.get("line_dash", "dot"),
                line_color=ref_line.get("line_color", "gray"),
                opacity=ref_line.get("opacity", 0.5),
                annotation_text=ref_line.get("annotation_text", ""),
                annotation_font_color=ref_line.get("annotation_font_color", "black"),
                row=1,
                col=1,
            )
            # Add to second row, first column
            fig.add_hline(
                y=ref_line.get("y"),
                line_dash=ref_line.get("line_dash", "dot"),
                line_color=ref_line.get("line_color", "gray"),
                opacity=ref_line.get("opacity", 0.5),
                row=2,
                col=1,
            )
            # Add to second row, second column
            fig.add_hline(
                y=ref_line.get("y"),
                line_dash=ref_line.get("line_dash", "dot"),
                line_color=ref_line.get("line_color", "gray"),
                opacity=ref_line.get("opacity", 0.5),
                row=2,
                col=2,
            )
            # Add to third row, first column
            fig.add_hline(
                y=ref_line.get("y"),
                line_dash=ref_line.get("line_dash", "dot"),
                line_color=ref_line.get("line_color", "gray"),
                opacity=ref_line.get("opacity", 0.5),
                row=3,
                col=1,
            )
            # Add to third row, second column
            fig.add_hline(
                y=ref_line.get("y"),
                line_dash=ref_line.get("line_dash", "dot"),
                line_color=ref_line.get("line_color", "gray"),
                opacity=ref_line.get("opacity", 0.5),
                row=3,
                col=2,
            )

        # Update layout
        fig.update_layout(
            height=rsi_config.get("height", 2500),
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=1, xanchor="center", x=0.5),
            plot_bgcolor="white",
        )
        
        # Move subplot titles slightly higher
        for annotation in fig.layout.annotations:
            if annotation.text:  # Only update subplot title annotations
                annotation.y += 0.01  # Move titles up by 2% of the plot height
        
        # Update axes labels and remove default grid for all subplots
        fig.update_xaxes(title_text="Date", showgrid=False, row=1, col=1)
        fig.update_xaxes(title_text="Date", showgrid=False, row=2, col=1)
        fig.update_xaxes(title_text="Date", showgrid=False, row=2, col=2)
        fig.update_xaxes(title_text="Date", showgrid=False, row=3, col=1)
        fig.update_xaxes(title_text="Date", showgrid=False, row=3, col=2)
        fig.update_yaxes(title_text=rsi_config.get("yaxis_title", "RSI"), showgrid=False, row=1, col=1)
        fig.update_yaxes(title_text=rsi_config.get("yaxis_title", "RSI"), showgrid=False, row=2, col=1)
        fig.update_yaxes(title_text=rsi_config.get("yaxis_title", "RSI"), showgrid=False, row=2, col=2)
        fig.update_yaxes(title_text=rsi_config.get("yaxis_title", "RSI"), showgrid=False, row=3, col=1)
        fig.update_yaxes(title_text=rsi_config.get("yaxis_title", "RSI"), showgrid=False, row=3, col=2)

        st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)

        # Create pie chart with latest RSI values
        if not rsi_df_original.empty:
            latest_rsi = rsi_df_original.iloc[-1].dropna()
            
            # Create pie chart
            pie_fig = go.Figure(data=[
                go.Pie(
                    labels=[f"{ticker} ({value:.1f})" for ticker, value in latest_rsi.items()],
                    values=latest_rsi.values,
                    marker=dict(
                        colors=[ticker_colors.get(ticker, default_colors[0]) for ticker in latest_rsi.index]
                    ),
                    textinfo="label+percent",
                    hole=0.3,  # Donut chart
                    showlegend=False,
                )
            ])
            
            pie_fig.update_layout(
                title="Latest RSI Values Distribution",
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            
            st.plotly_chart(pie_fig, config={"displayModeBar": False}, use_container_width=True)

    st.markdown("---")

    st.title("GLI : Global Liquidity Index")

    st.markdown(
        f"Visualize the **Global Liquidity Index** from a selected time period (minus {lookback_weeks} weeks) "
        "to **now**. Includes a forward-shifted (lagged) curve to explore potential lead/lag "
        "behavior, and overlays of asset prices normalized to 1 at the start of the period."
    )

    df = st_load_global_liquidity()
    _plot_global_liquidity(
        df,
        beginning_date=beginning_date,
        lag_weeks=lag_weeks,
        asset_config=asset_config,
        custom_symbols=custom_symbols,
        lookback_weeks=lookback_weeks,
        colors=colors,
        show_ema=show_ema,
        ema_period=ema_period,
        show_lag=show_lag,
    )

    st.markdown("---")
    st.title("Dominance Shifts")
    st.markdown("Percent change in price over different time periods. ")

    # Load dominance plots configuration (needed for pie charts)
    dominance_plots_list = macro_config.get("dominance_plots", {})
    dominance_plots = OmegaConf.to_container(dominance_plots_list, resolve=True) if dominance_plots_list else []

    _plot_dominance_pie_charts(dominance_plots)

    st.markdown("---")
    st.markdown(
        "**Region Dominance** metrics are calculated using daily data: the ratio of a regional ETF's 5-day price ratio "
        "to a benchmark ETF's 5-day price ratio, smoothed with a 14-day exponential moving average. "
        "A ratio > 1 indicates the regional market is outperforming the benchmark. "
        "Each plot shows the dominance ratio (primary axis) and normalized price curves (secondary axis)."
    )

    _plot_all_dominance_subplots(beginning_date=beginning_date, lookback_weeks=lookback_weeks, dominance_plots=dominance_plots)


if __name__ == "__main__":
    # Handle Hydra initialization - clear if already initialized
    if hydra.core.global_hydra.GlobalHydra().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    main()
