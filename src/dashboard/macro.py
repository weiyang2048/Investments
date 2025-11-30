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
import re
from matplotlib.colors import to_rgb

import hydra
from omegaconf import OmegaConf

from src.configurations.yaml import register_resolvers
from src.dashboard.create_page import setup_page_and_sidebar

from src.data.GLI import st_load_global_liquidity
from src.data.FearGreed import FearGreed
from src.data.TICKER import TICKERS


def _is_light_color(color: str) -> bool:
    """Determine if a color is light (returns True) or dark (returns False).
    
    Converts color to RGB and calculates luminance to determine if text should be dark or light.
    """
    try:
        rgb = to_rgb(color)  # Returns tuple of floats in 0-1 range
        # Calculate relative luminance (perceived brightness)
        # Using standard formula: 0.299*R + 0.587*G + 0.114*B
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return luminance > 0.5  # Light if luminance > 0.5
    except (ValueError, TypeError):
        return True  # Default to light (use dark text)


def _color_to_rgba(color: str, opacity: float = 0.5) -> str:
    """Convert color name or hex to rgba format with opacity.

    Uses matplotlib.colors.to_rgb() to handle color names and hex codes.
    Supports:
    - Hex colors: "#1f77b4", "1f77b4", "#f00", "f00"
    - Color names: "red", "blue", "seagreen", etc.
    - RGB strings: "rgb(255, 0, 0)"
    - RGBA strings: "rgba(255, 0, 0, 0.5)" (returns as-is)
    """
    # Check if it's already rgba
    if color.startswith("rgba("):
        return color

    # Check if it's rgb - extract values and add opacity
    rgb_match = re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", color)
    if rgb_match:
        r, g, b = rgb_match.groups()
        return f"rgba({r}, {g}, {b}, {opacity})"

    # Use matplotlib to convert color name or hex to RGB (returns 0-1 range)
    # Handles: "#1f77b4", "1f77b4", "#f00", "red", "seagreen", etc.
    try:
        rgb = to_rgb(color)  # Returns tuple of floats in 0-1 range
        r = int(rgb[0] * 255)
        g = int(rgb[1] * 255)
        b = int(rgb[2] * 255)
        return f"rgba({r}, {g}, {b}, {opacity})"
    except (ValueError, TypeError):
        # Fallback if color cannot be parsed (invalid hex, unknown color name, etc.)
        return f"rgba(128, 128, 128, {opacity})"


def _add_vs_pair_plot(
    fig: go.Figure,
    rsi_df: pd.DataFrame,
    challenger_config: dict,
    champion: str,
    ticker_colors: dict,
    rsi_config: dict,
    default_colors: list,
    price_df: pd.DataFrame = None,
):
    """Add a challenger vs champion plot (two tickers with difference curve) to the figure."""
    row = challenger_config.get("row")
    col = challenger_config.get("col")
    ticker1 = challenger_config.get("ticker")  # Challenger
    ticker2 = champion  # Champion is always the same

    if not ticker1 or not ticker2 or ticker1 not in rsi_df.columns or ticker2 not in rsi_df.columns:
        return

    # Get latest RSI values to determine annotation positions
    if not rsi_df.empty:
        latest_rsi1 = rsi_df[ticker1].iloc[-1] if ticker1 in rsi_df.columns else 50
        latest_rsi2 = rsi_df[ticker2].iloc[-1] if ticker2 in rsi_df.columns else 50
        # Determine which ticker has higher RSI
        higher_ticker = ticker1 if latest_rsi1 >= latest_rsi2 else ticker2
        lower_ticker = ticker2 if higher_ticker == ticker1 else ticker1
    else:
        higher_ticker = ticker1
        lower_ticker = ticker2

    # Calculate y-axis range for positioning annotations
    if not rsi_df.empty:
        rsi_min = min(rsi_df[ticker1].min(), rsi_df[ticker2].min()) if ticker1 in rsi_df.columns and ticker2 in rsi_df.columns else 0
        rsi_max = max(rsi_df[ticker1].max(), rsi_df[ticker2].max()) if ticker1 in rsi_df.columns and ticker2 in rsi_df.columns else 100
        rsi_range = rsi_max - rsi_min
        y_above = rsi_min + rsi_range * 0.75  # Position above middle
        y_below = rsi_min + rsi_range * 0.25  # Position below middle
    else:
        y_above = 75
        y_below = 25

    # Add individual RSI lines (always shown)
    for ticker_symbol in [ticker1, ticker2]:
        color = ticker_colors.get(ticker_symbol, default_colors[0])
        line_opacity = 0.5
        fig.add_trace(
            go.Scatter(
                x=rsi_df.index,
                y=rsi_df[ticker_symbol],
                mode="lines",
                name=f"{ticker_symbol} RSI",
                line=dict(
                    color=color,
                    width=rsi_config.get("line_width", 2),
                    dash="solid",
                ),
                opacity=line_opacity,
                showlegend=False,
                hovertemplate=f"{ticker_symbol}<br>RSI: %{{y:.1f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )

        # Add text annotation at the end of the line showing the ticker symbol
        # Position higher RSI ticker above middle, lower RSI ticker below middle
        if not rsi_df[ticker_symbol].empty:
            last_date = rsi_df.index[-1]
            # Use y_above for higher ticker, y_below for lower ticker
            annotation_y = y_above if ticker_symbol == higher_ticker else y_below
            fig.add_annotation(
                x=last_date,
                y=annotation_y,
                text=ticker_symbol,
                showarrow=False,
                xanchor="left",
                xshift=5,
                font=dict(color=color, size=14, family="Arial Black"),
                row=row,
                col=col,
            )

    # Add price information on primary axis with dotted lines
    if price_df is not None and not price_df.empty:
        for ticker_symbol in [ticker1, ticker2]:
            if ticker_symbol in price_df.columns:
                color = ticker_colors.get(ticker_symbol, default_colors[0])
                price_series = price_df[ticker_symbol].dropna()

                # Align price data with RSI data index
                price_aligned = price_series.reindex(rsi_df.index, method="ffill").dropna()
                price_normalized = price_aligned * 30 - 25

                fig.add_trace(
                    go.Scatter(
                        x=price_aligned.index,
                        y=price_normalized,
                        mode="lines",
                        name=f"{ticker_symbol} Price",
                        line=dict(
                            color=color,
                            width=2,
                        ),
                        opacity=0.7,
                        showlegend=False,
                        hovertemplate=f"{ticker_symbol} Price<br>%{{y:.1f}} (normalized)<extra></extra>",
                    ),
                    row=row,
                    col=col,
                    secondary_y=False,
                )

    # Calculate difference
    diff = rsi_df[ticker1] - rsi_df[ticker2]
    ticker1_color = ticker_colors.get(ticker1, default_colors[0])
    ticker2_color = ticker_colors.get(ticker2, default_colors[0])

    # Create positive and negative parts for fill
    diff_positive = diff.copy()
    diff_positive[diff_positive < 0] = 0
    diff_negative = diff.copy()
    diff_negative[diff_negative > 0] = 0

    # Add fill for positive area (ticker1 > ticker2) - use ticker1 color
    fig.add_trace(
        go.Scatter(
            x=diff.index,
            y=diff_positive,
            fill="tozeroy",
            fillcolor=_color_to_rgba(ticker1_color),
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
        secondary_y=True,
    )

    # Add fill for negative area (ticker1 < ticker2) - use ticker2 color
    fig.add_trace(
        go.Scatter(
            x=diff.index,
            y=diff_negative,
            fill="tozeroy",
            fillcolor=_color_to_rgba(ticker2_color),
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
        secondary_y=True,
    )

    # Add the difference curve line
    diff_name = challenger_config.get("diff_name", f"{ticker1} - {ticker2}")
    fig.add_trace(
        go.Scatter(
            x=rsi_df.index,
            y=diff,
            mode="lines",
            name=diff_name,
            line=dict(
                color="black",
                width=1,
                dash="solid",
            ),
            opacity=1.0,
            showlegend=False,
        ),
        row=row,
        col=col,
        secondary_y=True,
    )


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


def _add_macro_sidebar(macro_config: dict, rsi_config: dict):
    """Sidebar controls for the macro / liquidity view."""
    st.sidebar.header("Accelerators")
    accelerators_top_n = st.sidebar.number_input(
        "Top N Performers",
        min_value=1,
        max_value=20,
        value=rsi_config.get("accelerators_top_n", 7),
        step=1,
        help="Number of top performers to consider for 1 day and 1 week changes.",
        key="accelerators_top_n_input",
    )

    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.header("RSI Settings")
    use_alternative_tickers = st.sidebar.checkbox(
        "Use Alternative Tickers",
        value=rsi_config.get("use_alternative_tickers", False),
        help="When enabled, use the second ticker in lists (e.g., [TOPT, VOO] uses VOO). Otherwise uses the first ticker.",
        key="use_alternative_tickers_input",
    )

    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.header("GLI Settings")

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

    # Always show lag
    show_lag = True
    lag_weeks = st.sidebar.slider(
        "Lag / Look-ahead (weeks)",
        min_value=0,
        max_value=52,
        value=lookback_weeks,
        step=1,
        help="Shift the index forward in time by this many weeks to visualize a look-ahead effect.",
    )

    # Single global smoothing control, used for GLI EMA and overlays
    smoothing_span = st.sidebar.slider(
        "Smoothing window (EMA span)",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        help="Apply an EMA smoothing window to the Global Liquidity EMA and overlay series (1 = no smoothing, higher = smoother).",
        key="smoothing_span",
    )
    # Always show EMA; reuse smoothing_span as EMA period
    ema_period = int(smoothing_span)

    st.sidebar.markdown("**Custom Symbols**")
    custom_symbols_text = st.sidebar.text_input(
        "Add Custom Symbols",
        placeholder="Enter symbols separated by commas (e.g., AAPL, MSFT, GOOGL)",
        help="Enter custom ticker symbols separated by commas to overlay on the chart. Colors will be assigned automatically.",
        key="custom_symbols_input",
    )
    custom_symbols = parse_custom_symbols(custom_symbols_text)

    return beginning_date, int(lag_weeks), custom_symbols, int(ema_period), show_lag, int(accelerators_top_n), use_alternative_tickers


def _prepare_liquidity_data(
    df: pd.DataFrame,
    beginning_date: pd.Timestamp,
    lookback_weeks: int,
    lag_weeks: int,
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
        df_extended[f"Liquidity Index (EMA {ema_period})"] = df_extended["Liquidity Index"].ewm(span=ema_period, adjust=False).mean()

    if show_lag and lag_days > 0:
        df_extended["Liquidity Index (Lag)"] = df_extended["Liquidity Index"].shift(lag_days)
        df_extended[f"Liquidity Index (Lag EMA {ema_period})"] = df_extended["Liquidity Index (Lag)"].ewm(span=ema_period, adjust=False).mean()

    df_plot = df_extended.loc[df_extended.index <= end_date]
    return df_plot, start_date, end_date


def _add_liquidity_traces(
    fig: go.Figure,
    df_plot: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    lag_weeks: int,
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

    if f"Liquidity Index (EMA {ema_period})" in df_plot.columns:
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

        if f"Liquidity Index (Lag EMA {ema_period})" in df_plot.columns:
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

    # Use TICKERS class to get price data
    ticker_obj = TICKERS(tickers, period="max", normalize=False)
    assets_hist = ticker_obj.prices

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
    ema_period: int = 25,
    show_lag: bool = True,
):
    """Create and render the Global Liquidity plot with asset overlays (normalized)."""
    # Prepare data
    df_plot, start_date, end_date = _prepare_liquidity_data(df, beginning_date, lookback_weeks, lag_weeks, ema_period, show_lag)

    # Create figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add liquidity traces
    _add_liquidity_traces(fig, df_plot, start_date, end_date, lag_weeks, ema_period, show_lag)

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


def calculate_price_pct_change(price_data: pd.DataFrame, ticker: str, days_back: int) -> float:
    """Calculate percent change in price over a specific number of days using provided price data."""
    if price_data.empty or ticker not in price_data.columns:
        return 0.0

    end_date = pd.Timestamp.today()

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
    latest = price_filtered[ticker].iloc[-1] if len(price_filtered) > 0 else None
    if latest is None or pd.isna(latest):
        return 0.0

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


def _plot_dominance_pie_charts(
    price_data: pd.DataFrame,
    ticker_colors: dict,
    tickers_config: dict,
    ticker_to_display: dict = None,
    pct_change_1d: dict = None,
    pct_change_1w: dict = None,
):
    """Plot multiple pie charts showing price percent changes for different time periods.

    Args:
        price_data: DataFrame with price data
        ticker_colors: Dictionary mapping tickers to colors
        tickers_config: Configuration dictionary for tickers
        ticker_to_display: Dictionary mapping ticker symbols to display names
        pct_change_1d: Optional pre-calculated 1-day percent changes (dict: ticker -> value)
        pct_change_1w: Optional pre-calculated 1-week percent changes (dict: ticker -> value)
    """
    time_periods = [
        ("1 Day", 1),
        ("1 Week", 7),
        ("1 Month", 30),
        ("3 Months", 90),
        ("1 Year", 365),
    ]

    # Get tickers from price_data and build asset info from ticker config
    available_tickers = list(price_data.columns)
    assets = {}
    default_colors = ["#1f77b4", "#2ca02c", "purple", "goldenrod"]

    # Create reverse mapping if not provided
    if ticker_to_display is None:
        ticker_to_display = {}
        if isinstance(tickers_config, dict):
            for display_name, ticker_value in tickers_config.items():
                # Support both formats: simple string or dict with ticker key
                if isinstance(ticker_value, dict) and "ticker" in ticker_value:
                    actual_ticker = ticker_value.get("ticker")
                elif isinstance(ticker_value, str):
                    actual_ticker = ticker_value
                else:
                    actual_ticker = display_name
                ticker_to_display[actual_ticker] = display_name

    for idx, ticker in enumerate(available_tickers):
        # Get display name for this ticker (for config lookup)
        display_name = ticker_to_display.get(ticker, ticker)

        # Get name from ticker config if available
        if isinstance(tickers_config, dict) and display_name in tickers_config:
            ticker_value = tickers_config[display_name]
            # Support both formats: simple string or dict with name key
            if isinstance(ticker_value, dict):
                name = ticker_value.get("name", display_name)
            else:
                name = display_name
        else:
            name = display_name
        
        # Get color from ticker_colors dict (which is populated from main ticker configs)
        color = ticker_colors.get(ticker, default_colors[idx % len(default_colors)])

        assets[ticker] = {"name": name, "color": color}

    # Create Streamlit columns for the pie charts
    cols = st.columns(5)

    for col_idx, (period_name, days_back) in enumerate(time_periods):
        with cols[col_idx]:
            st.markdown(f"**{period_name}**")

            labels = []
            values = []
            colors_list = []
            text_labels = []
            hover_templates = []

            for ticker, info in assets.items():
                # Use pre-calculated values for 1d and 1w if available, otherwise calculate
                if days_back == 1 and pct_change_1d is not None and ticker in pct_change_1d:
                    pct_change = pct_change_1d[ticker]
                elif days_back == 7 and pct_change_1w is not None and ticker in pct_change_1w:
                    pct_change = pct_change_1w[ticker]
                else:
                    pct_change = calculate_price_pct_change(price_data, ticker, days_back)

                if pct_change > 0.01 and not pd.isna(pct_change):
                    labels.append(info["name"])
                    values.append(abs(pct_change))
                    colors_list.append(info["color"])
                    # Show actual percent change value
                    text_labels.append(f"{info['name']}<br>{pct_change:.1f}%")
                    # Custom hover template: Name, Ticker, Percentage
                    hover_templates.append(f"<b>{info['name']}</b><br>Ticker: {ticker}<br>Change: {pct_change:.1f}%<extra></extra>")

            # Create and display pie chart if we have data
            if values and sum(values) > 0:
                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=labels,
                            values=values,
                            marker=dict(colors=colors_list),
                            text=text_labels,
                            textinfo="text",
                            textfont=dict(size=14, family="Arial Black"),
                            showlegend=False,
                            textposition="inside",
                            hovertemplate="%{customdata}",
                            customdata=hover_templates,
                        )
                    ]
                )

                fig.update_layout(
                    height=350,
                    margin=dict(l=0, r=0, t=0, b=0),
                )
                st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)
            else:
                st.info("No data available")


def main():
    """Main entry point for the macro / Global Liquidity page."""
    register_resolvers()

    # Load Hydra config so we can reuse global dashboard styling
    with hydra.initialize(version_base=None, config_path="../../conf"):
        config = hydra.compose(config_name="main")

    # Load RSI configuration for sidebar
    rsi_config = config.get("macro", {}).get("rsi", {})

    beginning_date, lag_weeks, custom_symbols, ema_period, show_lag, accelerators_top_n, use_alternative_tickers = setup_page_and_sidebar(
        config["style_conf"],
        add_to_sidebar=lambda: _add_macro_sidebar(config["macro"], rsi_config),
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

    # Markets Strength (US Market RSI)
    with st.spinner("Loading market RSI data..."):
        # Extract ticker configuration (support both dict and list formats for backwards compatibility)
        tickers_config = rsi_config.get("tickers", {})

        # Create mapping from display name to actual ticker symbol
        # Also extract actual ticker symbols for data fetching
        display_to_ticker = {}  # Maps display name (e.g., "US") to actual ticker (e.g., "VTI")
        ticker_to_display = {}  # Maps actual ticker (e.g., "VTI") to display name (e.g., "US")
        actual_ticker_list = []

        # Use alternative tickers from sidebar (passed as parameter)
        if isinstance(tickers_config, dict):
            for display_name, ticker_value in tickers_config.items():
                # Support multiple formats:
                # 1. Simple format: US: TOPT (ticker_value is a string)
                # 2. List format: US: [TOPT, VOO] (ticker_value is a list)
                # 3. Dict format: US: {ticker: TOPT} (ticker_value is a dict)
                
                if isinstance(ticker_value, list):
                    # If list and use_alternative_tickers is True, use second item; otherwise use first
                    if use_alternative_tickers and len(ticker_value) >= 2:
                        actual_ticker = ticker_value[1]
                    elif len(ticker_value) >= 1:
                        actual_ticker = ticker_value[0]
                    else:
                        actual_ticker = display_name
                elif isinstance(ticker_value, dict) and "ticker" in ticker_value:
                    actual_ticker = ticker_value.get("ticker")
                elif isinstance(ticker_value, str):
                    actual_ticker = ticker_value
                else:
                    # Fallback: display name is the ticker
                    actual_ticker = display_name
                
                display_to_ticker[display_name] = actual_ticker
                ticker_to_display[actual_ticker] = display_name
                actual_ticker_list.append(actual_ticker)
        else:
            # Fallback to list format for backwards compatibility
            actual_ticker_list = tickers_config if isinstance(tickers_config, list) else ["VTI", "VXUS", "IBIT", "IAUM"]
            for t in actual_ticker_list:
                display_to_ticker[t] = t
                ticker_to_display[t] = t

        # Get champion and challengers from tickers config
        # First ticker is champion, rest are challengers
        if isinstance(tickers_config, dict) and len(tickers_config) > 0:
            ticker_items = list(tickers_config.items())
            champion_display = ticker_items[0][0]  # First display name
            challenger_displays = [item[0] for item in ticker_items[1:]]  # Rest are challengers
        else:
            # Fallback
            champion_display = "US"
            challenger_displays = []
        
        champion = display_to_ticker.get(champion_display, champion_display)
        challenger_tickers = set()
        num_cols = 3  # Always use 3 columns

        # Convert challengers list to list of dicts with row/col positions
        challengers = []
        for idx, challenger_display in enumerate(challenger_displays):
            # Resolve display name to actual ticker
            actual_challenger = display_to_ticker.get(challenger_display, challenger_display)
            challenger_tickers.add(actual_challenger)
            # Create challenger config dict with ticker and calculated position
            challengers.append({
                "ticker": actual_challenger,
                "row": (idx // num_cols) + 1,
                "col": (idx % num_cols) + 1,
            })

        # Ensure all tickers (champion and challengers) are included
        all_tickers = list(set(actual_ticker_list + [champion] + list(challenger_tickers)))

        # Load price and RSI data once - used for both Dominance Shifts and RSI plots
        ticker_obj = TICKERS(all_tickers, period=rsi_config.get("period", "360d"), normalize=rsi_config.get("normalize", True))
        ticker_obj.get_rsi()
        rsi_df = ticker_obj.rsis[rsi_config.get("rsi_period", 14)]
        rsi_df_original = rsi_df.copy()
        # RSI is not smoothed - keep raw RSI values
        # Note: ticker_obj.prices is loaded in TICKERS.__init__ and used for both:

        # Calculate latest RSI and RSI delta once - reused in Summary and RSI rank plot
        latest_rsi = None
        rsi_delta = None
        if not rsi_df_original.empty:
            latest_rsi = rsi_df_original.iloc[-1].dropna()
            # Calculate RSI delta (last RSI - previous RSI)
            if len(rsi_df_original) >= 2:
                previous_rsi = rsi_df_original.iloc[-2].dropna()
                rsi_delta = latest_rsi - previous_rsi.reindex(latest_rsi.index, fill_value=0)
            else:
                rsi_delta = pd.Series(0.0, index=latest_rsi.index)

        # Pre-calculate 1 day and 1 week percent changes once - reused in Summary and Dominance charts
        pct_change_1d = ticker_obj.calculate_price_pct_change(1) if not ticker_obj.prices.empty else {}
        pct_change_1w = ticker_obj.calculate_price_pct_change(7) if not ticker_obj.prices.empty else {}

        # Build ticker color dictionary
        # Get colors from main ticker configuration (regions.yaml, sectors.yaml, etc.)
        ticker_colors = {}
        default_colors = ["#1f77b4", "#2ca02c", "purple", "goldenrod"]
        
        # Access main ticker configs (merged from all ticker yaml files)
        main_tickers_config_raw = config.get("tickers", {})
        # Convert to regular dict if needed (OmegaConf DictConfig)
        main_tickers_config = OmegaConf.to_container(main_tickers_config_raw, resolve=True) if main_tickers_config_raw else {}

        # Populate colors for all tickers
        for ticker_symbol in all_tickers:
            if not ticker_symbol:
                continue
            # Look up color by actual ticker symbol in main ticker configs
            color = default_colors[0]  # Default fallback
            if isinstance(main_tickers_config, dict) and ticker_symbol in main_tickers_config:
                ticker_info = main_tickers_config[ticker_symbol]
                if isinstance(ticker_info, dict):
                    color = ticker_info.get("color", default_colors[0])
            ticker_colors[ticker_symbol] = color

        # Table of Contents
        st.title("Table of Contents")
        toc_html = """
        <div style="text-align: center; margin: 20px 0;">
            <a href="#summary" style="margin: 0 15px; text-decoration: none; color: #1f77b4;">Summary</a> |
            <a href="#dominance-shifts" style="margin: 0 15px; text-decoration: none; color: #1f77b4;">Dominance Shifts</a> |
            <a href="#markets-strength" style="margin: 0 15px; text-decoration: none; color: #1f77b4;">Markets Strength</a> |
            <a href="#gli-global-liquidity-index" style="margin: 0 15px; text-decoration: none; color: #1f77b4;">GLI: Global Liquidity Index</a>
        </div>
        """
        st.markdown(toc_html, unsafe_allow_html=True)
        st.markdown("---")

        # Calculate Summary: Accelerators
        # Get accelerators from TICKERS object
        accelerators = ticker_obj.get_accelerators(rsi_period=rsi_config.get("rsi_period", 14), top_n=accelerators_top_n)

        # Display Summary section
        st.markdown('<a id="summary"></a>', unsafe_allow_html=True)
        st.title("Summary")

        # Count number of tickers (regions/sectors) being monitored
        num_tickers = len(tickers_config) if isinstance(tickers_config, dict) else len(tickers_config) if isinstance(tickers_config, list) else 0
        st.metric("Tickers Monitored", num_tickers, help="Number of regions/sectors being monitored")

        if accelerators:
            st.subheader("Accelerators")
            st.markdown(f"Tickers with: 1 day & 1 week change in top {accelerators_top_n}, and (positive RSI delta or RSI > 50)")

            # Create a DataFrame for display
            accel_data = []
            for acc in accelerators:
                ticker = acc["ticker"]
                display_name = ticker_to_display.get(ticker, ticker)

                # Get name from config if available
                if isinstance(tickers_config, dict) and display_name in tickers_config:
                    ticker_value = tickers_config[display_name]
                    # Support both formats: simple string or dict with name key
                    if isinstance(ticker_value, dict):
                        name = ticker_value.get("name", display_name)
                    else:
                        name = display_name
                else:
                    name = display_name

                accel_data.append(
                    {
                        "Name": name,
                        "Ticker": ticker,
                        "RSI": f"{acc['rsi']:.1f}",
                        "RSI Δ": f"+{acc['rsi_delta']:.1f}",
                        "1D %": f"{acc['pct_1d']:.2f}%",
                        "1W %": f"{acc['pct_1w']:.2f}%",
                    }
                )

            accel_df = pd.DataFrame(accel_data)
            
            # Sort by 1W % (descending - highest first)
            # Extract numeric value from "1W %" column for sorting
            accel_df["1W %_sort"] = accel_df["1W %"].str.replace("%", "").astype(float)
            accel_df = accel_df.sort_values("1W %_sort", ascending=False).drop(columns=["1W %_sort"])
            
            # Style the dataframe: Name with ticker color background, Ticker with ticker color text
            def style_row(row):
                """Apply background color to Name, text color to Ticker."""
                ticker = row["Ticker"]
                color = ticker_colors.get(ticker, "#1f77b4")
                
                # Convert color to RGB format for CSS
                try:
                    rgb = to_rgb(color)
                    bg_color = f"rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})"
                    text_color = f"rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})"
                except (ValueError, TypeError):
                    bg_color = color  # Use as-is if conversion fails
                    text_color = color
                
                # Determine text color for Name column based on background brightness
                name_text_color = "black" if _is_light_color(color) else "white"
                
                # Return style: Name with background color, Ticker with text color, others no styling
                styles = []
                for col in accel_df.columns:
                    if col == "Name":
                        styles.append(f"background-color: {bg_color}; color: {name_text_color}; font-weight: bold")
                    elif col == "Ticker":
                        styles.append(f"color: {text_color}; font-weight: bold")
                    else:
                        styles.append("font-weight: bold")  # No background color, just bold text
                return styles
            
            styled_df = accel_df.style.apply(style_row, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.subheader("Accelerators")
            st.info("No accelerators found (no tickers meet all criteria)")

        # Show Dominance Shifts before RSI
        st.markdown('<a id="dominance-shifts"></a>', unsafe_allow_html=True)
        st.title("Dominance Shifts")
        st.markdown("% change in price over different time periods. ")
        _plot_dominance_pie_charts(
            ticker_obj.prices, ticker_colors, tickers_config, ticker_to_display, pct_change_1d=pct_change_1d, pct_change_1w=pct_change_1w
        )
        st.markdown('<a id="markets-strength"></a>', unsafe_allow_html=True)
        st.title("Markets Strength")

        # Filter to only show last 1 year of data for plotting
        one_year_ago = pd.Timestamp.today() - pd.Timedelta(days=365)
        rsi_df_plot = rsi_df.loc[rsi_df.index >= one_year_ago].copy()

        # Determine grid size from challengers (always 3 columns, rows fill first)
        num_cols = 3
        max_row = max([challenger.get("row", 1) for challenger in challengers], default=1)
        min_col = 1
        actual_max_col = num_cols

        # Build subplot titles in row-major order
        champion_display_name = ticker_to_display.get(champion, champion)
        subplot_titles = []
        for challenger in challengers:
            challenger_ticker = challenger.get("ticker", "")
            # Get display names for titles
            challenger_display_name = ticker_to_display.get(challenger_ticker, challenger_ticker)
            title = challenger.get("title", f"{challenger_display_name} vs {champion_display_name} RSI")
            subplot_titles.append(title)

        # Build specs: all positions need secondary_y=True in specs to support secondary y-axis
        specs = []
        for row_idx in range(1, max_row + 1):
            row_spec = []
            for col_idx in range(1, num_cols + 1):
                row_spec.append({"secondary_y": True})
            specs.append(row_spec)

        # Create the figure with subplots
        fig = make_subplots(
            rows=max_row,
            cols=num_cols,
            subplot_titles=subplot_titles,
            specs=specs,
            vertical_spacing=0.05,
            horizontal_spacing=0.01,
            shared_xaxes=True,
            shared_yaxes=True,
        )

        # Create bar chart with latest RSI values ranked and RSI delta (before subplots)
        if latest_rsi is not None and rsi_delta is not None:
            # Sort by RSI value in descending order (ranked)
            latest_rsi_sorted = latest_rsi.sort_values(ascending=False)
            # Sort delta by the same order as RSI
            rsi_delta_sorted = rsi_delta.reindex(latest_rsi_sorted.index)

            # Get display names and colors for sorted tickers
            labels = []
            colors_list = []
            text_labels = []
            hover_templates = []

            for ticker_symbol in latest_rsi_sorted.index:
                display_name = ticker_to_display.get(ticker_symbol, ticker_symbol)
                # Get name from config if available
                if isinstance(tickers_config, dict) and display_name in tickers_config:
                    ticker_value = tickers_config[display_name]
                    # Support both formats: simple string or dict with name key
                    if isinstance(ticker_value, dict):
                        name = ticker_value.get("name", display_name)
                    else:
                        name = display_name
                else:
                    name = display_name
                labels.append(name)
                colors_list.append(ticker_colors.get(ticker_symbol, default_colors[0]))
                # Combine name, actual ticker symbol, and RSI value for text inside bar
                text_labels.append(f"{name}<br>{ticker_symbol}<br>{latest_rsi_sorted[ticker_symbol]:.1f}")
                # Get RSI delta for this ticker
                delta_value = rsi_delta_sorted[ticker_symbol]
                delta_sign = "+" if delta_value >= 0 else ""
                # Custom hover template: Name, Ticker, RSI value, RSI delta (no tuple)
                hover_templates.append(f"<b>{name}</b><br>Ticker: {ticker_symbol}<br>RSI: {latest_rsi_sorted[ticker_symbol]:.1f}<br>RSI Δ: {delta_sign}{delta_value:.1f}<extra></extra>")

            # Create bar chart with secondary y-axis for delta arrows
            bar_fig = make_subplots(specs=[[{"secondary_y": False}]])

            # Add RSI bars on primary axis
            bar_fig.add_trace(
                go.Bar(
                    x=labels,
                    y=latest_rsi_sorted.values,
                    marker=dict(color=colors_list),
                    text=text_labels,
                    textposition="inside",
                    textfont=dict(size=14, weight="bold", family="Arial Black"),
                    insidetextanchor="middle",
                    name="RSI",
                    hovertemplate="%{customdata}",
                    customdata=hover_templates,
                ),
                secondary_y=False,
            )

            # Add arrows showing delta on secondary axis (starting at 0, pointing to delta value)
            for idx, ticker_symbol in enumerate(latest_rsi_sorted.index):
                delta_value = rsi_delta_sorted[ticker_symbol]
                # Green for positive, red for negative
                arrow_color = "green" if delta_value >= 0 else "red"
                label = labels[idx]

                # Add delta text label at the arrow end
                bar_fig.add_annotation(
                    x=label,
                    y=latest_rsi_sorted[ticker_symbol] - delta_value,
                    text=f"{abs(delta_value):.1f}",
                    xref="x",
                    yref="y1",
                    xanchor="center",
                    yanchor="bottom" if delta_value > 0 else "top",
                    font=dict(color=arrow_color, size=min(max(abs(delta_value) * 13, 5), 60), weight="bold"),
                    showarrow=False,
                )

            # Calculate secondary y-axis range to ensure 0 is visible and centered
            if not rsi_delta_sorted.empty:
                delta_max = abs(rsi_delta_sorted).max()
                delta_range = [-delta_max * 1.2, delta_max * 1.2] if delta_max > 0 else [-10, 10]
            else:
                delta_range = [-10, 10]

            # Update axes
            bar_fig.update_xaxes(title="", showticklabels=False)
            bar_fig.update_yaxes(title_text="RSI", secondary_y=False)
            bar_fig.update_yaxes(
                title_text="RSI Delta",
                secondary_y=True,
                range=delta_range,  # Set range to ensure 0 is centered
            )

            bar_fig.update_layout(
                title="Current RSI Rank",
                title_font=dict(size=24, family="Arial Black"),
                margin=dict(l=40, r=40, t=80, b=40),
            )

            st.plotly_chart(bar_fig, config={"displayModeBar": True}, use_container_width=True)
        st.subheader("Markets Strength Over Time")
        # Get price data for challenger plots (filtered to same time period as RSI)
        price_df_plot = None
        if hasattr(ticker_obj, "prices") and ticker_obj.prices is not None:
            price_df_plot = ticker_obj.prices.loc[ticker_obj.prices.index >= one_year_ago].copy()

        # Add all challenger vs champion plots
        for challenger in challengers:
            _add_vs_pair_plot(
                fig,
                rsi_df_plot,
                challenger,
                champion,
                ticker_colors,
                rsi_config,
                default_colors,
                price_df_plot,
            )

        # Add reference lines to all subplots
        for ref_line in rsi_config.get("reference_lines", []):
            for challenger in challengers:
                row = challenger.get("row")
                col = challenger.get("col")
                fig.add_hline(
                    y=ref_line.get("y"),
                    line_dash=ref_line.get("line_dash", "dot"),
                    line_color=ref_line.get("line_color", "gray"),
                    opacity=ref_line.get("opacity", 0.5),
                    row=row,
                    col=col,
                )

        # Calculate global ranges for shared axes
        # Get all RSI values across all pairs to determine global range
        all_rsi_values = []
        all_diff_values = []
        for challenger in challengers:
            challenger_ticker = challenger.get("ticker")
            if challenger_ticker in rsi_df_plot.columns and champion in rsi_df_plot.columns:
                all_rsi_values.extend([rsi_df_plot[challenger_ticker], rsi_df_plot[champion]])
                diff = rsi_df_plot[challenger_ticker] - rsi_df_plot[champion]
                all_diff_values.append(diff)

        # Calculate global primary y-axis range (RSI)
        if all_rsi_values:
            all_rsi_combined = pd.concat(all_rsi_values)
            rsi_min = all_rsi_combined.min()
            rsi_max = all_rsi_combined.max()
            rsi_span = rsi_max - rsi_min
            # Set primary axis range to be symmetric around 50
            global_primary_range = [max(0, 50 - rsi_span / 2 * 1.5), min(100, 50 + rsi_span / 2 * 1.5)]
        else:
            global_primary_range = [0, 100]

        # Calculate global secondary y-axis range (Difference)
        if all_diff_values:
            all_diff_combined = pd.concat(all_diff_values)
            diff_min = all_diff_combined.min()
            diff_max = all_diff_combined.max()
            diff_span = max(abs(diff_min), abs(diff_max))
            # Set secondary axis range to be symmetric around 0
            global_secondary_range = [-diff_span * 1.1, diff_span * 1.1]
        else:
            global_secondary_range = [-30, 30]

        # Update axes labels and ranges for all subplots
        for challenger in challengers:
            row = challenger.get("row")
            col = challenger.get("col")

            # Only show x-axis title on bottom row when axes are shared
            x_title = "Date" if row == max_row else ""
            fig.update_xaxes(title_text=x_title, showgrid=False, row=row, col=col)
            # Only show y-axis titles: primary on first column, secondary on last column
            primary_y_title = rsi_config.get("yaxis_title", "RSI") if col == min_col else ""
            secondary_y_title = "Difference" if col == actual_max_col else ""
            # Apply global primary y-axis range to all subplots
            fig.update_yaxes(
                title_text=primary_y_title,
                showgrid=False,
                row=row,
                col=col,
                secondary_y=False,
                range=global_primary_range,
            )
            # Apply global secondary y-axis range to all subplots
            # Only show ticks and labels on the last column
            show_secondary_ticks = col == actual_max_col
            fig.update_yaxes(
                title_text=secondary_y_title,
                showgrid=False,
                showticklabels=show_secondary_ticks,
                row=row,
                col=col,
                secondary_y=True,
                range=global_secondary_range,
            )
            fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=row, col=col, secondary_y=True)

        # Update layout
        fig.update_layout(
            height=rsi_config.get("height", 2500),
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="center", x=0.5),
            plot_bgcolor="white",
        )

        # Move subplot titles slightly higher
        for annotation in fig.layout.annotations:
            if annotation.text:
                annotation.y += 0.01

        st.plotly_chart(fig, config={"displayModeBar": True}, use_container_width=True)

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
        ema_period=ema_period,
        show_lag=show_lag,
    )


if __name__ == "__main__":
    # Handle Hydra initialization - clear if already initialized
    if hydra.core.global_hydra.GlobalHydra().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    main()
