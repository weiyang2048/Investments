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

import hydra
from omegaconf import OmegaConf

from src.configurations.yaml import register_resolvers
from src.dashboard.create_page import setup_page_and_sidebar

from src.data.GLI import st_load_global_liquidity
from src.data.P import st_get_tickers_close_prices
from src.data.FearGreed import FearGreed
from src.data.TICKER import TICKERS


def _color_to_rgba(color: str, opacity: float = 0.2) -> str:
    """Convert color name or hex to rgba format with opacity."""
    # Common color name to RGB mapping
    color_map = {
        "red": (255, 0, 0),
        "green": (0, 128, 0),
        "blue": (0, 0, 255),
        "purple": (128, 0, 128),
        "gray": (128, 128, 128),
        "grey": (128, 128, 128),
        "brown": (165, 42, 42),
        "goldenrod": (218, 165, 32),
        "orange": (255, 165, 0),
        "yellow": (255, 255, 0),
        "pink": (255, 192, 203),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "lightblue": (173, 216, 230),
        "lightgreen": (144, 238, 144),
        "coral": (255, 127, 80),
        "royalblue": (65, 105, 225),
        "gold": (255, 215, 0),
        "crimson": (220, 20, 60),
        "seagreen": (46, 139, 87),
    }
    
    # Check if it's a hex color
    if color.startswith("#"):
        # Remove # and convert hex to RGB
        hex_color = color.lstrip("#")
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return f"rgba({r}, {g}, {b}, {opacity})"
        elif len(hex_color) == 3:
            r = int(hex_color[0] * 2, 16)
            g = int(hex_color[1] * 2, 16)
            b = int(hex_color[2] * 2, 16)
            return f"rgba({r}, {g}, {b}, {opacity})"
    
    # Check if it's already rgba
    if color.startswith("rgba("):
        return color
    
    # Check if it's rgb
    rgb_match = re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", color)
    if rgb_match:
        r, g, b = rgb_match.groups()
        return f"rgba({r}, {g}, {b}, {opacity})"
    
    # Check color map
    color_lower = color.lower()
    if color_lower in color_map:
        r, g, b = color_map[color_lower]
        return f"rgba({r}, {g}, {b}, {opacity})"
    
    # Default fallback
    return f"rgba(128, 128, 128, {opacity})"


def _add_vs_pair_plot(
    fig: go.Figure,
    rsi_df: pd.DataFrame,
    pair_config: dict,
    ticker_colors: dict,
    rsi_config: dict,
    default_colors: list,
):
    """Add a vs pair plot (two tickers with difference curve) to the figure."""
    row = pair_config.get("row")
    col = pair_config.get("col")
    ticker1 = pair_config.get("ticker1")
    ticker2 = pair_config.get("ticker2")
    show_individual_lines = pair_config.get("show_individual_lines", True)
    
    if not ticker1 or not ticker2 or ticker1 not in rsi_df.columns or ticker2 not in rsi_df.columns:
        return
    
    # Add individual RSI lines
    # When show_individual_lines is False, still show them but with lower opacity
    for ticker_symbol in [ticker1, ticker2]:
        color = ticker_colors.get(ticker_symbol, default_colors[0])
        # Use lower opacity when show_individual_lines is False
        line_opacity = 0.5 if show_individual_lines else 0.3
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
            ),
            row=row,
            col=col,
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
    diff_name = pair_config.get("diff_name", f"{ticker1} - {ticker2}")
    fig.add_trace(
        go.Scatter(
            x=rsi_df.index,
            y=diff,
            mode="lines",
            name=diff_name,
            line=dict(
                color="black",
                width=rsi_config.get("line_width", 2),
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


def _calculate_price_pct_change_for_period(price_data: pd.DataFrame, ticker: str, days_back: int) -> float:
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


def _plot_dominance_pie_charts(price_data: pd.DataFrame, ticker_colors: dict, tickers_config: dict, ticker_to_display: dict = None):
    """Plot multiple pie charts showing price percent changes for different time periods."""
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
            for display_name, ticker_config in tickers_config.items():
                if isinstance(ticker_config, dict) and "ticker" in ticker_config:
                    actual_ticker = ticker_config.get("ticker")
                    ticker_to_display[actual_ticker] = display_name
                else:
                    ticker_to_display[display_name] = display_name
    
    for idx, ticker in enumerate(available_tickers):
        # Get display name for this ticker (for config lookup)
        display_name = ticker_to_display.get(ticker, ticker)
        
        # Get name and color from ticker config if available
        if isinstance(tickers_config, dict) and display_name in tickers_config:
            ticker_config = tickers_config[display_name]
            name = ticker_config.get("name", display_name)
            color = ticker_config.get("color", ticker_colors.get(ticker, default_colors[idx % len(default_colors)]))
        else:
            name = display_name
            color = ticker_colors.get(ticker, default_colors[idx % len(default_colors)])
        
        assets[ticker] = {"name": name, "color": color}

    # Create Streamlit columns for the pie charts
    cols = st.columns(5)

    for col_idx, (period_name, days_back) in enumerate(time_periods):
        with cols[col_idx]:
            st.markdown(f"**{period_name} Change**")

            labels = []
            values = []
            colors_list = []

            for ticker, info in assets.items():
                pct_change = _calculate_price_pct_change_for_period(price_data, ticker, days_back)

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
                            textfont=dict(size=14, family="Arial Black", color="black"),
                            hole=0.3,  # Donut chart
                            showlegend=False,
                            textposition="inside",
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
        
        # Create mapping from display name to actual ticker symbol
        # Also extract actual ticker symbols for data fetching
        display_to_ticker = {}  # Maps display name (e.g., "US") to actual ticker (e.g., "VTI")
        ticker_to_display = {}  # Maps actual ticker (e.g., "VTI") to display name (e.g., "US")
        actual_ticker_list = []
        
        if isinstance(tickers_config, dict):
            for display_name, ticker_config in tickers_config.items():
                # Check if config has a 'ticker' field (new pattern) or use display_name as ticker (old pattern)
                if isinstance(ticker_config, dict) and "ticker" in ticker_config:
                    actual_ticker = ticker_config.get("ticker")
                    display_to_ticker[display_name] = actual_ticker
                    ticker_to_display[actual_ticker] = display_name
                    actual_ticker_list.append(actual_ticker)
                else:
                    # Old pattern: display name is the ticker
                    display_to_ticker[display_name] = display_name
                    ticker_to_display[display_name] = display_name
                    actual_ticker_list.append(display_name)
        else:
            # Fallback to list format for backwards compatibility
            actual_ticker_list = tickers_config if isinstance(tickers_config, list) else ["VTI", "VXUS", "IBIT", "IAUM"]
            for t in actual_ticker_list:
                display_to_ticker[t] = t
                ticker_to_display[t] = t
        
        # Get vs_pairs from config and extract all required tickers
        vs_pairs = rsi_config.get("vs_pairs", [])
        vs_pair_tickers = set()
        for pair in vs_pairs:
            ticker1 = pair.get("ticker1")
            ticker2 = pair.get("ticker2")
            # Resolve display names to actual tickers
            actual_ticker1 = display_to_ticker.get(ticker1, ticker1)
            actual_ticker2 = display_to_ticker.get(ticker2, ticker2)
            vs_pair_tickers.add(actual_ticker1)
            vs_pair_tickers.add(actual_ticker2)
            # Update vs_pairs with actual ticker symbols
            pair["ticker1"] = actual_ticker1
            pair["ticker2"] = actual_ticker2
        
        # Ensure all tickers from vs_pairs are included
        all_tickers = list(set(actual_ticker_list + list(vs_pair_tickers)))
        
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

        # Filter to only show last 1 year of data for plotting
        one_year_ago = pd.Timestamp.today() - pd.Timedelta(days=365)
        rsi_df_plot = rsi_df.loc[rsi_df.index >= one_year_ago].copy()

        # Get vs_pairs configuration
        vs_pairs = rsi_config.get("vs_pairs", [])
        
        # Determine grid size from vs_pairs
        max_row = max([pair.get("row", 1) for pair in vs_pairs], default=1)
        max_col = max([pair.get("col", 1) for pair in vs_pairs], default=1)
        
        # Build subplot titles in row-major order (row 1 col 1, row 1 col 2, row 2 col 1, etc.)
        # Create a map of (row, col) -> title
        title_map = {}
        for pair in vs_pairs:
            row = pair.get("row")
            col = pair.get("col")
            ticker1 = pair.get("ticker1", "")
            ticker2 = pair.get("ticker2", "")
            # Get display names for titles
            display_name1 = ticker_to_display.get(ticker1, ticker1)
            display_name2 = ticker_to_display.get(ticker2, ticker2)
            title = pair.get("title", f"{display_name1} vs {display_name2} RSI")
            title_map[(row, col)] = title
        
        # Build subplot_titles list in row-major order (only for positions with vs_pairs)
        subplot_titles = []
        for row_idx in range(1, max_row + 1):
            for col_idx in range(1, max_col + 1):
                title = title_map.get((row_idx, col_idx))
                if title:
                    subplot_titles.append(title)
                else:
                    # Empty string for positions without pairs (shouldn't happen with our config)
                    subplot_titles.append("")
        
        # Build specs: all positions need secondary_y=True in specs to support secondary y-axis
        specs = []
        for row_idx in range(1, max_row + 1):
            row_spec = []
            for col_idx in range(1, max_col + 1):
                # Each subplot needs secondary_y=True in specs to allow secondary y-axis traces
                row_spec.append({"secondary_y": True})
            specs.append(row_spec)
        
        # Create the figure with subplots
        fig = make_subplots(
            rows=max_row,
            cols=max_col,
            subplot_titles=subplot_titles,
            specs=specs,
            vertical_spacing=0.08,
            horizontal_spacing=0.06,
        )

        # Build ticker color dictionary
        ticker_colors = {}
        default_colors = ["#1f77b4", "#2ca02c", "purple", "goldenrod"]
        
        # Get all unique tickers from vs_pairs
        all_vs_tickers = set()
        for pair in vs_pairs:
            all_vs_tickers.add(pair.get("ticker1"))
            all_vs_tickers.add(pair.get("ticker2"))
        
        # Populate colors for all tickers used in vs_pairs
        for ticker_symbol in all_vs_tickers:
            if not ticker_symbol:
                continue
            # Get display name for config lookup
            display_name = ticker_to_display.get(ticker_symbol, ticker_symbol)
            # Get color from config
            if isinstance(tickers_config, dict) and display_name in tickers_config:
                ticker_config = tickers_config[display_name]
                color = ticker_config.get("color", default_colors[0])
            else:
                color = default_colors[0]
            ticker_colors[ticker_symbol] = color

        # Create pie chart with latest RSI values (before subplots)
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
                    textfont=dict(size=18, family="Arial Black", color="black"),
                    hole=0.3,  # Donut chart
                    showlegend=False,
                )
            ])
            
            pie_fig.update_layout(
                title="Current RSI Values",
                title_font=dict(size=24, family="Arial Black"),
                height=600,
                margin=dict(l=40, r=40, t=80, b=40),
            )
            
            st.plotly_chart(pie_fig, config={"displayModeBar": True}, use_container_width=True)
            st.markdown("---")

        # Add all vs_pair plots
        for pair in vs_pairs:
            _add_vs_pair_plot(
                fig,
                rsi_df_plot,
                pair,
                ticker_colors,
                rsi_config,
                default_colors,
            )

        # Add reference lines to all subplots
        for ref_line in rsi_config.get("reference_lines", []):
            for pair in vs_pairs:
                row = pair.get("row")
                col = pair.get("col")
                # Only show annotation on first subplot
                annotation_text = ref_line.get("annotation_text", "") if (row == 1 and col == 1) else ""
                fig.add_hline(
                    y=ref_line.get("y"),
                    line_dash=ref_line.get("line_dash", "dot"),
                    line_color=ref_line.get("line_color", "gray"),
                    opacity=ref_line.get("opacity", 0.5),
                    annotation_text=annotation_text,
                    annotation_font_color=ref_line.get("annotation_font_color", "black") if annotation_text else None,
                    row=row,
                    col=col,
                )

        # Update axes labels for all subplots
        for pair in vs_pairs:
            row = pair.get("row")
            col = pair.get("col")
            ticker1 = pair.get("ticker1")
            ticker2 = pair.get("ticker2")
            
            fig.update_xaxes(title_text="Date", showgrid=False, row=row, col=col)
            fig.update_yaxes(title_text=rsi_config.get("yaxis_title", "RSI"), showgrid=False, row=row, col=col, secondary_y=False)
            
            # Align secondary y-axis so that RSI=50 aligns with Difference=0
            # Get RSI data range to calculate primary axis span
            if ticker1 in rsi_df_plot.columns and ticker2 in rsi_df_plot.columns:
                # Get combined RSI range
                rsi_combined = pd.concat([rsi_df_plot[ticker1], rsi_df_plot[ticker2]])
                rsi_min = rsi_combined.min()
                rsi_max = rsi_combined.max()
                rsi_span = rsi_max - rsi_min
                
                # Set primary axis range to be symmetric around 50
                primary_range = [max(0, 50 - rsi_span/2 * 1.5), min(100, 50 + rsi_span/2 * 1.5)]
                
                # Set secondary axis range to match primary scale, centered at 0
                secondary_span = rsi_span * 1.1
                secondary_range = [-secondary_span / 2, secondary_span / 2]
                
                # Update primary axis range
                fig.update_yaxes(
                    range=primary_range,
                    row=row,
                    col=col,
                    secondary_y=False
                )
            else:
                # Default range for secondary axis
                secondary_range = [-30, 30]
            
            fig.update_yaxes(
                title_text="Difference",
                showgrid=False,
                row=row,
                col=col,
                secondary_y=True,
                range=secondary_range
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

        st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)

    st.markdown("---")
    st.title("Dominance Shifts")
    st.markdown("Percent change in price over different time periods. ")

    # Plot dominance pie charts using the same tickers and price data from RSI section
    _plot_dominance_pie_charts(ticker.prices, ticker_colors, tickers_config, ticker_to_display)

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




if __name__ == "__main__":
    # Handle Hydra initialization - clear if already initialized
    if hydra.core.global_hydra.GlobalHydra().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    main()
