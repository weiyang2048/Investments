import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict
import numpy as np
from src.stats import Stats
from src.data import normalize_prices, compute_momentum, compute_annualized_momentum_sum
from typing import Callable
from src.configurations import get_random_style


def create_performance_plot(
    df: pd.DataFrame,
    symbols: List[str],
    look_back_days: List[int],
    colors_dict: Dict[str, str],
    line_styles_dict: Dict[str, str],
    equity_config: Dict[str, Dict],
    transformation: Callable[[float], float] = lambda x: x,
) -> Dict[str, any]:
    """
    Create a multi-subplot figure showing normalized performance.

    Args:
        df: DataFrame with price data
        symbols: List of symbols to plot
        look_back_days: List of lookback periods in days
        colors_dict: Dictionary mapping symbols to their colors
        line_styles_dict: Dictionary mapping symbols to their line styles
        equity_config: Configuration for equity symbols
        transformation: Function to transform the data
    Returns:
        Dictionary containing:
        - 'figure': Plotly figure object
        - 'normalized_data': DataFrame with normalized price data
        - 'symbols': List of symbols used
        - 'look_back_days': List of lookback periods used
    """
    # Adjust the subplot grid to 2 rows and 3 columns
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[f"{days} trading days" for days in look_back_days],
        # shared_yaxes=True,
        vertical_spacing=0.1,
        horizontal_spacing=0.03,
    )
    stats = Stats(normalize_prices(df))
    for i, days in enumerate(look_back_days):
        df_normalized = normalize_prices(df.iloc[-days:])
        # reorder the columns to match the order of the symbols
        df_normalized = df_normalized[["Date"] + symbols]
        stats = Stats(df_normalized)
        current_ratios = list(stats.ratios(transformation))
        # top 3 symbols by return
        symbols_in_this_plot = list(df_normalized.columns[1:])
        top_3_symbols = [symbols_in_this_plot[j] for j in np.argsort(current_ratios)[-1:-4:-1]]
        negative_symbols = [symbols_in_this_plot[j] for j in np.argsort(current_ratios) if current_ratios[j] < 0]
        visible_symbols = set(top_3_symbols + negative_symbols)
        for symbol in symbols:
            keys = ["name", "region", "industry", "n_holdings"]
            keys = [key for key in keys if key in equity_config.get(symbol, {})]
            size = 6 if df_normalized.shape[0] < 30 else 1
            fig.add_trace(
                go.Scatter(
                    x=df_normalized["Date"],
                    y=df_normalized[symbol],
                    name=symbol,
                    mode="lines+markers",
                    line=dict(color=colors_dict[symbol], dash=line_styles_dict[symbol]),
                    marker=dict(color=colors_dict[symbol], size=size),
                    legendgroup=symbol,
                    showlegend=(i == 0),
                    visible=True if symbol in visible_symbols else "legendonly",
                    hovertemplate=f"<b style='color: {colors_dict[symbol]}'>Symbol:</b> {symbol}<br>"
                    + (
                        f"<b style='color: {colors_dict[symbol]}'>ETF Name:</b> {equity_config[symbol].get('name', '-')}<br>"
                        if equity_config.get(symbol, {}).get("name")
                        else ""
                    )
                    + "".join([f"<b style='color: {colors_dict[symbol]}'>{key}:</b> {equity_config[symbol].get(key, '-')}<br>" for key in keys])
                    + f"<b style='color: {colors_dict[symbol]}'>Date:</b>"
                    + "%{x}<br>"
                    + f"<b style='color: {colors_dict[symbol]}'>Normalized Price:</b>"
                    + "%{y:.2f}<extra></extra>",
                ),
                row=(i // 2) + 1,  # Calculate row index
                col=(i % 2) + 1,  # Calculate column index
            )
            fig.update_xaxes(showgrid=False, row=i // 2 + 1, col=i % 2 + 1)
            fig.update_yaxes(showgrid=False, row=i // 2 + 1, col=i % 2 + 1)
        top_3_symbols_styled = [f'<span style="color: {colors_dict[symbol]}">' + symbol + "</span>" for symbol in top_3_symbols]
        negative_symbols_styled = [f'<span style="color: {colors_dict[symbol]}">' + symbol + "</span>" for symbol in negative_symbols]
        annotations = (
            "<span style='color: snow; opacity: 0.3'>|</span>"
            + f"{"<span style='color: snow; opacity: 0.3'>|</span>".join([f"<span style='color: {colors_dict[symbol]}'>{ratio*100:.0f}</span>{'<span style=\"color: snow; opacity: 0.3\">|</span><br>' if (k+1)!=1 and (k+1)%6==0 else ''}" for k,symbol, ratio in zip(range(len(symbols_in_this_plot)),symbols_in_this_plot, current_ratios)])}"
            + f"{'<span style=\"color: snow; opacity: 0.3\">|</span><br>' if len(symbols_in_this_plot) % 6 != 0 else ''}"
            + f"{' + '.join(top_3_symbols_styled)}<br>{' - ' if len(negative_symbols_styled) > 0 else ''}{' - '.join(negative_symbols_styled)}<br>"
        )
        fig.add_annotation(
            x=min(df_normalized["Date"]),
            y=min(df_normalized.iloc[-1, 1:]),
            text=annotations,
            font=dict(color="lightgreen" if stats.avg_return > 0 else "coral", size=10),
            opacity=1,
            bgcolor="black",
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            row=i // 2 + 1,
            col=i % 2 + 1,
        )
    fig.update_layout(
        height=800,
        showlegend=True,
        # title_text="Normalized Performadnce Comparison",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        hovermode="closest",
        autosize=True,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(color="white"),
            bgcolor="black",
        ),
    )

    return {"figure": fig, "normalized_data": df_normalized, "symbols": symbols, "look_back_days": look_back_days}


def create_momentum_plot(
    df: pd.DataFrame,
    symbols: List[str],
    window_sizes: List[int] = [7, 30, 90, 180, 360],
    colors_dict: Dict[str, str] = None,
    line_styles_dict: Dict[str, str] = None,
    equity_config: Dict[str, Dict] = None,
    target_return: float = 1.3,
) -> Dict[str, any]:
    """
    Create a momentum plot showing momentum and renormalized prices for different window sizes.

    Args:
        df: DataFrame with price data
        symbols: List of symbols to plot
        window_sizes: List of window sizes in days for momentum calculation
        colors_dict: Dictionary mapping symbols to their colors
        line_styles_dict: Dictionary mapping symbols to their line styles
        equity_config: Configuration for equity symbols
        target_return: Target annualized return for threshold calculation

    Returns:
        Dictionary containing:
        - 'figure': Plotly figure object
        - 'momentum_combined': DataFrame with combined momentum data
        - 'momentum_data': Dictionary of momentum data by window size
        - 'window_sizes': List of window sizes used
    """
    # Normalize the data first
    df_norm = normalize_prices(df)

    # Compute momentum for all window sizes and get momentum counts
    momentum_data, momentum_combined = compute_momentum(df_norm, window_sizes, target_return=target_return)

    n_windows = len(window_sizes)
    fig = make_subplots(
        rows=n_windows,
        cols=1,
        subplot_titles=[f"Momentum and Price (window={window} days)" for window in window_sizes],
        vertical_spacing=0.05,
        specs=[[{"secondary_y": True}] for _ in range(n_windows)],
    )

    # Default colors if not provided
    if colors_dict is None:
        colors_dict = {symbol: f"hsl({i*360/len(symbols)}, 70%, 50%)" for i, symbol in enumerate(symbols)}

    if line_styles_dict is None:
        line_styles_dict = {symbol: "solid" for symbol in symbols}

    for idx, window in enumerate(window_sizes):
        momentum_df = momentum_data[window]

        # Get the last window*3 rows for better visualization
        display_rows = min(window * 3, len(momentum_df))
        momentum_display = momentum_df.tail(display_rows)

        # Get top 3 symbols by momentum for this window
        momentum_values = []
        valid_symbols = []
        for symbol in symbols:
            if symbol in momentum_display.columns:
                last_momentum = momentum_display[symbol].iloc[-1]
                momentum_values.append(last_momentum)
                valid_symbols.append(symbol)

        # Get top 3 symbols by momentum
        if len(valid_symbols) >= 3:
            top_3_indices = np.argsort(momentum_values)[-3:]
            top_3_symbols = [valid_symbols[i] for i in top_3_indices]
        else:
            top_3_symbols = valid_symbols

        # Plot momentum on the left y-axis (all symbols, but only top 3 visible)
        for symbol in symbols:
            if symbol in momentum_display.columns:
                is_visible = symbol in top_3_symbols
                # Calculate annualized momentum
                annualized_momentum = (1 + momentum_display[symbol]) ** (252 / window) - 1
                fig.add_trace(
                    go.Scatter(
                        x=momentum_display["Date"],
                        y=momentum_display[symbol],
                        customdata=annualized_momentum,
                        name=f"{symbol}",
                        mode="lines+markers",
                        line=dict(color=colors_dict.get(symbol, "blue"), dash="solid"),
                        marker=dict(size=3),
                        legendgroup=symbol,  # Changed from f"{symbol}_momentum"
                        showlegend=(idx == 0),
                        visible=True if is_visible else "legendonly",
                        hovertemplate=f"<b>{symbol} Momentum</b><br>"
                        + (
                            f"<b>ETF Name:</b> {equity_config[symbol].get('name', '-')}<br>"
                            if equity_config and equity_config.get(symbol, {}).get("name")
                            else ""
                        )
                        + f"Date: %{{x}}<br>"
                        + f"Momentum: %{{y:.2%}}<br>"
                        + f"Annualized: %{{customdata:.1%}}<extra></extra>",
                    ),
                    row=idx + 1,
                    col=1,
                    secondary_y=False,
                )

        # Add horizontal line based on window size: (1+y1)^(252/window) = target_return
        y1_threshold = target_return ** (window / 252) - 1
        fig.add_hline(y=y1_threshold, line_dash="dash", line_color="lightgreen", opacity=0.7, row=idx + 1, col=1)

        # Plot renormalized price on the right y-axis (all symbols, but only top 3 visible)
        for symbol in symbols:
            if symbol in df.columns:
                # Select the last window*3 rows for the current window
                price_window = df[["Date", symbol]].tail(display_rows).copy()

                # Renormalize price in this window: set first value to 1
                if not price_window[symbol].isnull().all():
                    first_valid = price_window[symbol].first_valid_index()
                    if first_valid is not None:
                        base = price_window.loc[first_valid, symbol]
                        price_window["renorm"] = price_window[symbol] / base
                    else:
                        price_window["renorm"] = price_window[symbol]
                else:
                    price_window["renorm"] = price_window[symbol]
                is_visible = symbol in top_3_symbols
                fig.add_trace(
                    go.Scatter(
                        x=price_window["Date"],
                        y=price_window["renorm"],
                        name=f"{symbol} Renorm Price",
                        mode="lines",
                        line=dict(color=colors_dict.get(symbol, "blue"), dash="longdash"),
                        opacity=0.7,
                        legendgroup=symbol,  # Changed from f"{symbol}_price"
                        showlegend=False,
                        visible=True if is_visible else "legendonly",
                        hovertemplate=f"<b>{symbol} Renorm Price</b><br>"
                        + (
                            f"<b>ETF Name:</b> {equity_config[symbol].get('name', '-')}<br>"
                            if equity_config and equity_config.get(symbol, {}).get("name")
                            else ""
                        )
                        + f"Date: %{{x}}<br>"
                        + f"Price: %{{y:.2f}}<extra></extra>",
                    ),
                    row=idx + 1,
                    col=1,
                    secondary_y=True,
                )

        # Update axes for this subplot
        fig.update_xaxes(title_text="Date", showgrid=False, row=idx + 1, col=1)
        fig.update_yaxes(title_text="Momentum", showgrid=False, secondary_y=False, row=idx + 1, col=1)
        fig.update_yaxes(title_text="Price", showgrid=False, secondary_y=True, row=idx + 1, col=1)

    # Update layout
    fig.update_layout(
        height=300 * n_windows,
        showlegend=True,
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        hovermode="closest",
        autosize=True,
        margin=dict(l=50, r=50, t=60, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="white"),
            bgcolor="black",
        ),
    )

    return {"figure": fig, "momentum_combined": momentum_combined, "momentum_data": momentum_data, "window_sizes": window_sizes}


def create_momentum_ranking_display(
    df: pd.DataFrame,
    symbols: List[str],
    window_sizes: List[int] = [7, 30, 90, 180, 360],
    equity_config: Dict[str, Dict] = None,
) -> pd.DataFrame:
    """
    Create a ranking display showing the sum of annualized momentum across all windows.

    Args:
        df: DataFrame with price data
        symbols: List of symbols to analyze
        window_sizes: List of window sizes in days for momentum calculation
        equity_config: Configuration for equity symbols

    Returns:
        DataFrame with ranked symbols by summed annualized momentum (transposed)
    """
    # Normalize the data first
    df_norm = normalize_prices(df)

    # Compute the summed annualized momentum
    ranking_df = compute_annualized_momentum_sum(df_norm, window_sizes)

    # Add ETF names if available
    if equity_config:
        # ranking_df["ETF_Name"] = ranking_df["Symbol"].map(lambda x: equity_config.get(x, {}).get("name", "-"))
        # Reorder columns to show ETF name
        ranking_df = ranking_df[[ "Symbol", "agg_momemtum"]]

    # Transpose the DataFrame
    # Set Symbol as index for transposition
    ranking_df_transposed = ranking_df.set_index("Symbol").T
    ranking_df_transposed = ranking_df_transposed.round(2)
    
    # Set the index column as the index
    # ranking_df_transposed = ranking_df_transposed.set_index("index")

    return ranking_df_transposed
