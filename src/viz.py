import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict


def normalize_prices(
    df: pd.DataFrame,
    symbols: List[str],
) -> pd.DataFrame:
    """
    Normalize price data to start at 1.0.

    Args:
        df: DataFrame with price data
        symbols: List of symbols to normalize

    Returns:
        DataFrame with normalized prices
    """
    df_normalized = df.copy()
    for symbol in symbols:
        df_normalized[symbol] = df[symbol] / df[symbol].iloc[0]
    return df_normalized


def create_performance_plot(
    df: pd.DataFrame,
    symbols: List[str],
    look_back_days: List[int],
    colors_dict: Dict[str, str],
    line_styles_dict: Dict[str, str],
    equity_config: Dict[str, Dict],
) -> go.Figure:
    """
    Create a multi-subplot figure showing normalized performance.

    Args:
        df: DataFrame with price data
        symbols: List of symbols to plot
        look_back_days: List of lookback periods in days
        colors_dict: Dictionary mapping symbols to their colors
        line_styles_dict: Dictionary mapping symbols to their line styles
    Returns:
        Plotly figure object
    """
    # Adjust the subplot grid to 2 rows and 3 columns
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[f"{days} trading days" for days in look_back_days],
        # shared_yaxes=True,
        vertical_spacing=0.1,
        horizontal_spacing=0.03,
    )

    for i, days in enumerate(look_back_days):
        df_normalized = normalize_prices(df.iloc[-days:], symbols)
        for symbol in symbols:
            fig.add_trace(
                go.Scatter(
                    x=df_normalized["Date"],
                    y=df_normalized[symbol],
                    name=symbol,
                    line=dict(color=colors_dict[symbol], dash=line_styles_dict[symbol]),
                    legendgroup=symbol,
                    showlegend=i == 0,
                    hovertemplate=(
                        f"<b style='color: {colors_dict[symbol]}'>Symbol:</b> {symbol}<br>"
                        f"<b style='color: {colors_dict[symbol]}'>Name:</b> {equity_config[symbol].get('name','-')}<br>"
                        f"<b style='color: {colors_dict[symbol]}'>Region:</b> {equity_config[symbol].get('region', '-')}<br>"
                        f"<b style='color: {colors_dict[symbol]}'>Industry:</b> {equity_config[symbol].get('industry', '-')}<br>"
                        f"<b style='color: {colors_dict[symbol]}'>Date:</b>" + "%{x}<br>"
                        f"<b style='color: {colors_dict[symbol]}'>Normalized Price:</b>"
                        + "%{y:.2f}<extra></extra>"
                    ),
                ),
                row=(i // 3) + 1,  # Calculate row index
                col=(i % 3) + 1,  # Calculate column index
            )

    fig.update_layout(
        height=800,
        showlegend=True,
        # title_text="Normalized Performance Comparison",
        hovermode="closest",
        autosize=True,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
        ),
    )

    return fig
