import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict
import numpy as np
from src.stats import Stats
from src.data import normalize_prices


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
        # reorder the columns to match the order of the symbols
        df_normalized = df_normalized[["Date"] + symbols]
        for symbol in symbols:
            keys = ["name", "region", "industry", "n_holdings"]
            keys = [key for key in keys if key in equity_config[symbol]]
            fig.add_trace(
                go.Scatter(
                    x=df_normalized["Date"],
                    y=df_normalized[symbol],
                    name=symbol,
                    mode="lines",
                    line=dict(color=colors_dict[symbol], dash=line_styles_dict[symbol]),
                    legendgroup=symbol,
                    showlegend=(i == 0),
                    hovertemplate=f"<b style='color: {colors_dict[symbol]}'>Symbol:</b> {symbol}<br>"
                    + "".join(
                        [
                            f"<b style='color: {colors_dict[symbol]}'>{key}:</b> {equity_config[symbol].get(key, '-')}<br>"
                            for key in keys
                        ]
                    )
                    + f"<b style='color: {colors_dict[symbol]}'>Date:</b>"
                    + "%{x}<br>"
                    + f"<b style='color: {colors_dict[symbol]}'>Normalized Price:</b>"
                    + "%{y:.2f}<extra></extra>",
                ),
                row=(i // 3) + 1,  # Calculate row index
                col=(i % 3) + 1,  # Calculate column index
            )
            fig.update_xaxes(showgrid=False, row=i // 3 + 1, col=i % 3 + 1)
            fig.update_yaxes(showgrid=False, row=i // 3 + 1, col=i % 3 + 1)
            # add a text box with the average performance
        stats = Stats(df_normalized)
        annotations = (
            f"AVG: {"+" if stats.avg_return > 0 else "-"} {abs(stats.avg_return):.2%}, <span style='color: violet; '> {stats.weighted_mean_std()[1]*100:.2f}</span><br><br>"
            + "<span style='color: snow; opacity: 0.3'>|</span>"
            + f"{"<span style='color: snow; opacity: 0.3'>|</span>".join([f"<span style='color: {colors_dict[symbol]}'>{ratio*100:.0f}</span>{'<span style="color: snow; opacity: 0.3">|</span><br>' if (i+1)!=1 and (i+1)%10==0 else ''}" for i,symbol, ratio in zip(range(len(symbols)),symbols, stats.ratios)])}"
            + f"{'<span style="color: snow; opacity: 0.3">|</span><br>' if len(symbols) % 10 != 0 else ''}"
            + f"<br>ALT: {"+" if stats.alternative_return() > 0 else "-"} {abs(stats.alternative_return()):.2%}, <span style='color: violet; '> {stats.weighted_mean_std(stats.ratios)[1]*100:.2f}</span>"
        )
        fig.add_annotation(
            x=min(df_normalized["Date"]),
            y=max(df_normalized.iloc[-1, 1:]),
            text=annotations,
            font=dict(color="lightgreen" if stats.avg_return > 0 else "coral"),
            opacity=1,
            bgcolor="black",
            xanchor="left",
            yanchor="top",
            showarrow=False,
            row=i // 3 + 1,
            col=i % 3 + 1,
        )
    fig.update_layout(
        height=800,
        showlegend=True,
        # title_text="Normalized Performance Comparison",
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

    return fig
