import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict
import numpy as np


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
        final_row = df_normalized.iloc[-1, 1:] - 1
        avg_performance = final_row.mean()
        none_zero_ss = final_row[final_row > 0].apply(lambda x: x**2).sum()
        ratios = [x**2 / none_zero_ss if x > 0 else 0 for x in final_row]
        alternative_performance = np.dot(final_row, ratios)
        annotations = (
            f"AVG Perf: {"+" if avg_performance > 0 else "-"} {abs(avg_performance):.2%}<br>"
            + "<span style='color: snow; opacity: 0.3'>|</span>"
            + f"{"<span style='color: snow; opacity: 0.3'>|</span>".join([f"<span style='color: {colors_dict[symbol]}'>{int(ratio*100) if ratio>0 else ''}</span>{'<span style="color: snow; opacity: 0.3">|</span><br>' if i!=0 and i%9==0 else ''}" for i,symbol, ratio in zip(range(len(symbols)),symbols, ratios)])}"
            + f"{'<span style="color: snow; opacity: 0.3">|</span><br>' if len(symbols) % 10 != 0 else ''}"
            + f"Alt Perf: {"+" if alternative_performance > 0 else "-"} {abs(alternative_performance):.2%}"
        )
        fig.add_annotation(
            x=min(df_normalized["Date"]),
            y=max(df_normalized.iloc[-1, 1:]),
            text=annotations,
            font=dict(color="lightgreen" if avg_performance > 0 else "coral"),
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
