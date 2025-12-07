import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
import numpy as np
import yaml
import re
import os
from src.stats import Stats
from src.data import compute_symbol_metrics
from typing import Callable
from src.configurations import get_random_style

# Load plotly config
with open("conf/style_conf/plotly.yaml", "r") as f:
    plotly_config = yaml.safe_load(f)


def _create_hover_template(symbol: str, colors_dict: Dict[str, str], equity_config: Dict[str, Dict], 
                          keys: List[str], data_type: str = "Normalized Price") -> str:
    """Create hover template for plotly traces."""
    base_template = f"<b style='color: {colors_dict[symbol]}'>Symbol:</b> {symbol}<br>"
    
    for key in keys:
        if key in equity_config.get(symbol, {}):
            base_template += f"<b style='color: {colors_dict[symbol]}'>{key}:</b> {equity_config[symbol].get(key, '-')}<br>"
    
    base_template += f"<b style='color: {colors_dict[symbol]}'>Date:</b>%{{x}}<br>"
    base_template += f"<b style='color: {colors_dict[symbol]}'>{data_type}:</b>%{{y:.2f}}<extra></extra>"
    
    return base_template

def create_price_plot(
    df: pd.DataFrame,
    symbols: List[str],
    look_back_days: List[int],
    colors_dict: Dict[str, str],
    line_styles_dict: Dict[str, str],
    equity_config: Dict[str, Dict],
    visible_symbols: List[str] = None,
) -> go.Figure:
    """Create a price performance plot showing normalized prices for different lookback periods.
    
    Args:
        df: DataFrame with price data
        symbols: List of all symbols to plot
        look_back_days: List of lookback periods in days
        colors_dict: Dictionary mapping symbols to colors
        line_styles_dict: Dictionary mapping symbols to line styles
        equity_config: Configuration dictionary for symbols
        visible_symbols: List of symbols to display by default (others will be hidden but toggleable)
    """
    # Data is expected to be already normalized (from TICKERS with normalize=True)
    df_norm = df
    n_windows = len(look_back_days)
    
    # If visible_symbols not provided, show all symbols
    if visible_symbols is None:
        visible_symbols = symbols[:50] if len(symbols) > 50 else symbols
    
    # Create subplot titles
    subplot_titles = [f"Performance ({days}d)" for days in look_back_days]
    
    fig = make_subplots(
        rows=n_windows, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        specs=[[{"secondary_y": False}] for _ in range(n_windows)],
    )
    
    for idx, days in enumerate(look_back_days):
        # Get normalized data for this window
        df_window = df_norm.iloc[-days:] if len(df_norm) > days else df_norm
        df_window_norm = df_window[symbols]
        
        # Add price traces for all symbols
        for symbol in symbols:
            if symbol not in df_window_norm.columns:
                continue
                
            keys = [key for key in ["name", "region", "industry", "n_holdings"] 
                   if key in equity_config.get(symbol, {})]
            size = 6 if df_window_norm.shape[0] < 30 else 1
            
            # Determine visibility: top symbols are visible, others are legendonly
            is_visible = symbol in visible_symbols
            visible = True if is_visible else "legendonly"
            
            fig.add_trace(
                go.Scatter(
                    x=df_window_norm.index, 
                    y=df_window_norm[symbol],
                    name=symbol, 
                    mode="lines+markers",
                    line=dict(color=colors_dict.get(symbol, "blue"), dash=line_styles_dict.get(symbol, "solid")),
                    marker=dict(color=colors_dict.get(symbol, "blue"), size=size),
                    legendgroup=symbol, 
                    showlegend=(idx == 0),
                    visible=visible,
                    hovertemplate=_create_hover_template(symbol, colors_dict, equity_config, keys, "Normalized Price"),
                ),
                row=idx + 1, col=1,
            )
        
        # Add horizontal line at y=1
        fig.add_hline(y=1, line_dash="solid", line_color="darkgray", 
                     opacity=0.25, row=idx + 1, col=1)
        
        # Update axes
        fig.update_xaxes(title_text="Date", showgrid=False, row=idx + 1, col=1)
        fig.update_yaxes(title_text="Normalized Price", showgrid=False, row=idx + 1, col=1)
    
    fig.update_layout(
        # height=300 * n_windows, 
        showlegend=True, 
        plot_bgcolor="black", 
        paper_bgcolor="black",
        font=dict(color="white"), 
        hovermode="closest", 
        autosize=True,
        margin=dict(l=50, r=50, t=60, b=100),
        legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5,
                   font=dict(color="white"), bgcolor="black"),
    )
    
    return fig


def create_momentum_ranking_display(
    df: pd.DataFrame,
    window_sizes: List[int] = [7, 30, 90, 180, 360],
    metrics_order: List[str] = None,
    ticker_obj=None,
) -> pd.DataFrame:
    """Create a ranking display showing momentum (m) from TICKER.get_momentum with period [10,20,50,100,200]."""
    # Data is expected to be already normalized (from TICKERS with normalize=True)
    ranking_df = compute_symbol_metrics(df, window_sizes, ticker_obj=ticker_obj)
    
    # Get all columns except Rank and Symbol for transposition
    columns_to_keep = [col for col in ranking_df.columns if col not in ["Rank", "Symbol"]]
    
    # Keep all relevant columns and transpose
    ranking_df = ranking_df[["Symbol"] + columns_to_keep]
    transposed_df = ranking_df.set_index("Symbol").T
    
    # Load metrics_order from config if not provided
    if metrics_order is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "conf", "main.yaml")
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                metrics_order = config.get("lenses", {}).get("metrics_order", ["m", "sharpe", "p", "d0", "ema10", "ema20", "ema50", "ema200", "rsi_delta", "rsi", "macd_delta", "macd", "drawdown", "combined_score"])
        except Exception:
            # Fallback to default order
            metrics_order = ["m", "sharpe", "p", "d0", "ema10", "ema20", "ema50", "ema200", "rsi_delta", "rsi", "macd_delta", "macd", "drawdown", "combined_score"]
    
    # Create the desired row order from metrics_order
    # Exclude individual momentum rows (m10, m50, m200, m400, etc.) and acceleration rows
    ordered_rows = metrics_order.copy()
    
    # Filter to only include rows that exist in the dataframe and exclude any individual momentum rows (m followed by numbers)
    existing_rows = [row for row in ordered_rows if row in transposed_df.index]
    # Also filter out any remaining momentum rows that match pattern m\d+ (like m10, m50, m200, m400)
    momentum_pattern = re.compile(r'^m\d+$')
    existing_rows = [row for row in existing_rows if not momentum_pattern.match(row)]
    
    
    # Multiply d0 by 100 and format to 1 decimal place
    if "d0" in transposed_df.index:
        transposed_df.loc["d0"] = (transposed_df.loc["d0"] * 100).round(2)
    
    # Multiply drawdown by 100 (convert to percentage)
    if "drawdown" in transposed_df.index:

        drawdown_values = transposed_df.loc["drawdown"] * 100
        # Round to int, but handle NaN values by keeping them as NaN or converting to nullable int
        transposed_df.loc["drawdown"] = drawdown_values.round(0).astype("Int64")
    
    result_df = transposed_df.loc[existing_rows]
    # Replace _delta with triangle symbol (Δ) in index names
    result_df.index = result_df.index.str.replace("_delta", "Δ", regex=False)
    return result_df


def create_plotly_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    text: str,
    hover_data: dict,
    layout: dict = dict(),
):
    """Create a bar chart using Plotly Express."""
    fig = px.bar(df, x=x_col, y=y_col, text=text, hover_data=hover_data)
    hover_template = "<b>%{x:.2f if str(x).isnumeric() else x}</b><br> <b>%{y:.2f if str(y).isnumeric() else y}</b><extra></extra>"
    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        hovertemplate=hover_template,
    )
    if df[y_col].dtype == "string":
        fig.update_yaxes(type="category")
    plotly_config["layout"].update(layout)
    fig.update_layout(plotly_config["layout"])
    return fig


def create_plotly_choropleth(
    df: pd.DataFrame,
    locations_col: str,
    color_col: str,
    hover_name_col: str = None,
    color_scale: str = "YlOrRd",
    projection: str = "natural earth",
    layout: dict = dict(),
    locationmode: str = "country names",
    log_scale: bool = True,
    lower_bound: float = None,
):
    """Create a Plotly choropleth map."""
    df_copy = df.loc[df[color_col] > (lower_bound or -np.inf)].copy()
    
    if log_scale:
        df_copy[color_col] = df_copy[color_col] + (1 - df_copy[color_col].min()) + 0.00001
        df_copy[color_col + "_log"] = np.log(df_copy[color_col])

    fig = px.choropleth(
        df_copy,
        locations=locations_col,
        color=color_col + "_log" if log_scale else color_col,
        hover_name=hover_name_col,
        color_continuous_scale=color_scale,
        projection=projection,
        locationmode=locationmode,
        hover_data={"Country": False, color_col: ":.2f", color_col + "_log": False},
    )

    fig.update_layout(coloraxis_showscale=False)
    plotly_config["layout"].update(layout)
    fig.update_layout(plotly_config["layout"])
    return fig
