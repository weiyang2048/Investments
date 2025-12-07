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
from src.data import normalize_prices, compute_annualized_momentum_sum
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
) -> go.Figure:
    """Create a price performance plot showing normalized prices for different lookback periods."""
    df_norm = normalize_prices(df)
    n_windows = len(look_back_days)
    
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
        df_window_norm = normalize_prices(df_window)[symbols]
        
        # Limit to top symbols if too many
        visible_symbols = symbols[:50] if len(symbols) > 50 else symbols
        
        # Add price traces
        for symbol in symbols:
            if symbol not in df_window_norm.columns:
                continue
                
            keys = [key for key in ["name", "region", "industry", "n_holdings"] 
                   if key in equity_config.get(symbol, {})]
            size = 6 if df_window_norm.shape[0] < 30 else 1
            
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
                    visible=True if symbol in visible_symbols else "legendonly",
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
        height=300 * n_windows, 
        showlegend=True, 
        plot_bgcolor="black", 
        paper_bgcolor="black",
        font=dict(color="white"), 
        hovermode="closest", 
        autosize=True,
        margin=dict(l=50, r=150, t=60, b=50),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
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
    df_norm = normalize_prices(df)
    ranking_df = compute_annualized_momentum_sum(df_norm, window_sizes, ticker_obj=ticker_obj)
    
    # Get all columns except Rank and Symbol for transposition
    columns_to_keep = [col for col in ranking_df.columns if col not in ["Rank", "Symbol"]]
    
    # Keep all relevant columns and transpose
    ranking_df = ranking_df[["Symbol"] + columns_to_keep]
    transposed_df = ranking_df.set_index("Symbol").T.round(2)
    
    # Load metrics_order from config if not provided
    if metrics_order is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "conf", "main.yaml")
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                metrics_order = config.get("lenses", {}).get("metrics_order", ["m", "sharpe", "p", "ema20", "ema50", "ema200", "rsi_delta", "rsi", "macd_delta", "macd", "drawdown", "combined_score", "stride", "s0", "s1", "d0"])
        except Exception:
            # Fallback to default order
            metrics_order = ["m", "sharpe", "p", "ema20", "ema50", "ema200", "rsi_delta", "rsi", "macd_delta", "macd", "drawdown", "combined_score", "stride", "s0", "s1", "d0"]
    
    # Create the desired row order from metrics_order
    # Exclude individual momentum rows (m10, m50, m200, m400, etc.) and acceleration rows
    ordered_rows = metrics_order.copy()
    
    # Filter to only include rows that exist in the dataframe and exclude any individual momentum rows (m followed by numbers)
    existing_rows = [row for row in ordered_rows if row in transposed_df.index]
    # Also filter out any remaining momentum rows that match pattern m\d+ (like m10, m50, m200, m400)
    momentum_pattern = re.compile(r'^m\d+$')
    existing_rows = [row for row in existing_rows if not momentum_pattern.match(row)]
    
    integer_columns = ["s0", "s1"]
    transposed_df.loc[integer_columns] = transposed_df.loc[integer_columns].astype(int)
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
    colorbar_title: str = "Value",
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


def create_price_ratio_plot(
    df: pd.DataFrame,
    denominator_symbol: str,
    numerator_symbols: List[str],
    colors_dict: Dict[str, str],
    line_styles_dict: Dict[str, str],
    equity_config: Dict[str, Dict],
    look_back_days: List[int] = None,
    momentum_ranking: pd.DataFrame = None,
    top_n: int = 5,
) -> go.Figure:
    """
    Create price ratio plots organized by window size. Each row represents a different lookback period.
    First symbol is denominator, other symbols are numerators.
    
    Args:
        df: DataFrame with price data (columns: Date, symbols...)
        denominator_symbol: Symbol to use as denominator (first symbol)
        numerator_symbols: List of symbols to use as numerators (other symbols)
        colors_dict: Dictionary mapping symbols to colors
        line_styles_dict: Dictionary mapping symbols to line styles
        equity_config: Configuration dictionary for symbols
        look_back_days: List of lookback periods in days (each becomes a row)
        momentum_ranking: DataFrame with momentum rankings (transposed, columns=symbols, rows=metrics including 'combined_score')
        top_n: Number of top symbols to display by momentum (default: 5)
        
    Returns:
        plotly.graph_objects.Figure: The ratio plot with rows for different window sizes
    """
    # Filter out denominator_symbol from numerator_symbols if present
    numerator_symbols = [s for s in numerator_symbols if s != denominator_symbol]
    
    # Sort numerator symbols by momentum ranking (if available)
    sorted_numerator_symbols = numerator_symbols.copy()
    if momentum_ranking is not None and "combined_score" in momentum_ranking.index:
        # Get combined_score row (symbols are columns)
        combined_scores = momentum_ranking.loc["combined_score"]
        # Filter to only symbols that exist in numerator_symbols
        available_scores = combined_scores[combined_scores.index.isin(numerator_symbols)]
        # Sort by combined_score descending
        sorted_by_momentum = available_scores.sort_values(ascending=False).index.tolist()
        # Keep any symbols not in momentum_ranking at the end
        symbols_not_in_ranking = [s for s in numerator_symbols if s not in sorted_by_momentum]
        sorted_numerator_symbols = sorted_by_momentum + symbols_not_in_ranking
    
    # Determine top N symbols by momentum for visibility control
    top_symbols = set()
    if momentum_ranking is not None and "combined_score" in momentum_ranking.index:
        # Get combined_score row (symbols are columns)
        combined_scores = momentum_ranking.loc["combined_score"]
        # Filter to only symbols that exist in numerator_symbols
        available_scores = combined_scores[combined_scores.index.isin(numerator_symbols)]
        # Sort by combined_score descending and take top N
        top_symbols = set(available_scores.sort_values(ascending=False).head(top_n).index.tolist())
    
    if not numerator_symbols:
        return go.Figure().add_annotation(
            text="No numerator symbols provided", 
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16)
        )
    
    # Check if denominator symbol exists
    if denominator_symbol not in df.columns:
        return go.Figure().add_annotation(
            text=f"Denominator symbol {denominator_symbol} not found in data", 
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16)
        )
    
    # Default to all data if look_back_days not provided
    if look_back_days is None or len(look_back_days) == 0:
        look_back_days = [len(df)]
    
    n_subplots = len(look_back_days)
    n_cols = 2  # Fixed 2 columns
    n_rows = (n_subplots + n_cols - 1) // n_cols  # Calculate rows needed
    
    # Create subplots with 2 columns
    subplot_titles = [f"{days}d" for days in look_back_days]
    
    # Calculate vertical spacing dynamically
    vertical_spacing = min(0.12, 0.9 / max(n_rows - 1, 1))
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=0.1,
    )
    
    # Add traces for each window size (each subplot)
    for subplot_idx, days in enumerate(look_back_days):
        row = (subplot_idx // n_cols) + 1
        col = (subplot_idx % n_cols) + 1
        
        # Get data for this window (last N days) and normalize prices within this window
        df_window = df.tail(days).copy()
        df_window_norm = normalize_prices(df_window)
        
        # Add AVG trace first: average of all numerator symbols (except denominator)
        valid_numerator_symbols = [s for s in numerator_symbols if s in df_window_norm.columns]
        if valid_numerator_symbols and denominator_symbol in df_window_norm.columns:
            # Compute average of normalized prices for all numerators
            avg_numerator_norm = df_window_norm[valid_numerator_symbols].mean(axis=1)
            # Compute average ratio
            avg_ratio = avg_numerator_norm / df_window_norm[denominator_symbol]
            
            # Create hover template for AVG
            hover_template_avg = f"<b style='color: gold'>Symbol:</b> AVG<br>"
            hover_template_avg += f"<b style='color: gold'>Date:</b>%{{x}}<br>"
            hover_template_avg += f"<b>Average of {len(valid_numerator_symbols)} symbols</b><br>"
            hover_template_avg += f"<b>Denominator ({denominator_symbol}):</b>%{{customdata[1]:.2f}}<br>"
            hover_template_avg += f"<b>Average Ratio:</b>%{{y:.4f}}<extra></extra>"
            
            # Add AVG trace (always visible) - added first to appear first in legend
            fig.add_trace(
                go.Scatter(
                    x=df_window_norm.index,
                    y=avg_ratio,
                    name="AVG",  # Show as AVG in legend
                    mode="lines",
                    line=dict(color="gold", width=2, dash="dash"),
                    customdata=list(zip(avg_numerator_norm, df_window_norm[denominator_symbol])),
                    hovertemplate=hover_template_avg,
                    visible=True,  # Always visible
                    showlegend=(subplot_idx == 0),  # Only show legend for first subplot
                    legendgroup="AVG",
                ),
                row=row, col=col,
            )
        
        # Add traces for ALL ratios in this window (sorted by momentum, AVG already added first)
        for numerator_symbol in sorted_numerator_symbols:
            if numerator_symbol not in df_window_norm.columns:
                continue
            
            # Calculate the ratio from normalized prices (both start at 1.0 in this window)
            ratio = df_window_norm[numerator_symbol] / df_window_norm[denominator_symbol]
            
            # Create hover template for ratio
            keys = [key for key in ["name", "region", "industry", "n_holdings"] 
                    if key in equity_config.get(numerator_symbol, {})]
            hover_template_ratio = _create_hover_template(numerator_symbol, colors_dict, equity_config, keys, "Ratio")
            hover_template_ratio += f"<b>Denominator ({denominator_symbol}):</b>%{{customdata[1]:.2f}}<br>"
            hover_template_ratio += f"<b>Ratio ({numerator_symbol}/{denominator_symbol}):</b>%{{y:.4f}}<extra></extra>"
            
            # Determine visibility: top N by momentum are visible, others are legendonly
            is_visible = numerator_symbol in top_symbols if top_symbols else True
            visible = True if is_visible else "legendonly"
            
            # Add ratio line
            fig.add_trace(
                go.Scatter(
                    x=df_window_norm.index,
                    y=ratio,
                    name=numerator_symbol,  # Only show numerator symbol in legend (denominator is implicit)
                    mode="lines",
                    line=dict(color=colors_dict.get(numerator_symbol, "blue"), width=3),
                    customdata=list(zip(df_window_norm[numerator_symbol], df_window_norm[denominator_symbol])),
                    hovertemplate=hover_template_ratio,
                    visible=visible,
                    showlegend=(subplot_idx == 0),  # Only show legend for first subplot
                    legendgroup=numerator_symbol,
                ),
                row=row, col=col,
            )
        
        # Add horizontal line at ratio = 1 for this subplot
        fig.add_hline(
            y=1, 
            line_dash="dash", 
            line_color="crimson", 
            opacity=0.5,
            row=row, col=col,
        )
    
    # Update layout
    fig.update_layout(
        title=f"Price Ratio Analysis: */{denominator_symbol} (by window size)",
        height=500 * n_rows,  # Adjust height based on number of rows
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    
    # Set axes titles
    for subplot_idx in range(n_subplots):
        row = (subplot_idx // n_cols) + 1
        col = (subplot_idx % n_cols) + 1
        fig.update_yaxes(title_text="Ratio", row=row, col=col)
        # Set x-axis title only for bottom row subplots
        if row == n_rows:
            fig.update_xaxes(title_text="Date", row=row, col=col)
    
    return fig