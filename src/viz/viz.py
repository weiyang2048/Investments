import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
import numpy as np
import yaml
from src.stats import Stats
from src.data import normalize_prices, compute_momentum, compute_annualized_momentum_sum
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

def create_momentum_ranking_display(
    df: pd.DataFrame,
    window_sizes: List[int] = [7, 30, 90, 180, 360],
) -> pd.DataFrame:
    """Create a ranking display showing the sum of annualized momentum across all windows."""
    df_norm = normalize_prices(df)
    ranking_df = compute_annualized_momentum_sum(df_norm, window_sizes)
    
    # Get all columns except Rank and Symbol for transposition
    columns_to_keep = [col for col in ranking_df.columns if col not in ["Rank", "Symbol"]]
    
    # Keep all relevant columns and transpose
    ranking_df = ranking_df[["Symbol"] + columns_to_keep]
    transposed_df = ranking_df.set_index("Symbol").T.round(2)
    
    # Reorder rows: am first, then momentum rows, then acceleration rows, then average acceleration
    momentum_rows = [f"m{window}" for window in window_sizes]
    acceleration_rows = [f"a{window}" for window in window_sizes[1:-1]]  # Exclude last window
    
    # Acceleration data is already computed in compute_annualized_momentum_sum
    
    # Create the desired row order: m first, then a, then d0, then d6, then combined_score, then put-call ratio, then stride, then streak rows, then average consecutive movements, then average percentage movements, then momentum rows (hide individual acceleration rows)
    pct_change_row = f"d{min(window_sizes)}"
    ordered_rows = ["m"] + ["a"] + ["d0"] + [pct_change_row] + ["combined_score"] + ["pcr_m1"] + ["stride", "s0", "s1", "s2", "avg_s+", "avg_s-", "avg%+", "avg%-"] + momentum_rows
    
    # Filter to only include rows that exist in the dataframe
    existing_rows = [row for row in ordered_rows if row in transposed_df.index]
    integer_columns = ["s0", "s1", "s2"]
    transposed_df.loc[integer_columns] = transposed_df.loc[integer_columns].astype(int)
    return transposed_df.loc[existing_rows]


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


def create_combined_performance_momentum_plot(
    df: pd.DataFrame,
    symbols: List[str],
    look_back_days: List[int],
    colors_dict: Dict[str, str],
    line_styles_dict: Dict[str, str],
    equity_config: Dict[str, Dict],
    target_return: float = 1.4,
    transformation: Callable[[float], float] = lambda x: x,
    momentum_ranking: pd.DataFrame = None,
) -> Dict[str, any]:
    """Create a combined plot showing performance and momentum in a single subplot with 2 columns and rows for windows."""
    # Prepare data for both performance and momentum
    df_norm = normalize_prices(df)
    momentum_data, momentum_combined = compute_momentum(df_norm, look_back_days, target_return=target_return)
    
    # Sort symbols by momentum ranking if provided and get top 4 globally
    top_4_symbols_global = []
    if momentum_ranking is not None and "m" in momentum_ranking.index:
        # Get momentum ranking values and sort symbols by them (descending order)
        ranking_values = momentum_ranking.loc["m"]
        sorted_symbols = ranking_values.sort_values(ascending=False).index.tolist()
        # Keep only symbols that exist in our data
        symbols = [s for s in sorted_symbols if s in symbols]
        # Get top 4 symbols globally
        top_4_symbols_global = symbols[:5]
    
    n_windows = len(look_back_days)
    # Create subplot titles with performance and momentum for each row
    subplot_titles = []
    for i, days in enumerate(look_back_days):
        subplot_titles.append(f"Performance ({days}d)")
        # Only add momentum title if not the final row
        if i < n_windows - 1:
            subplot_titles.append(f"Momentum ({days}d)")
        else:
            subplot_titles.append("")  # Empty title for final row momentum plot
    
    fig = make_subplots(
        rows=n_windows, cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08, horizontal_spacing=0.1,
        specs=[[{"secondary_y": False}, {"secondary_y": False}] for _ in range(n_windows)],
    )
    
    for idx, days in enumerate(look_back_days):
        # Performance plot (left column)
        df_normalized = normalize_prices(df.iloc[-days:])[["Date"] + symbols]
        
        # Use global top 4 symbols for visibility
        visible_symbols = set(top_4_symbols_global) if top_4_symbols_global else set(symbols[:5])
        
        # Add performance traces
        for symbol in symbols:
            keys = [key for key in ["name", "region", "industry", "n_holdings"] 
                   if key in equity_config.get(symbol, {})]
            size = 6 if df_normalized.shape[0] < 30 else 1
            
            fig.add_trace(
                go.Scatter(
                    x=df_normalized["Date"], y=df_normalized[symbol],
                    name=symbol, mode="lines+markers",
                    line=dict(color=colors_dict[symbol], dash=line_styles_dict.get(symbol, "solid")),
                    marker=dict(color=colors_dict[symbol], size=size),
                    legendgroup=symbol, showlegend=(idx == 0),
                    visible=True if symbol in visible_symbols else "legendonly",
                    hovertemplate=_create_hover_template(symbol, colors_dict, equity_config, keys, "Normalized Price"),
                ),
                row=idx + 1, col=1, secondary_y=False,
            )
        
        # Add horizontal line at y=1 for all performance plots (first column)
        fig.add_hline(y=1, line_dash="solid", line_color="darkgray", 
                     opacity=0.25, row=idx + 1, col=1)
        
        # Momentum plot (right column) - skip for the final row
        if idx < n_windows - 1:  # Not the final row
            momentum_df = momentum_data[days]
            display_rows = min(days, len(momentum_df))
            momentum_display = momentum_df.tail(display_rows)
            
            # Use global top 4 symbols for momentum visibility
            top_4_momentum_symbols = top_4_symbols_global if top_4_symbols_global else symbols[:5]
            
            # Plot momentum traces
            for symbol in symbols:
                if symbol in momentum_display.columns:
                    annualized_momentum = (1 + momentum_display[symbol]) ** (252 / days) - 1
                    fig.add_trace(
                        go.Scatter(
                            x=momentum_display["Date"], y=momentum_display[symbol],
                            customdata=annualized_momentum, name=f"{symbol} Momentum", mode="lines+markers",
                            line=dict(color=colors_dict.get(symbol, "blue"), dash= "dashdot"),
                            marker=dict(size=3), legendgroup=symbol, showlegend=False,
                            visible=True if symbol in top_4_momentum_symbols else "legendonly",
                            hovertemplate=_create_hover_template(symbol, colors_dict, equity_config or {}, 
                                                              ["name"], "Momentum") + 
                                       f"<br>Momentum: %{{y:.2%}}<br>Annualized: %{{customdata:.1%}}<extra></extra>",
                        ),
                        row=idx + 1, col=2, secondary_y=False,
                    )
            
            # Add threshold line for momentum
            y1_threshold = target_return ** (days / 252) - 1
            fig.add_hline(y=y1_threshold, line_dash="dash", line_color="lightgreen", 
                         opacity=0.7, row=idx + 1, col=2)
        
        # Update axes
        fig.update_xaxes(title_text="Date", showgrid=False, row=idx + 1, col=1)
        fig.update_yaxes(title_text="Normalized Price", showgrid=False, row=idx + 1, col=1)
        
        fig.update_xaxes(title_text="Date", showgrid=False, row=idx + 1, col=2)
        fig.update_yaxes(title_text="Momentum", showgrid=False, row=idx + 1, col=2)
    
    fig.update_layout(
        height=300 * n_windows, showlegend=True, plot_bgcolor="black", paper_bgcolor="black",
        font=dict(color="white"), hovermode="closest", autosize=True,
        margin=dict(l=50, r=50, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                   font=dict(color="white"), bgcolor="black"),
    )
    
    return {
        "figure": fig, 
        "normalized_data": df_normalized, 
        "momentum_combined": momentum_combined, 
        "momentum_data": momentum_data,
        "symbols": symbols, 
        "look_back_days": look_back_days
    }


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
    symbol1: str,
    symbol2: str,
    colors_dict: Dict[str, str],
    line_styles_dict: Dict[str, str],
    equity_config: Dict[str, Dict],
    look_back_days: List[int] = None,
) -> go.Figure:
    """
    Create a price ratio plot for two symbols.
    
    Args:
        df: DataFrame with price data (columns: Date, symbol1, symbol2, ...)
        symbol1: First symbol (numerator)
        symbol2: Second symbol (denominator)
        colors_dict: Dictionary mapping symbols to colors
        line_styles_dict: Dictionary mapping symbols to line styles
        equity_config: Configuration dictionary for symbols
        look_back_days: List of lookback periods to display (not used, kept for compatibility)
        
    Returns:
        plotly.graph_objects.Figure: The ratio plot
    """
    # Use the maximum available data (all data)
    df_period = df.copy()
    
    # Calculate the ratio
    if symbol1 in df_period.columns and symbol2 in df_period.columns:
        ratio = df_period[symbol1] / df_period[symbol2]
        
        # Create hover template
        hover_template = f"<b>Date:</b>%{{x}}<br>"
        hover_template += f"<b style='color: {colors_dict.get(symbol1, 'black')}'>{symbol1}:</b>%{{customdata[0]:.2f}}<br>"
        hover_template += f"<b style='color: {colors_dict.get(symbol2, 'black')}'>{symbol2}:</b>%{{customdata[1]:.2f}}<br>"
        hover_template += f"<b>Ratio ({symbol1}/{symbol2}):</b>%{{y:.4f}}<extra></extra>"
        
        # Create the figure with secondary y-axis
        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add ratio line (left y-axis)
        fig.add_trace(
            go.Scatter(
                x=df_period["Date"],
                y=ratio,
                name=f"Ratio {symbol1}/{symbol2}",
                mode="lines+markers",
                line=dict(color=colors_dict.get(symbol1, "blue"), width=3, dash="dash"),
                marker=dict(size=5),
                customdata=list(zip(df_period[symbol1], df_period[symbol2])),
                hovertemplate=hover_template,
            ),
            secondary_y=False,
        )
        
        # Add horizontal line at ratio = 1
        fig.add_hline(
            y=1, 
            line_dash="dash", 
            line_color="gray", 
            opacity=0.5,
            annotation_text="Ratio = 1"
        )
        
        # Add individual price lines (right y-axis)
        # Symbol 1 (numerator) - solid line
        fig.add_trace(
            go.Scatter(
                x=df_period["Date"],
                y=df_period[symbol1],
                name=f"{symbol1} Price",
                mode="lines",
                line=dict(color=colors_dict.get(symbol1, "blue"), width=2),
                opacity=0.8,
            ),
            secondary_y=True,
        )
        
        # Symbol 2 (denominator) - solid line
        fig.add_trace(
            go.Scatter(
                x=df_period["Date"],
                y=df_period[symbol2],
                name=f"{symbol2} Price",
                mode="lines",
                line=dict(color=colors_dict.get(symbol2, "red"), width=2),
                opacity=0.8,
            ),
            secondary_y=True,
        )
        
        # Update layout
        fig.update_layout(
            title=f"Price Ratio Analysis: {symbol1}/{symbol2}",
            height=500,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Date",
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Ratio", secondary_y=False)
        fig.update_yaxes(title_text="Price ($)", secondary_y=True)
        
        return fig
    else:
        # Return empty figure if symbols not found
        return go.Figure().add_annotation(text="One or both symbols not found in data", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, 
                                        showarrow=False, font_size=16)