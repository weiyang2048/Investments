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
    
    if equity_config.get(symbol, {}).get("name"):
        base_template += f"<b style='color: {colors_dict[symbol]}'>ETF Name:</b> {equity_config[symbol].get('name', '-')}<br>"
    
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
    
    # Dynamically create column names based on window sizes
    momentum_columns = [f"m{window}" for window in window_sizes]
    columns_to_keep = ["Symbol", "am"] + momentum_columns
    
    # Keep only Symbol, am, and dynamic momentum columns, then transpose
    ranking_df = ranking_df[columns_to_keep]
    return ranking_df.set_index("Symbol").T.round(2)


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
    if momentum_ranking is not None and "am" in momentum_ranking.index:
        # Get momentum ranking values and sort symbols by them (descending order)
        ranking_values = momentum_ranking.loc["am"]
        sorted_symbols = ranking_values.sort_values(ascending=False).index.tolist()
        # Keep only symbols that exist in our data
        symbols = [s for s in sorted_symbols if s in symbols]
        # Get top 4 symbols globally
        top_4_symbols_global = symbols[:4]
    
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
        visible_symbols = set(top_4_symbols_global) if top_4_symbols_global else set(symbols[:4])
        
        # Add performance traces
        for symbol in symbols:
            keys = [key for key in ["name", "region", "industry", "n_holdings"] 
                   if key in equity_config.get(symbol, {})]
            size = 6 if df_normalized.shape[0] < 30 else 1
            
            fig.add_trace(
                go.Scatter(
                    x=df_normalized["Date"], y=df_normalized[symbol],
                    name=symbol, mode="lines+markers",
                    line=dict(color=colors_dict[symbol], dash="solid"),
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
            display_rows = min(days * 3, len(momentum_df))
            momentum_display = momentum_df.tail(display_rows)
            
            # Use global top 4 symbols for momentum visibility
            top_4_momentum_symbols = top_4_symbols_global if top_4_symbols_global else symbols[:4]
            
            # Plot momentum traces
            for symbol in symbols:
                if symbol in momentum_display.columns:
                    annualized_momentum = (1 + momentum_display[symbol]) ** (252 / days) - 1
                    fig.add_trace(
                        go.Scatter(
                            x=momentum_display["Date"], y=momentum_display[symbol],
                            customdata=annualized_momentum, name=f"{symbol} Momentum", mode="lines+markers",
                            line=dict(color=colors_dict.get(symbol, "blue"), dash="dash"),
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
