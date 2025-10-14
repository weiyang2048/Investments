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


def _get_default_colors(symbols: List[str]) -> Dict[str, str]:
    """Get default colors for symbols."""
    return {symbol: f"hsl({i*360/len(symbols)}, 70%, 50%)" for i, symbol in enumerate(symbols)}


def _get_default_line_styles(symbols: List[str]) -> Dict[str, str]:
    """Get default line styles for symbols."""
    return {symbol: "solid" for symbol in symbols}


def create_performance_plot(
    df: pd.DataFrame,
    symbols: List[str],
    look_back_days: List[int],
    colors_dict: Dict[str, str],
    line_styles_dict: Dict[str, str],
    equity_config: Dict[str, Dict],
    transformation: Callable[[float], float] = lambda x: x,
) -> Dict[str, any]:
    """Create a multi-subplot figure showing normalized performance."""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[f"{days} trading days" for days in look_back_days],
        vertical_spacing=0.1, horizontal_spacing=0.03,
    )
    
    for i, days in enumerate(look_back_days):
        df_normalized = normalize_prices(df.iloc[-days:])[["Date"] + symbols]
        stats = Stats(df_normalized)
        current_ratios = list(stats.ratios(transformation))
        
        # Get visible symbols (top 3 + negative)
        symbols_in_plot = list(df_normalized.columns[1:])
        top_3_symbols = [symbols_in_plot[j] for j in np.argsort(current_ratios)[-1:-4:-1]]
        negative_symbols = [symbols_in_plot[j] for j in np.argsort(current_ratios) if current_ratios[j] < 0]
        visible_symbols = set(top_3_symbols + negative_symbols)
        
        # Add traces for each symbol
        for symbol in symbols:
            keys = [key for key in ["name", "region", "industry", "n_holdings"] 
                   if key in equity_config.get(symbol, {})]
            size = 6 if df_normalized.shape[0] < 30 else 1
            
            fig.add_trace(
                go.Scatter(
                    x=df_normalized["Date"], y=df_normalized[symbol],
                    name=symbol, mode="lines+markers",
                    line=dict(color=colors_dict[symbol], dash=line_styles_dict[symbol]),
                    marker=dict(color=colors_dict[symbol], size=size),
                    legendgroup=symbol, showlegend=(i == 0),
                    visible=True if symbol in visible_symbols else "legendonly",
                    hovertemplate=_create_hover_template(symbol, colors_dict, equity_config, keys),
                ),
                row=(i // 2) + 1, col=(i % 2) + 1,
            )
            fig.update_xaxes(showgrid=False, row=i // 2 + 1, col=i % 2 + 1)
            fig.update_yaxes(showgrid=False, row=i // 2 + 1, col=i % 2 + 1)
        
        # Add annotation with performance summary
        top_3_styled = [f'<span style="color: {colors_dict[symbol]}">{symbol}</span>' 
                       for symbol in top_3_symbols]
        negative_styled = [f'<span style="color: {colors_dict[symbol]}">{symbol}</span>' 
                          for symbol in negative_symbols]
        
        ratio_text = "".join([
            f"<span style='color: {colors_dict[symbol]}'>{ratio*100:.0f}</span>"
            f"{'<span style=\"color: snow; opacity: 0.3\">|</span><br>' if (k+1) % 6 == 0 else ''}"
            for k, symbol, ratio in zip(range(len(symbols_in_plot)), symbols_in_plot, current_ratios)
        ])
        
        annotations = (
            "<span style='color: snow; opacity: 0.3'>|</span>" + ratio_text +
            f"{'<span style=\"color: snow; opacity: 0.3\">|</span><br>' if len(symbols_in_plot) % 6 != 0 else ''}"
            f"{' + '.join(top_3_styled)}<br>{' - ' if negative_styled else ''}{' - '.join(negative_styled)}<br>"
        )
        
        fig.add_annotation(
            x=min(df_normalized["Date"]), y=min(df_normalized.iloc[-1, 1:]),
            text=annotations,
            font=dict(color="lightgreen" if stats.avg_return > 0 else "coral", size=10),
            opacity=1, bgcolor="black", xanchor="left", yanchor="bottom",
            showarrow=False, row=i // 2 + 1, col=i % 2 + 1,
        )
    
    fig.update_layout(
        height=800, showlegend=True, plot_bgcolor="black", paper_bgcolor="black",
        font=dict(color="white"), hovermode="closest", autosize=True,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5,
                   font=dict(color="white"), bgcolor="black"),
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
    """Create a momentum plot showing momentum and renormalized prices for different window sizes."""
    df_norm = normalize_prices(df)
    momentum_data, momentum_combined = compute_momentum(df_norm, window_sizes, target_return=target_return)
    
    colors_dict = colors_dict or _get_default_colors(symbols)
    line_styles_dict = line_styles_dict or _get_default_line_styles(symbols)
    
    n_windows = len(window_sizes)
    fig = make_subplots(
        rows=n_windows, cols=1,
        subplot_titles=[f"Momentum and Price (window={window} days)" for window in window_sizes],
        vertical_spacing=0.05,
        specs=[[{"secondary_y": True}] for _ in range(n_windows)],
    )

    for idx, window in enumerate(window_sizes):
        momentum_df = momentum_data[window]
        display_rows = min(window * 3, len(momentum_df))
        momentum_display = momentum_df.tail(display_rows)

        # Get top 3 symbols by momentum
        valid_symbols = [s for s in symbols if s in momentum_display.columns]
        momentum_values = [momentum_display[s].iloc[-1] for s in valid_symbols]
        top_3_symbols = [valid_symbols[i] for i in np.argsort(momentum_values)[-3:]] if len(valid_symbols) >= 3 else valid_symbols

        # Plot momentum traces
        for symbol in symbols:
            if symbol in momentum_display.columns:
                annualized_momentum = (1 + momentum_display[symbol]) ** (252 / window) - 1
                fig.add_trace(
                    go.Scatter(
                        x=momentum_display["Date"], y=momentum_display[symbol],
                        customdata=annualized_momentum, name=symbol, mode="lines+markers",
                        line=dict(color=colors_dict.get(symbol, "blue"), dash="solid"),
                        marker=dict(size=3), legendgroup=symbol, showlegend=(idx == 0),
                        visible=True if symbol in top_3_symbols else "legendonly",
                        hovertemplate=_create_hover_template(symbol, colors_dict, equity_config or {}, 
                                                          ["name"], "Momentum") + 
                                   f"<br>Momentum: %{{y:.2%}}<br>Annualized: %{{customdata:.1%}}<extra></extra>",
                    ),
                    row=idx + 1, col=1, secondary_y=False,
                )

        # Add threshold line
        y1_threshold = target_return ** (window / 252) - 1
        fig.add_hline(y=y1_threshold, line_dash="dash", line_color="lightgreen", 
                     opacity=0.7, row=idx + 1, col=1)

        # Plot renormalized price traces
        for symbol in symbols:
            if symbol in df.columns:
                price_window = df[["Date", symbol]].tail(display_rows).copy()
                if not price_window[symbol].isnull().all():
                    first_valid = price_window[symbol].first_valid_index()
                    if first_valid is not None:
                        base = price_window.loc[first_valid, symbol]
                        price_window["renorm"] = price_window[symbol] / base
                    else:
                        price_window["renorm"] = price_window[symbol]
                else:
                    price_window["renorm"] = price_window[symbol]
                
                fig.add_trace(
                    go.Scatter(
                        x=price_window["Date"], y=price_window["renorm"],
                        name=f"{symbol} Renorm Price", mode="lines",
                        line=dict(color=colors_dict.get(symbol, "blue"), dash="longdash"),
                        opacity=0.7, legendgroup=symbol, showlegend=False,
                        visible=True if symbol in top_3_symbols else "legendonly",
                        hovertemplate=_create_hover_template(symbol, colors_dict, equity_config or {}, 
                                                          ["name"], "Price") + 
                                   f"<br>Price: %{{y:.2f}}<extra></extra>",
                    ),
                    row=idx + 1, col=1, secondary_y=True,
                )

        # Update axes
        fig.update_xaxes(title_text="Date", showgrid=False, row=idx + 1, col=1)
        fig.update_yaxes(title_text="Momentum", showgrid=False, secondary_y=False, row=idx + 1, col=1)
        fig.update_yaxes(title_text="Price", showgrid=False, secondary_y=True, row=idx + 1, col=1)

    fig.update_layout(
        height=300 * n_windows, showlegend=True, plot_bgcolor="black", paper_bgcolor="black",
        font=dict(color="white"), hovermode="closest", autosize=True,
        margin=dict(l=50, r=50, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                   font=dict(color="white"), bgcolor="black"),
    )

    return {"figure": fig, "momentum_combined": momentum_combined, "momentum_data": momentum_data, "window_sizes": window_sizes}


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
