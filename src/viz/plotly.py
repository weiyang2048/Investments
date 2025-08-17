import numpy as np
import plotly.express as px
import pandas as pd


def create_plotly_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    text: str,
    hover_data: dict,
    layout: dict = None,
):
    """
    Create a bar chart using Plotly Express.
    #  Example:
    ```python
    fig = create_plotly_bar_chart(
        df,
        x_col="Country",
        y_col="Exposure %",
        text="Exposure %",
        hover_data={"Country": True, "Exposure %": ":.2f"},
        layout={
            "xaxis_tickangle": -45,
            "yaxis_title": "Exposure (%)",
            "xaxis_title": "Country",
            "margin": {"l": 20, "r": 20, "t": 40, "b": 80},
            "height": 400,
        },
    )
    ```
    """
    fig = px.bar(df, x=x_col, y=y_col, text=text, hover_data=hover_data)
    hover_template = "<b>%{x}</b><br>{y_col}: %{y:.2f}<extra></extra>".replace("{y_col}", y_col)
    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        hovertemplate=hover_template,
    )
    if layout:
        fig.update_layout(layout)
    return fig


def create_plotly_choropleth(
    df: pd.DataFrame,
    locations_col: str,
    color_col: str,
    hover_name_col: str = None,
    color_scale: str = "YlOrRd",
    projection: str = "natural earth",
    layout: dict = None,
    locationmode: str = "country names",
    colorbar_title: str = "Value",
    log_scale: bool = True,
    lower_bound: float = 0,
):
    """
    Create a Plotly choropleth map.

    Args:
        df: DataFrame containing the data.
        locations_col: Column with country names or codes.
        color_col: Column with values to color by.
        hover_name_col: Column to use for hover labels (optional).
        color_scale: Color scale for the map.
        projection: Map projection.
        layout: Optional dict for layout updates.
        locationmode: 'country names', 'ISO-3', or 'ISO-2'.
        colorbar_title: Title for the colorbar.

    Returns:
        Plotly Figure object.
    """
    df_copy = df.loc[df[color_col] > lower_bound].copy()

    if log_scale:
        df_copy[color_col] = df_copy[color_col] + (1 - df_copy[color_col].min()) + 0.00001
        df_copy[color_col + "_log"] = np.log(df_copy[color_col])
    custom_data = [color_col + "_log" if log_scale else color_col]
    hover_template = (
        "<b>%{hover_name}</b><br>%{custom_data[0]}: %{custom_data[0]:.2f}<extra></extra>"
    )
    fig = px.choropleth(
        df_copy,
        locations=locations_col,
        color=color_col + "_log" if log_scale else color_col,
        hover_name=hover_name_col,
        color_continuous_scale=color_scale,
        projection=projection,
        locationmode=locationmode,
        custom_data=custom_data,
    )
    fig.update_traces(hovertemplate=hover_template)
    # no colorbar
    # fig.update_layout(coloraxis_showscale=False)

    if layout:
        fig.update_layout(layout)
    return fig
