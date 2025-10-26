import streamlit as st
import pandas as pd
from typing import Optional
import numpy as np
from functools import partial


def highlight_row(row):
    max_abs_value = np.max(abs(row))
    if "pcr_m1" in row.name:
        row = row
    else:
        row = row / max_abs_value

    result = []
    if row.name == "a":
        for value in row:
            if pd.isnull(value):
                result.append("")
                continue
            if value <= 0:
                color = f"background-color: rgba(0, 0, 255, {abs(value):.2f}); color: white; font-weight: bold;"
            elif value > 0:
                color = f"background-color: rgba(255, 0, 255, {abs(value):.2f}); color: white; font-weight: bold;"
            else:
                color = ""
            result.append(color)
        return result
    elif row.name in ["s0", "s1", "s2"]:
        for value in row:
            if pd.isnull(value):
                result.append("")
                continue
            color = f"background-color: white; color: rgba({abs(value)*255 if value < 0 else 0}, {value*255 if value > 0 else 0}, 0, {0.5+abs(value)/2:.2f});"
            result.append(color)
        return result
    elif row.name == "pcr_m1":
        for value in row:
            if pd.isnull(value):
                result.append("")
                continue
            # Normalize value between 0 and 1 for color intensity (assuming typical range 0-3)
            if value > 1:
                # Orange for bearish sentiment (more puts than calls)
                color = f"background-color: rgba(255, 165, 0, {np.clip(value/2, 0, 1):.2f}); border: 1px solid rgba(255, 165, 0, {np.clip(value/2, 0, 1):.2f})"
            elif value < 1:
                # Seagreen for bullish sentiment (more calls than puts)
                color = f"background-color: rgba(46, 139, 120, {np.clip(1-value, 0, 1):.2f}); border: 1px solid rgba(46, 139, 87, {np.clip(1-value/2, 0, 1):.2f})"
            elif value == 1:
                color = f"background-color: white; color: black"
            else:
                color = ""
            result.append(color)
        return result
    elif row.name in ["avg%+", "avg_s+", "avg_s-", "avg%-"]:
        for value in row:
            if pd.isnull(value):
                result.append("")
                continue
            background_color = "white" if row.name in ["avg%+", "avg%-"] else "black"
            text_color = f"rgb(0, {value*255}, 0)" if row.name in ["avg%+", "avg_s+"] else f"rgb({abs(value)*255}, 0, 0)"
            color = f"background-color: {background_color}; color: {text_color}; font-weight: bold"
            result.append(color)
        return result
    elif row.name == "stride":
        for value in row:
            if pd.isnull(value):
                result.append("")
                continue
            color = f"background-color: rgb({abs(value)*255 if value < 0 else 0}, {value*255 if value > 0 else 0}, 0); color: white; font-weight: bold"
            result.append(color)
        return result
    elif row.name.startswith("d") and row.name[1:].isdigit():  # Handle d6, d7, etc.
        for value in row:
            if pd.isnull(value):
                result.append("")
                continue
            # Normalize value for color intensity (assuming typical range -10% to +10%)
            intensity = min(abs(value) *10, 1)  # Cap at 1 for very large values
            
            if value > 0:
                # Green gradient for positive changes
                color = f"background-color: rgba(0, {int(100 + intensity * 155)}, 0, {0.3 + intensity * 0.7}); color: white; font-weight: bold; border: 1px solid rgba(0, {int(100 + intensity * 155)}, 0, 0.8)"
            elif value < 0:
                # Red gradient for negative changes
                color = f"background-color: rgba({int(100 + intensity * 155)}, 0, 0, {0.3 + intensity * 0.7}); color: white; font-weight: bold; border: 1px solid rgba({int(100 + intensity * 155)}, 0, 0, 0.8)"
            else:
                # Neutral color for zero change
                color = f"background-color: rgba(128, 128, 128, 0.3); color: white; font-weight: bold"
            result.append(color)
        return result
    else:
        return [""] * len(row)


def display_dataframe(
    df: pd.DataFrame,
    symbol_type: Optional[str] = None,
    data_type: Optional[str] = None,
    centered: bool = True,
    hide_index: bool = False,
    hide_rows: Optional[list] = None,
    cmap: str = "RdYlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    **style_kwargs,
) -> None:
    """Display DataFrame with optional styling, centering, and vmin for colormap.
    
    Args:
        df: DataFrame to display
        symbol_type: Type of symbols (for styling)
        data_type: Type of data (for styling)
        centered: Whether to center the display
        hide_index: Whether to hide the DataFrame index
        hide_rows: List of row names to hide (optional, no rows hidden by default)
        cmap: Colormap for background gradient
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        **style_kwargs: Additional styling arguments
    """
    df = df.copy()
    
    # Hide specified rows if provided
    if hide_rows:
        df = df.drop(index=hide_rows, errors='ignore')
    
    if symbol_type and data_type:
        styled_df = (
            df.style.set_properties(**{"font-weight": "bold"})
            .background_gradient(cmap=cmap, vmin=vmin, vmax=vmax)
            .apply(partial(highlight_row), axis=1)
            .format("{:.2f}", subset=[col for col in df.columns if df[col].dtype == "float64"])
            .format("{:.0f}", subset=[col for col in df.columns if df[col].dtype == "int64" or df[col].dtype == "int32"])
        )

    else:
        styled_df = df

    if centered:
        center_cols = st.columns([1, 6, 1])
        with center_cols[1]:
            st.dataframe(styled_df, hide_index=hide_index, height=35 * len(df) + 38, width="stretch", **style_kwargs)
    else:
        st.dataframe(styled_df, hide_index=hide_index, height=35 * len(df) + 38, width="stretch", **style_kwargs)


def display_table_of_contents(sections: Optional[list] = None) -> None:
    """Display table of contents with sections."""
    st.markdown("<h2>Table of Contents</h2>", unsafe_allow_html=True)

    cols = st.columns(min(len(sections), 3))
    for i, section in enumerate(sections):
        with cols[i % 3]:
            st.markdown(f"- [{section}](#{section.lower()})")


def display_section_header(header: str, anchor: Optional[str] = None) -> None:
    """Display a section header with optional anchor for navigation."""
    anchor = anchor or header
    st.markdown(f"<h3 id='{anchor.lower()}'>{header}</h3>", unsafe_allow_html=True)
