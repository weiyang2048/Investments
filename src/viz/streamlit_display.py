import streamlit as st
import pandas as pd
from typing import Optional
import numpy as np
from functools import partial


def highlight_row(row, row_name: str):
    if row.name == row_name:
        max_abs_value = np.max(abs(row))
        row = row / max_abs_value
        result = []
        for value in row:
            try:
                x = abs(value) 
            except Exception:
                x = 0
            if pd.isnull(value):
                result.append('')
                continue
            if value <= 0:
                color = f'background-color: rgba(0, 0, 255, {x:.2f})'
            elif value > 0:
                color = f'background-color: rgba(255, 0, 255, {x:.2f})'
            else:
                color = ''
            result.append(color)
        return result
    else:
        return [''] * len(row)

def display_dataframe(
    df: pd.DataFrame,
    symbol_type: Optional[str] = None,
    data_type: Optional[str] = None,
    centered: bool = True,
    hide_index: bool = False,
    cmap: str = "RdYlGn",
    vmin: Optional[float] = None,
    **style_kwargs,
) -> None:
    """Display DataFrame with optional styling, centering, and vmin for colormap."""
    df = df.copy()
    if 'avg_a' in df.index:
        df.loc['avg_a'] = df.loc['avg_a']
    if symbol_type and data_type:
        styled_df = (
            df.style.set_properties(**{"font-weight": "bold"})
            .background_gradient(
                cmap=cmap,
                vmin=min(vmin, np.max(df)) if vmin else None,
                vmax=np.max(df) if vmin else None
            )
            .apply(partial(highlight_row, row_name="avg_a"), axis=1)
            .format("{:.2}", subset=[col for col in df.columns if df[col].dtype == "float64"])
            .format("{:.0f}", subset=[col for col in df.columns if df[col].dtype == "int64"])
        )

    else:
        styled_df = df

    if centered:
        center_cols = st.columns([1, 6, 1])
        with center_cols[1]:
            st.dataframe(styled_df, hide_index=hide_index, **style_kwargs)
    else:
        st.dataframe(styled_df, hide_index=hide_index, **style_kwargs)


def display_table_of_contents(sections: Optional[list] = None) -> None:
    """Display table of contents with sections."""
    st.markdown("---")
    st.markdown("<h2>Table of Contents</h2>", unsafe_allow_html=True)
    
    cols = st.columns(min(len(sections), 3))
    for i, section in enumerate(sections):
        with cols[i % 3]:
            st.markdown(f"- [{section}](#{section.lower()})")


def display_section_header(header: str, anchor: Optional[str] = None) -> None:
    """Display a section header with optional anchor for navigation."""
    anchor = anchor or header
    st.markdown(f"<h3 id='{anchor.lower()}'>{header}</h3>", unsafe_allow_html=True)
