import streamlit as st
import pandas as pd
from typing import Optional
import numpy as np


def display_dataframe(
    df: pd.DataFrame,
    symbol_type: Optional[str] = None,
    data_type: Optional[str] = None,
    centered: bool = True,
    hide_index: bool = False,
    cmap: str = "RdYlGn",
    caption: Optional[str] = None,
    vmin: Optional[float] = None,
    **style_kwargs,
) -> None:
    """Display DataFrame with optional styling, centering, and vmin for colormap."""
    if symbol_type and data_type:
        styled_df = (
            df.style.set_properties(**{"font-weight": "bold"})
            .background_gradient(
                cmap=cmap,
                axis=1 if df.shape[1] > df.shape[0] else 0,
                vmin=min(vmin, np.max(df)) if vmin else None,
                vmax=np.max(df) if vmin else None
            )
            .set_caption(caption or f"{symbol_type} - {data_type}")
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
