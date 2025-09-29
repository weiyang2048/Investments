import streamlit as st
import pandas as pd
from typing import Optional
from src.viz.cmaps import custom_cmap


def display_dataframe(
    df: pd.DataFrame,
    symbol_type: Optional[str] = None,
    data_type: Optional[str] = None,
    centered: bool = True,
    hide_index: bool = False,
    cmap: str = "RdYlGn",
    caption: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    **style_kwargs,
) -> None:
    """
    Unified helper function to display DataFrames with flexible styling options.

    Args:
        df: DataFrame to display
        symbol_type: Type of symbol (e.g., "ETF", "Stock") for caption generation
        data_type: Type of data (e.g., "Performance", "Momentum") for caption generation
        centered: Whether to center the DataFrame in a column layout
        hide_index: Whether to hide the DataFrame index
        check_empty: Whether to skip display if DataFrame is empty
        cmap: Color map for background gradient
        caption: Custom caption (overrides symbol_type + data_type)
        width: Width of the DataFrame display
        height: Height of the DataFrame display
        **style_kwargs: Additional arguments passed to st.dataframe()

    Returns:
        None
    """

    # Determine if we need to create styling or use pre-styled DataFrame
    if symbol_type and data_type:
        # Create styling for the DataFrame
        styled_df = (
            df.style.set_properties(**{"font-weight": "bold"})
            .background_gradient(cmap=cmap, axis=1)
            .set_caption(caption or f"{symbol_type} - {data_type}")
        )
    else:
        # Use the DataFrame as-is (assumes it's already styled)
        styled_df = df

    # Display with or without centering
    if centered:
        center_cols = st.columns([1, 6, 1])
        with center_cols[1]:
            _display_dataframe_with_size(styled_df, hide_index, width, height, **style_kwargs)
    else:
        _display_dataframe_with_size(styled_df, hide_index, width, height, **style_kwargs)


def _display_dataframe_with_size(
    df,
    hide_index: bool = False,
    width: Optional[int] = None,
    height: Optional[int] = None,
    **style_kwargs,
) -> None:
    """Helper function to display DataFrame with optional size constraints."""
    if width or height:
        # Use container_width and container_height for size control
        st.dataframe(df, hide_index=hide_index, use_container_width=True if width is None else False, **style_kwargs)
    else:
        st.dataframe(df, hide_index=hide_index, **style_kwargs)


def display_table_of_contents(sections: Optional[list] = None) -> None:

    st.markdown("---")
    st.markdown("<h2>Table of Contents</h2>", unsafe_allow_html=True)

    # Create columns dynamically based on number of sections
    num_sections = len(sections)
    if num_sections <= 3:
        cols = st.columns(num_sections)
    else:
        # For more than 3 sections, use 3 columns and wrap
        cols = st.columns(3)

    for i, section in enumerate(sections):
        col_idx = i % 3 if num_sections > 3 else i
        with cols[col_idx]:
            st.markdown(f"- [{section}](#{section.lower()})")


def display_section_header(header: str, anchor: Optional[str] = None) -> None:
    """
    Display a section header with optional anchor for navigation.

    Args:
        header: Header text to display
        anchor: Optional anchor ID for navigation
    """
    anchor = anchor or header
    st.markdown(f"<h3 id='{anchor.lower()}'>{header}</h3>", unsafe_allow_html=True)
