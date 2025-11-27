import streamlit as st
import pandas as pd
from typing import Optional
import numpy as np
import yaml
from pyhere import here

def rg0_centered(value,center=0):
    return [int(255 * abs(value - center)) if value <= center else 0, int(255 * (value - center)) if value > center else 0, 0]

def highlight_row(row, df=None):
    """Apply conditional formatting to DataFrame rows based on row type."""
    row_original = row.copy()
    min_value_abs, max_value_abs = abs(row.min()), abs(row.max())
    normalize_value_centered = [value / max_value_abs if value > 0 else value / min_value_abs for value in row]
    result = []

    if row.name in ["a", "rsi_delta"]:  # * accelerations
        for value, normalized_value in zip(row, normalize_value_centered):
            color = f"color: rgb({','.join(map(str, rg0_centered(normalized_value, center=0)))}); background-color: black; font-weight: bold;"
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
            if value > 1:
                intensity = np.clip(value / 2, 0, 1)
                color = f"background-color: rgba(255, 165, 0, {intensity:.2f}); border: 1px solid rgba(255, 165, 0, {intensity:.2f})"
            elif value < 1:
                intensity = np.clip(1 - value, 0, 1)
                color = f"background-color: rgba(46, 139, 120, {intensity:.2f}); border: 1px solid rgba(46, 139, 87, {intensity:.2f})"
            else:
                color = "background-color: white; color: black"
            result.append(color)
        return result

    elif row.name in ["avg%+", "avg_s+", "avg_s-", "avg%-"]:
        for value in row:
            if pd.isnull(value):
                result.append("")
                continue
            bg_color = "white" if row.name in ["avg%+", "avg%-"] else "black"
            text_color = f"rgb(0, {value*255}, 0)" if row.name in ["avg%+", "avg_s+"] else f"rgb({abs(value)*255}, 0, 0)"
            color = f"background-color: {bg_color}; color: {text_color}; font-weight: bold"
            result.append(color)
        return result

    elif row.name == "stride":
        for value in row:
            if pd.isnull(value):
                result.append("")
                continue
            color = (
                f"background-color: rgb({abs(value)*255 if value < 0 else 0}, {value*255 if value > 0 else 0}, 0); color: white; font-weight: bold"
            )
            result.append(color)
        return result

    elif row.name.startswith("d") and row.name[1:].isdigit():
        for value in row:
            if pd.isnull(value):
                result.append("")
                continue
            intensity = min(abs(value) * 10, 1)
            if value > 0:
                green = int(100 + intensity * 155)
                color = f"background-color: rgba(0, {green}, 0, {0.3 + intensity * 0.7}); color: white; font-weight: bold; border: 1px solid rgba(0, {green}, 0, 0.8)"
            elif value < 0:
                red = int(100 + intensity * 155)
                color = f"background-color: rgba({red}, 0, 0, {0.3 + intensity * 0.7}); color: white; font-weight: bold; border: 1px solid rgba({red}, 0, 0, 0.8)"
            else:
                color = "background-color: rgba(128, 128, 128, 0.3); color: white; font-weight: bold"
            result.append(color)
        return result

    elif row.name in ["p", "ema50", "ema200"]:
        if df is not None and "p" in df.index and "ema50" in df.index:
            for col in row_original.index:
                p_val = df.loc["p", col] if "p" in df.index else None
                ema50_val = df.loc["ema50", col] if "ema50" in df.index else None
                ema200_val = df.loc["ema200", col] if "ema200" in df.index and row.name == "ema200" else None

                if pd.isnull(row_original[col]):
                    result.append("")
                    continue

                if row.name in ["ema50"]:
                    if p_val < ema50_val:
                        color = "background-color: rgb(220, 20, 60); color: white; font-weight: bold"  # crimson
                    else:
                        color = "background-color: white; color: black; font-weight: bold"
                elif row.name == "ema200":
                    # ema200: green if ema200 < ema50, else violet if ema200 > ema50
                    if pd.isnull(ema200_val) or pd.isnull(ema50_val):
                        result.append("")
                        continue
                    if ema200_val <= ema50_val:
                        color = "background-color: white; color: black; font-weight: bold"
                    else:
                        color = "background-color: rgb(148, 0, 211); color: white; font-weight: bold"

                else:
                    color = "background-color: white; color: black; font-weight: bold"
                result.append(color)
        else:
            for value in row_original:
                result.append("" if pd.isnull(value) else "background-color: rgba(200, 200, 200, 0.3); color: black; font-weight: bold")
        return result

    elif row.name == "rsi":
        for value in row_original:
            if pd.isnull(value):
                result.append("")
                continue
            rsi_value = float(value)
            distance = abs(rsi_value - 50)
            intensity = min(distance / 50.0, 1.0)

            if rsi_value > 50.0:
                green = int(100 + intensity * 155)
                color = f"background-color: white; color: rgb(0, {green}, 0); font-weight: bold"
            elif rsi_value < 50.0:
                red = int(100 + intensity * 155)
                color = f"background-color: white; color: rgb({red}, 0, 0); font-weight: bold"
            else:
                color = "background-color: white; color: black; font-weight: bold"
            result.append(color)
        return result

    # elif row.name == "rsi_delta":
    #     for value, value in zip(row, normalize_value_centered):
    #         color = f"color: rgb({','.join(map(str, rg0_centered(value, center=0)))}); background-color: black; font-weight: bold;"
    #         result.append(color)
    #     return result

    elif row.name == "macd":
        for value in row_original:
            if pd.isnull(value):
                result.append("")
                continue
            macd_value = float(value)
            # Color based on direction: green for positive (bullish), red for negative (bearish)
            # Intensity based on magnitude
            abs_macd = abs(macd_value)
            intensity = min(abs_macd / 5.0, 1.0)  # Normalize to 0-1, assuming max MACD ~5

            if macd_value > 0:
                # Green for positive MACD (bullish)
                green = int(100 + intensity * 155)
                color = f"background-color: white; color: rgb(0, {green}, 0); font-weight: bold"
            elif macd_value < 0:
                # Red for negative MACD (bearish)
                red = int(100 + intensity * 155)
                color = f"background-color: white; color: rgb({red}, 0, 0); font-weight: bold"
            else:
                # Neutral for zero
                color = "background-color: white; color: black; font-weight: bold"
            result.append(color)
        return result

    elif row.name == "drawdown":
        for value in row_original:
            if pd.isnull(value):
                result.append("")
                continue
            red_intensity = int(np.clip(value, 0, 1) * 255)
            color = f"background-color: white; color: rgb({red_intensity}, 0, 0); font-weight: bold"
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
    """Display DataFrame with optional styling, centering, and vmin for colormap."""
    df = df.copy()

    if hide_rows:
        df = df.drop(index=hide_rows, errors="ignore")

    # Load yaml suffix mapping
    yaml_suffix_map = {}
    try:
        with open(here("conf/main.yaml"), "r") as f:
            yaml_suffix_map = yaml.safe_load(f).get("yaml_suffix_map", {})
    except Exception:
        yaml_suffix_map = {"r.yaml": ")", "f.yaml": "*"}

    # Load ticker sets for each yaml file
    yaml_ticker_sets = {}
    for yaml_file, suffix in yaml_suffix_map.items():
        try:
            with open(here(f"conf/tickers/{yaml_file}"), "r") as f:
                yaml_data = yaml.safe_load(f)
                yaml_ticker_sets[suffix] = set(yaml_data.keys()) if yaml_data else set()
        except Exception:
            yaml_ticker_sets[suffix] = set()

    # Append suffixes to column names
    def get_column_suffix(col):
        return "".join(suffix for suffix, ticker_set in yaml_ticker_sets.items() if ticker_set and col in ticker_set)

    df.columns = [f"{col}{get_column_suffix(col)}" if get_column_suffix(col) else col for col in df.columns]

    # Add prefixes based on "m" value for columns with suffixes
    def get_column_prefix(col):
        if not any(col.endswith(suffix) for suffix in yaml_suffix_map.values()) or "m" not in df.index:
            return ""
        m_value = df.loc["m", col]
        if pd.isna(m_value):
            return ""
        return "=" if m_value < 0.2 else "-" if m_value < 0.4 else ""

    df.columns = [f"{prefix}{col}" if (prefix := get_column_prefix(col)) else col for col in df.columns]

    # Style and display main table
    if symbol_type and data_type:
        styled_df = (
            df.style.set_properties(**{"font-weight": "bold"})
            .background_gradient(cmap=cmap, vmin=vmin, vmax=vmax)
            .apply(lambda row: highlight_row(row, df=df), axis=1)
            .format("{:.2f}", subset=[col for col in df.columns if df[col].dtype == "float64"])
            .format("{:.0f}", subset=[col for col in df.columns if df[col].dtype in ["int64", "int32"]])
        )
    else:
        styled_df = df

    # Display main table
    display_kwargs = {"hide_index": hide_index, "height": 35 * len(df) + 38, "width": "stretch", **style_kwargs}
    if centered:
        center_cols = st.columns([1, 6, 1])
        with center_cols[1]:
            st.dataframe(styled_df, **display_kwargs)
    else:
        st.dataframe(styled_df, **display_kwargs)

    # Display marked symbols subtable if applicable
    all_suffixes = list(yaml_suffix_map.values())
    marked_columns = [col for col in df.columns if any(col.endswith(suffix) for suffix in all_suffixes)]
    unmarked_columns = [col for col in df.columns if col not in marked_columns]

    if marked_columns and unmarked_columns:
        marked_df = df[marked_columns].copy()
        with st.expander("📌 Marked Symbols Only", expanded=False):
            if symbol_type and data_type:
                marked_styled_df = (
                    marked_df.style.set_properties(**{"font-weight": "bold"})
                    .background_gradient(cmap=cmap, vmin=vmin, vmax=vmax)
                    .apply(lambda row: highlight_row(row, df=marked_df), axis=1)
                    .format("{:.2f}", subset=[col for col in marked_df.columns if marked_df[col].dtype == "float64"])
                    .format("{:.0f}", subset=[col for col in marked_df.columns if marked_df[col].dtype in ["int64", "int32"]])
                )
            else:
                marked_styled_df = marked_df

            suffix_legend = ", ".join([f"{suffix} = {yaml_file.replace('.yaml', '')}" for yaml_file, suffix in yaml_suffix_map.items()])
            st.caption(f"Symbols marked with: {suffix_legend}")

            if centered:
                marked_center_cols = st.columns([1, 6, 1])
                with marked_center_cols[1]:
                    st.dataframe(marked_styled_df, **display_kwargs)
            else:
                st.dataframe(marked_styled_df, **display_kwargs)


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
