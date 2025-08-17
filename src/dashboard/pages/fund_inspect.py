from yahooquery import Ticker
import yaml
import folium
import country_converter as coco
import pandas as pd
from typing import Dict
import mstarpy as ms
import streamlit as st
from streamlit_folium import st_folium
from src.data import get_fund_snap
import requests
from src.viz import create_plotly_bar_chart, create_plotly_choropleth

with open("conf/mstar.yaml", "r") as f:
    mstar_config = yaml.safe_load(f)


def create_fund_symbol_selector() -> str:
    """
    Create a hybrid symbol selector for fund inspection that allows users to:
    1. Select from predefined popular funds/ETFs
    2. Enter custom symbols manually

    Returns:
        Selected symbol string
    """
    # Predefined popular funds/ETFs
    popular_funds = {
        "Global ETFs": ["VT", "VTI", "VXUS", "EFA", "VWO", "ACWI"],
        "US ETFs": ["SPY", "QQQ", "IWM", "VTV", "VUG", "VBR"],
        "International ETFs": ["VEA", "VWO", "EFA", "EEM", "IEFA", "IEMG"],
        "Bond ETFs": ["BND", "BNDX", "AGG", "LQD", "HYG", "TLT"],
        "Sector ETFs": ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP"],
        "Commodity ETFs": ["GLD", "SLV", "USO", "UNG", "DBA", "DBC"],
    }

    # % Option 1: Select from popular categories
    selected_category = st.sidebar.selectbox(
        "Select Category:",
        list(popular_funds.keys()),
        help="Choose from predefined fund categories",
    )

    selected_fund = st.sidebar.selectbox(
        "Select Fund:",
        popular_funds[selected_category],
        help="Choose a specific fund from the selected category",
    )

    # % Option 2: Custom symbol input
    st.sidebar.subheader("Or")
    custom_symbol = st.sidebar.text_input(
        "Enter custom symbol:",
        value=selected_fund,
        placeholder="e.g., VT, VTI, AAPL",
        help="Enter any fund/ETF symbol you want to analyze",
    )

    if custom_symbol.strip():
        final_symbol = custom_symbol.strip().upper()
    else:
        final_symbol = selected_fund

    return final_symbol


def get_country_exposure(snap: dict) -> Dict[str, float]:
    """
    Get country exposure data for a given symbol using Morningstar API.

    Args:
        symbol: The fund/ETF symbol

    Returns:
        Dictionary mapping country names to exposure percentages
    """
    try:
        # Get fund data from Morningstar
        if not snap or "Portfolios" not in snap:
            return {}

        portfolios = snap.get("Portfolios", [])
        if not portfolios:
            return {}

        country_exposure = portfolios[0].get("CountryExposure", [])
        if not country_exposure:
            return {}

        breakdown_values = country_exposure[0].get("BreakdownValues", [])

        # Convert to dictionary mapping country names to percentages
        exposure_dict = {}
        for item in breakdown_values:
            country_code = item.get("Type", "")
            percentage = item.get("Value", 0.0)

            if country_code and percentage is not None:
                # Convert country code to full name
                country_name = mstar_config.get("country_code_to_name", {}).get(
                    country_code, country_code
                )
                exposure_dict[country_name] = percentage

        return exposure_dict

    except Exception as e:
        print(f"Error getting country exposure for {snap.get('Symbol')}: {e}")
        return {}


def create_country_exposure_table(exposure_df: pd.DataFrame):
    """
    Create a table showing country exposure for a given symbol.
    """
    fig = create_plotly_bar_chart(
        exposure_df,
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

    return fig


def display_country_exposure_map_streamlit(snap: dict):
    """
    Display the country exposure map directly in Streamlit.

    Args:
        symbol: The fund/ETF symbol
    """
    st.markdown(f"### Country Exposure for {snap.get('Symbol')}")

    with st.spinner(f"Loading country exposure data for {snap.get('Symbol')}..."):
        exposure_data = get_country_exposure(snap)

        if not exposure_data:
            st.warning(f"No country exposure data found for {snap.get('Symbol')}")
            return

        exposure_df = pd.DataFrame(list(exposure_data.items()), columns=["Country", "Exposure %"])
        exposure_df = exposure_df.sort_values("Exposure %", ascending=False)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(create_country_exposure_table(exposure_df), use_container_width=True)
        with col2:
            world_map = create_plotly_choropleth(
                exposure_df,
                "Country",
                "Exposure %",
                hover_name_col="Country",
                color_scale="YlOrRd",
                layout={
                    "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
                    "paper_bgcolor": "rgba(255, 255, 255, 1)",
                    "plot_bgcolor": "rgba(255, 255, 255, 1)",
                },
            )
            st.plotly_chart(world_map, use_container_width=True)
        # button show df data
        if st.button("Show Data"):
            st.write(exposure_df.to_dict(orient="records"))


def main_fund_inspect_page(selections: list):
    """
    Main function for the fund inspection page with hybrid symbol selection.
    """
    st.title("Fund Inspector")

    # % Sidebar: Symbol Selector
    selected_symbol = selections

    # Display the country exposure map
    snap = get_fund_snap(selected_symbol)

    # Display the selected fund's symbol and name
    col1, col2 = st.columns(2)
    with col1:
        basic_info = (
            f"**Selected Symbol:** {snap.get('Symbol')}  \n**Fund Name:** {snap.get('Name')}"
        )
        st.markdown(basic_info)

    with col2:
        investment_strategy = snap.get("InvestmentStrategy") or snap.get("Investment Strategy")
        if investment_strategy:
            st.markdown("#### Investment Strategy")
            st.info(investment_strategy)
        else:
            st.markdown("**No investment strategy information available.**")
    display_country_exposure_map_streamlit(snap)

    if st.button("Show All Data"):
        st.write(snap)


# Example usage and testing
if __name__ == "__main__":
    # Test with a sample symbol
    import hydra

    with hydra.initialize(version_base=None, config_path="../../../conf"):
        config = hydra.compose(
            config_name="main",
            # overrides=["+style_conf=FundInspect"],
        )
    from src.dashboard.create_page import setup_page_and_sidebar

    selections = setup_page_and_sidebar(config["style_conf"], create_fund_symbol_selector)

    test_symbol = "VT"
    print(f"Creating country exposure map for {test_symbol}...")

    main_fund_inspect_page(selections)
