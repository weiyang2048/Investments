import os
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
from src.data.mstar import get_number_of_holdings
from src.viz import create_plotly_bar_chart, create_plotly_choropleth
import random
import plotly.express as px

with open("conf/mstar.yaml", "r") as f:
    mstar_config = yaml.safe_load(f)


def create_fund_symbol_selector() -> str:
    """
    Create a hybrid symbol selector for fund inspection that allows users to:
    1. Select from predefined popular funds/ETFs
    2. Enter custom symbols manually
    3. Randomly select a category and fund

    Returns:
        Selected symbol string
    """
    # Predefined popular funds/ETFs
    portfolios = dict()
    for lense in config["lenses"]:
        for category in config["lenses"][lense]:
            portfolios[category] = config["lenses"][lense][category]
    popular_funds = {portfolio: [fund for fund in portfolios[portfolio]] for portfolio in portfolios}
    # % Option 1: Select from popular categories
    fund_list = list(popular_funds.keys())

    selected_category = st.sidebar.selectbox(
        "Select Category:",
        fund_list,
        index=st.session_state.get("random_category_index", 0),
        help="Choose from predefined fund categories",
        placeholder="Select Category",
    )

    selected_fund = st.sidebar.selectbox(
        "Select Fund:",
        popular_funds[selected_category],
        index=st.session_state.get("random_fund_index", 0),
        help="Choose a specific fund from the selected category",
        placeholder="Select Fund",
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

    def random_button_click():
        random_category_index = random.choice(range(len(fund_list)))
        random_category = fund_list[random_category_index]
        random_fund_index = random.choice(range(len(popular_funds[random_category])))
        st.session_state.random_category_index = random_category_index
        st.session_state.random_fund_index = random_fund_index
        # st.rerun()

    # Add random selection button
    st.button("🎲", on_click=random_button_click, key="random_button", help="Randomly select a category and fund")

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
                country_name = mstar_config.get("country_code_to_name", {}).get(country_code, country_code)
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
        exposure_df.sort_values("Exposure %", ascending=True),
        x_col="Exposure %",
        y_col="Country",
        text="Exposure %",
        hover_data={"Country": True, "Exposure %": ":.2f"},
    )

    return fig


def display_country_exposure_map_streamlit(snap: dict):
    """
    Display the country exposure map directly in Streamlit.

    Args:
        symbol: The fund/ETF symbol
    """
    st.markdown(f"### Country Exposure for <span style='color:gold; font-weight:bold;'>{snap.get('Symbol')}</span>", unsafe_allow_html=True)

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
            )
            st.plotly_chart(world_map, use_container_width=True)


def main_fund_inspect_page(selections: list):
    """
    Main function for the fund inspection page with hybrid symbol selection.
    """
    st.title("Fund Inspector")

    # % Sidebar: Symbol Selector
    selected_symbol = selections

    # Display the country exposure map
    snap = get_fund_snap(selected_symbol)
    if not snap:
        st.error(f"No data found for {selected_symbol}")
        return
    # Display the selected fund's symbol and name
    col1, col2, col3 = st.columns(3)
    with col1:
        basic_info = (
            f"<span class='field'>Symbol : </span> <span class='data' style='color:gold; font-weight:bold;'>{snap.get('Symbol')}</span>"
            + f"<br><span class='field'>Fund Name : </span> <span class='data'>{snap.get('Name')}</span>"
            + f"<br><span class='field'>Number of Holdings : </span> <span class='data'>{get_number_of_holdings(snap)}</span>"
            + f"<br><span class='field'>Total Expense : </span> <span class='data'>{snap.get('TotalExpenseRatio')}%</span>"
            + f"<br><span class='field'>Yield : </span> <span class='data'>{snap.get('YieldHistory').get('Value')}% [{snap.get('YieldHistory').get('Type')}]</span>"
            + (
                f"<br><span class='field'>Sector : </span> <span class='data'>{snap.get('Sector').get('SectorName')} [{snap.get('Sector').get('SectorCode')}]</span>"
                if snap.get("Sector")
                else ""
            )
            + (
                f"<br><span class='field'>Industry : </span> <span class='data'>{snap.get('Industry').get('IndustryName')} [{snap.get('Industry').get('IndustryCode')}]</span>"
                if snap.get("Industry")
                else ""
            )
            + f"<br><span class='field'>Company : </span> <span class='data'>{snap.get('CompanyName', snap.get('BrandingCompanyName'))}</span>"
        )
        st.markdown(basic_info, unsafe_allow_html=True)

    with col2:
        style_box_breakdown = snap.get("Portfolios")[0].get("StyleBoxBreakdown")[0].get("BreakdownValues")
        # Prepare matrix
        matrix = [[None for _ in range(3)] for _ in range(3)]  # rows: Large/Mid/Small, cols: Value/Blend/Growth
        labels_row = ["Large", "Mid", "Small"]
        labels_col = ["Value", "Blend", "Growth"]
        for i, cell in enumerate(style_box_breakdown):
            row = i // 3
            col = i % 3
            matrix[row][col] = cell["Value"]
        style_box_df = pd.DataFrame(matrix, index=labels_row, columns=labels_col)
        st.markdown("<span class='field'>Style Box Breakdown </span>", unsafe_allow_html=True)
        st.dataframe(
            style_box_df.style.format("{:.2f}").background_gradient(cmap="Reds", axis=None),
            use_container_width=False,
            width=250,
        )

    with col3:
        investment_strategy = snap.get("InvestmentStrategy") or snap.get("Investment Strategy")
        investment_strategy = "<span style='color:lightblue'>" + investment_strategy + "</span>"
        if investment_strategy:
            st.markdown("<span class='field'>Investment Strategy </span>", unsafe_allow_html=True)
            st.markdown(investment_strategy, unsafe_allow_html=True)
        else:
            st.markdown("**No investment strategy information available.**")
    display_country_exposure_map_streamlit(snap)

    # display pandas df from records snap.get("GrowthOf10K")
    if snap.get("GrowthOf10K"):
        st.markdown("### Performances")
        growth_df = pd.DataFrame(snap.get("GrowthOf10K")[0]["HistoryDetails"])
        initial_value = growth_df.iloc[0]["Value"]
        end_value = growth_df.iloc[-1]["Value"]
        annualized_return = ((end_value / initial_value) ** (12 / len(growth_df))) - 1
        fig = px.line(
            growth_df,
            x="EndDate",
            y="Value",
            markers=True,
            title=f"Growth of $10K Over Time, Annualized Return: {annualized_return:.2%}",
            labels={"EndDate": "Date", "Value": "Value"},
            hover_data={"EndDate": True, "Value": ":.2f"},
        )

        st.plotly_chart(fig, use_container_width=True)

    if "weiya" in os.path.expanduser("~"):
        # if st.button("Show All Snapshot Data", key="show_all_snapshot_data"):
        st.write({key: snap[key] for key in random.sample(list(snap.keys()), 30) if key not in mstar_config["not_useful_keys"][0]["snapshot"]})

        if st.button("Show Config", key="show_config"):
            keys = {key: config[key].keys() for key in config.keys()}
            st.write(keys)


# Example usage and testing
if __name__ == "__main__":
    # Test with a sample symbol
    import hydra
    from src.configurations.yaml import register_resolvers

    if hydra.core.global_hydra.GlobalHydra().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_resolvers()

    with hydra.initialize(version_base=None, config_path="../../../conf"):
        config = hydra.compose(
            config_name="main",
            # overrides=["+style_conf=FundInspect"],
            # overrides=["lenses=[regional, sectoral, zoo]"],
        )
    from src.dashboard.create_page import setup_page_and_sidebar

    selections = setup_page_and_sidebar(config["style_conf"], create_fund_symbol_selector)
    portfolios = config["lenses"]
    popular_funds = {portfolio: [fund for fund in portfolios[portfolio]] for portfolio in portfolios}

    main_fund_inspect_page(selections)
