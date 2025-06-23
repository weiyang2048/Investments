import streamlit as st
import pandas as pd
from typing import List
from src.data import get_daily_prices_list

from src.viz import create_performance_plot


def setup_page(dashboard_config: dict) -> None:
    """Configure the Streamlit page settings."""
    st.set_page_config(
        **dashboard_config["page_config"],
    )
    st.title(dashboard_config["title"])
    # st.markdown(dashboard_config["description"])

    # st.sidebar.header("Controls")
    st.markdown(dashboard_config["style_string"], unsafe_allow_html=True)

    # add a link to my page on the sidebar
    st.sidebar.markdown(
        "<a id='homepage-link' href='https://www.noWei.us' target='_blank'>Homepage:  <b><i>noWei.us</i></b></a>",
        unsafe_allow_html=True,
    )
    # link to my linkedin page
    st.sidebar.markdown(
        "<a id='linkedin-link'  href='https://www.linkedin.com/in/weiyang2048/' target='_blank'>LinkedIn: <b><i>weiyang2048</i></b></a>",
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data(symbols: List[str], period: str = "10y") -> pd.DataFrame:
    """
    Load and pivot price data for the given symbols.

    Args:
        symbols: List of symbols to load data for
        period: Time period to load (e.g., "1y", "5y")

    Returns:
        Pivoted DataFrame with dates as index and symbols as columns
    """
    df = get_daily_prices_list(symbols, period)
    df.reset_index(inplace=True)
    return df.pivot(index="Date", columns="Symbol", values="Close").reset_index()


def show_market_performance(
    equity_config: dict, portfolio_config: dict, dashboard_config: dict
) -> None:
    """Function to show the market performance dashboard."""
    setup_page(dashboard_config)

    # Symbol selection using tabs instead of sidebar radio
    symbol_types = [key for key in portfolio_config.keys()]
    tabs = st.tabs(symbol_types)

    # Create content for each tab
    for i, symbol_type in enumerate(symbol_types):
        with tabs[i]:
            symbols = portfolio_config[symbol_type]
            period = "7y"

            # Load and process data
            df_pivot = load_data(list(symbols), period)

            # Create and display plot
            look_back_days = dashboard_config["look_back_days"]
            colors_dict = {
                symbol: equity_config.get(symbol, {}).get("color", "snow") for symbol in symbols
            }
            line_styles_dict = {
                symbol: equity_config.get(symbol, {}).get("line_style", "solid")
                for symbol in symbols
            }
            fig = create_performance_plot(
                df_pivot, symbols, look_back_days, colors_dict, line_styles_dict, equity_config
            )
            st.plotly_chart(fig, use_container_width=True)

            # Display raw data option
            if st.checkbox("Show Raw Data", key=f"raw_data_{symbol_type}"):
                st.dataframe(df_pivot)


def show_geographical_analysis(
    equity_config: dict, portfolio_config: dict, dashboard_config: dict
) -> None:
    """Function to show geographical analysis with Folium maps."""
    setup_page(dashboard_config)

    # Symbol selection using tabs instead of sidebar radio
    symbol_types = [key for key in portfolio_config.keys()]
    symbol_tabs = st.tabs(symbol_types)

    # Create content for each symbol type tab
    for i, symbol_type in enumerate(symbol_types):
        with symbol_tabs[i]:
            symbols = portfolio_config[symbol_type]
            period = "7y"

            # Load and process data
            df_pivot = load_data(list(symbols), period)

            # Colors dictionary
            colors_dict = {
                symbol: equity_config.get(symbol, {}).get("color", "snow") for symbol in symbols
            }

            # Create tabs for different map views
            tab1, tab2 = st.tabs(["📍 Individual Symbols", "🌍 Regional Performance"])

            with tab1:
                st.subheader("Geographical Distribution of Investments")
                st.markdown("Interactive map showing individual symbol locations and performance")

                # Create geographical map
                geo_map = create_geographical_map(df_pivot, symbols, equity_config, colors_dict)
                folium_static(geo_map, width=800, height=600)

                # Add some statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Symbols", len(symbols))
                with col2:
                    regions = set(
                        equity_config.get(symbol, {}).get("region", "Global") for symbol in symbols
                    )
                    st.metric("Regions Covered", len(regions))
                with col3:
                    industries = set(
                        equity_config.get(symbol, {}).get("industry", "N/A") for symbol in symbols
                    )
                    st.metric("Industries", len(industries))

            with tab2:
                st.subheader("Regional Performance Overview")
                st.markdown("Choropleth-style map showing average performance by region")

                # Create regional performance map
                regional_map = create_regional_performance_map(df_pivot, symbols, equity_config)
                folium_static(regional_map, width=800, height=600)

                # Regional performance summary
                st.subheader("Regional Performance Summary")

                # Calculate regional performance
                region_performance = {}
                for symbol in symbols:
                    if symbol in equity_config and symbol in df_pivot.columns:
                        region = equity_config[symbol].get("region", "Global")
                        recent_data = df_pivot[symbol].dropna()

                        if len(recent_data) >= 30:
                            performance = (recent_data.iloc[-1] / recent_data.iloc[-30] - 1) * 100

                            if region not in region_performance:
                                region_performance[region] = []
                            region_performance[region].append(performance)

                # Display regional performance table
                if region_performance:
                    performance_df = pd.DataFrame(
                        [
                            {
                                "Region": region,
                                "Avg Performance (%)": sum(perfs) / len(perfs),
                                "Symbols Count": len(perfs),
                                "Best Performer": max(perfs),
                                "Worst Performer": min(perfs),
                            }
                            for region, perfs in region_performance.items()
                        ]
                    )
                    performance_df = performance_df.sort_values(
                        "Avg Performance (%)", ascending=False
                    )
                    st.dataframe(performance_df, use_container_width=True)

    # add a link to my page on the sidebar
    st.sidebar.markdown(
        "<a id='homepage-link' href='https://www.noWei.us' target='_blank'>Homepage:  <i>noWei.us</i></a>",
        unsafe_allow_html=True,
    )
    # link to my linkedin page
    st.sidebar.markdown(
        "<a id='linkedin-link'  href='https://www.linkedin.com/in/weiyang2048/' target='_blank'>LinkedIn: <i>weiyang2048</i></a>",
        unsafe_allow_html=True,
    )
