from re import T
from src.configurations.yaml import register_resolvers
import streamlit as st

import hydra
import numpy as np
from src.data import pivot_data
from src.performence.aggregation import aggregate_performance
from src.viz.viz import create_performance_plot, create_momentum_plot
from src.configurations.style_picker import get_random_style
from src.dashboard.create_page import setup_page_and_sidebar
import pandas as pd
from src.association import pivoted_to_corr
from src.viz.cmaps import custom_cmap


def sidebar(config):
    lenses = config["lenses"]
    lense_option = st.sidebar.radio(
        "Lense",
        lenses.keys(),
        help="Choose the lense to display\n",
        key="lense_option",
    )
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("Correlation Denoising")
    marchenko_pastur = st.sidebar.checkbox(
        "Marchenko Pastur",
        value=True,
        key="marchenko_pastur_input",
    )
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    initial_lookback_days = st.sidebar.number_input(
        "Initial Lookback Days",
        min_value=1,
        max_value=3650,
        value=7,
        step=1,
        help="Enter the initial number of days for the analysis period. / Entrez le nombre initial de jours pour la période d'analyse. (Français: 'jours de retour en arrière')",
        key="lookback_days_input",
    )
    momentum_base_factor = st.sidebar.slider(
        "Momentum Base Factor",
        min_value=1.1,
        max_value=2.5,
        value=1.5,
        step=0.1,
        help="Base factor for momentum threshold calculation. / Facteur de base pour le calcul du seuil de momentum. (Français: 'facteur de base')",
        key="momentum_base_factor_input",
    )
    lookback_factor = st.sidebar.number_input(
        "Lookback Factor",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Enter the factor to multiply the lookback days. / Entrez le facteur pour multiplier les jours de retour en arrière. (Français: 'facteur')",
        key="lookback_factor_input",
    )

    return marchenko_pastur, initial_lookback_days, lookback_factor, lense_option, momentum_base_factor


def show_market_performance(
    equity_config: dict,
    portfolio_config: dict,
    marchenko_pastur: bool = True,
    initial_lookback_days: int = 5,
    lookback_factor: int = 3,
    momentum_base_factor: float = 1.3,
) -> None:
    """Function to show the market performance dashboard."""
    # transformation = sidebar()

    # Symbol selection using tabs instead of sidebar radio
    symbol_types = [key for key in portfolio_config.keys()]
    symbol_types = symbol_types + ["Summary"]
    tabs = st.tabs(symbol_types)

    # Generate look_back_days list based on user input
    look_back_days = [int(initial_lookback_days * (lookback_factor**i)) for i in range(6)]
    summaries = dict()
    momentum_summaries = dict()
    dfs = dict()
    for i, symbol_type in enumerate(symbol_types):
        with tabs[i]:
            if symbol_type == "Summary":
                # Display performance summaries
                for idx, symbol_type in enumerate(summaries.keys()):
                    count_df = summaries[symbol_type]
                    center_cols = st.columns([1, 6, 1])
                    with center_cols[1]:
                        styled_df = (
                            count_df.style.set_properties(**{"font-weight": "bold"})
                            .background_gradient(cmap="RdYlGn", vmin=-10, vmax=20, axis=1)
                            .set_caption(f"{symbol_type} - Performance")
                        )
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)

                # Display momentum summaries
                for idx, symbol_type in enumerate(momentum_summaries.keys()):
                    momentum_df = momentum_summaries[symbol_type]
                    if not momentum_df.empty:
                        center_cols = st.columns([1, 6, 1])
                        with center_cols[1]:
                            styled_momentum = (
                                momentum_df.style.set_properties(**{"font-weight": "bold"})
                                .background_gradient(cmap="RdYlGn", vmin=0, vmax=5, axis=1)
                                .set_caption(f"{symbol_type} - Momentum")
                            )
                            st.dataframe(styled_momentum, use_container_width=True, hide_index=True)

                for symbol_type in dfs.keys():
                    df_pivot = dfs[symbol_type].copy()
                    pivoted_to_corr(df_pivot, plot=True, streamlit=True, marchenko_pastur=marchenko_pastur)
            else:
                symbols = portfolio_config[symbol_type]
                period = f"{look_back_days[-1]}d"

                # Load and process data
                df_pivot = pivot_data(list(symbols), period, streamlit=True)
                dfs[symbol_type] = df_pivot
                # Create and display plot
                colors_dict = {symbol: equity_config.get(symbol, {}).get("color", get_random_style("color")) for symbol in symbols}
                line_styles_dict = {symbol: equity_config.get(symbol, {}).get("line_style", get_random_style("line_style")) for symbol in symbols}
                fig, df_normalized, count_dict = create_performance_plot(
                    df_pivot,
                    symbols,
                    look_back_days,
                    colors_dict,
                    line_styles_dict,
                    equity_config,
                )
                count_df = pd.DataFrame(list(count_dict.items()), columns=["Symbol", "Count"])
                count_df.query("Count != 0", inplace=True)
                count_df["weight"] = count_df["Count"]
                count_df = count_df[["Symbol", "weight"]]
                count_df = count_df.sort_values(by="weight", ascending=False)
                # Transpose so symbols are columns
                count_df_t = count_df.set_index("Symbol").T
                summaries[symbol_type] = count_df_t
                momentum_fig, momentum_summary = create_momentum_plot(
                    df_pivot,
                    symbols,
                    window_sizes=[7, 30, 90, 180, 360],
                    colors_dict=colors_dict,
                    line_styles_dict=line_styles_dict,
                    equity_config=equity_config,
                    momentum_base_factor=momentum_base_factor,
                )
                # Store momentum summary for Summary tab
                momentum_summaries[symbol_type] = momentum_summary
                center_cols = st.columns([1, 6, 1])
                with center_cols[1]:
                    st.dataframe(
                        count_df_t.style.set_properties(**{"font-weight": "bold"}).background_gradient(cmap="RdYlGn", vmin=-3, vmax=3, axis=1),
                        use_container_width=True,
                        hide_index=True,
                    )
                if not momentum_summary.empty:
                    center_cols = st.columns([1, 6, 1])
                    with center_cols[1]:
                        styled_momentum = momentum_summary.style.set_properties(**{"font-weight": "bold"}).background_gradient(
                            cmap="RdYlGn", vmin=0, vmax=5, axis=1
                        )
                        st.dataframe(styled_momentum, use_container_width=True, hide_index=True)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                st.markdown("### 📈 Momentum Analysis")

                # Display momentum summary table

                st.plotly_chart(momentum_fig, use_container_width=True, config={"displayModeBar": False})

                # @ aggregations
                melt_df = df_pivot.melt(id_vars=["Date"], var_name="Symbol", value_name="Price")
                melt_df.sort_values(by=["Symbol", "Date"], inplace=True, ascending=True)
                center_cols = st.columns([1, 6, 1])

                pivoted_to_corr(df_pivot, plot=True, streamlit=True, marchenko_pastur=marchenko_pastur)

                with center_cols[1]:
                    stats_df = aggregate_performance(melt_df)
                    styled = stats_df.style.background_gradient(cmap=custom_cmap, axis=0, vmin=-0.2, vmax=0.2, gmap=None).format("{:.2%}")
                    st.dataframe(
                        styled,
                        use_container_width=True,
                        hide_index=False,
                    )


if __name__ == "__main__":

    if hydra.core.global_hydra.GlobalHydra().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_resolvers()

    with hydra.initialize(version_base=None, config_path="../../conf"):
        config = hydra.compose(config_name="main")
    marchenko_pastur, initial_lookback_days, lookback_factor, lense_option, momentum_base_factor = setup_page_and_sidebar(
        config["style_conf"], add_to_sidebar=lambda: sidebar(config)
    )
    st.title(lense_option)

    show_market_performance(
        config["tickers"], config["lenses"][lense_option], marchenko_pastur, initial_lookback_days, lookback_factor, momentum_base_factor
    )
