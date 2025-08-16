from src.dashboard.create_page import setup_page, show_market_performance
import hydra
import streamlit as st

if __name__ == "__main__":
    if hydra.core.global_hydra.GlobalHydra().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    with hydra.initialize(version_base=None, config_path="../../../conf"):
        config = hydra.compose(
            config_name="main",
            overrides=[
                # "+style_conf=Zoo",
                "portfolio=porfolios_zoo"
            ],
        )
    st.title("Zoo")
    show_market_performance(config["tickers"], config["portfolio"], config["style_conf"])
