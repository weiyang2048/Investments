from src.configurations.yaml import register_resolvers
import streamlit as st

register_resolvers()

if __name__ == "__main__":
    from src.dashboard.create_page import setup_page, show_market_performance
    import hydra

    if hydra.core.global_hydra.GlobalHydra().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    with hydra.initialize(version_base=None, config_path="../../conf"):
        config = hydra.compose(
            config_name="main",
            overrides=[
                # "+style_conf=main",
                "portfolio=regions",
                # "~tickers.insurance_stocks",
            ],
        )
    # OmegaConf.resolve(config)
    st.title("Regions")
    show_market_performance(config["tickers"], config["portfolio"], config["style_conf"])
