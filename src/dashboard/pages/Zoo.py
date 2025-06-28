from src.dashboard.create_page import setup_page, show_market_performance
import hydra


if __name__ == "__main__":
    if hydra.core.global_hydra.GlobalHydra().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    with hydra.initialize(version_base=None, config_path="../../../conf"):
        config = hydra.compose(
            config_name="main",
            overrides=["+dashboard_layout=Zoo", "portfolio=porfolios_zoo"],
        )

    show_market_performance(config["tickers"], config["portfolio"], config["dashboard_layout"])
