if __name__ == "__main__":
    from src.dashboard.create_page import setup_page, show_market_performance
    import hydra

    with hydra.initialize(version_base=None, config_path="../../conf"):
        config = hydra.compose(
            config_name="config",
            overrides=[
                "+dashboard_layout=main",
                "portfolio=etfs_macro",
                # "~tickers.insurance_stocks",
            ],
        )

    show_market_performance(config["tickers"], config["portfolio"], config["dashboard_layout"])
