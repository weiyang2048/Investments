from conf import load_config, load_portfolios_conf, load_dashboard_conf
from src.dashboard.main import show_market_performance
import hydra

if __name__ == "__main__":
    from pyhere import here

    with hydra.initialize(version_base=None, config_path="../../../conf"):
        dashboard_config = hydra.compose(
            config_name="config", overrides=["+dashboard_layout=Sectors"]
        )["dashboard_layout"]

    etf_config = load_config("etf")
    stock_config = load_config("stock")
    equity_config = {**etf_config["etfs"], **stock_config["stocks"]}
    portfolio_config = load_portfolios_conf("sectors/portfolio")
    show_market_performance(equity_config, portfolio_config, dashboard_config)
