from src.create_page import setup_page, show_market_performance
import hydra
from conf.config_loader import load_config, load_portfolios_conf


if __name__ == "__main__":

    with hydra.initialize(version_base=None, config_path="../../../conf"):
        dashboard_config = hydra.compose(
            config_name="config", overrides=["+dashboard_layout=Sectors"]
        )["dashboard_layout"]

    etf_config = load_config("etf")
    stock_config = load_config("stock")
    equity_config = {**etf_config["etfs"], **stock_config["stocks"]}
    portfolio_config = load_portfolios_conf("sectors/portfolio")
    show_market_performance(equity_config, portfolio_config, dashboard_config)
