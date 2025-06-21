from conf import load_config, load_portfolios_conf, load_dashboard_conf
from src.dashboard.main import show_market_performance

if __name__ == "__main__":
    etf_config = load_config("etf")
    stock_config = load_config("stock")
    equity_config = {**etf_config["etfs"], **stock_config["stocks"]}
    portfolio_config = load_portfolios_conf("sectors/portfolio")
    dashboard_config = load_dashboard_conf("sectors/page")
    show_market_performance(equity_config, portfolio_config, dashboard_config)
