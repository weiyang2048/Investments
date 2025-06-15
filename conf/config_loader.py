"""
Configuration loader module for the Investments project.
Handles loading and parsing of YAML configuration files.
"""

import os
import yaml
from pathlib import Path
from typing import List


def load_config(config_name: str) -> dict:
    """
    Load a YAML configuration file from the conf directory.

    Args:
        config_name (str): Name of the configuration file (without .yaml extension)

    Returns:
        dict: Configuration data
    """
    conf_dir = Path(__file__).parent
    config_path = conf_dir / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {config_name}.yaml not found in {conf_dir}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_portfolios_conf(portfolio_name: str) -> dict:
    conf_dir = Path(__file__).parent
    portfolio_path = conf_dir / f"{portfolio_name}.yaml"
    if not portfolio_path.exists():
        raise FileNotFoundError(f"Portfolio file {portfolio_name}.yaml not found in {conf_dir}")
    with open(portfolio_path, "r") as f:
        return yaml.safe_load(f)


def load_dashboard_conf(dashboard_name: str) -> dict:
    conf_dir = Path(__file__).parent
    dashboard_path = conf_dir / f"{dashboard_name}.yaml"
    if not dashboard_path.exists():
        raise FileNotFoundError(f"Dashboard file {dashboard_name}.yaml not found in {conf_dir}")
    with open(dashboard_path, "r") as f:
        return yaml.safe_load(f)


def get_symbols(symbol_type: str, portfolio_config: dict) -> List[str]:
    """
    Get symbols based on the selected type.

    Args:
        symbol_type: Type of symbols to retrieve ("Markets", "Sectors", or "Regions")
        etf_config: ETF configuration dictionary

    Returns:
        List of symbol strings
    """
    return portfolio_config[symbol_type]
