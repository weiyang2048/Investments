"""
Configuration loader module for the Investments project.
Handles loading and parsing of YAML configuration files.
"""

import os
import yaml
from pathlib import Path


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
        raise FileNotFoundError(
            f"Configuration file {config_name}.yaml not found in {conf_dir}"
        )

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_portfolios(portfolio_name: str) -> dict:
    conf_dir = Path(__file__).parent
    portfolio_path = conf_dir / f"{portfolio_name}.yaml"
    if not portfolio_path.exists():
        raise FileNotFoundError(
            f"Portfolio file {portfolio_name}.yaml not found in {conf_dir}"
        )
    with open(portfolio_path, "r") as f:
        return yaml.safe_load(f)
