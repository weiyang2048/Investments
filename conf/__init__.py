"""
Configuration package for the Investments project.
This module provides utilities for loading and managing configuration files.
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


def get_symbols_by(by: str, value: str) -> list:
    """
    Get symbols by a given key-value pair in the configuration file.

    Args:
        by (str): Key to filter by
        value (str): Value to match

    Returns:
        list: List of symbols matching the criteria
    """
    etf_config = load_config("etf")
    return [x for x in etf_config["etfs"] if etf_config["etfs"][x].get(by) == value]


# Example usage:
# etf_config = load_config('etf')
