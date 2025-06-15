"""
Configuration package for the Investments project.
This module provides utilities for loading and managing configuration files.
"""

from .config_loader import load_config, get_symbols, load_portfolios_conf, load_dashboard_conf

__all__ = ["load_config", "get_symbols", "load_portfolios_conf", "load_dashboard_conf"]
