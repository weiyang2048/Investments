"""
Configuration package for the Investments project.
This module provides utilities for loading and managing configuration files.
"""

from .config_loader import load_config
from .symbol_utils import get_symbols

__all__ = ['load_config', 'get_symbols']
