"""
Symbol utilities module for the Investments project.
Handles symbol-related operations and filtering.
"""

from typing import List

def get_symbols(symbol_type: str, etf_config: dict) -> List[str]:
    """
    Get symbols based on the selected type.

    Args:
        symbol_type: Type of symbols to retrieve ("Markets", "Sectors", or "Regions")
        etf_config: ETF configuration dictionary

    Returns:
        List of symbol strings
    """
    return etf_config[symbol_type] 