import pandas as pd
import numpy as np
from typing import Union, List


def compute_momentum(
    prices: pd.Series,
    period: Union[int, List[int]] = 14,
    trading_days: int = 252
) -> float:
    """
    Compute the latest annualized momentum from a price series.
    
    Momentum measures the rate of change in price over a specified period.
    When a list of periods is provided, computes momentum for each and returns the average.
    
    Args:
        prices: Series of prices with datetime index
        period: Number of periods to look back, or list of periods (default: 14)
        trading_days: Number of trading days per year for annualization (default: 252)
    
    Returns:
        Latest annualized momentum value: (1 + momentum)^(trading_days/period) - 1
        If period is a list, returns the average of annualized momentums.
        Returns np.nan if insufficient data is available.
    
    Examples:
        >>> prices = pd.Series([100, 102, 105, 103, 108], 
        ...                    index=pd.date_range('2024-01-01', periods=5))
        >>> momentum = compute_momentum(prices, period=2)
        >>> # Returns annualized percentage change from 2 periods ago to latest
        >>> momentum = compute_momentum(prices, period=[7, 14, 30])
        >>> # Returns average of annualized momentums for periods 7, 14, and 30
    """
    if (prices <= 0).any():
        raise ValueError("All prices must be positive for momentum calculation")
    
    # Convert single period to list for uniform processing
    periods = [period] if isinstance(period, int) else period
    
    if not periods:
        return np.nan
    
    annualized_momentums = []
    
    for p in periods:
        if len(prices) < p + 1:
            continue
        
        # Calculate latest momentum: (P_t / P_{t-period}) - 1
        current_price = prices.iloc[-1]
        past_price = prices.iloc[-(p + 1)]
        
        if pd.isna(current_price) or pd.isna(past_price) or past_price <= 0:
            continue
        
        momentum = (current_price / past_price) - 1
        
        # Annualize momentum: (1 + momentum)^(trading_days/period) - 1
        annualized_momentum = (1 + momentum) ** (trading_days / p) - 1
        annualized_momentums.append(annualized_momentum)
    
    if not annualized_momentums:
        return np.nan
    
    # Return average if list was provided, single value if int was provided
    return np.mean(annualized_momentums)
