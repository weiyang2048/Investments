import pandas as pd

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) using Wilder's smoothing method.
    
    Parameters
    ----------
    series : pd.Series
        Input time series data (typically closing prices).
    window : int, optional
        The lookback period for RSI calculation (default is 14).
    
    Returns
    -------
    pd.Series
        The RSI series.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Wilder's smoothing (equivalent to EMA with alpha = 1/window)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi
