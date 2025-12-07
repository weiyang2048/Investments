import pandas as pd

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """
    Compute the Exponential Moving Average (EMA) of a pandas Series.

    Parameters
    ----------
    series : pd.Series
        Input time series data.
    span : int
        The span for the EMA.

    Returns
    -------
    pd.Series
        The EMA series.
    """
    return series.ewm(span=span, adjust=False).mean()

def compute_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute rolling volatility (standard deviation) of a pandas Series.

    Parameters
    ----------
    series : pd.Series
        Input time series data.
    window : int, optional
        The size of the rolling window (default 20).

    Returns
    -------
    pd.Series
        Rolling volatility series.
    """
    return series.pct_change().rolling(window=window, min_periods=1).std()
