import pandas as pd


def latest_percent_change(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the latest percent change of a DataFrame."""
    return df.iloc[-1] / df.iloc[-2] - 1


def last_week_percent_change(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the last week percent change of a DataFrame."""
    return df.iloc[-1] / df.iloc[-5] - 1


def last_month_percent_change(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the last month percent change of a DataFrame."""
    return df.iloc[-1] / df.iloc[-5 * 5] - 1


def last_quarter_percent_change(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the last quarter percent change of a DataFrame."""
    return df.iloc[-1] / df.iloc[-5 * 5 * 4] - 1


def last_year_percent_change(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the last year percent change of a DataFrame."""
    return df.iloc[-1] / df.iloc[-min(5 * 5 * 4 * 12, len(df) - 1)] - 1


def aggregate_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate performance of a DataFrame."""
    df = df.copy()
    df.dropna(inplace=True)
    return (
        df.groupby("Symbol")
        .agg(
            Latest_Pct_Chg=("Price", latest_percent_change),
            Last_W_Pct_Chg=("Price", last_week_percent_change),
            Last_M_Pct_Chg=("Price", last_month_percent_change),
            Last_Q_Pct_Chg=("Price", last_quarter_percent_change),
            Last_Y_Pct_Chg=("Price", last_year_percent_change),
        )
        .sort_values(by="Latest_Pct_Chg", ascending=False)
    )
