import streamlit as st
import pandas as pd
from liquidity.models.liquidity import GlobalLiquidity

def load_global_liquidity() -> pd.DataFrame:
    """Load the Global Liquidity index once and cache the DataFrame."""
    model = GlobalLiquidity()
    # model.show()  # Optional: opens its own window; avoid in Streamlit
    df = model.df.copy()
    # Ensure we have a DateTimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df
@st.cache_resource(show_spinner=True)
def st_load_global_liquidity() -> pd.DataFrame:
    return load_global_liquidity()