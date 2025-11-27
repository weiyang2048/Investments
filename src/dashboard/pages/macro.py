"""
Streamlit macro page for visualizing the Global Liquidity Index.

This page uses the `liquidity` package's GlobalLiquidity model and
displays the index from a selected beginning date (minus 15 weeks) to
now. It also overlays VT ETF price normalized to 1 at the start of
the range.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import hydra
import yfinance as yf
from liquidity.models.liquidity import GlobalLiquidity

from src.configurations.yaml import register_resolvers
from src.dashboard.create_page import setup_page_and_sidebar

from src.data.GLI import st_load_global_liquidity



@st.cache_resource(show_spinner=False)
def load_vt_history() -> pd.DataFrame:
    """Load VT price history (Close) and cache it."""
    vt = yf.Ticker("VT")
    hist = vt.history(period="max")
    print(hist.head())
    # Align timezone with liquidity index (use tz-naive)
    if hasattr(hist.index, "tz") and hist.index.tz is not None:
        hist = hist.tz_convert(None)
    return hist[["Close"]].copy()


def _add_macro_sidebar():
    """Sidebar controls for the macro / liquidity view."""
    st.sidebar.header("Macro Settings")
    beginning_date = st.sidebar.date_input(
        "Beginning Date",
        value=pd.Timestamp.today() - pd.Timedelta(weeks=52),
        min_value=pd.Timestamp("1990-01-01"),
        max_value=pd.Timestamp.today(),
        help="Beginning date of the visualization window. Data will be shown from 15 weeks before this date to now.",
    )
    lag_weeks = st.sidebar.slider(
        "Lag / Look-ahead (weeks)",
        min_value=0,
        max_value=52,
        value=12,
        step=1,
        help="Shift the index forward in time by this many weeks to visualize a look-ahead effect.",
    )
    show_lag = st.sidebar.checkbox(
        "Show lagged series",
        value=True,
        help="Display the forward-shifted (lagged) Global Liquidity Index.",
    )
    return pd.Timestamp(beginning_date), int(lag_weeks), bool(show_lag)


def _plot_global_liquidity(df: pd.DataFrame, beginning_date: pd.Timestamp, lag_weeks: int, show_lag: bool):
    """Create and render the Global Liquidity plot with VT overlay (normalized)."""
    # Calculate date range: beginning_date - 15 weeks to now
    lookback_weeks = 15
    lookback_days = lookback_weeks * 7
    start_date = beginning_date - pd.Timedelta(days=lookback_days)
    end_date = pd.Timestamp.today()  # Always plot to now
    
    lag_days = lag_weeks * 7

    # Build extended date range so the lagged curve can extend into the future
    max_date = end_date + pd.Timedelta(days=lag_days)
    date_range_full = pd.date_range(start_date, max_date, freq="D")
    
    # Check if we have data in this range
    df_range = df[(df.index >= start_date) & (df.index <= end_date)]
    if df_range.empty:
        st.warning(f"No Global Liquidity data available for the period {start_date.date()} to {end_date.date()}.")
        return

    # Reindex and interpolate to daily frequency
    df_extended = df.reindex(date_range_full).interpolate(method="linear")

    if show_lag and lag_days > 0:
        df_extended["Liquidity Index (Lag)"] = df_extended["Liquidity Index"].shift(lag_days)

    # Only keep the plotting window (start_date to max_date with lag)
    plot_range = (date_range_full >= start_date) & (date_range_full <= max_date)
    df_plot = df_extended.loc[plot_range]

    # Create plotly figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add unlagged Global Liquidity Index with opacity 0.25
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot["Liquidity Index"],
            name=f"Global Liquidity Index ({start_date.date()} to {end_date.date()})",
            line=dict(color="blue", width=2),
            opacity=0.25,
        ),
        secondary_y=False,
    )

    # Add lagged Global Liquidity Index if enabled
    if show_lag and "Liquidity Index (Lag)" in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot["Liquidity Index (Lag)"],
                name=f"Liquidity Index (Lag {lag_weeks}w)",
                line=dict(color="green", width=2, dash="dash"),
                opacity=1.0,
            ),
            secondary_y=False,
        )

    # Load VT history and overlay normalized price on secondary axis
    vt_hist = load_vt_history()
    if vt_hist.empty:
        st.info("VT price data could not be loaded (yfinance returned no data).")
    else:
        # Filter VT data to the plotting range
        vt_mask = (vt_hist.index >= start_date) & (vt_hist.index <= max_date)
        vt_plot = vt_hist.loc[vt_mask]

        if not vt_plot.empty:
            # Normalize by earliest price in the plotting range
            base_price = vt_plot["Close"].iloc[0]
            vt_norm = vt_plot["Close"] / base_price
            
            fig.add_trace(
                go.Scatter(
                    x=vt_plot.index,
                    y=vt_norm,
                    name="VT (price / first price)",
                    line=dict(color="red", width=2),
                    opacity=1.0,
                ),
                secondary_y=True,
            )
        else:
            # No VT data in this range
            st.info(
                "No VT price data available for the selected period. "
                "Try a later year (VT starts trading around 2008)."
            )

    # Set axis labels
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Global Liquidity Index", secondary_y=False)
    fig.update_yaxes(title_text="VT (normalized price)", secondary_y=True)

    # Update layout
    fig.update_layout(
        height=500,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, config={"displayModeBar": False})


def main():
    """Main entry point for the macro / Global Liquidity page."""
    register_resolvers()

    # Load Hydra config so we can reuse global dashboard styling
    with hydra.initialize(version_base=None, config_path="../../../conf"):
        config = hydra.compose(config_name="main")

    beginning_date, lag_weeks, show_lag = setup_page_and_sidebar(
        config["style_conf"],
        add_to_sidebar=_add_macro_sidebar,
    )

    st.title("🌐 Global Liquidity Index")
    st.markdown(
        "Visualize the **Global Liquidity Index** from a selected beginning date (minus 15 weeks) "
        "to **now**. Includes an optional forward-shifted (lagged) curve to explore potential "
        "lead/lag behavior, and an overlay of **VT** ETF price normalized to 1 at the start of "
        "the period."
    )

    df = st_load_global_liquidity()
    _plot_global_liquidity(df, beginning_date=beginning_date, lag_weeks=lag_weeks, show_lag=show_lag)


if __name__ == "__main__":
    # Handle Hydra initialization - clear if already initialized
    if hydra.core.global_hydra.GlobalHydra().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    main()


