"""
Streamlit page for displaying Fidelity Portfolio Positions CSV.

This page allows users to view their portfolio positions from a CSV file.
It checks for Portfolio_Positions.csv in the root directory, and if not found,
provides an option to upload the file.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from pyhere import here
import hydra
from src.configurations.yaml import register_resolvers
from src.dashboard.create_page import setup_page_and_sidebar
from cli.update_portfolio_yaml import (
    load_account_mapping,
    read_portfolio_csv,
    filter_by_account,
    update_yaml_file,
)


def clean_currency_value(value):
    """Clean currency values from CSV (remove $ and commas, handle empty/null)."""
    if pd.isna(value) or value == "":
        return None
    if isinstance(value, str):
        # Remove $, commas, and whitespace
        cleaned = value.replace("$", "").replace(",", "").strip()
        if cleaned == "" or cleaned == "-":
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return float(value)


def clean_percentage_value(value):
    """Clean percentage values from CSV (remove %, handle empty/null)."""
    if pd.isna(value) or value == "":
        return None
    if isinstance(value, str):
        cleaned = value.replace("%", "").replace(",", "").strip()
        if cleaned == "" or cleaned == "-":
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return float(value)


def load_portfolio_positions(csv_path: Path) -> pd.DataFrame:
    """Load and clean portfolio positions CSV."""
    # Use utf-8-sig to handle BOM character
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    
    # Normalize column names (strip BOM and whitespace)
    df.columns = [col.strip().lstrip('\ufeff') for col in df.columns]
    
    # Clean currency columns
    currency_columns = [
        "Last Price",
        "Last Price Change",
        "Current Value",
        "Today's Gain/Loss Dollar",
        "Total Gain/Loss Dollar",
        "Cost Basis Total",
        "Average Cost Basis",
    ]
    
    for col in currency_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_currency_value)
    
    # Clean percentage columns
    percentage_columns = [
        "Today's Gain/Loss Percent",
        "Total Gain/Loss Percent",
        "Percent Of Account",
    ]
    
    for col in percentage_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_percentage_value)
    
    # Clean Quantity column
    if "Quantity" in df.columns:
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    
    return df


def filter_positions(df: pd.DataFrame, show_cash: bool = False) -> pd.DataFrame:
    """Filter out cash positions if requested."""
    if not show_cash:
        # Filter out cash positions (symbols ending with ** or ***)
        df = df[~df["Symbol"].astype(str).str.endswith(("**", "***"))]
    return df


def calc_gain_loss_pct(gain_loss: float, value: float) -> float:
    """Calculate gain/loss percentage."""
    return (gain_loss / (value - gain_loss) * 100) if (value - gain_loss) != 0 else 0


def calc_subset_stats(subset_df: pd.DataFrame):
    """Calculate statistics for a subset of positions (excluding cash)."""
    # Filter to only include valid accounts (non-empty positions or positive cash)
    valid_accounts = get_valid_accounts(subset_df)
    if valid_accounts:
        subset_df = subset_df[subset_df["Account Number"].isin(valid_accounts)]
    
    non_cash = filter_non_cash(subset_df)
    value = subset_df["Current Value"].sum() if "Current Value" in subset_df.columns else 0
    gain_loss = subset_df["Total Gain/Loss Dollar"].sum() if "Total Gain/Loss Dollar" in subset_df.columns else 0
    gain_loss_pct = calc_gain_loss_pct(gain_loss, value)
    positions = len(non_cash)
    accounts = subset_df["Account Number"].nunique() if "Account Number" in subset_df.columns and len(subset_df) > 0 else 0
    return value, gain_loss, gain_loss_pct, positions, accounts


def display_portfolio_summary(df: pd.DataFrame) -> None:
    """Display portfolio summary statistics with Short-Term and Long-Term breakdown."""
    # Filter to only include valid accounts (non-empty positions or positive cash)
    valid_accounts = get_valid_accounts(df)
    print(df)
    if valid_accounts:
        df = df[df["Account Number"].isin(valid_accounts)]
    
    df_classified = classify_account_type(df)
    
    # Calculate totals
    total_value = df["Current Value"].sum() if "Current Value" in df.columns else 0
    total_gain_loss = df["Total Gain/Loss Dollar"].sum() if "Total Gain/Loss Dollar" in df.columns else 0
    total_gain_loss_pct = calc_gain_loss_pct(total_gain_loss, total_value)
    num_positions = len(filter_non_cash(df))
    num_accounts = df["Account Number"].nunique() if "Account Number" in df.columns else 0
    
    # Calculate Short-Term and Long-Term stats
    short_term_df = df_classified[df_classified["Account Type"] == "Short-Term"]
    short_term_value, short_term_gain_loss, short_term_gain_loss_pct, short_term_positions, short_term_accounts = calc_subset_stats(short_term_df)
    
    long_term_df = df_classified[df_classified["Account Type"] == "Long-Term"]
    long_term_value, long_term_gain_loss, long_term_gain_loss_pct, long_term_positions, long_term_accounts = calc_subset_stats(long_term_df)
    
    # Display overall summary, Short-Term, and Long-Term in the same row
    st.subheader("📊 Portfolio Summary")
    col1, col2, col3 = st.columns(3)
    
    # Overall Summary column
    with col1:
        st.markdown("**📊 Overall Summary**")
        st.metric("Total Value", f"${total_value:,.2f}" if total_value else "N/A")
        st.metric(
            "Total Gain/Loss",
            f"${total_gain_loss:,.2f} ({total_gain_loss_pct:.2f}%)" if total_gain_loss else "N/A",
            delta=f"{total_gain_loss_pct:.2f}%" if total_gain_loss else None,
        )
        st.caption(f"Positions: {num_positions} | Accounts: {num_accounts}")
    
    # Short-Term column
    with col2:
        st.markdown("**🔄 Short-Term**")
        st.metric("Total Value", f"${short_term_value:,.2f}" if short_term_value else "$0.00")
        st.metric(
            "Gain/Loss",
            f"${short_term_gain_loss:,.2f} ({short_term_gain_loss_pct:.2f}%)" if short_term_gain_loss else "N/A",
            delta=f"{short_term_gain_loss_pct:.2f}%" if short_term_gain_loss else None,
        )
        st.caption(f"Positions: {short_term_positions} | Accounts: {short_term_accounts}")
    
    # Long-Term column
    with col3:
        st.markdown("**📅 Long-Term**")
        st.metric("Total Value", f"${long_term_value:,.2f}" if long_term_value else "$0.00")
        st.metric(
            "Gain/Loss",
            f"${long_term_gain_loss:,.2f} ({long_term_gain_loss_pct:.2f}%)" if long_term_gain_loss else "N/A",
            delta=f"{long_term_gain_loss_pct:.2f}%" if long_term_gain_loss else None,
        )
        st.caption(f"Positions: {long_term_positions} | Accounts: {long_term_accounts}")


def is_cash_position(symbol: str) -> bool:
    """Check if a symbol represents a cash position."""
    if pd.isna(symbol):
        return False
    symbol_str = str(symbol)
    return symbol_str.endswith(("**", "***"))


def get_valid_accounts(df: pd.DataFrame) -> set:
    """Get set of account numbers that have non-empty positions or positive cash."""
    if "Account Number" not in df.columns or len(df) == 0:
        return set()
    
    # Identify cash positions
    df_copy = df.copy()
    if "Symbol" in df_copy.columns:
        df_copy["Is Cash"] = df_copy["Symbol"].apply(is_cash_position)
    else:
        df_copy["Is Cash"] = False
    
    # Group by account and calculate cash and non-cash values
    valid_accounts = set()
    
    if "Current Value" in df_copy.columns:
        # Calculate cash value per account
        cash_df = df_copy[df_copy["Is Cash"]]
        if not cash_df.empty:
            cash_by_account = cash_df.groupby("Account Number")["Current Value"].sum()
            # Accounts with positive cash
            valid_accounts.update(cash_by_account[cash_by_account > 0].index.tolist())
        
        # Calculate non-cash value per account
        non_cash_df = df_copy[~df_copy["Is Cash"]]
        if not non_cash_df.empty:
            non_cash_by_account = non_cash_df.groupby("Account Number")["Current Value"].sum()
            # Accounts with non-empty positions (positive value)
            valid_accounts.update(non_cash_by_account[non_cash_by_account > 0].index.tolist())
    
    return valid_accounts


def filter_non_cash(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out cash positions from dataframe."""
    if "Symbol" not in df.columns:
        return df
    return df[~df["Symbol"].astype(str).str.endswith(("**", "***"))]


def format_description(desc: str) -> str:
    """Format description: first 10 chars ... last 10 chars."""
    desc_str = str(desc)
    if len(desc_str) <= 20:
        return desc_str
    return desc_str[:10] + "..." + desc_str[-10:]


def get_default_display_columns() -> list:
    """Get default columns to display in position tables."""
    return [
        "Account Name",
        "Symbol",
        "Description",
        "Last Price",
        "Current Value",
        "Today's Gain/Loss Dollar",
        "Today's Gain/Loss Percent",
        "Total Gain/Loss Dollar",
        "Total Gain/Loss Percent",
        "% Account",
        "% Total",
    ]


def create_format_dict(df: pd.DataFrame) -> dict:
    """Create format dictionary for numeric columns."""
    format_dict = {}
    numeric_formats = {
        "Quantity": "{:,.0f}",
        "Last Price": "${:,.2f}",
        "Current Value": "${:,.2f}",
        "% Account": "{:.2f}%",
        "% Total": "{:.2f}%",
        "Today's Gain/Loss Dollar": "${:,.2f}",
        "Total Gain/Loss Dollar": "${:,.2f}",
        "Today's Gain/Loss Percent": "{:.2f}%",
        "Total Gain/Loss Percent": "{:.2f}%",
    }
    for col, fmt in numeric_formats.items():
        if col in df.columns:
            format_dict[col] = fmt
    return format_dict


def apply_gain_loss_gradient(styled_df, df: pd.DataFrame, col_name: str, default_abs_max: float):
    """Apply red-green gradient to gain/loss column."""
    if col_name not in df.columns or not df[col_name].notna().any():
        return styled_df
    vmin = float(df[col_name].min())
    vmax = float(df[col_name].max())
    abs_max = max(abs(vmin), abs(vmax)) if vmin is not None and vmax is not None else default_abs_max
    return styled_df.background_gradient(
        subset=[col_name],
        cmap="RdYlGn",
        vmin=-abs_max,
        vmax=abs_max,
    )


def apply_positive_gradient(styled_df, df: pd.DataFrame, col_name: str, cmap: str = "Greens", default_max: float = 10000):
    """Apply gradient to positive-value columns."""
    if col_name not in df.columns or not df[col_name].notna().any():
        return styled_df
    vmax = float(df[col_name].max())
    return styled_df.background_gradient(subset=[col_name], cmap=cmap, vmin=0, vmax=vmax)


def classify_account_type(df: pd.DataFrame) -> pd.DataFrame:
    """Add Account Type column (Short-Term or Long-Term)."""
    df = df.copy()
    df["Account Type"] = "Long-Term"
    # if len of Account Number is >9, or begin with a letter, then it is short-term
    df["Account Type"] = np.where((df["Account Number"].astype(str).str.len() > 9) | df["Account Number"].astype(str).str.startswith(("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")), "Short-Term", "Long-Term")
   
    
    return df


def display_account_breakdown(df: pd.DataFrame, include_cash: bool = True) -> None:
    """Display breakdown by account, including cash positions."""
    if "Account Number" not in df.columns or "Account Name" not in df.columns:
        return
    
    # Filter to only include valid accounts (non-empty positions or positive cash)
    valid_accounts = get_valid_accounts(df)
    if valid_accounts:
        df = df[df["Account Number"].isin(valid_accounts)]
    
    st.subheader("📊 Account Breakdown")
    
    # Classify account types
    df_classified = classify_account_type(df)
    
    # Identify cash positions
    df_with_cash_flag = df_classified.copy()
    df_with_cash_flag["Is Cash"] = df_with_cash_flag["Symbol"].apply(is_cash_position)
    
    # Group by account (including cash positions)
    # First determine account type: short-term if ANY position in account is short-term
    account_type_map = df_with_cash_flag.groupby(["Account Number", "Account Name"])["Account Type"].apply(
        lambda x: "Short-Term" if (x == "Short-Term").any() else "Long-Term"
    ).reset_index()
    account_type_map.columns = ["Account Number", "Account Name", "Account Type"]
    
    # Count non-cash positions only
    non_cash_df = df_with_cash_flag[~df_with_cash_flag["Is Cash"]]
    
    account_summary = df_with_cash_flag.groupby(["Account Number", "Account Name"]).agg({
        "Current Value": "sum",
        "Total Gain/Loss Dollar": "sum",
        "Is Cash": lambda x: (x.sum() > 0),  # True if account has any cash positions
    }).reset_index()
    
    # Count non-cash positions per account
    non_cash_counts = non_cash_df.groupby(["Account Number", "Account Name"]).size().reset_index(name="Positions")
    account_summary = account_summary.merge(
        non_cash_counts,
        on=["Account Number", "Account Name"],
        how="left"
    )
    account_summary["Positions"] = account_summary["Positions"].fillna(0).astype(int)
    
    # Calculate cash value per account
    cash_by_account = df_with_cash_flag[df_with_cash_flag["Is Cash"]].groupby(
        ["Account Number", "Account Name"]
    )["Current Value"].sum().reset_index()
    cash_by_account.columns = ["Account Number", "Account Name", "Cash Value"]
    
    # Merge cash values and account type
    account_summary = account_summary.merge(
        cash_by_account,
        on=["Account Number", "Account Name"],
        how="left"
    )
    account_summary = account_summary.merge(
        account_type_map,
        on=["Account Number", "Account Name"],
        how="left"
    )
    account_summary["Account Type"] = account_summary["Account Type"].fillna("Long-Term")
    account_summary["Cash Value"] = account_summary["Cash Value"].fillna(0)
    
    # Ensure Cash Value column exists
    if "Cash Value" not in account_summary.columns:
        account_summary["Cash Value"] = 0
    
    # Rename columns properly
    account_summary = account_summary.rename(columns={
        "Current Value": "Total Value",
        "Total Gain/Loss Dollar": "Total Gain/Loss",
    })
    
    # Calculate investment value (total minus cash)
    if "Total Value" in account_summary.columns and "Cash Value" in account_summary.columns:
        account_summary["Investment Value"] = account_summary["Total Value"] - account_summary["Cash Value"]
    else:
        account_summary["Investment Value"] = account_summary.get("Total Value", 0)
    
    # Ensure all required columns exist
    if "Cash Value" not in account_summary.columns:
        account_summary["Cash Value"] = 0
    
    # Calculate gain/loss percentage (only for investment positions, not cash)
    account_summary["Gain/Loss %"] = account_summary.apply(
        lambda row: calc_gain_loss_pct(row["Total Gain/Loss"], row["Investment Value"]),
        axis=1
    ).round(2).fillna(0)
    
    # Sort by total value
    account_summary = account_summary.sort_values("Total Value", ascending=False)
    
    # Reorder columns for better display
    display_cols = [
        "Account Name",
        "Total Value",
        "Investment Value",
        "Cash Value",
        "Total Gain/Loss",
        "Gain/Loss %",
        "Positions",
    ]
    # Only select columns that actually exist in the dataframe
    available_display_cols = [col for col in display_cols if col in account_summary.columns]
    account_summary = account_summary[available_display_cols]
    
    # Keep numeric values for gradient
    display_account_summary = account_summary.copy()
    
    # Ensure numeric columns are numeric type
    numeric_cols = ["Total Value", "Investment Value", "Cash Value", "Total Gain/Loss", "Gain/Loss %", "Positions"]
    for col in numeric_cols:
        if col in display_account_summary.columns:
            display_account_summary[col] = pd.to_numeric(display_account_summary[col], errors="coerce")
    
    # Apply color gradients to account breakdown
    styled_account = display_account_summary.style
    
    # Apply gradients
    for col in ["Total Value", "Investment Value"]:
        styled_account = apply_positive_gradient(styled_account, display_account_summary, col, "Greens", 10000)
    styled_account = apply_positive_gradient(styled_account, display_account_summary, "Cash Value", "Blues", 1000)
    styled_account = apply_gain_loss_gradient(styled_account, display_account_summary, "Total Gain/Loss", 1000)
    styled_account = apply_gain_loss_gradient(styled_account, display_account_summary, "Gain/Loss %", 50)
    
    
    # Format columns
    account_format_dict = {
        "Total Value": "${:,.2f}",
        "Investment Value": "${:,.2f}",
        "Cash Value": "${:,.2f}",
        "Total Gain/Loss": "${:,.2f}",
        "Gain/Loss %": "{:.2f}%",
        "Positions": "{:,.0f}",
    }
    styled_account = styled_account.format({k: v for k, v in account_format_dict.items() if k in display_account_summary.columns})
    
    st.dataframe(styled_account, width='stretch', hide_index=True)
    
    # Add note about cash positions
    st.caption("💡 Cash positions (FDRXX**, SPAXX**, USD***, etc.) are included in Total Value and Cash Value columns.")


def main_portfolio_page():
    """Main function for portfolio positions page."""
    # Setup page
    register_resolvers()
    
    with hydra.initialize(version_base=None, config_path="../../../conf"):
        config = hydra.compose(config_name="main")
    
    # Define threshold controls function to be added before webpage links
    def add_threshold_settings():
        st.sidebar.header("Threshold Settings")
        up_threshold = st.sidebar.slider(
            "Up Threshold (%)",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=0.5,
            help="Positions with gain % >= this value will be marked"
        )
        down_threshold = st.sidebar.slider(
            "Down Threshold (%)",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=0.5,
            help="Positions with loss % <= -this value will be marked"
        )
        return up_threshold, down_threshold
    
    # Get threshold values from sidebar (added before webpage links)
    up_threshold, down_threshold = setup_page_and_sidebar(config["style_conf"], add_to_sidebar=add_threshold_settings)
    
    st.title("📈 Portfolio Positions")
    st.markdown("View your Fidelity portfolio positions from CSV export.")
    st.info("ℹ️ **Note:** Currently only Fidelity positions files are supported.")
    
    # Check for CSV file in root
    root_csv_path = here("Portfolio_Positions.csv")
    uploaded_file = None
    
    file_status_msg = ""
    if root_csv_path.exists():
        file_status_msg = "✅ Found Portfolio_Positions.csv in project root"
        csv_path = root_csv_path
        
        # Run inv-port update if CSV file exists
        try:
            with st.spinner("🔄 Updating portfolio YAML files..."):
                # Load account mapping from config
                account_mapping = load_account_mapping()
                
                # Read portfolio positions
                positions = read_portfolio_csv(str(csv_path))
                
                total_tickers = 0
                
                # Process each account
                for account_number, yaml_file in account_mapping.items():
                    tickers = filter_by_account(positions, account_number)
                    
                    if tickers:
                        total_tickers += len(tickers)
                        # Create a simple echo function for Streamlit
                        def streamlit_echo(message):
                            pass  # Silently process, we'll show summary at the end
                        update_yaml_file(yaml_file, tickers, echo=streamlit_echo)
                
                if total_tickers > 0:
                    st.success(f"✅ Portfolio YAML files updated successfully! Updated {total_tickers} tickers across {len(account_mapping)} accounts")
                else:
                    st.info("ℹ️ No tickers found to update")
        except FileNotFoundError as e:
            st.warning(f"⚠️ Could not update portfolio YAML files: {e}")
        except Exception as e:
            st.warning(f"⚠️ Could not update portfolio YAML files: {e}")
        
    else:
        st.info("📤 No Portfolio_Positions.csv found in root. Please upload your portfolio positions file.")
        uploaded_file = st.file_uploader(
            "Upload Portfolio Positions CSV",
            type=["csv"],
            help="Upload your Fidelity portfolio positions CSV export",
        )
        
        if uploaded_file is None:
            st.stop()
        
        # Save uploaded file temporarily
        csv_path = Path("temp_portfolio_positions.csv")
        with open(csv_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    # Load data
    load_msg = ""
    try:
        with st.spinner("Loading portfolio positions..."):
            df = load_portfolio_positions(csv_path)
            load_msg = f"✅ Loaded {len(df)} positions"
    except Exception as e:
        st.error(f"❌ Error loading CSV file: {e}")
        st.stop()
    
    # Drop rows with missing symbols
    dropped_msg = ""
    if "Symbol" in df.columns:
        initial_count = len(df)
        df = df[df["Symbol"].notna() & (df["Symbol"].astype(str).str.strip() != "")]
        dropped_count = initial_count - len(df)
        if dropped_count > 0:
            dropped_msg = f"ℹ️ Dropped {dropped_count} row(s) with missing symbols"
    
    # Display combined messages on same line
    combined_parts = []
    if file_status_msg:
        combined_parts.append(file_status_msg)
    if load_msg:
        combined_parts.append(load_msg)
    if dropped_msg:
        combined_parts.append(dropped_msg)
    
    if combined_parts:
        st.success(" | ".join(combined_parts))
    
    # Always show cash positions (no filtering)
    df = filter_positions(df, show_cash=True)
    
    # Mark positions based on thresholds
    if "Total Gain/Loss Percent" in df.columns:
        df["Marked"] = (
            (df["Total Gain/Loss Percent"] >= up_threshold) |
            (df["Total Gain/Loss Percent"] <= -down_threshold)
        )
    else:
        df["Marked"] = False
    
    # Display summary
    display_portfolio_summary(df)
    
    st.divider()
    
    # Display account breakdown (including all accounts with cash positions)
    # Use original dataframe before filtering to show all accounts
    if root_csv_path.exists() or uploaded_file:
        try:
            original_df = load_portfolio_positions(csv_path)
            # Drop rows with missing symbols for account breakdown too
            if "Symbol" in original_df.columns:
                original_df = original_df[original_df["Symbol"].notna() & (original_df["Symbol"].astype(str).str.strip() != "")]
            display_account_breakdown(original_df, include_cash=True)
        except Exception:
            # Fallback to filtered df if original load fails
            display_account_breakdown(df, include_cash=True)
    else:
        display_account_breakdown(df, include_cash=True)
    
    st.divider()
    
    # Display marked positions table (if any)
    marked_df = df[df["Marked"] == True] if "Marked" in df.columns else pd.DataFrame()
    if len(marked_df) > 0:
        st.subheader(f"⚠️ Marked Positions (Gain ≥ {up_threshold}% or Loss ≤ -{down_threshold}%)")
        
        # Rename Percent Of Account to % Account if it exists in original dataframe
        if "Percent Of Account" in marked_df.columns:
            marked_df = marked_df.rename(columns={"Percent Of Account": "% Account"})
        
        # Select columns to display for marked positions
        default_cols = get_default_display_columns()
        columns_to_show = [col for col in default_cols if col in marked_df.columns]
        marked_display_df = marked_df[columns_to_show].copy()
        
        # Calculate % Total
        if "Current Value" in marked_display_df.columns:
            total_portfolio_value = df["Current Value"].sum() if "Current Value" in df.columns else 1
            marked_display_df["% Total"] = (marked_display_df["Current Value"] / total_portfolio_value * 100).round(2)
        
        # Format Description
        if "Description" in marked_display_df.columns:
            marked_display_df["Description"] = marked_display_df["Description"].apply(format_description)
        
        # Sort by Total Gain/Loss Percent (absolute value, descending)
        if "Total Gain/Loss Percent" in marked_display_df.columns:
            marked_display_df = marked_display_df.reindex(
                marked_display_df["Total Gain/Loss Percent"].abs().sort_values(ascending=False).index
            )
        
        # Apply styling
        if len(marked_display_df) > 0:
            styled_marked_df = marked_display_df.style
            
            # Apply gradients to gain/loss columns
            for col, default_max in [("Today's Gain/Loss Dollar", 100), ("Today's Gain/Loss Percent", 10),
                                     ("Total Gain/Loss Dollar", 1000), ("Total Gain/Loss Percent", 50)]:
                styled_marked_df = apply_gain_loss_gradient(styled_marked_df, marked_display_df, col, default_max)
            
            # Apply gradients to positive-value columns
            styled_marked_df = apply_positive_gradient(styled_marked_df, marked_display_df, "Current Value", "Greens", 10000)
            styled_marked_df = apply_positive_gradient(styled_marked_df, marked_display_df, "% Account", "Blues", 100)
            styled_marked_df = apply_positive_gradient(styled_marked_df, marked_display_df, "% Total", "Blues", 100)
            
            # Format numeric columns
            format_dict = create_format_dict(marked_display_df)
            if format_dict:
                styled_marked_df = styled_marked_df.format(format_dict)
            
            st.dataframe(styled_marked_df, width='stretch', hide_index=True)
        
        st.divider()
    
    # Display detailed positions table
    st.subheader("📋 Detailed Positions")
    
    # Rename Percent Of Account to % Account if it exists in original dataframe
    if "Percent Of Account" in df.columns:
        df = df.rename(columns={"Percent Of Account": "% Account"})
    
    # Select columns and prepare display dataframe
    default_cols = get_default_display_columns()
    columns_to_show = [col for col in default_cols if col in df.columns]
    display_df = df[columns_to_show].copy()
    
    # Filter out cash positions
    display_df = filter_non_cash(display_df)
    
    # Calculate % Total
    if "Current Value" in display_df.columns:
        total_portfolio_value = df["Current Value"].sum() if "Current Value" in df.columns else 1
        display_df["% Total"] = (display_df["Current Value"] / total_portfolio_value * 100).round(2)
    
    if "Description" in display_df.columns:
        display_df["Description"] = display_df["Description"].apply(format_description)
    
    # Format the display dataframe
    if len(display_df) > 0:
        # Sort by current value (descending)
        if "Current Value" in display_df.columns:
            display_df = display_df.sort_values("Current Value", ascending=False, na_position="last")
        
        # Apply color gradients
        styled_df = display_df.style
        
        # Apply gradients to gain/loss columns
        for col, default_max in [("Today's Gain/Loss Dollar", 100), ("Today's Gain/Loss Percent", 10),
                                 ("Total Gain/Loss Dollar", 1000), ("Total Gain/Loss Percent", 50)]:
            styled_df = apply_gain_loss_gradient(styled_df, display_df, col, default_max)
        
        # Apply gradients to positive-value columns
        styled_df = apply_positive_gradient(styled_df, display_df, "Current Value", "Greens", 10000)
        styled_df = apply_positive_gradient(styled_df, display_df, "% Account", "Blues", 100)
        styled_df = apply_positive_gradient(styled_df, display_df, "% Total", "Blues", 100)
        
        # Format numeric columns
        format_dict = create_format_dict(display_df)
        if format_dict:
            styled_df = styled_df.format(format_dict)
        
        # Display the styled dataframe
        st.dataframe(
            styled_df,
            width='stretch',
            hide_index=True,
            height=600,
        )
        
        # Download button
        csv_data = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv_data,
            file_name="portfolio_positions_filtered.csv",
            mime="text/csv",
        )
    else:
        st.warning("No positions found matching the selected filters.")
    
    # Clean up temporary file if uploaded
    if uploaded_file and csv_path.exists() and csv_path.name == "temp_portfolio_positions.csv":
        csv_path.unlink()


if __name__ == "__main__":
    main_portfolio_page()

