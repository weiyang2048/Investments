"""
Script to update portfolio YAML files from Fidelity CSV export.

This script reads a Fidelity portfolio positions CSV file and updates
the corresponding YAML files based on account numbers:
- 264114448 → r.yaml (Regions)
- 259472507 → s.yaml (Sectors)
- X96769741 → f.yaml (Fidelity)

Usage:
    python cli/update_portfolio_yaml.py <csv_file_path>
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from pyhere import here


def load_account_mapping() -> Dict[str, str]:
    """Load account number to YAML file mapping from config file."""
    config_path = here("conf/account_mapping.json")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Account mapping config not found: {config_path}\n"
            "Please create conf/account_mapping.json with account number mappings."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_portfolio_csv(csv_path: str) -> List[Dict]:
    """Read portfolio positions from CSV file."""
    positions = []
    # Use utf-8-sig to automatically handle BOM character
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # Normalize column names to strip BOM and whitespace
        reader.fieldnames = [col.strip().lstrip('\ufeff') for col in reader.fieldnames] if reader.fieldnames else None
        
        for row in reader:
            # Skip empty rows and footer text
            if not row.get("Symbol") or row.get("Symbol").startswith("*"):
                continue
            # Skip cash positions (SPAXX**, FDRXX**, USD***)
            symbol = row.get("Symbol", "").strip()
            if symbol.endswith("**") or symbol.endswith("***"):
                continue
            positions.append(row)
    return positions


def filter_by_account(positions: List[Dict], account_number: str) -> List[Tuple[str, str]]:
    """Filter positions by account number and return (symbol, description) tuples."""
    result = []
    for pos in positions:
        # Handle BOM in account number field - try both with and without normalization
        account_num = pos.get("Account Number", "").strip().lstrip('\ufeff')
        if account_num == account_number:
            symbol = pos.get("Symbol", "").strip()
            description = pos.get("Description", "").strip()
            if symbol and description:
                result.append((symbol, description))
    return result


def update_yaml_file(yaml_file: str, tickers: List[Tuple[str, str]], echo=None) -> None:
    """Update YAML file with new ticker symbols and descriptions.
    
    Args:
        yaml_file: Name of the YAML file to update
        tickers: List of (symbol, description) tuples
        echo: Optional output function (defaults to print)
    """
    if echo is None:
        echo = print
    
    yaml_path = here(f"conf/tickers/{yaml_file}")
    
    # Load existing YAML file if it exists
    existing_data = {}
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            existing_data = yaml.safe_load(f) or {}
    
    # Update with new tickers (new ones will overwrite old ones with same symbol)
    for symbol, description in tickers:
        existing_data[symbol] = {"name": description}
    
    # Sort by symbol for consistency
    sorted_data = dict(sorted(existing_data.items()))
    
    # Write back to YAML file
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(
            sorted_data,
            f,
            default_flow_style=False,
            sort_keys=False,  # Already sorted
            allow_unicode=True,
            width=1000,  # Prevent line wrapping
        )
    
    echo(f"✓ Updated {yaml_file} with {len(tickers)} tickers ({len(sorted_data)} total)")


def main(csv_file_path: str) -> None:
    """Main function to process CSV and update YAML files."""
    csv_path = Path(csv_file_path)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_file_path}")
        sys.exit(1)
    
    # Load account mapping from config
    try:
        account_mapping = load_account_mapping()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Reading portfolio positions from: {csv_file_path}")
    positions = read_portfolio_csv(csv_path)
    print(f"Found {len(positions)} positions")
    
    # Process each account
    for account_number, yaml_file in account_mapping.items():
        print(f"\nProcessing account {account_number} → {yaml_file}")
        tickers = filter_by_account(positions, account_number)
        if tickers:
            print(f"  Found {len(tickers)} tickers:")
            for symbol, description in tickers:
                print(f"    - {symbol}: {description}")
            update_yaml_file(yaml_file, tickers)
        else:
            print(f"  No tickers found for account {account_number}")


if __name__ == "__main__":
    # python cli/update_portfolio_yaml.py Notebooks/Portfolio_Positions.csv
    if len(sys.argv) < 2:
        print("Usage: python cli/update_portfolio_yaml.py <csv_file_path>")
        print("\nExample:")
        print("  python cli/update_portfolio_yaml.py Notebooks/Portfolio_Positions.csv")
        sys.exit(1)
    
    main(sys.argv[1])

