"""CLI for portfolio management commands."""

import click
from pathlib import Path

from pyhere import here
from cli.update_portfolio_yaml import (
    load_account_mapping,
    read_portfolio_csv,
    filter_by_account,
    update_yaml_file,
)


@click.group()
def cli():
    """Portfolio management CLI - Update YAML files from Fidelity CSV exports."""
    pass


@cli.command()
@click.argument("csv_file", required=False, type=click.Path(path_type=Path))
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed output for each ticker found",
)
def update(csv_file, verbose):
    """Update portfolio YAML files from Fidelity CSV export.
    
    Reads a Fidelity portfolio positions CSV file and updates the corresponding
    YAML files (r.yaml, s.yaml, f.yaml) based on account numbers configured
    in conf/account_mapping.json.
    
    If CSV_FILE is not provided, defaults to Portfolio_Positions.csv in the project root.
    
    Example:
        inv-port update
        inv-port update Notebooks/Portfolio_Positions_Nov-01-2025.csv
    """
    # Default to Portfolio_Positions.csv in project root if not provided
    if csv_file is None:
        csv_file = here("Portfolio_Positions.csv")
    
    # Validate file exists
    if not csv_file.exists():
        click.echo(f"❌ Error: CSV file not found: {csv_file}", err=True)
        click.echo(f"   Please provide a valid CSV file path or place Portfolio_Positions.csv in the project root.", err=True)
        raise click.Abort()
    # Load account mapping from config
    try:
        account_mapping = load_account_mapping()
    except FileNotFoundError as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()
    
    click.echo(f"📊 Reading portfolio positions from: {csv_file}")
    positions = read_portfolio_csv(csv_file)
    click.echo(f"   Found {len(positions)} positions")
    
    total_tickers = 0
    
    # Process each account
    for account_number, yaml_file in account_mapping.items():
        click.echo(f"\n🔍 Processing account {account_number} → {yaml_file}")
        tickers = filter_by_account(positions, account_number)
        
        if tickers:
            total_tickers += len(tickers)
            if verbose:
                click.echo(f"   Found {len(tickers)} tickers:")
                for symbol, description in tickers:
                    click.echo(f"     • {symbol}: {description}")
            else:
                click.echo(f"   Found {len(tickers)} tickers")
            update_yaml_file(yaml_file, tickers, echo=click.echo)
        else:
            click.echo(f"   ⚠️  No tickers found for account {account_number}")
    
    click.echo(f"\n✅ Completed! Updated {total_tickers} tickers across {len(account_mapping)} accounts")


if __name__ == "__main__":
    cli()

