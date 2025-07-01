import click
import streamlit.web.cli as stcli
import sys
import os

@click.group()
def cli():
    """Investment Dashboard CLI - A command line interface for managing your investment dashboard."""
    pass

@cli.command()
@click.option('--port', default=8501, help='Port to run the dashboard on')
def run(port):
    """Run the Streamlit dashboard."""
    click.echo(f"🚀 Starting dashboard on port {port}...")
    sys.argv = ["streamlit", "run", "src/dashboard/Regions.py", "--server.port", str(port)]
    sys.exit(stcli.main())

@cli.command()
def update():
    """Update the investment data."""
    click.echo("📊 Updating investment data...")
    # Add your data update logic here
    click.echo("✅ Data update completed!")

@cli.command()
def clean():
    """Clean temporary files and cache."""
    click.echo("🧹 Cleaning temporary files...")
    # Add your cleanup logic here
    click.echo("✨ Cleanup completed!")

if __name__ == '__main__':
    cli() 