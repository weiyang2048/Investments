import click
import streamlit.web.cli as stcli
import sys
import os


@click.group()
def cli():
    """Investment Dashboard CLI - A command line interface for managing your investment dashboard."""
    pass


@cli.command()
@click.option("--port", default=8501, help="Port to run the dashboard on")
def run(port):
    """Run the Streamlit dashboard."""
    click.echo(f"🚀 Starting dashboard on port {port}...")
    sys.argv = ["streamlit", "run", "src/dashboard/Lenses.py", "--server.port", str(port)]
    sys.exit(stcli.main())


if __name__ == "__main__":
    cli()
