"""Command for showing version information.

This module implements the 'tinker version' command.
"""

import click


@click.command()
def cli():
    """Show version information."""
    try:
        # Lazy import version only when needed
        from tinker._version import __version__

        click.echo(f"tinker {__version__}")
    except ImportError:
        click.echo("tinker (version unavailable)")
