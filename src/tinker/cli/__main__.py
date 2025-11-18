"""Tinker CLI - Command-line interface for the Tinker SDK.

This module implements lazy loading to ensure fast startup times.
Only Click is imported at the module level. All other dependencies
including command modules are imported on-demand.

Enable execution of the CLI via python -m tinker.cli
"""

import sys
import click
from .lazy_group import LazyGroup
from .context import CLIContext
from .exceptions import TinkerCliError


@click.group(
    cls=LazyGroup,
    lazy_subcommands={
        "checkpoint": "tinker.cli.commands.checkpoint:cli",
        "run": "tinker.cli.commands.run:cli",
        "version": "tinker.cli.commands.version:cli",
    },
    context_settings=dict(help_option_names=["-h", "--help"]),
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format (default: table)",
)
@click.pass_context
def main_cli(ctx: click.Context, format: str) -> None:
    """Tinker management CLI."""
    # Store format in context for subcommands to access
    ctx.obj = CLIContext(format=format)  # type: ignore[assignment]


def main():
    try:
        main_cli()
    except TinkerCliError as e:
        # Print error message to stderr
        if e.message:
            print(f"Error: {e.message}", file=sys.stderr)
        if e.details:
            print(e.details, file=sys.stderr)
        sys.exit(e.exit_code)
    except KeyboardInterrupt:
        # Standard Unix exit code for Ctrl+C
        sys.exit(130)


# Make main available for entry point
cli = main


if __name__ == "__main__":
    main()
