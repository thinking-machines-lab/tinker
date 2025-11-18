"""Lazy loading support for Click commands.

This module provides a LazyGroup class that extends Click's Group to support
lazy loading of subcommands, ensuring fast CLI startup times.
"""

import importlib
from typing import Any, Dict, List
import click


class LazyGroup(click.Group):
    """A Click Group that supports lazy loading of subcommands.

    This allows the CLI to have fast startup times by only importing
    command modules when they are actually invoked, not when the CLI
    is first loaded or when help is displayed.
    """

    def __init__(
        self, *args: Any, lazy_subcommands: Dict[str, str] | None = None, **kwargs: Any
    ) -> None:
        """Initialize the LazyGroup.

        Args:
            lazy_subcommands: A dictionary mapping command names to import paths.
                             Format: {"command": "module.path:attribute_name"}
        """
        super().__init__(*args, **kwargs)
        self.lazy_subcommands = lazy_subcommands or {}

    def list_commands(self, ctx: click.Context) -> List[str]:
        """Return a list of all command names.

        This includes both eagerly loaded commands and lazy commands.
        """
        # Get any eagerly loaded commands
        base = super().list_commands(ctx)
        # Add lazy command names
        lazy = sorted(self.lazy_subcommands.keys())
        return base + lazy

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        """Get a command by name, loading it lazily if necessary.

        Args:
            ctx: The Click context
            cmd_name: The name of the command to retrieve

        Returns:
            The Click command object, or None if not found
        """
        # Check if it's a lazy command
        if cmd_name in self.lazy_subcommands:
            return self._lazy_load(cmd_name)
        # Fall back to normal command loading
        return super().get_command(ctx, cmd_name)

    def _lazy_load(self, cmd_name: str) -> click.Command:
        """Lazily load a command by importing its module.

        Args:
            cmd_name: The name of the command to load

        Returns:
            The loaded Click command object

        Raises:
            ValueError: If the imported object is not a Click Command
        """
        # Get the import path for this command
        import_path = self.lazy_subcommands[cmd_name]

        # Split into module path and attribute name
        module_name, attr_name = import_path.rsplit(":", 1)

        # Import the module
        mod = importlib.import_module(module_name)

        # Get the command object
        cmd_object = getattr(mod, attr_name)

        # Verify it's a Click command
        if not isinstance(cmd_object, click.Command):
            raise ValueError(
                f"Lazy loading of {import_path} failed: '{attr_name}' is not a Click Command"
            )

        return cmd_object
