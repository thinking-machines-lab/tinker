"""Context object for the Tinker CLI.

This module provides a dataclass for sharing configuration and state
between CLI commands.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class CLIContext:
    """Context object for sharing state between CLI commands.

    This dataclass is passed through the Click command hierarchy
    using the @click.pass_obj decorator, allowing commands to access
    shared configuration without needing to traverse the context tree.

    Attributes:
        format: Output format for command results ('table' or 'json')
    """

    format: Literal["table", "json"] = "table"
