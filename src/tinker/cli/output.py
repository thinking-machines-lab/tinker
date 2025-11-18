"""Output formatting utilities for the Tinker CLI.

This module provides a base class for structured output that can be
rendered as either a table (using rich) or as JSON. Each command
defines its own output class that inherits from OutputBase.
"""

import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Callable, Union


class OutputBase(ABC):
    """Virtual base class for all command outputs.

    Subclasses must implement methods to convert data to various formats.
    The base class provides the common print() method that handles
    format selection and rendering.
    """

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert output to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the output data
        """
        pass

    @abstractmethod
    def get_table_columns(self) -> List[str]:
        """Return list of column names for table output.

        Returns:
            List of column header strings
        """
        pass

    @abstractmethod
    def get_table_rows(self) -> List[List[str]]:
        """Return list of rows for table output.

        Each row should be a list of string values corresponding
        to the columns returned by get_table_columns().

        Returns:
            List of rows, where each row is a list of string values
        """
        pass

    def get_title(self) -> str | None:
        """Optional title for the output display.

        Override this method to provide a title for the table.

        Returns:
            Title string or None
        """
        return None

    def print(self, format: str = "table") -> None:
        """Print the output in the specified format.

        Args:
            format: Output format - either "table" or "json"
        """
        if format == "json":
            self._print_json()
        else:
            self._print_table()

    def _print_json(self) -> None:
        """Print output as JSON."""
        import json

        data = self.to_dict()
        json.dump(data, sys.stdout, indent=2, default=str)
        print()  # Add newline after JSON output

    def _print_table(self) -> None:
        """Print output as a rich table."""
        # Lazy import rich to avoid slow startup
        from rich.console import Console
        from rich.table import Table

        console = Console()

        # Create table with optional title
        title = self.get_title()
        table = Table(title=title) if title else Table()

        # Add columns
        columns = self.get_table_columns()
        for col in columns:
            # First column (usually ID) gets special styling
            if col == columns[0]:
                table.add_column(col, style="bright_cyan", no_wrap=True)
            else:
                table.add_column(col)

        # Add rows
        rows = self.get_table_rows()
        for row in rows:
            table.add_row(*row)

        # Print the table
        console.print(table)


# Utility formatting functions


def format_size(bytes: int) -> str:
    """Format bytes as human-readable size.

    Args:
        bytes: Size in bytes

    Returns:
        Human-readable size string (e.g., "1.2 GB")
    """
    if bytes < 0:
        return "N/A"

    size = float(bytes)
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if size < 1024.0:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0

    return f"{size:.1f} EB"


def format_timestamp(dt: Union[datetime, str, None]) -> str:
    """Format datetime as relative time or absolute date.

    Args:
        dt: datetime object, ISO string, or None

    Returns:
        Formatted time string (e.g., "2 hours ago", "2024-01-15")
    """
    if not dt:
        return "N/A"

    # Lazy import datetime
    from datetime import datetime, timezone

    # Handle different input types
    if isinstance(dt, str):
        # Try to parse ISO format string
        try:
            from datetime import datetime

            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return str(dt)

    if not hasattr(dt, "replace"):
        # Not a datetime object
        return str(dt)

    try:
        # Get current time
        now = datetime.now(timezone.utc)

        # Ensure dt has timezone info
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)

        # Calculate time difference
        delta = now - dt

        # Format based on age
        if delta.days > 30:
            return dt.strftime("%Y-%m-%d")
        elif delta.days > 7:
            weeks = delta.days // 7
            return f"{weeks} week{'s' if weeks > 1 else ''} ago"
        elif delta.days > 0:
            return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
        elif delta.seconds > 3600:
            hours = delta.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif delta.seconds > 60:
            minutes = delta.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "just now"

    except Exception:
        # If any error occurs, just return string representation
        return str(dt)


def format_bool(value: bool) -> str:
    """Format boolean for display.

    Args:
        value: Boolean value

    Returns:
        "Yes" or "No"
    """
    return "Yes" if value else "No"


def format_optional(value: Any, formatter: Callable[[Any], str] | None = None) -> str:
    """Format an optional value.

    Args:
        value: Value to format (may be None)
        formatter: Optional formatting function to apply if value is not None

    Returns:
        Formatted string or "N/A" if value is None
    """
    if value is None:
        return "N/A"
    if formatter:
        return formatter(value)
    return str(value)
