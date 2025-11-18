"""Custom exceptions for the Tinker CLI.

This module defines exceptions used throughout the CLI for consistent
error handling and graceful exits.
"""


class TinkerCliError(Exception):
    """Custom exception for CLI errors that should exit gracefully.

    This exception is caught at the top level of the CLI and converted
    to appropriate error messages and exit codes.

    Attributes:
        message: The main error message to display
        details: Optional additional details or suggestions
        exit_code: The exit code to use (default: 1)
    """

    def __init__(self, message: str, details: str | None = None, exit_code: int = 1):
        """Initialize a TinkerCliError.

        Args:
            message: The main error message (will be prefixed with "Error: ")
            details: Optional additional details or help text
            exit_code: The exit code to use when exiting (default: 1)
        """
        self.message = message
        self.details = details
        self.exit_code = exit_code
        super().__init__(message)
