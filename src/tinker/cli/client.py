"""Client utilities for tinker CLI - handles SDK client creation and configuration.

This module provides functions for creating and configuring the Tinker SDK
client, with proper error handling for common issues like authentication
and network errors.
"""

import sys
from functools import wraps
from typing import TypeVar, Callable, Any, cast, TYPE_CHECKING

from .exceptions import TinkerCliError

if TYPE_CHECKING:
    from tinker.lib.public_interfaces.rest_client import RestClient


def create_rest_client() -> "RestClient":
    """Create and configure a RestClient instance with proper error handling.

    This function handles the creation of the ServiceClient and RestClient,
    with appropriate error messages for common failure cases.

    Returns:
        A configured RestClient instance

    Raises:
        TinkerCliError: If client creation fails
    """
    # Lazy import to avoid slow startup
    from tinker import ServiceClient

    try:
        service_client = ServiceClient()
        return service_client.create_rest_client()
    except ImportError as e:
        raise TinkerCliError(
            f"Failed to import Tinker SDK: {e}",
            "Please ensure the tinker package is properly installed.",
        )
    except ValueError as e:
        # Often indicates missing or invalid API key
        raise TinkerCliError(
            f"Configuration error: {e}", "Please check your Tinker API credentials."
        )
    except Exception as e:
        # Catch-all for other errors
        raise TinkerCliError(
            f"Failed to connect to Tinker API: {e}",
            "Please check your network connection and API configuration.",
        )


# Type variable for decorator
F = TypeVar("F", bound=Callable[..., Any])


def handle_api_errors(func: F) -> F:
    """Decorator for handling common API errors.

    This decorator catches common exceptions from the Tinker API
    and provides user-friendly error messages.

    Args:
        func: Function to wrap with error handling

    Returns:
        Wrapped function with error handling
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Lazy import to avoid slow startup
        from tinker._exceptions import (
            APIError,
            BadRequestError,
            AuthenticationError,
            PermissionDeniedError,
            NotFoundError,
            UnprocessableEntityError,
            RateLimitError,
            InternalServerError,
            APIStatusError,
            APIConnectionError,
            APITimeoutError,
        )

        try:
            return func(*args, **kwargs)
        except NotFoundError as e:
            details = f"Details: {e.message}" if hasattr(e, "message") else None
            raise TinkerCliError("Resource not found", details)
        except AuthenticationError as e:
            details = "Please check your API key or authentication credentials."
            if hasattr(e, "message"):
                details += f"\nDetails: {e.message}"
            raise TinkerCliError("Authentication failed", details)
        except PermissionDeniedError as e:
            details = "You don't have permission to access this resource."
            if hasattr(e, "message"):
                details += f"\nDetails: {e.message}"
            raise TinkerCliError("Permission denied", details)
        except BadRequestError as e:
            details = f"Details: {e.message}" if hasattr(e, "message") else None
            raise TinkerCliError("Invalid request", details)
        except UnprocessableEntityError as e:
            details = f"Details: {e.message}" if hasattr(e, "message") else None
            raise TinkerCliError("Invalid data provided", details)
        except RateLimitError as e:
            details = "Please wait a moment before trying again."
            if hasattr(e, "message"):
                details += f"\nDetails: {e.message}"
            raise TinkerCliError("Rate limit exceeded", details)
        except InternalServerError as e:
            details = "The Tinker API encountered an internal error. Please try again later."
            if hasattr(e, "message"):
                details += f"\nDetails: {e.message}"
            raise TinkerCliError("Internal server error", details)
        except APITimeoutError as e:
            details = "The request to Tinker API timed out. Please try again."
            if hasattr(e, "message"):
                details += f"\nDetails: {e.message}"
            raise TinkerCliError("Request timeout", details)
        except APIConnectionError as e:
            details = "Could not connect to the Tinker API. Please check your network connection."
            if hasattr(e, "message"):
                details += f"\nDetails: {e.message}"
            raise TinkerCliError("Connection failed", details)
        except APIStatusError as e:
            status = e.status_code if hasattr(e, "status_code") else "unknown"
            details = f"Details: {e.message}" if hasattr(e, "message") else None
            raise TinkerCliError(f"API error (status {status})", details)
        except APIError as e:
            # Generic API error
            details = f"Details: {e.message}" if hasattr(e, "message") else None
            raise TinkerCliError("API error occurred", details)
        except TinkerCliError:
            # Re-raise our own errors without modification
            raise
        except KeyboardInterrupt:
            # Re-raise keyboard interrupt to be handled by main
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            import traceback

            details = None
            if sys.stderr.isatty():
                # Only include traceback if stderr is a terminal (for debugging)
                import io

                tb_str = io.StringIO()
                traceback.print_exc(file=tb_str)
                details = tb_str.getvalue()
            raise TinkerCliError(f"Unexpected error occurred: {e}", details)

    return cast(F, wrapper)
