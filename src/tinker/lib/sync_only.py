"""
Decorator to prevent synchronous methods from being called in async contexts.

This helps users avoid a common footgun where calling sync methods from async code
can cause deadlocks and performance issues.
"""

import asyncio
import logging
import traceback
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_jupyter() -> bool:
    """Check if code is running in a Jupyter notebook."""
    try:
        get_ipython  # type: ignore
    except NameError:
        return False
    shell = get_ipython().__class__.__name__  # type: ignore
    if shell in ("ZMQInteractiveShell", "Shell"):
        return True  # Jupyter notebook or qtconsole
    return False  # Other type of shell


def is_in_async_context() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def make_error_message(  # noqa: UP047
    func: Callable[..., T], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> str:
    # If we get here, we're in an async context - this is bad!
    method_name = func.__name__
    async_method_name = f"{method_name}_async"

    # Get the class name for a better error message
    class_name = ""
    if args and hasattr(args[0], "__class__"):
        class_name = f"{args[0].__class__.__name__}."

    return (
        f"Synchronous method '{class_name}{method_name}()' called from async context. "
        f"Use '{class_name}{async_method_name}()' instead.\n"
        f"Calling sync methods from async code can cause deadlocks and performance issues."
    )


def sync_only(func: Callable[..., T]) -> Callable[..., T]:  # noqa: UP047
    """Decorator to ensure a method is only called from sync context.

    This helps prevent a common footgun where users accidentally call
    sync methods from async code, which can cause deadlocks and performance issues.

    Assumes (in error message) that the wrapped method has a corresponding method name
    {method_name}_async.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        if is_in_async_context() and not is_jupyter():
            error_message = make_error_message(func, args, kwargs)
            logger.warning(error_message)
            logger.warning(
                f"===== Stack for calling sync from async ===== \n{traceback.format_stack()}\n ==========="
            )

        return func(*args, **kwargs)

    return wrapper
