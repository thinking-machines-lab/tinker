"""
Decorator for methods that block on the SDK's event loop.

Calling one from async code blocks that code's event loop; calling one on the SDK's own
event loop waits on itself forever, so that is refused.
"""

import asyncio
import logging
import traceback
from functools import wraps
from typing import Any, Callable, TypeVar

from .public_interfaces.api_future import APIFuture

logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_jupyter() -> bool:
    """Check if code is running in a Jupyter notebook."""
    try:
        get_ipython  # type: ignore
    except NameError:
        return False
    shell = get_ipython().__class__.__name__  # type: ignore  # noqa: F821
    return shell in ("ZMQInteractiveShell", "Shell")  # Jupyter notebook or qtconsole


def is_in_async_context() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _is_on_sdk_loop() -> bool:
    from .internal_client_holder import _internal_client_holder_thread_singleton

    return asyncio.get_running_loop() is _internal_client_holder_thread_singleton.loop_if_started()


def make_error_message(  # noqa: UP047
    func: Callable[..., T], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> str:
    method_name = func.__name__

    # Get the class name for a better error message
    class_name = ""
    if args and hasattr(args[0], "__class__"):
        class_name = f"{args[0].__class__.__name__}."

    return (
        f"Synchronous method '{class_name}{method_name}()' called from async context. "
        "It blocks the event loop until the result arrives; await the future instead "
        "(`await future`, or `await future.result_async()`)."
    )


def _is_finished_future(args: tuple[Any, ...]) -> bool:
    """A finished future's result never waits, so it may be read anywhere, the SDK's own
    loop included: done callbacks run there."""
    future = args[0] if args else None
    return isinstance(future, APIFuture) and future.future().done()


def sync_only(func: Callable[..., T]) -> Callable[..., T]:  # noqa: UP047
    """Decorator for a method that blocks until a result arrives from the SDK's event loop.

    Called from async code, the method logs a warning: it blocks that code's event loop.
    Called on the SDK's own event loop, it raises instead, since it would wait on itself
    forever. On a future that has already finished, the method returns at once and may be
    called anywhere.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        if is_in_async_context() and not _is_finished_future(args):
            error_message = make_error_message(func, args, kwargs)
            if _is_on_sdk_loop():
                raise RuntimeError(
                    f"{error_message} This call is on the Tinker SDK's own event loop, "
                    "where it would wait on itself forever."
                )
            if is_jupyter():
                return func(*args, **kwargs)
            logger.warning(error_message)
            logger.warning(
                f"===== Stack for calling sync from async ===== \n{traceback.format_stack()}\n ==========="
            )

        return func(*args, **kwargs)

    return wrapper
