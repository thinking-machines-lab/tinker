"""Attributes every request a client sends to the public SDK method that sent it.

The server records the `x-sdk-version` and `x-sdk-method` headers per request to see
which SDK versions and methods still hit an endpoint before deprecating it. The clients
do their work on the holder's event loop thread, across several requests per call, so
the method name travels in a context variable: `run_coroutine_threadsafe` copies the
calling thread's context into the task it schedules, where `AsyncTinker` reads the name
into each request's headers.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
from collections.abc import Callable, Iterator
from typing import Any, ParamSpec, TypeVar

from typing_extensions import deprecated

from tinker._version import __version__ as tinker_sdk_version

SDK_VERSION_HEADER = "x-sdk-version"
SDK_METHOD_HEADER = "x-sdk-method"

C = TypeVar("C", bound=type)
P = ParamSpec("P")
R = TypeVar("R")

_current_sdk_method: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tinker_sdk_method", default=None
)


def current_sdk_method_headers() -> dict[str, str]:
    """Headers attributing a request to the SDK version and the public method call in
    progress, or empty outside one."""
    sdk_method = _current_sdk_method.get()
    if sdk_method is None:
        return {}
    return {SDK_VERSION_HEADER: tinker_sdk_version, SDK_METHOD_HEADER: sdk_method}


def context_without_sdk_method() -> contextvars.Context:
    """The current context minus the method in progress, for a task that outlives the call
    starting it (a polling or refresh loop), so its requests aren't attributed to that call."""
    context = contextvars.copy_context()
    context.run(_current_sdk_method.set, None)
    return context


def without_sdk_method_telemetry(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator: `func` runs in `context_without_sdk_method`, so neither the requests it
    sends nor the tasks it starts are attributed to the public method that called it.

    For set-up a method triggers lazily on behalf of the whole client, such as
    constructing the holder with its session and the background loops that outlive the
    call."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return context_without_sdk_method().run(func, *args, **kwargs)

    return wrapper


def with_sdk_method_telemetry(cls: C) -> C:
    """Class decorator: requests sent while a public method of `cls` runs carry its name,
    qualified by the class, such as `TrainingClient.forward_backward`.

    Covers the methods `tracked_sdk_method_names` lists, each under its own name, so
    `sample` and `sample_async` are distinct. Only the outermost call counts: a public
    method that calls another one, on the same client or another, keeps its own name.

    A public method marked `@deprecated` or `@experimental` warns at its caller's line.
    """
    for name in _public_function_names(cls):
        setattr(cls, name, _tracked(vars(cls)[name], f"{cls.__name__}.{name}"))
    return cls


def tracked_sdk_method_names(cls: type) -> list[str]:
    """The `x-sdk-method` values `with_sdk_method_telemetry` makes instances of `cls`
    send: the public methods `cls` defines itself, sync and async, qualified by the class
    name. Static methods and properties are not included."""
    return [f"{cls.__name__}.{name}" for name in _public_function_names(cls)]


def _public_function_names(cls: type) -> list[str]:
    return sorted(
        name
        for name, member in vars(cls).items()
        if not name.startswith("_") and inspect.isfunction(member)
    )


# The code shared by every wrapper `deprecated` (and so `experimental`) puts around a function.
_DEPRECATED_WRAPPER_CODE = deprecated("")(lambda: None).__code__


def _is_deprecated_wrapper(func: Callable[..., Any]) -> bool:
    return getattr(func, "__code__", None) is _DEPRECATED_WRAPPER_CODE


def _tracked(func: Callable[..., Any], sdk_method: str) -> Callable[..., Any]:
    if _is_deprecated_wrapper(func):
        # The wrapper `deprecated` adds warns a fixed number of frames up, so it stays
        # outermost and calls the tracked method in place of the method.
        deprecated_wrapper: Any = func
        method = deprecated_wrapper.__wrapped__
        tracked = _tracked(method, sdk_method)
        for cell in deprecated_wrapper.__closure__:
            if cell.cell_contents is method:
                cell.cell_contents = tracked
        return func

    # Decorators such as `sync_only` wrap an `async def` in a plain function; the
    # innermost function says whether the method is async.
    if inspect.iscoroutinefunction(inspect.unwrap(func)):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with _entered(sdk_method):
                return await func(*args, **kwargs)

        return async_wrapper

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with _entered(sdk_method):
            return func(*args, **kwargs)

    return wrapper


@contextlib.contextmanager
def _entered(sdk_method: str) -> Iterator[None]:
    if _current_sdk_method.get() is not None:
        yield
        return
    token = _current_sdk_method.set(sdk_method)
    try:
        yield
    finally:
        _current_sdk_method.reset(token)
