"""Marks SDK APIs that are not yet stable."""

from typing_extensions import deprecated


class ExperimentalWarning(FutureWarning):
    """Emitted when calling an API that may change or be removed in any release.

    A `FutureWarning`, so it is shown to end users by default. Silence it with
    `warnings.filterwarnings("ignore", category=tinker.ExperimentalWarning)`.
    """


experimental = deprecated
"""Decorator for an API that may change or be removed in any release without notice.

Always call it with a string-literal message and `category=ExperimentalWarning`::

    @experimental(
        "RestClient.search_sessions is experimental: it may change or be removed in any "
        "release without a deprecation period.",
        category=ExperimentalWarning,
    )
    def search_sessions(...): ...

Type checkers (pyright, and mypy with its `deprecated` error code) then report each
use as deprecated, and each call emits an `ExperimentalWarning`. Both checkers need
the call and the literal message at the decorated function: a message held in a
constant is not reported by mypy, and a decorator object built once and reused bare
is not reported by mypy at all. Remove the decorator once the API is stable.
"""
