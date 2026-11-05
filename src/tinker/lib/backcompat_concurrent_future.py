"""Backward compatibility: `SamplingClient.sample` and `compute_logprobs` returned a
`concurrent.futures.Future` before they returned an `APIFuture`.

Everything in this module is temporary and goes away with that compatibility window.
"""

from __future__ import annotations

import asyncio
import warnings
from concurrent.futures import Future as ConcurrentFuture
from typing import Callable, TypeVar

from tinker.lib.public_interfaces.api_future import APIFuture

T = TypeVar("T")


class _AwaitableFutureCompatWrapper(ConcurrentFuture[T], APIFuture[T]):  # pyright: ignore[reportUnusedClass]
    """Temporary adapter for `SamplingClient.sample` and `compute_logprobs`, which used to
    return a `concurrent.futures.Future` and are now annotated `APIFuture`.

    Typed callers see the `APIFuture` and reach for `future()` when they
    need a standard future. Untyped callers written against the old return type keep
    working because this is still a `concurrent.futures.Future` (its state mirrors the
    real future's), and its Future methods warn them off. Not public: remove it, and
    unwrap the two return sites, once the compatibility window closes.
    """

    def __init__(self, future: ConcurrentFuture[T]):
        super().__init__()
        self._future: ConcurrentFuture[T] = future
        self._warned = False
        future.add_done_callback(self._mirror)

    def _mirror(self, future: ConcurrentFuture[T]) -> None:
        # Copy the finished state onto this future, so the `concurrent.futures` machinery
        # (`wait`, `as_completed`, done callbacks, `asyncio.wrap_future`) sees it complete.
        if future.cancelled():
            if ConcurrentFuture.cancel(self):
                self.set_running_or_notify_cancel()
        elif (exception := future.exception()) is not None:
            self.set_exception(exception)
        else:
            self.set_result(future.result())

    def result(self, timeout: float | None = None) -> T:
        return self._future.result(timeout)

    async def result_async(self, timeout: float | None = None) -> T:
        async with asyncio.timeout(timeout):
            return await asyncio.wrap_future(self._future)

    def future(self) -> ConcurrentFuture[T]:
        return self._future

    def __await__(self):
        return APIFuture.__await__(self)

    # The `concurrent.futures.Future` API, kept working for old callers and warning them off.
    def _warn(self, method: str) -> None:
        # Once per future: the stdlib bridges (`asyncio.wrap_future`) call `done()`,
        # `exception()` and `cancel()` from inside callbacks, where a warning raised as an
        # error would be swallowed and leave their future pending; the first use, in the
        # caller's own stack, has already warned by then.
        if self._warned:
            return
        self._warned = True
        warnings.warn(
            f"DEPRECATED: .{method}() on this future uses a temporary compatibility wrapper "
            "for tinker's APIFuture, which a subsequent release removes. Await the future "
            "or call .result(); if you need a concurrent.futures.Future, call "
            ".future().",
            FutureWarning,
            stacklevel=3,
        )

    def cancel(self) -> bool:
        self._warn("cancel")
        return self._future.cancel()

    def cancelled(self) -> bool:
        self._warn("cancelled")
        return self._future.cancelled()

    def running(self) -> bool:
        self._warn("running")
        return self._future.running()

    def done(self) -> bool:
        self._warn("done")
        return self._future.done()

    def exception(self, timeout: float | None = None) -> BaseException | None:
        self._warn("exception")
        return self._future.exception(timeout)

    def add_done_callback(self, fn: Callable[[ConcurrentFuture[T]], object]) -> None:
        self._warn("add_done_callback")
        super().add_done_callback(fn)
