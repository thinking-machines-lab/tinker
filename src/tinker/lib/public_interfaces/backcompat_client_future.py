"""The Future half of `BackcompatTrainingClientFuture` and `BackcompatSamplingClientFuture`."""

from __future__ import annotations

import warnings
from concurrent.futures import Future as ConcurrentFuture
from typing import Any, TypeVar

from ..sync_only import sync_only
from .api_future import APIFuture, AwaitableConcurrentFuture

C = TypeVar("C")


class BackcompatClientFuture(APIFuture[C]):
    """An `APIFuture` of a client that can also be used, deprecated, as the client.

    The two subclasses pair this class with the client's class and forward every public
    method of the client to the real client explicitly, one deprecated method each. This
    base supplies what explicit forwarders cannot:

    - The `APIFuture` API (`result`, `result_async`, `future`), implemented
      once over the pending client rather than once per wrapper, and making the wrapper an
      `APIFuture` for `isinstance` checks and annotations.
    - The client's instance attributes. A wrapper never runs the client's `__init__`, so
      the attributes it sets, such as `model_id`, do not exist on the wrapper;
      `__getattr__` reads them from the finished client, deprecated like the methods.
    - A `repr` that shows the future's state instead of a client that was never set up.
    """

    def __init__(self, future: AwaitableConcurrentFuture[C]) -> None:
        self._future = future

    @sync_only
    def result(self, timeout: float | None = None) -> C:
        return self._future.result(timeout)

    async def result_async(self, timeout: float | None = None) -> C:
        return await self._future.result_async(timeout)

    def future(self) -> ConcurrentFuture[C]:
        return self._future.future()

    def __repr__(self) -> str:
        future = self._future.future()
        if not future.done():
            state = "pending"
        elif future.cancelled():
            state = "cancelled"
        elif (exception := future.exception()) is not None:
            state = f"failed: {exception!r}"
        else:
            state = repr(future.result())
        return f"<{type(self).__name__}: {state}>"

    def __getattr__(self, name: str) -> Any:
        # Attributes the client's __init__ sets, such as `model_id`, which the forwarding
        # methods don't cover.
        if name.startswith("_"):
            raise AttributeError(name)
        warnings.warn(
            f"Reading `.{name}` from the future returned by a client-creation method is "
            "deprecated and will stop working in a future tinker SDK version. Call "
            ".result() on the future, or await it, to get the client first.",
            FutureWarning,
            stacklevel=2,
        )
        return getattr(self._awaited_client(), name)

    def _awaited_client(self) -> C:
        """The client, for a deprecated forwarding method."""
        return self.result()

    async def _awaited_client_async(self) -> C:
        return await self._future.result_async()
