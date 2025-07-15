from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from contextlib import AbstractContextManager
from typing import Any, Protocol, TypeVar

from tinker._client import AsyncTinker

from .public_interfaces.api_future import AwaitableConcurrentFuture
from .client_connection_pool_type import ClientConnectionPoolType

T = TypeVar("T")


class AsyncTinkerProvider(Protocol):
    # both of the following methods should be threadsafe
    def get_loop(self) -> asyncio.AbstractEventLoop: ...

    def run_coroutine_threadsafe(
        self,
        coro: Coroutine[Any, Any, T],
    ) -> AwaitableConcurrentFuture[T]: ...

    # must be called and used within the provided event loop
    def aclient(
        self, client_pool_type: ClientConnectionPoolType
    ) -> AbstractContextManager[AsyncTinker]: ...
