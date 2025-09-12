from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from contextlib import AbstractContextManager
from enum import Enum
from typing import Any, Protocol, TypeVar

import tinker

from .public_interfaces.api_future import AwaitableConcurrentFuture

T = TypeVar("T")


class ClientConnectionPoolType(Enum):
    SAMPLE = "sample"
    TRAIN = "train"
    RETRIEVE_PROMISE = "retrieve_promise"
    TELEMETRY = "telemetry"


class AsyncTinkerProvider(Protocol):
    # both of the following methods should be threadsafe
    def get_loop(self) -> asyncio.AbstractEventLoop: ...

    def run_coroutine_threadsafe(
        self,
        coro: Coroutine[Any, Any, T],
    ) -> AwaitableConcurrentFuture[T]:
        return AwaitableConcurrentFuture(asyncio.run_coroutine_threadsafe(coro, self.get_loop()))

    # must be called and used within the provided event loop
    def aclient(self, client_pool_type: ClientConnectionPoolType) -> AbstractContextManager[tinker.AsyncTinker]: ...
