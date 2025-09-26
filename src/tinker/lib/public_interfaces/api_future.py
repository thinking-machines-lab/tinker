"""
API Future classes for handling async operations with retry logic.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import Future as ConcurrentFuture
from typing import Generic, TypeVar

T = TypeVar("T")


class APIFuture(ABC, Generic[T]):
    @abstractmethod
    async def result_async(self, timeout: float | None = None) -> T:
        raise NotImplementedError

    @abstractmethod
    def result(self, timeout: float | None = None) -> T:
        raise NotImplementedError

    def __await__(self):
        return self.result_async().__await__()


class AwaitableConcurrentFuture(APIFuture[T]):
    def __init__(self, future: ConcurrentFuture[T]):
        self._future: ConcurrentFuture[T] = future

    def result(self, timeout: float | None = None) -> T:
        return self._future.result(timeout)

    async def result_async(self, timeout: float | None = None) -> T:
        async with asyncio.timeout(timeout):
            return await asyncio.wrap_future(self._future)

    def future(self) -> ConcurrentFuture[T]:
        return self._future
