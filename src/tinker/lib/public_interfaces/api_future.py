"""
API Future classes for handling async operations with retry logic.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future as ConcurrentFuture
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Dict, Generic, List, Type, TypeVar, cast

import tinker
from tinker import types

from ..._models import BaseModel
from ..retry_handler import RetryableException
from ..sync_only import sync_only

if TYPE_CHECKING:
    from tinker.lib.internal_client_holder import InternalClientHolder

logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")

# Sentinel object to indicate that the function hasn't been called yet
_UNCOMPUTED = object()


class ResolvedFuture(Generic[T], ConcurrentFuture[T]):
    def __init__(self, result: T):
        self._result = result

    def result(self, timeout: float | None = None) -> T:
        # This is typed to not be None, but it might be valid to return None for some T
        return self._result  # type: ignore


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


class _APIFuture(APIFuture[T]):  # pyright: ignore[reportUnusedClass]
    def __init__(
        self,
        model_cls: Type[T],
        holder: InternalClientHolder,
        untyped_future: types.UntypedAPIFuture,
        request_start_time: float,
        request_type: str,
    ):
        self.model_cls = model_cls
        self.holder = holder
        self.untyped_future = untyped_future
        self.request_type = request_type
        self._cached_result: Any = _UNCOMPUTED

        # This helps us collect telemetry about how long (1) it takes the
        # client to serialize the request, (2) round-trip time to the server
        # and back, and (3) how long the server takes to process the request.
        # We send this delta in a header to the server when retrieving the promise
        # result.
        self.request_start_time = request_start_time
        self.request_future_start_time = time.time()
        self.request_queue_roundtrip_time = self.request_future_start_time - request_start_time
        self._future = self.holder.run_coroutine_threadsafe(self._result_async())

    async def _result_async(self, timeout: float | None = None) -> T:
        """Get the result of this future, with automatic retries for transient errors."""
        if self._cached_result is not _UNCOMPUTED:
            return cast(T, self._cached_result)

        start_time = time.time()
        iteration = -1
        connection_error_retries = 0

        while True:
            iteration += 1

            if timeout is not None and time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Timeout of {timeout} seconds reached while waiting for result of {self.request_id=}"
                )

            # Headers for telemetry
            headers = {
                "X-Tinker-Request-Iteration": str(iteration),
                "X-Tinker-Request-Type": self.request_type,
            }
            if iteration == 0:
                headers["X-Tinker-Create-Promise-Roundtrip-Time"] = str(
                    self.request_queue_roundtrip_time
                )

            # Function hasn't been called yet, execute it now
            try:
                with self.holder.aclient() as client:
                    response = await client.futures.with_raw_response.retrieve(
                        request_id=self.request_id, timeout=45, extra_headers=headers, max_retries=0
                    )
            except tinker.APIStatusError as e:
                connection_error_retries = 0
                # Retry 408s until we time out
                if e.status_code == 408:
                    continue
                if e.status_code == 410:
                    raise RetryableException(
                        message=f"Promise expired/broken for request {self.untyped_future.request_id}"
                    ) from e
                if e.status_code in range(500, 600):
                    continue
                raise ValueError(
                    f"Error retrieving result: {e} with status code {e.status_code=} for {self.request_id=} and expected type {self.model_cls=}"
                ) from e
            except tinker.APIConnectionError:
                # Retry all connection errors with exponential backoff
                await asyncio.sleep(min(2**connection_error_retries, 30))
                connection_error_retries += 1
                continue

            # Function hasn't been called yet, execute it now
            result_dict: Dict[str, Any] = await response.json()  # type: ignore

            if "type" in result_dict and result_dict["type"] == "try_again":
                logger.warning(f"Retrying request {self.request_id=} because of try_again")
                continue

            if "error" in result_dict:
                raise ValueError(
                    f"Error retrieving result: {result_dict} for {self.request_id=} and expected type {self.model_cls=}"
                )

            try:
                # Check if model_cls is a BaseModel subclass before calling model_validate
                if inspect.isclass(self.model_cls) and issubclass(self.model_cls, BaseModel):
                    self._cached_result = self.model_cls.model_validate(result_dict)
                else:
                    # For non-BaseModel types, just return the result directly
                    self._cached_result = result_dict
                return cast(T, self._cached_result)
            except Exception as e:
                raise ValueError(
                    f"Error retrieving result: {e} for {self.request_id=} and expected type {self.model_cls=}"
                ) from e

    @property
    def request_id(self) -> str:
        return self.untyped_future.request_id

    @sync_only
    def result(self, timeout: float | None = None) -> T:
        return self._future.result(timeout)

    async def result_async(self, timeout: float | None = None) -> T:
        return await asyncio.wait_for(self._future, timeout)


class _CombinedAPIFuture(APIFuture[T]):  # pyright: ignore[reportUnusedClass]
    def __init__(
        self,
        futures: List[APIFuture[T]],
        transform: Callable[[List[T]], T],
        holder: InternalClientHolder,
    ):
        self.futures = futures
        self.transform = transform
        self.holder = holder

    @sync_only
    def result(self, timeout: float | None = None) -> T:
        return self.holder.run_coroutine_threadsafe(self.result_async(timeout)).result()

    async def result_async(self, timeout: float | None = None) -> T:
        results = await asyncio.gather(*[future.result_async(timeout) for future in self.futures])
        return self.transform(results)


class _AsyncFuture(Generic[T]):  # pyright: ignore[reportUnusedClass]
    """
    A simpler async future for direct async operations that don't need promise retrieval.
    Used for telemetry operations.
    """

    def __init__(
        self,
        holder: InternalClientHolder,
        coro: Coroutine[Any, Any, T],
        request_type: str,
    ):
        self.holder = holder
        self.request_type = request_type
        self._cached_result: Any = _UNCOMPUTED
        self._future = self.holder.run_coroutine_threadsafe(self._execute(coro))

    async def _execute(self, coro: Coroutine[Any, Any, T]) -> T:
        """Execute the coroutine with retry logic."""
        if self._cached_result is not _UNCOMPUTED:
            return cast(T, self._cached_result)

        max_retries = 3
        retry_count = 0

        while retry_count <= max_retries:
            try:
                self._cached_result = await coro
                return cast(T, self._cached_result)
            except tinker.APIStatusError as e:
                # Retry 5xx errors
                if e.status_code in range(500, 600) and retry_count < max_retries:
                    retry_count += 1
                    await asyncio.sleep(min(2**retry_count, 10))
                    continue
                raise
            except tinker.APIConnectionError:
                # Retry connection errors with exponential backoff
                if retry_count < max_retries:
                    retry_count += 1
                    await asyncio.sleep(min(2**retry_count, 10))
                    continue
                raise

        raise RuntimeError(f"Max retries exceeded for {self.request_type}")

    @sync_only
    def result(self, timeout: float | None = None) -> T:
        return self._future.result(timeout)

    async def result_async(self, timeout: float | None = None) -> T:
        if timeout is not None:
            return await asyncio.wait_for(self._future, timeout)
        return await self._future
