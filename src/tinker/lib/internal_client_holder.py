"""Internal client holder for managing AsyncTinker clients."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
import uuid
from collections.abc import Coroutine, Generator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Awaitable, Callable, TypeVar

import httpx

from tinker._client import AsyncTinker
from tinker._exceptions import APIConnectionError, APIStatusError
from tinker.lib.async_tinker_provider import AsyncTinkerProvider
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.public_interfaces.api_future import AwaitableConcurrentFuture
from tinker.lib.telemetry import Telemetry, init_telemetry
from tinker.lib.telemetry_provider import TelemetryProvider

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_REQUESTS_PER_HTTPX_CLIENT = 50


class ClientConnectionPool:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        max_requests_per_client: int,
        constructor_kwargs: dict[str, Any],
    ):
        self._loop = loop
        self._max_requests_per_client = max_requests_per_client
        self._constructor_kwargs = constructor_kwargs
        self._clients: list[AsyncTinker] = []
        self._client_active_refcount: list[int] = []

    @contextmanager
    def aclient(self) -> Generator[AsyncTinker, None, None]:
        assert _current_loop() is self._loop, "AsyncTinker client called from incorrect event loop"
        client_idx = -1
        for i, ref_count in enumerate(self._client_active_refcount):
            if ref_count < self._max_requests_per_client:
                client_idx = i
                break
        if client_idx == -1:
            self._clients.append(AsyncTinker(**self._constructor_kwargs))
            client_idx = len(self._clients) - 1
            self._client_active_refcount.append(0)

        self._client_active_refcount[client_idx] += 1
        try:
            yield self._clients[client_idx]
        finally:
            self._client_active_refcount[client_idx] -= 1


class InternalClientHolderThreadSingleton:
    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started: bool = False
        self._lifecycle_lock: threading.Lock = threading.Lock()

    def _ensure_started(self):
        if self._started:
            return

        with self._lifecycle_lock:
            if self._started:
                return
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._background_thread_func, daemon=True)
            self._thread.start()
            self._started = True

    def _background_thread_func(self):
        assert self._loop is not None, "Loop must not be None"
        self._loop.run_forever()

    def get_loop(self) -> asyncio.AbstractEventLoop:
        self._ensure_started()
        assert self._loop is not None, "Loop must not be None"
        return self._loop


_internal_client_holder_thread_singleton = InternalClientHolderThreadSingleton()


class InternalClientHolder(AsyncTinkerProvider, TelemetryProvider):
    def __init__(self, **kwargs: Any) -> None:
        self._constructor_kwargs = kwargs
        # So we can use async eventloop for parallel sampling requests
        # in sync code.
        self._loop: asyncio.AbstractEventLoop = _internal_client_holder_thread_singleton.get_loop()
        self._client_pools: dict[ClientConnectionPoolType, ClientConnectionPool] = {}
        self._sample_backoff_until: float | None = None
        self._sample_dispatch_semaphore: asyncio.Semaphore = asyncio.Semaphore(400)
        self._session_id: str = str(uuid.uuid4())
        self._telemetry: Telemetry | None = init_telemetry(self, session_id=self._session_id)

        self._training_client_counter: int = 0
        self._training_client_lock: threading.Lock = threading.Lock()

        self._unordered_id_counter: int = 0

    def _get_client_connection_pool(
        self, client_pool_type: ClientConnectionPoolType
    ) -> ClientConnectionPool:
        if client_pool_type not in self._client_pools:
            max_requests_per_client = (
                1
                if client_pool_type == ClientConnectionPoolType.TRAIN
                else MAX_REQUESTS_PER_HTTPX_CLIENT
            )
            self._client_pools[client_pool_type] = ClientConnectionPool(
                self.get_loop(), max_requests_per_client, self._constructor_kwargs
            )
        return self._client_pools[client_pool_type]

    def get_training_client_id(self) -> int:
        with self._training_client_lock:
            training_client_id = self._training_client_counter
            self._training_client_counter += 1
            return training_client_id

    def aclient(
        self, client_pool_type: ClientConnectionPoolType
    ) -> AbstractContextManager[AsyncTinker]:
        return self._get_client_connection_pool(client_pool_type).aclient()

    def get_loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    def get_telemetry(self) -> Telemetry | None:
        return self._telemetry

    def run_coroutine_threadsafe(
        self,
        coro: Coroutine[Any, Any, T],
    ) -> AwaitableConcurrentFuture[T]:
        return AwaitableConcurrentFuture(asyncio.run_coroutine_threadsafe(coro, self.get_loop()))

    def close(self):
        if telemetry := self._telemetry:
            telemetry.stop()

    def __del__(self):
        self.close()

    def make_training_client_idempotency_key(self, training_client_id: int, request_id: int) -> str:
        return f"{self._session_id}:{training_client_id}:{request_id}"

    def make_idempotency_key(self) -> str:
        self._unordered_id_counter += 1
        return f"{self._session_id}:unordered:{self._unordered_id_counter}"

    @staticmethod
    def _is_retryable_status_code(status_code: int) -> bool:
        return status_code in (408, 409, 429) or (500 <= status_code < 600)

    @staticmethod
    def _is_retryable_exception(exception: Exception) -> bool:
        RETRYABLE_EXCEPTIONS = (
            asyncio.TimeoutError,
            APIConnectionError,
            httpx.TimeoutException,
        )
        if isinstance(exception, RETRYABLE_EXCEPTIONS):
            return True
        if isinstance(exception, APIStatusError):
            return InternalClientHolder._is_retryable_status_code(exception.status_code)
        return False

    async def execute_with_retries(
        self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        MAX_WAIT_TIME = 60 * 5
        start_time = time.time()
        attempt_count = 0
        while True:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                is_retryable = self._is_retryable_exception(e)
                current_time = time.time()
                elapsed_time = current_time - start_time
                if telemetry := self.get_telemetry():
                    telemetry.log(
                        "InternalClientHolder.execute_with_retries.exception",
                        event_data={
                            "func": getattr(
                                func, "__qualname__", getattr(func, "__name__", type(func).__name__)
                            ),
                            "exception": str(e),
                            "exception_type": type(e).__name__,
                            "exception_stack": "".join(
                                traceback.format_exception(type(e), e, e.__traceback__)
                            )
                            if e.__traceback__
                            else None,
                            "status_code": getattr(e, "status_code", None),
                            "is_retryable": is_retryable,
                            "attempt_count": attempt_count,
                            "start_time": start_time,
                            "current_time": current_time,
                            "elapsed_time": elapsed_time,
                        },
                        severity="WARNING" if is_retryable else "ERROR",
                    )
                if is_retryable and elapsed_time < MAX_WAIT_TIME:
                    # Apply exponential backoff
                    time_to_wait = min(2**attempt_count, 30)
                    attempt_count += 1
                    # Don't wait too long if we're almost at the max wait time
                    time_to_wait = min(time_to_wait, start_time + MAX_WAIT_TIME - current_time)
                    await asyncio.sleep(time_to_wait)
                    continue

                raise e


def _current_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None
