"""Internal client holder for managing AsyncTinker clients."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import time
import traceback
from collections.abc import Coroutine, Generator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Awaitable, Callable, TypeVar

import httpx

from tinker import types
from tinker._client import AsyncTinker
from tinker._exceptions import APIConnectionError, APIStatusError
from tinker._version import __version__ as tinker_sdk_version
from tinker.lib.async_tinker_provider import AsyncTinkerProvider
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.public_interfaces.api_future import AwaitableConcurrentFuture
from tinker.lib.telemetry import Telemetry, init_telemetry, is_user_error
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
    def __init__(self, user_metadata: dict[str, str] | None = None, **kwargs: Any) -> None:
        self._constructor_kwargs = kwargs
        # So we can use async eventloop for parallel sampling requests
        # in sync code.
        self._loop: asyncio.AbstractEventLoop = _internal_client_holder_thread_singleton.get_loop()
        self._client_pools: dict[ClientConnectionPoolType, ClientConnectionPool] = {}
        self._sample_backoff_until: float | None = None
        self._sample_dispatch_semaphore: asyncio.Semaphore = asyncio.Semaphore(400)
        self._telemetry: Telemetry | None = None
        session_id, session_heartbeat_task = self.run_coroutine_threadsafe(
            self._create_session(user_metadata)
        ).result()
        self._session_id: str = session_id
        self._session_heartbeat_task: asyncio.Task[None] = session_heartbeat_task
        self._telemetry = init_telemetry(self, session_id=self._session_id)

        self._training_client_counter: int = 0
        self._training_client_lock: threading.Lock = threading.Lock()

        self._sampling_client_counter: int = 0

    async def _session_heartbeat(self, session_id: str):
        SESSION_HEARTBEAT_PERIOD_SEC = 10
        SESSION_MISSED_HEARTBEAT_WARNING_THRESHOLD_SEC = 60 * 2
        last_heartbeat_time = time.monotonic()
        while True:
            await asyncio.sleep(SESSION_HEARTBEAT_PERIOD_SEC)

            last_exception: str | None = None
            try:
                with self.aclient(ClientConnectionPoolType.SESSION) as client:
                    await client.service.session_heartbeat(
                        session_id=session_id, max_retries=0, timeout=10
                    )
                last_heartbeat_time = time.monotonic()
            except Exception as e:
                last_exception = f"{type(e).__name__}: {str(e)}"
                pass
            if (
                time.monotonic() - last_heartbeat_time
                > SESSION_MISSED_HEARTBEAT_WARNING_THRESHOLD_SEC
            ):
                logger.warning(
                    f"Session heartbeat failed for {time.monotonic() - last_heartbeat_time} seconds for session {session_id}. Last exception: {last_exception}.\n"
                    + "Your connection may be unreliable or Tinker is down. If this persists, the session will be terminated."
                )

    async def _create_sampling_session(
        self, model_path: str | None = None, base_model: str | None = None
    ) -> str:
        sampling_session_seq_id = self._sampling_client_counter
        self._sampling_client_counter += 1
        with self.aclient(ClientConnectionPoolType.SESSION) as client:
            request = types.CreateSamplingSessionRequest(
                session_id=self._session_id,
                sampling_session_seq_id=sampling_session_seq_id,
                model_path=model_path,
                base_model=base_model,
            )
            result = await client.service.create_sampling_session(request=request)
            return result.sampling_session_id

    async def _create_session(
        self, user_metadata: dict[str, str] | None = None
    ) -> tuple[str, asyncio.Task[None]]:
        if (tags_str := os.environ.get("TINKER_TAGS")) is not None:
            tags: set[str] = set(tags_str.split(","))
        else:
            tags = set()
        with self.aclient(ClientConnectionPoolType.SESSION) as client:
            request = types.CreateSessionRequest(
                tags=list(tags), user_metadata=user_metadata or {}, sdk_version=tinker_sdk_version
            )
            result = await client.service.create_session(request=request)
        if result.info_message:
            logger.info(result.info_message)
        if result.warning_message:
            logger.warning(result.warning_message)
        if result.error_message:
            logger.error(result.error_message)
        session_id = result.session_id
        session_heartbeat_task = asyncio.create_task(self._session_heartbeat(session_id))
        return session_id, session_heartbeat_task

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

    def get_session_id(self) -> str:
        return self._session_id

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
        self.run_coroutine_threadsafe(self._async_cleanup()).result()
        if telemetry := self._telemetry:
            telemetry.stop()

    def __del__(self):
        self.close()

    async def _async_cleanup(self):
        if self._session_heartbeat_task:
            self._session_heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._session_heartbeat_task

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
                user_error = is_user_error(e)
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
                            "is_user_error": user_error,
                            "attempt_count": attempt_count,
                            "start_time": start_time,
                            "current_time": current_time,
                            "elapsed_time": elapsed_time,
                        },
                        severity="WARNING" if is_retryable or user_error else "ERROR",
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
