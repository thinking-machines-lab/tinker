"""Internal client holder for managing AsyncTinker clients."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import threading
import time
import traceback
import weakref
from collections.abc import Coroutine, Generator
from contextlib import AbstractContextManager, asynccontextmanager, contextmanager
from typing import Any, Awaitable, Callable, TypeVar

import httpx

from tinker import types
from tinker._client import AsyncTinker
from tinker._exceptions import APIConnectionError, APIStatusError
from tinker._version import __version__ as tinker_sdk_version
from tinker.lib._auth_token_provider import (
    ApiKeyAuthProvider,
    AuthTokenProvider,
    resolve_auth_provider,
)
from tinker.lib._jwt_auth import JwtAuthProvider
from tinker.lib.async_tinker_provider import AsyncTinkerProvider
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.public_interfaces.api_future import AwaitableConcurrentFuture
from tinker.lib.telemetry import Telemetry, init_telemetry, is_user_error
from tinker.lib.telemetry_provider import TelemetryProvider

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_REQUESTS_PER_HTTPX_CLIENT = 50
MAX_CONNECTION_ERROR_RETRIES = 16


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
        self._connection_error_retries_remaining: int = MAX_CONNECTION_ERROR_RETRIES

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
            if self._connection_error_retries_remaining < MAX_CONNECTION_ERROR_RETRIES:
                self._connection_error_retries_remaining += 1
        except APIStatusError as e:
            # This indicates request rejected by Cloudflare. Reset the connection and retry
            if e.status_code == 400 and e.response.headers.get("content-length", "0") == "0":
                # Ensure a new connection gets opened
                self._clients[client_idx] = AsyncTinker(**self._constructor_kwargs)
                if self._connection_error_retries_remaining > 0:
                    self._connection_error_retries_remaining -= 1
                    raise APIConnectionError(request=e.request) from e
            raise e
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

    def _set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Inject an external event loop (e.g. the sidecar subprocess loop).

        Must be called before any InternalClientHolder is created.
        Prevents _ensure_started from spawning a background thread — the
        caller's loop is used directly.
        """
        with self._lifecycle_lock:
            if self._started:
                raise RuntimeError("Cannot set_loop after singleton has started")
            self._loop = loop
            self._started = True  # prevent _ensure_started from creating a thread

    def get_loop(self) -> asyncio.AbstractEventLoop:
        self._ensure_started()
        assert self._loop is not None, "Loop must not be None"
        return self._loop


_internal_client_holder_thread_singleton = InternalClientHolderThreadSingleton()


class _ShadowHolderSingleton:
    """Singleton to cache shadow InternalClientHolders by constructor args."""

    def __init__(self):
        self._lock: threading.Lock = threading.Lock()
        # Key is (session_id, json-serialized kwargs)
        self._cache: dict[tuple[str, str], weakref.ref[InternalClientHolder]] = {}

    def get_or_create(self, session_id: str, kwargs: dict[str, Any]) -> InternalClientHolder:
        key = (session_id, json.dumps(kwargs, sort_keys=True))
        with self._lock:
            if key in self._cache:
                holder = self._cache[key]()
                if holder is not None:
                    return holder
            holder = InternalClientHolder(session_id=session_id, **kwargs)
            self._cache[key] = weakref.ref(holder)
            return holder


_shadow_holder_singleton = _ShadowHolderSingleton()


class BytesSemaphore:
    def __init__(self, max_bytes: int):
        self._bytes: int = max_bytes
        self._condition: asyncio.Condition = asyncio.Condition()
        self._release_task: asyncio.Task[None] | None = None

    async def _release(self):
        async with self._condition:
            self._condition.notify_all()

    @asynccontextmanager
    async def acquire(self, bytes: int):
        async with self._condition:
            while self._bytes < 0:
                await self._condition.wait()
        self._bytes -= bytes

        try:
            yield
        finally:
            self._bytes += bytes
            # Make sure the release task is never cancelled.
            self._release_task = asyncio.create_task(self._release())


class InternalClientHolder(AsyncTinkerProvider, TelemetryProvider):
    def __init__(
        self,
        user_metadata: dict[str, str] | None = None,
        project_id: str | None = None,
        *,
        session_id: str | None = None,
        api_key: str | None = None,
        _client_config: dict[str, str | int | bool] | None = None,
        _jwt_auth_seed: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Resolve from env now so shadow_kwargs carries the actual credential
        # across pickle boundaries (workers may not have the env var set).
        self._api_key = api_key or os.environ.get("TINKER_API_KEY")
        self._constructor_kwargs = dict(kwargs)
        self._loop: asyncio.AbstractEventLoop = _internal_client_holder_thread_singleton.get_loop()
        self._client_pools: dict[ClientConnectionPoolType, ClientConnectionPool] = {}
        self._sample_backoff_until: float | None = None
        self._sample_dispatch_semaphore: asyncio.Semaphore = asyncio.Semaphore(400)
        self._sample_dispatch_throttled_semaphore: asyncio.Semaphore = asyncio.Semaphore(10)
        self._training_client_lock: threading.Lock = threading.Lock()
        self._telemetry: Telemetry | None = None

        # Fetch server-side client config before any server contact so that
        # flags are available for subsequent setup steps.  Shadow holders
        # receive the config via kwargs to avoid a redundant fetch (and
        # potential deadlock on the event loop thread).
        if _client_config is not None:
            self._client_config = types.ClientConfigResponse.model_validate(_client_config)
        else:
            self._assert_not_on_event_loop("fetch client config")
            config_auth = resolve_auth_provider(api_key, enforce_cmd=False)
            self._client_config = self.run_coroutine_threadsafe(
                self._fetch_client_config(config_auth)
            ).result()

        self._sample_dispatch_bytes_semaphore: BytesSemaphore = BytesSemaphore(
            self._client_config.sample_dispatch_bytes_semaphore_size
        )
        self._inflight_response_bytes_semaphore: BytesSemaphore = BytesSemaphore(
            self._client_config.inflight_response_bytes_semaphore_size
        )

        if not self._client_config.pjwt_auth_enabled:
            # Without JWT exchange, only API keys are accepted by the server.
            # Replace any cmd-based provider with a plain API key provider.
            self._default_auth = ApiKeyAuthProvider(api_key=self._api_key)
        else:
            # Create a dedicated pool for JWT exchange with the appropriate
            # credential provider.  The lambda captures the pool so it stays alive.
            use_cmd = self._client_config.credential_default_source == "credential_cmd"
            auth_pool_auth = resolve_auth_provider(self._api_key, use_cmd)
            auth_kwargs = {**self._constructor_kwargs, "_auth": auth_pool_auth}
            auth_pool = ClientConnectionPool(self.get_loop(), 1, auth_kwargs)
            auth_aclient = lambda: auth_pool.aclient()  # noqa: E731
            self._default_auth = JwtAuthProvider(auth_aclient, seed_token=_jwt_auth_seed)
            if _jwt_auth_seed:
                # Shadow holder: start refresh in background, don't block.
                self.run_coroutine_threadsafe(self._default_auth.init())
            else:
                # Primary holder: must have a valid JWT before proceeding.
                self._assert_not_on_event_loop("exchange JWT")
                self.run_coroutine_threadsafe(
                    self.execute_with_retries(self._default_auth.init)
                ).result()

        if session_id is not None:
            # Shadow mode: reuse existing session, can't create new clients
            self._session_id: str = session_id
            self._training_client_counter: int | None = None
            self._sampling_client_counter: int | None = None
        else:
            # Normal mode: create new session.
            self._assert_not_on_event_loop("create a new session")
            self._session_id = self.run_coroutine_threadsafe(
                self._create_session(user_metadata=user_metadata, project_id=project_id)
            ).result()
            self._training_client_counter = 0
            self._sampling_client_counter = 0

        if self._loop.is_running() and _current_loop() is self._loop:
            # Already on the event loop thread — .result() would deadlock.
            # Create the heartbeat task directly instead of via run_coroutine_threadsafe.
            self._session_heartbeat_task: asyncio.Task[None] = asyncio.create_task(
                self._session_heartbeat(self._session_id)
            )
        else:
            self._session_heartbeat_task = self.run_coroutine_threadsafe(
                self._start_heartbeat()
            ).result()
        self._telemetry: Telemetry | None = init_telemetry(self, session_id=self._session_id)

    @classmethod
    def get_shadow_holder(cls, session_id: str, kwargs: dict[str, Any]) -> InternalClientHolder:
        """Get or create a shadow holder from the singleton cache."""
        return _shadow_holder_singleton.get_or_create(session_id, kwargs)

    def _assert_not_on_event_loop(self, action: str) -> None:
        """Raise if called from the event loop thread (would deadlock on .result())."""
        if self._loop.is_running() and _current_loop() is self._loop:
            raise RuntimeError(
                f"Cannot {action} from the event loop thread. "
                "Use session_id= to create a shadow holder instead."
            )

    @property
    def shadow_kwargs(self) -> dict[str, Any]:
        """Constructor kwargs for shadow holders, including cached server config and JWT seed."""
        result = {
            **self._constructor_kwargs,
            "api_key": self._api_key,
            "_client_config": self._client_config.model_dump(),
        }
        if isinstance(self._default_auth, JwtAuthProvider):
            result["_jwt_auth_seed"] = self._default_auth._token
        return result

    @asynccontextmanager
    async def _sample_dispatch_count_rate_limit(self):
        async with self._sample_dispatch_semaphore:
            yield

    @asynccontextmanager
    async def _sample_dispatch_count_throttled_rate_limit(self):
        async with self._sample_dispatch_throttled_semaphore:
            yield

    def _sample_backoff_requested_recently(self) -> bool:
        return (
            self._sample_backoff_until is not None
            and time.monotonic() - self._sample_backoff_until < 10
        )

    @asynccontextmanager
    async def _sample_dispatch_bytes_rate_limit(self, bytes: int):
        if self._sample_backoff_requested_recently():
            # Rate limit more aggressively if we received backoff response recently
            bytes *= 20
        async with self._sample_dispatch_bytes_semaphore.acquire(bytes):
            yield

    @asynccontextmanager
    async def sample_dispatch_rate_limit(self, estimated_bytes_count: int):
        async with contextlib.AsyncExitStack() as stack:
            await stack.enter_async_context(self._sample_dispatch_count_rate_limit())
            if self._sample_backoff_requested_recently():
                await stack.enter_async_context(self._sample_dispatch_count_throttled_rate_limit())
            await stack.enter_async_context(
                self._sample_dispatch_bytes_rate_limit(estimated_bytes_count)
            )

            yield

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
        if model_path and not model_path.startswith("tinker://"):
            raise ValueError("model_path must start with 'tinker://'")
        # _create_sampling_session can only be called via a ServiceClient.
        # ServiceClient will never have a shadow holder, so we can safely assert.
        assert self._sampling_client_counter is not None
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

    async def _start_heartbeat(self) -> asyncio.Task[None]:
        """Start the session heartbeat task."""
        return asyncio.create_task(self._session_heartbeat(self._session_id))

    async def _fetch_client_config(self, auth: AuthTokenProvider) -> types.ClientConfigResponse:
        """Call /api/v1/client/config and return server feature flags.

        Creates a one-off connection pool with the given auth.  Retries
        transient failures via execute_with_retries.
        """
        kwargs = {**self._constructor_kwargs, "_auth": auth}
        pool = ClientConnectionPool(self.get_loop(), 1, kwargs)

        async def _once() -> types.ClientConfigResponse:
            with pool.aclient() as client:
                return await client.service.client_config(
                    request=types.ClientConfigRequest(sdk_version=tinker_sdk_version)
                )

        return await self.execute_with_retries(_once)

    async def _create_session(
        self,
        user_metadata: dict[str, str] | None = None,
        project_id: str | None = None,
    ) -> str:
        if (tags_str := os.environ.get("TINKER_TAGS")) is not None:
            tags: set[str] = set(tags_str.split(","))
        else:
            tags = set()
        with self.aclient(ClientConnectionPoolType.SESSION) as client:
            request = types.CreateSessionRequest(
                tags=list(tags),
                user_metadata=user_metadata or {},
                sdk_version=tinker_sdk_version,
                project_id=project_id,
            )
            result = await client.service.create_session(request=request)
        if result.info_message:
            logger.info(result.info_message)
        if result.warning_message:
            logger.warning(result.warning_message)
        if result.error_message:
            logger.error(result.error_message)
        return result.session_id

    def _get_client_connection_pool(
        self, client_pool_type: ClientConnectionPoolType
    ) -> ClientConnectionPool:
        if client_pool_type not in self._client_pools:
            max_requests_per_client = (
                1
                if client_pool_type == ClientConnectionPoolType.TRAIN
                else MAX_REQUESTS_PER_HTTPX_CLIENT
            )
            kwargs = {**self._constructor_kwargs, "_auth": self._default_auth}
            self._client_pools[client_pool_type] = ClientConnectionPool(
                self.get_loop(), max_requests_per_client, kwargs
            )
        return self._client_pools[client_pool_type]

    def get_session_id(self) -> str:
        return self._session_id

    def get_training_client_id(self) -> int:
        # get_training_client_id can only be called via a ServiceClient.
        # ServiceClient will never have a shadow holder, so we can safely assert.
        assert self._training_client_counter is not None
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
        self.run_coroutine_threadsafe(self._async_cleanup())
        if telemetry := getattr(self, "_telemetry", None):
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

    def estimate_bytes_count_in_chunk(self, chunk: types.ModelInputChunk) -> int:
        if isinstance(chunk, types.ImageChunk):
            return len(chunk.data)
        if isinstance(chunk, types.ImageAssetPointerChunk):
            return len(chunk.location)
        return chunk.length * 10

    def estimate_bytes_count_in_model_input(self, model_input: types.ModelInput) -> int:
        return sum(self.estimate_bytes_count_in_chunk(chunk) for chunk in model_input.chunks)


def _current_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None
