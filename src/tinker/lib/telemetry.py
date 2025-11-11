import asyncio
import contextlib
import functools
import inspect
import logging
import os
import platform
import threading
import traceback
from collections import deque
from collections.abc import Awaitable
from datetime import datetime, timezone
from typing import (
    Callable,
    ParamSpec,
    TypeVar,
    cast,
    overload,
)
from uuid import uuid4

from tinker._exceptions import APIError, RequestFailedError
from tinker._version import __version__
from tinker.types import RequestErrorCategory
from tinker.types.generic_event import GenericEvent
from tinker.types.session_end_event import SessionEndEvent
from tinker.types.session_start_event import SessionStartEvent
from tinker.types.severity import Severity
from tinker.types.telemetry_batch import TelemetryBatch
from tinker.types.telemetry_event import TelemetryEvent
from tinker.types.telemetry_response import TelemetryResponse
from tinker.types.telemetry_send_request import TelemetrySendRequest
from tinker.types.unhandled_exception_event import UnhandledExceptionEvent

from .async_tinker_provider import AsyncTinkerProvider
from .client_connection_pool_type import ClientConnectionPoolType
from .sync_only import sync_only
from .telemetry_provider import TelemetryProvider

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE: int = 100
FLUSH_INTERVAL: float = 10.0
FLUSH_TIMEOUT: float = 30.0
MAX_QUEUE_SIZE: int = 10000
HTTP_TIMEOUT_SECONDS: float = 5.0


class Telemetry:
    def __init__(self, tinker_provider: AsyncTinkerProvider, session_id: str):
        self._tinker_provider: AsyncTinkerProvider = tinker_provider
        self._session_id: str = session_id
        self._session_start: datetime = datetime.now(timezone.utc)
        self._session_index: int = 0
        self._session_index_lock: threading.Lock = threading.Lock()
        self._queue: deque[TelemetryEvent] = deque()
        self._queue_lock: threading.Lock = threading.Lock()
        self._task: asyncio.Task[None] | None = None
        self._flush_event: asyncio.Event | None = None
        self._push_counter: int = 0
        self._flush_counter: int = 0
        self._counter_lock: threading.Lock = threading.Lock()
        _ = self._log(self._session_start_event())
        self._start()

    def _start(self):
        def cb():
            self._flush_event = asyncio.Event()
            self._task = asyncio.create_task(self._periodic_flush(), name="tinker-telemetry")

        _ = self._tinker_provider.get_loop().call_soon_threadsafe(cb)

    def stop(self):
        def cb():
            if task := self._task:
                _ = task.cancel()

        _ = self._tinker_provider.get_loop().call_soon_threadsafe(cb)

    async def _periodic_flush(self):
        while True:
            if self._flush_event:
                try:
                    _ = await asyncio.wait_for(self._flush_event.wait(), timeout=FLUSH_INTERVAL)
                except TimeoutError:
                    pass
                finally:
                    self._flush_event.clear()
            await self._flush()

    async def _flush(self):
        while True:
            with self._queue_lock:
                if not self._queue:
                    break
                batch_size = min(MAX_BATCH_SIZE, len(self._queue))
                events = [self._queue.popleft() for _ in range(batch_size)]
                batch = self._batch(events)
            try:
                _ = await self._send_batch_with_retry(batch)
            finally:
                # increment counter even if we fail to send the batch so we're not blocking
                # on a flush for non-APIErrors (e.g. missing API key)
                with self._counter_lock:
                    self._flush_counter += len(events)

    def _trigger_flush(self):
        if self._flush_event:
            _ = self._tinker_provider.get_loop().call_soon_threadsafe(self._flush_event.set)

    async def _wait_until_drained(self) -> bool:
        with self._counter_lock:
            target_count = self._push_counter
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < FLUSH_TIMEOUT:
            with self._counter_lock:
                if self._flush_counter >= target_count:
                    return True
            await asyncio.sleep(0.1)
        return False

    def _wait_until_drained_sync(self) -> bool:
        try:
            return asyncio.run_coroutine_threadsafe(
                self._wait_until_drained(), self._tinker_provider.get_loop()
            ).result(timeout=FLUSH_TIMEOUT)
        except (TimeoutError, asyncio.CancelledError):
            return False

    async def _send_batch_with_retry(self, batch: TelemetryBatch) -> TelemetryResponse:
        while True:
            try:
                return await self._send_batch(batch)
            except APIError as e:
                logger.warning("Failed to send telemetry batch", exc_info=e)
                await asyncio.sleep(1)
                continue

    async def _send_batch(self, batch: TelemetryBatch) -> TelemetryResponse:
        with self._tinker_provider.aclient(ClientConnectionPoolType.TELEMETRY) as client:
            request = _to_send_request(batch)
            return await client.telemetry.send(request=request, timeout=HTTP_TIMEOUT_SECONDS)

    def _log(self, *events: TelemetryEvent) -> bool:
        with self._queue_lock:
            if len(self._queue) + len(events) > MAX_QUEUE_SIZE:
                logger.warning("Telemetry queue full, dropping events")
                return False
            self._queue.extend(events)
        with self._counter_lock:
            self._push_counter += len(events)
        return True

    def log(
        self,
        event_name: str,  # should be low cardinality
        event_data: dict[str, object] | None = None,
        severity: Severity = "INFO",
    ) -> bool:
        return self._log(self._generic_event(event_name, event_data, severity))

    async def log_exception(self, exception: BaseException, severity: Severity = "ERROR") -> bool:
        logged = self._log(self._exception_or_user_error_event(exception, severity))
        # trigger flush but don't block on it
        self._trigger_flush()
        return logged

    async def log_fatal_exception(
        self, exception: BaseException, severity: Severity = "ERROR"
    ) -> bool:
        logged = self._log(
            self._exception_or_user_error_event(exception, severity), self._session_end_event()
        )
        self._trigger_flush()
        # wait for the flush to complete
        _ = await self._wait_until_drained()
        if logged:
            self._notify_exception_logged()
        return logged

    @sync_only
    def log_exception_sync(self, exception: BaseException, severity: Severity = "ERROR") -> bool:
        logged = self._log(self._exception_or_user_error_event(exception, severity))
        # trigger flush but don't block on it
        self._trigger_flush()
        return logged

    @sync_only
    def log_fatal_exception_sync(
        self, exception: BaseException, severity: Severity = "ERROR"
    ) -> bool:
        logged = self._log(
            self._exception_or_user_error_event(exception, severity), self._session_end_event()
        )
        self._trigger_flush()
        # wait for the flush to complete
        if _current_loop() is None:
            _ = self._wait_until_drained_sync()
        if logged:
            self._notify_exception_logged()
        return logged

    def _notify_exception_logged(self):
        logger.info(f"Exception logged for session ID: {self._session_id}")

    def _batch(self, events: list[TelemetryEvent]) -> TelemetryBatch:
        return TelemetryBatch(
            platform=platform.system(),
            sdk_version=__version__,
            session_id=self._session_id,
            events=events,
        )

    def _generic_event(
        self,
        event_name: str,
        event_data: dict[str, object] | None = None,
        severity: Severity = "INFO",
    ) -> GenericEvent:
        return GenericEvent(
            event="GENERIC_EVENT",
            event_id=str(uuid4()),
            event_session_index=self._next_session_index(),
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            event_name=event_name,
            event_data=event_data or {},
        )

    def _session_start_event(self) -> SessionStartEvent:
        return SessionStartEvent(
            event="SESSION_START",
            event_id=str(uuid4()),
            event_session_index=self._next_session_index(),
            timestamp=self._session_start,
            severity="INFO",
        )

    def _session_end_event(self) -> SessionEndEvent:
        end_time = datetime.now(timezone.utc)
        return SessionEndEvent(
            event="SESSION_END",
            event_id=str(uuid4()),
            event_session_index=self._next_session_index(),
            timestamp=end_time,
            severity="INFO",
            duration=str(end_time - self._session_start),
        )

    def _exception_or_user_error_event(
        self, exception: BaseException, severity: Severity
    ) -> TelemetryEvent:
        return (
            self._user_error_event(exception)
            if is_user_error(exception)
            else self._exception_event(exception, severity)
        )

    def _exception_event(
        self, exception: BaseException, severity: Severity
    ) -> UnhandledExceptionEvent:
        return UnhandledExceptionEvent(
            event="UNHANDLED_EXCEPTION",
            event_id=str(uuid4()),
            event_session_index=self._next_session_index(),
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            error_type=exception.__class__.__name__,
            error_message=str(exception),
            traceback="".join(
                traceback.format_exception(type(exception), exception, exception.__traceback__)
            )
            if exception.__traceback__
            else None,
        )

    def _user_error_event(self, exception: BaseException) -> GenericEvent:
        data: dict[str, object] = {"error_type": exception.__class__.__name__}
        if message := str(exception):
            data["message"] = message
        if user_error := _get_user_error(exception):
            status_code = getattr(user_error, "status_code", None)
            if isinstance(status_code, int):
                data["status_code"] = status_code
            if body := getattr(user_error, "body", None):
                data["body"] = body
        return self._generic_event("user_error", data, "WARNING")

    def _next_session_index(self) -> int:
        with self._session_index_lock:
            idx = self._session_index
            self._session_index += 1
            return idx

    @contextlib.contextmanager
    def capture_exceptions(self, fatal: bool = False, severity: Severity = "ERROR"):
        try:
            yield
        except Exception as e:
            if fatal:
                _ = self.log_fatal_exception_sync(e, severity)
            else:
                _ = self.log_exception_sync(e, severity)
            raise

    @contextlib.asynccontextmanager
    async def acapture_exceptions(self, fatal: bool = False, severity: Severity = "ERROR"):
        try:
            yield
        except Exception as e:
            if fatal:
                _ = await self.log_fatal_exception(e, severity)
            else:
                _ = await self.log_exception(e, severity)
            raise


def _is_telemetry_enabled() -> bool:
    return os.environ.get("TINKER_TELEMETRY", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def init_telemetry(tinker_provider: AsyncTinkerProvider, session_id: str) -> Telemetry | None:
    try:
        return Telemetry(tinker_provider, session_id) if _is_telemetry_enabled() else None
    except Exception as e:
        logger.warning(f"Error initializing telemetry: {e}")
        return None


P = ParamSpec("P")
R = TypeVar("R")


# Decorator to capture exceptions. Class must implement TelemetryProvider.
# Pass fatal=True to log a session end event in addition to the exception.
#
# Example:
# @capture_exceptions
# def my_method(self):
#     pass
#
# @capture_exceptions(fatal=True, severity="CRITICAL")
# def my_method(self):
#     pass
@overload
def capture_exceptions(
    func: Callable[P, R], *, fatal: bool = False, severity: Severity = "ERROR"
) -> Callable[P, R]: ...


@overload
def capture_exceptions(
    *, fatal: bool = False, severity: Severity = "ERROR"
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def capture_exceptions(
    func: Callable[P, R] | None = None, *, fatal: bool = False, severity: Severity = "ERROR"
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    def _get_telemetry(func: Callable[..., object], args: tuple[object, ...]) -> Telemetry | None:
        if args and isinstance(args[0], TelemetryProvider):
            return args[0].get_telemetry()
        with contextlib.suppress(TypeError, AttributeError):
            self = inspect.getclosurevars(func).nonlocals.get("self")
            if isinstance(self, TelemetryProvider):
                return self.get_telemetry()
        logger.warning("@capture_exceptions used without TelemetryProvider: %s", func.__name__)
        return None

    def _decorate(func: Callable[P, R]) -> Callable[P, R]:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def _awrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                telemetry = _get_telemetry(func, args)
                if telemetry is None:
                    return await cast(Callable[..., Awaitable[R]], func)(*args, **kwargs)
                async with telemetry.acapture_exceptions(fatal=fatal, severity=severity):
                    return await cast(Callable[..., Awaitable[R]], func)(*args, **kwargs)

            return cast(Callable[P, R], _awrapper)

        @functools.wraps(func)
        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            telemetry = _get_telemetry(func, args)
            if telemetry is None:
                return func(*args, **kwargs)
            with telemetry.capture_exceptions(fatal=fatal, severity=severity):
                return func(*args, **kwargs)

        return cast(Callable[P, R], _wrapper)

    return _decorate if func is None else _decorate(func)


def is_user_error(exception: BaseException) -> bool:
    return _get_user_error(exception) is not None


def _get_user_error(
    exception: BaseException, visited: set[int] | None = None
) -> BaseException | None:
    visited = set() if visited is None else visited
    if id(exception) in visited:
        return None
    visited.add(id(exception))

    if (
        isinstance(exception, RequestFailedError)
        and exception.category is RequestErrorCategory.User
    ):
        return exception

    status_code = getattr(exception, "status_code", None)
    if isinstance(status_code, int) and 400 <= status_code < 500 and status_code != 408:
        return exception

    if (cause := getattr(exception, "__cause__", None)) is not None and (
        user_error := _get_user_error(cause, visited)
    ) is not None:
        return user_error

    if (context := getattr(exception, "__context__", None)) is not None:
        return _get_user_error(context, visited)

    return None


def _to_send_request(batch: TelemetryBatch) -> TelemetrySendRequest:
    return TelemetrySendRequest(**batch.model_dump())


def _current_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None
