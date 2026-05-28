from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, List, Type, TypeVar, cast

import tinker
from tinker import types
from tinker._exceptions import RequestFailedError
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.public_interfaces.api_future import APIFuture
from tinker.lib.telemetry import Telemetry
from tinker.types import RequestErrorCategory
from tinker.types.future_retrieve_request import FutureRetrieveRequest

from ._pydantic_conv import deserialize_json_response
from .retryable_exception import RetryableException
from .sync_only import sync_only

if TYPE_CHECKING:
    from tinker.lib.internal_client_holder import InternalClientHolder

from tinker.proto.response_conv import PROTO_SUPPORTED_TYPES, deserialize_proto_response

logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")

# Sentinel object to indicate that the function hasn't been called yet
_UNCOMPUTED = object()


class QueueState(Enum):
    ACTIVE = "active"
    PAUSED_RATE_LIMIT = "paused_rate_limit"
    PAUSED_CAPACITY = "paused_capacity"
    UNKNOWN = "unknown"


class QueueStateObserver(ABC):
    @abstractmethod
    def on_queue_state_change(
        self, queue_state: QueueState, queue_state_reason: str | None
    ) -> None:
        raise NotImplementedError


# Internal outcome types — `_fetch_via_rest` produces one of these, and
# `_handle_outcome` dispatches.


@dataclass
class _SuccessProto:
    proto_bytes: bytes


@dataclass
class _SuccessJson:
    result_dict: dict[str, Any]


@dataclass
class _TryAgain:
    queue_state: QueueState
    queue_state_reason: str | None = None


@dataclass
class _MetadataOnly:
    payload_size: int


@dataclass
class _Failed:
    error_message: str
    error_category: RequestErrorCategory


_Outcome = _SuccessProto | _SuccessJson | _TryAgain | _MetadataOnly | _Failed


@dataclass
class _LoopState:
    """Mutable state carried across retry iterations of _result_async."""

    allow_metadata_only: bool = True
    connection_error_retries: int = 0
    bad_request_retries: int = 0


_MAX_BAD_REQUEST_RETRIES = 3


class _TransportErrorKind(Enum):
    """How the outer loop should react to a transport-layer error."""

    # Immediate retry without backoff. Maps to HTTP 408/5xx.
    RETRY = "retry"
    # Retry with exponential backoff. Maps to connection errors.
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    # Retry up to _MAX_BAD_REQUEST_RETRIES. Maps to bare HTTP 400, which
    # an upstream proxy may inject spuriously.
    RETRY_IF_BUDGET = "retry_if_budget"
    # Raise RetryableException so the outer-outer client can retry the
    # *original* request (promise is gone/corrupt). Maps to HTTP 410.
    RETRYABLE_EXCEPTION = "retryable_exception"
    # Everything else — auth/permission/not_found/etc. Raise ValueError.
    FATAL = "fatal"


@dataclass
class _TransportError:
    """Transport-layer error.

    `_fetch_via_rest` translates exceptions into this shape;
    `_handle_transport_error` picks the retry action based on `.kind` and
    emits the telemetry event in one place.
    """

    kind: _TransportErrorKind
    # HTTP status. 0 means "transport error with no HTTP response"
    # (e.g. connection refused).
    status_code: int
    detail: str
    exception: Exception
    # Telemetry event name. "api_status_error" is the catch-all;
    # "connection_error" identifies cases where no usable response came
    # back from the server (APIConnectionError).
    event_name: str = "api_status_error"
    # HTTP 408 carries queue_state in the body. Plumb it through so the
    # handler can still notify the observer.
    try_again_body: dict[str, Any] | None = None
    # Post-mortem context for terminal errors (auth headers, error body).
    response_headers: dict[str, str] | None = None
    request_headers: dict[str, str] | None = None
    response_body: object | None = None


_REST_QUEUE_STATE_TO_ENUM: dict[str, QueueState] = {
    "active": QueueState.ACTIVE,
    "paused_capacity": QueueState.PAUSED_CAPACITY,
    "paused_rate_limit": QueueState.PAUSED_RATE_LIMIT,
}


def _rest_queue_state_to_enum(value: str) -> QueueState:
    return _REST_QUEUE_STATE_TO_ENUM.get(value, QueueState.UNKNOWN)


def _rest_status_error_to_transport_error(e: tinker.APIStatusError) -> _TransportError:
    status = e.response.status_code
    try_again_body: dict[str, Any] | None = None
    if status == 408:
        # 408 is a try_again pretending to be an HTTP error; the body
        # carries queue_state. Capture it here so the shared handler can
        # notify the observer even though the outcome isn't a _TryAgain.
        with contextlib.suppress(Exception):
            try_again_body = e.response.json()
        kind = _TransportErrorKind.RETRY
    elif status == 410:
        kind = _TransportErrorKind.RETRYABLE_EXCEPTION
    elif 500 <= status < 600:
        kind = _TransportErrorKind.RETRY
    elif status == 429:
        # Rate limit / capacity. Back off so the retry has a chance of
        # finding the bucket refilled.
        kind = _TransportErrorKind.RETRY_WITH_BACKOFF
    elif status == 400:
        kind = _TransportErrorKind.RETRY_IF_BUDGET
    else:
        kind = _TransportErrorKind.FATAL

    # Capture request/response headers + body for post-mortems on errors
    # that don't have a clean retry path: FATAL (auth/permission/not_found)
    # and RETRY_IF_BUDGET (bare 400 from an upstream proxy — useful to
    # identify which L7 hop injected the 400). Skipped on simple-retry codes
    # to keep retry-loop events light.
    response_headers: dict[str, str] | None = None
    request_headers: dict[str, str] | None = None
    response_body: object | None = None
    if kind in (_TransportErrorKind.FATAL, _TransportErrorKind.RETRY_IF_BUDGET):
        response_headers = dict(e.response.headers)
        request_headers = dict(e.request.headers)
        response_body = e.body

    return _TransportError(
        kind=kind,
        status_code=status,
        detail=str(e),
        exception=e,
        try_again_body=try_again_body,
        response_headers=response_headers,
        request_headers=request_headers,
        response_body=response_body,
    )


class _APIFuture(APIFuture[T]):  # pyright: ignore[reportUnusedClass]
    def __init__(
        self,
        model_cls: Type[T],
        holder: InternalClientHolder,
        untyped_future: types.UntypedAPIFuture,
        request_start_time: float,
        request_type: str,
        queue_state_observer: QueueStateObserver | None = None,
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
        self._queue_state_observer: QueueStateObserver | None = queue_state_observer

    async def _result_async(self, timeout: float | None = None) -> T:
        """Get the result of this future, with automatic retries for transient errors."""
        if self._cached_result is not _UNCOMPUTED:
            return cast(T, self._cached_result)

        start_time = time.time()
        iteration = -1
        state = _LoopState()

        async with contextlib.AsyncExitStack() as stack:
            while True:
                iteration += 1
                self._check_timeout(timeout, iteration, start_time)

                fetched = await self._fetch_via_rest(state, iteration)
                if isinstance(fetched, _TransportError):
                    await self._handle_transport_error(fetched, state, iteration, start_time)
                    continue

                result = await self._handle_outcome(fetched, state, stack, iteration, start_time)
                if result is None:
                    # Business-level retry (try_again / metadata-only peek).
                    continue
                return result

        # Unreachable: the while-True either returns or raises.
        raise AssertionError("unreachable")

    async def _handle_transport_error(
        self,
        err: _TransportError,
        state: _LoopState,
        iteration: int,
        start_time: float,
    ) -> None:
        """Retry / telemetry / raise dispatch for transport errors.
        Returns (loop continues) or raises.
        """
        # Billing-pause: while the holder says we're still inside the
        # max-pause window, sleep and retry silently (no telemetry — the
        # events stream stays clean for these). Once the window is
        # exceeded, fall through to the regular dispatch which raises.
        if err.status_code == 402 and self.holder._should_pause_on_billing(
            err.status_code, err.detail
        ):
            await asyncio.sleep(5)
            return

        # FATAL and RETRY_IF_BUDGET log at ERROR so retry-loop dashboards
        # catch spurious 400 spikes from upstream proxies; other retryable
        # kinds stay at WARNING.
        severity = (
            "ERROR"
            if err.kind in (_TransportErrorKind.FATAL, _TransportErrorKind.RETRY_IF_BUDGET)
            else "WARNING"
        )
        is_user = 400 <= err.status_code < 500 and err.status_code != 408
        if telemetry := self.get_telemetry():
            event_data: dict[str, object] = {
                "request_id": self.request_id,
                "request_type": self.request_type,
                "status_code": err.status_code,
                "detail": err.detail,
                "kind": err.kind.value,
                "should_retry": err.kind != _TransportErrorKind.FATAL,
                "is_user_error": is_user,
                "exception": str(err.exception),
                "exception_type": type(err.exception).__name__,
                "iteration": iteration,
                "elapsed_time": time.time() - start_time,
                "bad_request_retries": state.bad_request_retries,
                "connection_error_retries": state.connection_error_retries,
            }
            # FATAL errors carry full request/response context for
            # post-mortems (auth headers, error body, etc.).
            if err.response_headers is not None:
                event_data["response_headers"] = err.response_headers
            if err.request_headers is not None:
                event_data["request_headers"] = err.request_headers
            if err.response_body is not None:
                event_data["response_body"] = err.response_body
            telemetry.log(
                f"APIFuture.result_async.{err.event_name}",
                event_data=event_data,
                severity=severity,
            )

        # Getting any HTTP response (even an error one) proves the connection
        # is alive, so reset the connection-backoff exponent. Without this,
        # a transient outage that incremented connection_error_retries stays
        # inflated after the server recovers, producing unnecessarily long
        # sleeps on subsequent UNAVAILABLE-ish errors.
        if err.kind != _TransportErrorKind.RETRY_WITH_BACKOFF:
            state.connection_error_retries = 0

        if err.kind == _TransportErrorKind.RETRY:
            # HTTP 408 carries queue_state in the body; surface it to the
            # observer so clients can render progress. Reset the
            # bad_request budget too — a 408 is a successful round-trip and
            # an earlier 400 shouldn't fence off subsequent retries.
            if err.try_again_body is not None:
                state.bad_request_retries = 0
            if err.try_again_body and self._queue_state_observer is not None:
                with contextlib.suppress(Exception):
                    qs = err.try_again_body.get("queue_state")
                    if qs:
                        self._queue_state_observer.on_queue_state_change(
                            _rest_queue_state_to_enum(qs),
                            err.try_again_body.get("queue_state_reason"),
                        )
            return

        if err.kind == _TransportErrorKind.RETRY_WITH_BACKOFF:
            await asyncio.sleep(min(2**state.connection_error_retries, 30))
            state.connection_error_retries += 1
            return

        if err.kind == _TransportErrorKind.RETRY_IF_BUDGET:
            if state.bad_request_retries < _MAX_BAD_REQUEST_RETRIES:
                state.bad_request_retries += 1
                return
            raise ValueError(
                f"Error retrieving result: {err.detail} (status {err.status_code}) "
                f"for {self.request_id=} — exceeded {_MAX_BAD_REQUEST_RETRIES} retries"
            ) from err.exception

        if err.kind == _TransportErrorKind.RETRYABLE_EXCEPTION:
            raise RetryableException(
                message=f"Promise expired/broken for request {self.untyped_future.request_id}"
            ) from err.exception

        # FATAL
        raise ValueError(
            f"Error retrieving result: {err.detail} with status {err.status_code} for "
            f"{self.request_id=} and expected type {self.model_cls=}"
        ) from err.exception

    def _check_timeout(self, timeout: float | None, iteration: int, start_time: float) -> None:
        if timeout is None or time.time() - start_time <= timeout:
            return
        if telemetry := self.get_telemetry():
            telemetry.log(
                "APIFuture.result_async.timeout",
                event_data={
                    "request_id": self.request_id,
                    "request_type": self.request_type,
                    "timeout": timeout,
                    "iteration": iteration,
                    "elapsed_time": time.time() - start_time,
                },
                severity="ERROR",
            )
        raise TimeoutError(
            f"Timeout of {timeout} seconds reached while waiting for result of {self.request_id=}"
        )

    async def _fetch_via_rest(
        self, state: _LoopState, iteration: int
    ) -> _Outcome | _TransportError:
        """Fetch retrieve_future over REST. Returns an _Outcome on success
        or a _TransportError (normalized) for the shared handler. No retry
        or telemetry logic here — kept per-transport only what's
        transport-specific (header building + error translation).
        """
        headers = {
            "X-Tinker-Request-Iteration": str(iteration),
            "X-Tinker-Request-Type": self.request_type,
        }
        if self.model_cls in PROTO_SUPPORTED_TYPES:
            headers["Accept"] = "application/x-protobuf, application/json"
        if iteration == 0:
            headers["X-Tinker-Create-Promise-Roundtrip-Time"] = str(
                self.request_queue_roundtrip_time
            )

        try:
            with self.holder.aclient(ClientConnectionPoolType.RETRIEVE_PROMISE) as client:
                response = await client.futures.with_raw_response.retrieve(
                    request=FutureRetrieveRequest(
                        request_id=self.request_id,
                        allow_metadata_only=state.allow_metadata_only,
                    ),
                    timeout=45,
                    extra_headers=headers,
                    max_retries=0,
                )
        except tinker.APIStatusError as e:
            return _rest_status_error_to_transport_error(e)
        except tinker.APIConnectionError as e:
            return _TransportError(
                kind=_TransportErrorKind.RETRY_WITH_BACKOFF,
                status_code=0,
                detail=str(e),
                exception=e,
                event_name="connection_error",
            )

        if "application/x-protobuf" in response.headers.get("content-type", ""):
            return _SuccessProto(proto_bytes=response.http_response.content)

        result_dict: Any = await response.json()

        if result_dict.get("type") == "try_again":
            logger.warning(f"Retrying request {self.request_id=} because of try_again")
            qs_str = result_dict.get("queue_state") or ""
            return _TryAgain(
                queue_state=_rest_queue_state_to_enum(qs_str),
                queue_state_reason=result_dict.get("queue_state_reason"),
            )
        if result_dict.get("status") == "complete_metadata":
            return _MetadataOnly(payload_size=result_dict.get("response_payload_size") or 0)
        if "error" in result_dict:
            error_category = RequestErrorCategory.Unknown
            with contextlib.suppress(Exception):
                error_category = RequestErrorCategory(result_dict.get("category"))
            return _Failed(
                error_message=result_dict["error"],
                error_category=error_category,
            )
        return _SuccessJson(result_dict=result_dict)

    async def _handle_outcome(
        self,
        outcome: _Outcome,
        state: _LoopState,
        stack: contextlib.AsyncExitStack,
        iteration: int,
        start_time: float,
    ) -> T | None:
        """Dispatch the resolved outcome. Returns T on success, None when
        the caller should loop (try_again / metadata-only peek). Raises for
        terminal failure or deserialization error.
        """
        # Any outcome means we got a complete response from the server.
        # Reset the connection-backoff exponent so a recovered server
        # doesn't keep paying for an earlier outage.
        state.connection_error_retries = 0

        if isinstance(outcome, _TryAgain):
            # Skip notification when the server didn't supply a queue_state:
            # UNKNOWN with no reason means there's nothing observer-worthy.
            if self._queue_state_observer is not None and (
                outcome.queue_state != QueueState.UNKNOWN or outcome.queue_state_reason is not None
            ):
                self._queue_state_observer.on_queue_state_change(
                    outcome.queue_state, outcome.queue_state_reason
                )
            return None
        if isinstance(outcome, _MetadataOnly):
            assert state.allow_metadata_only, (
                "got metadata_only but the flag was off — server should not emit this twice"
            )
            state.allow_metadata_only = False
            await stack.enter_async_context(
                self.holder._inflight_response_bytes_semaphore.acquire(outcome.payload_size)
            )
            return None
        if isinstance(outcome, _Failed):
            if telemetry := self.get_telemetry():
                is_user = outcome.error_category is RequestErrorCategory.User
                telemetry.log(
                    "APIFuture.result_async.application_error",
                    event_data={
                        "request_id": self.request_id,
                        "request_type": self.request_type,
                        "error": outcome.error_message,
                        "error_category": outcome.error_category.name,
                        "is_user_error": is_user,
                        "iteration": iteration,
                        "elapsed_time": time.time() - start_time,
                    },
                    severity="WARNING" if is_user else "ERROR",
                )
            raise RequestFailedError(
                f"Request failed: {outcome.error_message} for {self.request_id=} "
                f"and expected type {self.model_cls=}",
                request_id=self.request_id,
                category=outcome.error_category,
            )
        if isinstance(outcome, _SuccessProto):
            try:
                self._cached_result = deserialize_proto_response(
                    outcome.proto_bytes, self.model_cls
                )
                return cast(T, self._cached_result)
            except Exception as e:
                if telemetry := self.get_telemetry():
                    telemetry.log(
                        "APIFuture.result_async.proto_deserialization_error",
                        event_data={
                            "request_id": self.request_id,
                            "request_type": self.request_type,
                            "exception": str(e),
                            "exception_type": type(e).__name__,
                            "proto_bytes_len": len(outcome.proto_bytes),
                            "model_cls": str(self.model_cls),
                            "iteration": iteration,
                            "elapsed_time": time.time() - start_time,
                        },
                        severity="ERROR",
                    )
                raise ValueError(
                    f"Proto deserialization failed: {e} for {self.request_id=} "
                    f"and expected type {self.model_cls=}"
                ) from e
        # _SuccessJson
        try:
            self._cached_result = deserialize_json_response(outcome.result_dict, self.model_cls)
            return cast(T, self._cached_result)
        except Exception as e:
            if telemetry := self.get_telemetry():
                telemetry.log(
                    "APIFuture.result_async.validation_error",
                    event_data={
                        "request_id": self.request_id,
                        "request_type": self.request_type,
                        "exception": str(e),
                        "exception_type": type(e).__name__,
                        "exception_stack": (
                            "".join(traceback.format_exception(type(e), e, e.__traceback__))
                            if e.__traceback__
                            else None
                        ),
                        "model_cls": str(self.model_cls),
                        "iteration": iteration,
                        "elapsed_time": time.time() - start_time,
                    },
                    severity="ERROR",
                )
            raise ValueError(
                f"Error retrieving result: {e} for {self.request_id=} "
                f"and expected type {self.model_cls=}"
            ) from e

    @property
    def request_id(self) -> str:
        return self.untyped_future.request_id

    @sync_only
    def result(self, timeout: float | None = None) -> T:
        return self._future.result(timeout)

    async def result_async(self, timeout: float | None = None) -> T:
        try:
            return await asyncio.wait_for(self._future, timeout)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            self._future.future().cancel()
            raise

    def get_telemetry(self) -> Telemetry | None:
        return self.holder.get_telemetry()


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
