from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time
import traceback
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, List, Type, TypeVar, cast

import tinker
from tinker import types
from tinker._exceptions import RequestFailedError
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.public_interfaces.api_future import APIFuture
from tinker.lib.telemetry import Telemetry, is_user_error
from tinker.types import RequestErrorCategory
from tinker.types.future_retrieve_request import FutureRetrieveRequest

from .._models import BaseModel
from .retryable_exception import RetryableException
from .sync_only import sync_only

if TYPE_CHECKING:
    from tinker.lib.internal_client_holder import InternalClientHolder

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
    def on_queue_state_change(self, queue_state: QueueState) -> None:
        raise NotImplementedError


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
        connection_error_retries = 0

        while True:
            iteration += 1

            if timeout is not None and time.time() - start_time > timeout:
                if telemetry := self.get_telemetry():
                    current_time = time.time()
                    telemetry.log(
                        "APIFuture.result_async.timeout",
                        event_data={
                            "request_id": self.request_id,
                            "request_type": self.request_type,
                            "timeout": timeout,
                            "iteration": iteration,
                            "elapsed_time": current_time - start_time,
                        },
                        severity="ERROR",
                    )
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
                with self.holder.aclient(ClientConnectionPoolType.RETRIEVE_PROMISE) as client:
                    response = await client.futures.with_raw_response.retrieve(
                        request=FutureRetrieveRequest(request_id=self.request_id),
                        timeout=45,
                        extra_headers=headers,
                        max_retries=0,
                    )
            except tinker.APIStatusError as e:
                connection_error_retries = 0
                should_retry = e.status_code == 408 or e.status_code in range(500, 600)
                user_error = is_user_error(e)
                if telemetry := self.get_telemetry():
                    current_time = time.time()
                    telemetry.log(
                        "APIFuture.result_async.api_status_error",
                        event_data={
                            "request_id": self.request_id,
                            "request_type": self.request_type,
                            "status_code": e.status_code,
                            "exception": str(e),
                            "should_retry": should_retry,
                            "is_user_error": user_error,
                            "iteration": iteration,
                            "elapsed_time": current_time - start_time,
                        },
                        severity="WARNING" if should_retry or user_error else "ERROR",
                    )

                # Retry 408s until we time out
                if e.status_code == 408:
                    if self._queue_state_observer is not None:
                        with contextlib.suppress(Exception):
                            response = e.response.json()
                            if queue_state_str := response.get("queue_state", None):
                                if queue_state_str == "active":
                                    queue_state = QueueState.ACTIVE
                                elif queue_state_str == "paused_rate_limit":
                                    queue_state = QueueState.PAUSED_RATE_LIMIT
                                elif queue_state_str == "paused_capacity":
                                    queue_state = QueueState.PAUSED_CAPACITY
                                else:
                                    queue_state = QueueState.UNKNOWN
                                self._queue_state_observer.on_queue_state_change(queue_state)
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
            except tinker.APIConnectionError as e:
                if telemetry := self.get_telemetry():
                    current_time = time.time()
                    telemetry.log(
                        "APIFuture.result_async.connection_error",
                        event_data={
                            "request_id": self.request_id,
                            "request_type": self.request_type,
                            "exception": str(e),
                            "connection_error_retries": connection_error_retries,
                            "iteration": iteration,
                            "elapsed_time": current_time - start_time,
                        },
                        severity="WARNING",
                    )

                # Retry all connection errors with exponential backoff
                await asyncio.sleep(min(2**connection_error_retries, 30))
                connection_error_retries += 1
                continue

            # Function hasn't been called yet, execute it now
            result_dict: Any = await response.json()

            if "type" in result_dict and result_dict["type"] == "try_again":
                logger.warning(f"Retrying request {self.request_id=} because of try_again")
                continue

            if "error" in result_dict:
                error_category = RequestErrorCategory.Unknown
                with contextlib.suppress(Exception):
                    error_category = RequestErrorCategory(result_dict.get("category"))

                user_error = error_category is RequestErrorCategory.User
                if telemetry := self.get_telemetry():
                    current_time = time.time()
                    telemetry.log(
                        "APIFuture.result_async.application_error",
                        event_data={
                            "request_id": self.request_id,
                            "request_type": self.request_type,
                            "error": result_dict["error"],
                            "error_category": error_category.name,
                            "is_user_error": user_error,
                            "iteration": iteration,
                            "elapsed_time": current_time - start_time,
                        },
                        severity="WARNING" if user_error else "ERROR",
                    )

                error_message = result_dict["error"]
                raise RequestFailedError(
                    f"Request failed: {error_message} for {self.request_id=} and expected type {self.model_cls=}",
                    request_id=self.request_id,
                    category=error_category,
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
                if telemetry := self.get_telemetry():
                    current_time = time.time()
                    telemetry.log(
                        "APIFuture.result_async.validation_error",
                        event_data={
                            "request_id": self.request_id,
                            "request_type": self.request_type,
                            "exception": str(e),
                            "exception_type": type(e).__name__,
                            "exception_stack": "".join(
                                traceback.format_exception(type(e), e, e.__traceback__)
                            )
                            if e.__traceback__
                            else None,
                            "model_cls": str(self.model_cls),
                            "iteration": iteration,
                            "elapsed_time": current_time - start_time,
                        },
                        severity="ERROR",
                    )

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
