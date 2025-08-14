"""
Generalizable retry handler for API requests with connection limiting,
progress tracking, and exponential backoff.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, Type, TypeVar

import httpx
import tinker

from .._constants import (
    DEFAULT_CONNECTION_LIMITS,
    INITIAL_RETRY_DELAY,
    MAX_RETRY_DELAY,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_retryable_status_code(status_code: int) -> bool:
    return status_code in (408, 409, 429) or (500 <= status_code < 600)


class RetryableException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


@dataclass
class RetryConfig:
    max_connections: int = DEFAULT_CONNECTION_LIMITS.max_connections or 100
    progress_timeout: float = 30 * 60  # Very long straggler
    retry_delay_base: float = INITIAL_RETRY_DELAY
    retry_delay_max: float = MAX_RETRY_DELAY
    jitter_factor: float = 0.25
    enable_retry_logic: bool = True
    retryable_exceptions: tuple[Type[Exception], ...] = (
        asyncio.TimeoutError,
        tinker.APIConnectionError,
        httpx.TimeoutException,
        RetryableException,
    )

    def __post_init__(self):
        if self.max_connections <= 0:
            raise ValueError(f"max_connections must be positive, got {self.max_connections}")

    def __hash__(self):
        return hash(
            (
                self.max_connections,
                self.progress_timeout,
                self.retry_delay_base,
                self.retry_delay_max,
                self.jitter_factor,
                self.enable_retry_logic,
                self.retryable_exceptions,
            )
        )


class RetryHandler(Generic[T]):  # noqa: UP046
    """
    A generalizable retry handler for API requests.

    Features:
    - Connection limiting with semaphores
    - Global progress timeout tracking
    - Exponential backoff with jitter
    - Configurable error classification

    Usage:
        handler = RetryHandler(config=retry_config)
        result = await handler.execute(my_function, *args, **kwargs)
    """

    def __init__(self, config: RetryConfig = RetryConfig(), name: str = "default"):
        self.config = config
        self.name = name

        current_time = time.time()
        self._last_global_progress = current_time
        self._last_printed_progress = current_time
        self._processed_count = 0
        self._waiting_at_semaphore_count = 0
        self._in_retry_loop_count = 0
        self._retry_count = 0
        self._exception_counts = {}  # Track exception types and their counts

        self._errors_since_last_retry: defaultdict[str, int] = defaultdict(int)

        # The semaphore is used to limit the number of concurrent requests.
        # Without a semaphore, progress can grind to a halt as requests fight
        # for limited httpx connections.
        self._semaphore = asyncio.Semaphore(config.max_connections)

    async def execute(
        self, func: Callable[..., Awaitable[T]], request_timeout: float, *args: Any, **kwargs: Any
    ) -> T:
        """Use as a direct function call."""

        self._waiting_at_semaphore_count += 1
        async with self._semaphore:
            self._waiting_at_semaphore_count -= 1
            self._in_retry_loop_count += 1
            try:
                return await self._execute_with_retry(func, request_timeout, *args, **kwargs)
            finally:
                self._in_retry_loop_count -= 1

    async def _execute_with_retry(
        self, func: Callable[..., Awaitable[T]], request_timeout: float, *args: Any, **kwargs: Any
    ) -> T:
        """Main retry logic."""
        # Fast path: skip all retry logic if disabled
        if not self.config.enable_retry_logic:
            return await func(*args, **kwargs)

        attempt_count = 0
        while True:
            current_time = time.time()
            # Only check timeout after first failure
            elapsed_since_last_progress = current_time - self._last_global_progress
            if (attempt_count > 0) and (elapsed_since_last_progress > self.config.progress_timeout):
                # Create a dummy request for the exception (required by APIConnectionError)
                dummy_request = httpx.Request("GET", "http://localhost")
                raise tinker.APIConnectionError(
                    message=f"No progress made in {self.config.progress_timeout}s. "
                    f"Requests appear to be stuck.",
                    request=dummy_request,
                )
            elapsed_since_last_printed_progress = current_time - self._last_printed_progress
            if elapsed_since_last_printed_progress > 2:
                print(
                    f"[{self.name}]: {self._waiting_at_semaphore_count} waiting, {self._in_retry_loop_count} in retry loop, {self._processed_count} completed, {self._retry_count} retries"
                )
                if self._errors_since_last_retry:
                    sorted_items = sorted(
                        self._errors_since_last_retry.items(), key=lambda x: x[1], reverse=True
                    )
                    print(f"[{self.name}]: Errors since last retry: {sorted_items}")
                self._last_printed_progress = current_time
                self._errors_since_last_retry.clear()

            try:
                attempt_count += 1
                logger.debug(f"Attempting request (attempt #{attempt_count})")
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=request_timeout,
                )

            except Exception as e:
                exception_str = f"{type(e).__name__}: {str(e) or 'No error message'}"
                self._errors_since_last_retry[exception_str] += 1

                if not self._should_retry(e):
                    logger.error(f"Request failed with non-retryable error: {exception_str}")
                    raise

                self._log_retry_reason(e, attempt_count, request_timeout=request_timeout)
                self._retry_count += 1

                # Calculate retry delay with exponential backoff and jitter
                retry_delay = self._calculate_retry_delay(attempt_count - 1)
                logger.debug(f"Retrying in {retry_delay:.2f}s")
                await asyncio.sleep(retry_delay)
            else:
                logger.debug(f"Request succeeded after {attempt_count} attempts")
                self._processed_count += 1
                self._last_global_progress = time.time()
                return result

    def _should_retry(self, exception: Exception) -> bool:
        """Determine if an exception should trigger a retry."""
        # Check if it's a generally retryable exception type
        if isinstance(exception, self.config.retryable_exceptions):
            return True

        # Check for API status errors with retryable status codes
        if isinstance(exception, tinker.APIStatusError):
            return is_retryable_status_code(exception.status_code)

        return False

    def _log_retry_reason(self, exception: Exception, attempt_count: int, request_timeout: float):
        """Log the reason for retrying."""
        if isinstance(exception, asyncio.TimeoutError):
            logger.debug(f"Request timed out after {request_timeout}s")
        elif isinstance(exception, tinker.APIConnectionError):
            logger.debug(f"Request failed with connection error: {exception}")
        elif isinstance(exception, tinker.APIStatusError):
            logger.debug(
                f"Request attempt #{attempt_count} failed with status {exception.status_code}"
            )
        else:
            logger.debug(f"Request attempt #{attempt_count} failed with error: {exception}")

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        delay = self.config.retry_delay_max
        try:
            delay = min(self.config.retry_delay_base * (2**attempt), self.config.retry_delay_max)
        except OverflowError:
            # There are two possible overflow errors:
            # (1) `min` tries to convert the value to a float, which can overflow
            #      if the integer value gets too large
            # (2) If the attempt number is too large, the `2 ** attempt` will overflow
            delay = self.config.retry_delay_max

        jitter = delay * self.config.jitter_factor * (2 * random.random() - 1)
        # Ensure the final delay doesn't exceed the maximum, even with jitter
        return max(0, min(delay + jitter, self.config.retry_delay_max))
