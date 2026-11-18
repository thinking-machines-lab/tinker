"""Detection of requests that a caller cancels and then submits again.

The SDK retries failed requests itself. Wrapping `SamplingClient.sample` in a timeout,
cancelling it and submitting the same request again does not get its result sooner: the
new request joins the back of the queue, and the cancelled one may already have used
compute. This module spots that pattern so the SDK can warn the caller and report it.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from pydantic_core import to_json

from tinker import types

from .telemetry_provider import TelemetryProvider

logger = logging.getLogger(__name__)

TELEMETRY_EVENT_NAME = "request_resubmitted_after_cancel"

_WARNING = (
    "A request was cancelled and then submitted again %.0fs later. Resubmitting does not "
    "get the result sooner: the new request joins the back of the queue, and the cancelled "
    "one may already have used compute. The SDK retries failed requests itself; if you need "
    "a deadline, keep the future and wait on it again (future.result(timeout=...)) rather "
    "than cancelling it and submitting the request again."
)


class RetryDetector:
    """Registry of the requests a caller cancelled, shared by every client of one holder,
    that flags an identical request submitted within
    `retry_detector_cancelled_request_ttl_sec` of the cancellation. That TTL and the
    interval between warnings come from the server's dynamic config, read on every use so
    a server-side change applies at the next refresh; a TTL of 0 turns the detector off.

    A request is identified by a key its caller chooses (anything `pydantic_core.to_json`
    can serialize, such as a tuple of the arguments it was made with); the registry holds a
    digest of the key, never the key itself. Entries expire as others are added or looked
    up, so the registry is empty unless a request was cancelled recently, and until then
    `watch` costs one config read and one truthiness check per request. Every method runs
    on the holder's event loop and keeps no lock; a cancellation and the request
    resubmitted right after it reach the loop in that order, both through
    `call_soon_threadsafe`, so the cancellation is recorded first. Tracking never fails
    the request it watches: its own errors are logged and dropped.
    """

    def __init__(
        self,
        telemetry_provider: TelemetryProvider,
        *,
        clock: Callable[[], float],
        dynamic_config: Callable[[], types.ClientDynamicConfigResponse],
    ) -> None:
        self._telemetry_provider = telemetry_provider
        self._clock = clock
        self._dynamic_config = dynamic_config
        # Request digest -> clock reading at its cancellation, oldest first.
        self._cancelled: OrderedDict[bytes, float] = OrderedDict()
        self._last_warned_at: float | None = None

    @asynccontextmanager
    async def watch(self, key: object) -> AsyncIterator[None]:
        """Watch the request awaited in the body of the `async with`. On entry, flag the
        request if it resubmits a recently cancelled one; if the caller cancels the body,
        record the request. Two requests are the same request when their `key`s serialize
        the same."""
        if self._dynamic_config().retry_detector_cancelled_request_ttl_sec <= 0:
            # The kill switch: nothing is checked or recorded.
            yield
            return
        if self._cancelled:
            self._flag_if_resubmitted(key)
        try:
            yield
        except asyncio.CancelledError:
            self._record_cancelled(key)
            raise

    def _flag_if_resubmitted(self, key: object) -> None:
        try:
            now = self._clock()
            self._expire(now)
            cancelled_at = self._cancelled.pop(_fingerprint(key), None)
            if cancelled_at is None:
                return
            self._report(now, seconds_since_cancel=now - cancelled_at)
        except Exception:
            logger.debug("Could not check a request for resubmission", exc_info=True)

    def _record_cancelled(self, key: object) -> None:
        try:
            now = self._clock()
            self._expire(now)
            digest = _fingerprint(key)
            self._cancelled.pop(digest, None)
            self._cancelled[digest] = now
        except Exception:
            logger.debug("Could not record a cancelled request", exc_info=True)

    def _expire(self, now: float) -> None:
        ttl = self._dynamic_config().retry_detector_cancelled_request_ttl_sec
        while self._cancelled:
            oldest_cancelled_at = next(iter(self._cancelled.values()))
            if now - oldest_cancelled_at < ttl:
                return
            self._cancelled.popitem(last=False)

    def _report(self, now: float, seconds_since_cancel: float) -> None:
        interval = self._dynamic_config().retry_detector_warning_interval_sec
        if self._last_warned_at is None or now - self._last_warned_at >= interval:
            self._last_warned_at = now
            logger.warning(_WARNING, seconds_since_cancel)
        if (telemetry := self._telemetry_provider.get_telemetry()) is not None:
            telemetry.log(
                TELEMETRY_EVENT_NAME,
                event_data={
                    "seconds_since_cancel": seconds_since_cancel,
                    "other_cancelled_requests_tracked": len(self._cancelled),
                },
                severity="WARNING",
            )


def _fingerprint(key: object) -> bytes:
    return hashlib.sha256(to_json(key, bytes_mode="base64")).digest()
