from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from tinker._exceptions import APIStatusError, TinkerError
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.types.future_completion import FutureCompletion
from tinker.types.futures_retrieve_request import (
    FuturesRetrieveRequest,
    SamplingSessionFuturesTarget,
)

if TYPE_CHECKING:
    from tinker.lib.internal_client_holder import InternalClientHolder

logger = logging.getLogger(__name__)

# Client-side timeout for one retrieve_futures poll. The server long-polls up to
# ~30s, so this must comfortably exceed that.
_POLL_HTTP_TIMEOUT_SEC = 45.0

# A bare HTTP 400 can be injected spuriously by an upstream proxy; retry a
# bounded number of times before treating it as terminal.
_MAX_BAD_REQUEST_RETRIES = 3


class SessionFuturesPoller:
    """One background poller per (sampling session, cloned-sampler) pair.

    Long-polls ``/api/v1/retrieve_futures`` in a loop, recording each completed
    request id (with its metadata) and waking any waiter registered for it. A
    sample's result future calls :meth:`wait_for` to block until its request
    completes instead of polling ``retrieve_future`` itself.

    Lives entirely on the holder's event loop; every method must be awaited from
    that loop.
    """

    def __init__(
        self,
        holder: InternalClientHolder,
        *,
        sampling_session_id: str,
        cloned_sampler_id: int,
    ) -> None:
        self._holder = holder
        self._sampling_session_id = sampling_session_id
        self._cloned_sampler_id = cloned_sampler_id
        self._cursor = 0
        # request_id -> event fired once the request is known complete.
        self._events: dict[str, asyncio.Event] = {}
        # request_id -> its completion metadata, kept until a waiter consumes it.
        self._completions: dict[str, FutureCompletion] = {}
        self._task: asyncio.Task[None] | None = None
        # Set once the poll loop hits a non-retryable error. Terminal: every
        # current and future waiter fails with this message.
        self._failure: str | None = None
        # Signals the poll loop that there is at least one waiter to poll for,
        # so it can idle instead of long-polling while no requests are active.
        self._has_work: asyncio.Event = asyncio.Event()

    def _ensure_running(self) -> None:
        if self._failure is not None:
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(
                self._poll_loop(), name="tinker_session_futures_poller"
            )

    async def wait_for(self, request_id: str) -> FutureCompletion:
        """Block until ``request_id`` completes, returning its metadata.

        Consumes the completion (and any registered event) so the poller's maps
        don't grow without bound over a long-lived session.

        Raises :class:`TinkerError` if the poll loop has failed (or fails while
        waiting) with a non-retryable error, unless this request's completion was
        already recorded, in which case it is returned regardless.
        """
        # An already-recorded completion always wins over a terminal failure: a
        # request known complete should proceed to its result fetch.
        completion = self._completions.pop(request_id, None)
        if completion is not None:
            self._events.pop(request_id, None)
            return completion
        if self._failure is not None:
            raise TinkerError(self._failure)
        self._ensure_running()

        # Each request has exactly one waiter, and its id is unique, so there is
        # never a pre-existing registration to reuse.
        assert request_id not in self._events, f"duplicate waiter for {request_id}"
        event = asyncio.Event()
        self._events[request_id] = event
        self._has_work.set()
        try:
            await event.wait()
        finally:
            # The poll loop removes the entry before notifying on completion;
            # this drops it on the cancellation and failure paths instead.
            self._events.pop(request_id, None)
        completion = self._completions.pop(request_id, None)
        if completion is not None:
            return completion
        # Woken by a terminal failure with no completion recorded for us.
        assert self._failure is not None
        raise TinkerError(self._failure)

    def _fail(self, message: str) -> None:
        """Record a terminal failure and wake every registered waiter.

        Waiters re-check ``self._failure`` after waking and raise. New waiters
        raise on entry, so the failure fans out to all of them.
        """
        self._failure = message
        for event in self._events.values():
            event.set()

    async def _poll_loop(self) -> None:
        target = SamplingSessionFuturesTarget(
            sampling_session_id=self._sampling_session_id,
            cloned_sampler_id=self._cloned_sampler_id,
        )
        bad_request_retries = 0
        while True:
            # Idle while no request is waiting rather than long-polling for
            # completions that cannot arrive; wait_for wakes us when one registers.
            if not self._events:
                self._has_work.clear()
                await self._has_work.wait()
            try:
                request = FuturesRetrieveRequest(target=target, prev_cursor=self._cursor)
                with self._holder.aclient(ClientConnectionPoolType.RETRIEVE_FUTURES) as client:
                    response = await client.futures.retrieve_multi(
                        request=request,
                        timeout=_POLL_HTTP_TIMEOUT_SEC,
                        max_retries=0,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # While billing is paused, retry silently until the holder's
                # max-pause window is exceeded, at which point
                # _should_pause_on_billing returns False and we fall through.
                if (
                    isinstance(e, APIStatusError)
                    and e.status_code == 402
                    and self._holder._should_pause_on_billing(e.status_code, e.message)
                ):
                    await asyncio.sleep(5)
                    continue

                if self._holder._is_retryable_exception(e):
                    await asyncio.sleep(1)
                    continue

                if (
                    isinstance(e, APIStatusError)
                    and e.status_code == 400
                    and bad_request_retries < _MAX_BAD_REQUEST_RETRIES
                ):
                    bad_request_retries += 1
                    await asyncio.sleep(1)
                    continue

                message = (
                    f"retrieve_futures polling failed for sampling session "
                    f"{self._sampling_session_id} (clone {self._cloned_sampler_id}): "
                    f"{type(e).__name__}: {e}"
                )
                logger.error(message)
                self._fail(message)
                return

            # A completed round-trip clears the transient-400 budget.
            bad_request_retries = 0
            self._cursor = response.cursor
            for completion in response.completions:
                self._completions[completion.request_id] = completion
                # Remove the registration before notifying so the top-of-loop
                # idle check sees an empty map once the last waiter is served,
                # rather than issuing another long poll for nothing.
                event = self._events.pop(completion.request_id, None)
                if event is not None:
                    event.set()

    def close(self) -> None:
        """Cancel the background poll task (best effort)."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            self._task = None
