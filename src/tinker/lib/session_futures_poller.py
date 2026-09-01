from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

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

    def _ensure_running(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(
                self._poll_loop(), name="tinker_session_futures_poller"
            )

    async def wait_for(self, request_id: str) -> FutureCompletion:
        """Block until ``request_id`` completes, returning its metadata.

        Consumes the completion (and any registered event) so the poller's maps
        don't grow without bound over a long-lived session.
        """
        self._ensure_running()
        completion = self._completions.pop(request_id, None)
        if completion is not None:
            self._events.pop(request_id, None)
            return completion

        event = self._events.get(request_id)
        if event is None:
            event = asyncio.Event()
            self._events[request_id] = event
        try:
            await event.wait()
        finally:
            # On normal wake and on cancellation, drop our registration.
            self._events.pop(request_id, None)
        return self._completions.pop(request_id)

    async def _poll_loop(self) -> None:
        target = SamplingSessionFuturesTarget(
            sampling_session_id=self._sampling_session_id,
            cloned_sampler_id=self._cloned_sampler_id,
        )
        while True:
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
                logger.warning(
                    "retrieve_futures poll failed for %s (clone %s): %s",
                    self._sampling_session_id,
                    self._cloned_sampler_id,
                    e,
                )
                await asyncio.sleep(1)
                continue

            self._cursor = response.cursor
            for completion in response.completions:
                self._completions[completion.request_id] = completion
                event = self._events.get(completion.request_id)
                if event is not None:
                    event.set()

    def close(self) -> None:
        """Cancel the background poll task (best effort)."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            self._task = None
