"""Tests for SessionFuturesPoller."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import pytest

from tinker._exceptions import APIStatusError
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.internal_client_holder import InternalClientHolder
from tinker.lib.session_futures_poller import SessionFuturesPoller
from tinker.types.future_completion import FutureFinished
from tinker.types.futures_retrieve_request import FuturesRetrieveRequest
from tinker.types.futures_retrieve_response import FuturesRetrieveResponse


def _status_error(status_code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://example.invalid/api/v1/retrieve_futures")
    response = httpx.Response(status_code, request=request, json={"error": "boom"})
    return APIStatusError("boom", response=response, body=None)


class _FakeFutures:
    def __init__(self, responses: list[FuturesRetrieveResponse | Exception]) -> None:
        self._responses = list(responses)
        self.seen_cursors: list[int] = []

    async def retrieve_multi(
        self, *, request: FuturesRetrieveRequest, timeout: float, max_retries: int
    ) -> FuturesRetrieveResponse:
        self.seen_cursors.append(request.prev_cursor)
        if self._responses:
            item = self._responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        # Emulate the server's long-poll returning nothing: block until cancelled.
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")


class _FakeClient:
    def __init__(self, futures: _FakeFutures) -> None:
        self.futures = futures


class _FakeHolder:
    def __init__(self, client: _FakeClient) -> None:
        self._client = client
        self.pause_calls = 0
        # Bound the billing pause so the loop eventually gives up like the real
        # holder does once the max-pause window is exceeded.
        self._max_pause_calls = 2

    @contextmanager
    def aclient(self, _pool_type: ClientConnectionPoolType) -> Iterator[_FakeClient]:
        yield self._client

    def _is_retryable_exception(self, e: Exception) -> bool:
        return InternalClientHolder._is_retryable_exception(e)

    def _should_pause_on_billing(self, status_code: int, detail: str) -> bool:
        self.pause_calls += 1
        return self.pause_calls <= self._max_pause_calls


def _resp(request_id: str, cursor: int, size: int = 10) -> FuturesRetrieveResponse:
    return FuturesRetrieveResponse(
        completions=[
            FutureFinished(request_id=request_id, response_payload_uncompressed_size=size)
        ],
        cursor=cursor,
    )


@pytest.mark.asyncio
async def test_wait_for_returns_completion_and_threads_cursor() -> None:
    futures = _FakeFutures([_resp("s:sample:0:1", cursor=1)])
    holder = _FakeHolder(_FakeClient(futures))
    poller = SessionFuturesPoller(
        holder,  # type: ignore[arg-type]
        sampling_session_id="s:sample:0",
        cloned_sampler_id=0,
    )
    try:
        completion = await asyncio.wait_for(poller.wait_for("s:sample:0:1"), timeout=5)
        assert completion.request_id == "s:sample:0:1"
        assert completion.state == "finished"
        assert completion.response_payload_uncompressed_size == 10
        assert futures.seen_cursors[0] == 0
    finally:
        poller.close()


@pytest.mark.asyncio
async def test_wait_for_blocks_until_a_later_poll_completes_it() -> None:
    # First poll: empty (cursor unchanged). Second poll: the completion arrives
    # and advances the cursor for the next poll.
    futures = _FakeFutures(
        [
            FuturesRetrieveResponse(completions=[], cursor=0),
            _resp("s:sample:0:2", cursor=5),
        ]
    )
    holder = _FakeHolder(_FakeClient(futures))
    poller = SessionFuturesPoller(
        holder,  # type: ignore[arg-type]
        sampling_session_id="s:sample:0",
        cloned_sampler_id=0,
    )
    try:
        completion = await asyncio.wait_for(poller.wait_for("s:sample:0:2"), timeout=5)
        assert completion.request_id == "s:sample:0:2"
        # Two polls, both at cursor 0; the loop then idles (no waiter left)
        # rather than issuing a third poll at the advanced cursor.
        assert futures.seen_cursors == [0, 0]
        assert poller._cursor == 5
    finally:
        poller.close()


@pytest.mark.asyncio
async def test_completion_consumed_once() -> None:
    futures = _FakeFutures([_resp("s:sample:0:1", cursor=1)])
    holder = _FakeHolder(_FakeClient(futures))
    poller = SessionFuturesPoller(
        holder,  # type: ignore[arg-type]
        sampling_session_id="s:sample:0",
        cloned_sampler_id=0,
    )
    try:
        await asyncio.wait_for(poller.wait_for("s:sample:0:1"), timeout=5)
        # The completion (and its event) were consumed, so the maps don't leak.
        assert "s:sample:0:1" not in poller._completions
        assert "s:sample:0:1" not in poller._events
    finally:
        poller.close()


@pytest.mark.asyncio
async def test_non_retryable_error_fails_existing_waiter() -> None:
    # 403 is not retryable — a blocked waiter must fail rather than hang.
    futures = _FakeFutures([_status_error(403)])
    holder = _FakeHolder(_FakeClient(futures))
    poller = SessionFuturesPoller(
        holder,  # type: ignore[arg-type]
        sampling_session_id="s:sample:0",
        cloned_sampler_id=0,
    )
    try:
        with pytest.raises(Exception, match="retrieve_futures polling failed"):
            await asyncio.wait_for(poller.wait_for("s:sample:0:1"), timeout=5)
    finally:
        poller.close()


@pytest.mark.asyncio
async def test_non_retryable_error_fails_new_waiter() -> None:
    # A waiter registering after the terminal failure fails immediately too.
    futures = _FakeFutures([_status_error(403)])
    holder = _FakeHolder(_FakeClient(futures))
    poller = SessionFuturesPoller(
        holder,  # type: ignore[arg-type]
        sampling_session_id="s:sample:0",
        cloned_sampler_id=0,
    )
    try:
        with pytest.raises(Exception, match="retrieve_futures polling failed"):
            await asyncio.wait_for(poller.wait_for("s:sample:0:1"), timeout=5)
        # New waiter, after the loop already failed: raises without re-polling.
        with pytest.raises(Exception, match="retrieve_futures polling failed"):
            await poller.wait_for("s:sample:0:2")
    finally:
        poller.close()


@pytest.mark.asyncio
async def test_billing_402_pauses_then_recovers() -> None:
    # 402s pause (holder says stay paused), then a real completion arrives.
    futures = _FakeFutures(
        [
            _status_error(402),
            _status_error(402),
            _resp("s:sample:0:1", cursor=1),
        ]
    )
    holder = _FakeHolder(_FakeClient(futures))
    poller = SessionFuturesPoller(
        holder,  # type: ignore[arg-type]
        sampling_session_id="s:sample:0",
        cloned_sampler_id=0,
    )
    try:
        completion = await asyncio.wait_for(poller.wait_for("s:sample:0:1"), timeout=15)
        assert completion.request_id == "s:sample:0:1"
        assert holder.pause_calls == 2
    finally:
        poller.close()


@pytest.mark.asyncio
async def test_billing_402_fails_once_pause_window_exceeded() -> None:
    # Endless 402s: once the holder stops pausing, the waiter fails.
    futures = _FakeFutures([_status_error(402) for _ in range(10)])
    holder = _FakeHolder(_FakeClient(futures))
    poller = SessionFuturesPoller(
        holder,  # type: ignore[arg-type]
        sampling_session_id="s:sample:0",
        cloned_sampler_id=0,
    )
    try:
        with pytest.raises(Exception, match="retrieve_futures polling failed"):
            await asyncio.wait_for(poller.wait_for("s:sample:0:1"), timeout=15)
    finally:
        poller.close()


@pytest.mark.asyncio
async def test_retryable_error_keeps_polling() -> None:
    # 500 is retryable — the loop retries and the later completion still lands.
    futures = _FakeFutures([_status_error(500), _resp("s:sample:0:1", cursor=1)])
    holder = _FakeHolder(_FakeClient(futures))
    poller = SessionFuturesPoller(
        holder,  # type: ignore[arg-type]
        sampling_session_id="s:sample:0",
        cloned_sampler_id=0,
    )
    try:
        completion = await asyncio.wait_for(poller.wait_for("s:sample:0:1"), timeout=15)
        assert completion.request_id == "s:sample:0:1"
        assert holder.pause_calls == 0
    finally:
        poller.close()


@pytest.mark.asyncio
async def test_transient_400_is_retried_before_completing() -> None:
    # A spurious 400 is retried within budget, then the completion arrives.
    futures = _FakeFutures(
        [
            _status_error(400),
            _status_error(400),
            _resp("s:sample:0:1", cursor=1),
        ]
    )
    holder = _FakeHolder(_FakeClient(futures))
    poller = SessionFuturesPoller(
        holder,  # type: ignore[arg-type]
        sampling_session_id="s:sample:0",
        cloned_sampler_id=0,
    )
    try:
        completion = await asyncio.wait_for(poller.wait_for("s:sample:0:1"), timeout=15)
        assert completion.request_id == "s:sample:0:1"
    finally:
        poller.close()


@pytest.mark.asyncio
async def test_persistent_400_fails_once_budget_exhausted() -> None:
    # 400s past the retry budget are terminal.
    futures = _FakeFutures([_status_error(400) for _ in range(10)])
    holder = _FakeHolder(_FakeClient(futures))
    poller = SessionFuturesPoller(
        holder,  # type: ignore[arg-type]
        sampling_session_id="s:sample:0",
        cloned_sampler_id=0,
    )
    try:
        with pytest.raises(Exception, match="retrieve_futures polling failed"):
            await asyncio.wait_for(poller.wait_for("s:sample:0:1"), timeout=15)
    finally:
        poller.close()


@pytest.mark.asyncio
async def test_recorded_completion_wins_over_terminal_failure() -> None:
    # The completion lands, then a later poll fails terminally. A waiter for the
    # already-recorded completion must still get it, not the failure.
    futures = _FakeFutures([_resp("s:sample:0:1", cursor=1), _status_error(403)])
    holder = _FakeHolder(_FakeClient(futures))
    poller = SessionFuturesPoller(
        holder,  # type: ignore[arg-type]
        sampling_session_id="s:sample:0",
        cloned_sampler_id=0,
    )
    try:
        # An unrelated waiter keeps the loop polling: it records the :1
        # completion, then the next poll fails terminally.
        keepalive = asyncio.ensure_future(poller.wait_for("s:sample:0:keepalive"))
        for _ in range(50):
            await asyncio.sleep(0)
            if poller._failure is not None:
                break
        assert poller._failure is not None
        completion = await asyncio.wait_for(poller.wait_for("s:sample:0:1"), timeout=5)
        assert completion.request_id == "s:sample:0:1"
        with pytest.raises(Exception, match="retrieve_futures polling failed"):
            await keepalive
    finally:
        poller.close()


@pytest.mark.asyncio
async def test_idles_until_a_waiter_registers() -> None:
    # With no active request, the loop must not poll; the first poll happens
    # only once a waiter registers.
    futures = _FakeFutures([_resp("s:sample:0:1", cursor=1)])
    holder = _FakeHolder(_FakeClient(futures))
    poller = SessionFuturesPoller(
        holder,  # type: ignore[arg-type]
        sampling_session_id="s:sample:0",
        cloned_sampler_id=0,
    )
    try:
        poller._ensure_running()
        for _ in range(20):
            await asyncio.sleep(0)
        assert futures.seen_cursors == []  # idle: nothing polled yet
        completion = await asyncio.wait_for(poller.wait_for("s:sample:0:1"), timeout=5)
        assert completion.request_id == "s:sample:0:1"
        assert futures.seen_cursors  # polling started once a waiter registered
    finally:
        poller.close()


@pytest.mark.asyncio
async def test_no_further_poll_after_last_request_completes() -> None:
    # Once the only waiter is served, the loop must idle rather than issue
    # another long poll — even with a second response queued and ready.
    futures = _FakeFutures([_resp("s:sample:0:1", cursor=1), _resp("s:sample:0:2", cursor=2)])
    holder = _FakeHolder(_FakeClient(futures))
    poller = SessionFuturesPoller(
        holder,  # type: ignore[arg-type]
        sampling_session_id="s:sample:0",
        cloned_sampler_id=0,
    )
    try:
        completion = await asyncio.wait_for(poller.wait_for("s:sample:0:1"), timeout=5)
        assert completion.request_id == "s:sample:0:1"
        # Give the loop ample chances to (wrongly) poll again.
        for _ in range(20):
            await asyncio.sleep(0)
        assert futures.seen_cursors == [0]  # exactly one poll, then idle
    finally:
        poller.close()
