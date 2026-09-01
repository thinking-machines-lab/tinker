"""Tests for SessionFuturesPoller."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.session_futures_poller import SessionFuturesPoller
from tinker.types.future_completion import FutureFinished
from tinker.types.futures_retrieve_request import FuturesRetrieveRequest
from tinker.types.futures_retrieve_response import FuturesRetrieveResponse


class _FakeFutures:
    def __init__(self, responses: list[FuturesRetrieveResponse]) -> None:
        self._responses = list(responses)
        self.seen_cursors: list[int] = []

    async def retrieve_multi(
        self, *, request: FuturesRetrieveRequest, timeout: float, max_retries: int
    ) -> FuturesRetrieveResponse:
        self.seen_cursors.append(request.prev_cursor)
        if self._responses:
            return self._responses.pop(0)
        # Emulate the server's long-poll returning nothing: block until cancelled.
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")


class _FakeClient:
    def __init__(self, futures: _FakeFutures) -> None:
        self.futures = futures


class _FakeHolder:
    def __init__(self, client: _FakeClient) -> None:
        self._client = client

    @contextmanager
    def aclient(self, _pool_type: ClientConnectionPoolType) -> Iterator[_FakeClient]:
        yield self._client


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
    # and the next poll must carry the advanced cursor.
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
        # Polls: 0 (initial), 0 (after empty), 5 (after completion, blocks).
        assert futures.seen_cursors[:2] == [0, 0]
        assert 5 in futures.seen_cursors
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
