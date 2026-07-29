from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Iterator
from typing import TypeVar

import pytest

from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.public_interfaces.rest_client import (
    RestClient,
    _SessionTraceExportPollResponse,
)

T = TypeVar("T")


class _FakeClient:
    def __init__(self, responses: list[_SessionTraceExportPollResponse]) -> None:
        self.responses = responses
        self.requests: list[str] = []

    async def get(self, path: str, *, cast_to: type) -> _SessionTraceExportPollResponse:
        assert cast_to is _SessionTraceExportPollResponse
        self.requests.append(path)
        return self.responses[min(len(self.requests) - 1, len(self.responses) - 1)]


class _FakeHolder:
    def __init__(self, client: _FakeClient) -> None:
        self.client = client
        self.used_pool_type: ClientConnectionPoolType | None = None

    @contextlib.contextmanager
    def aclient(self, pool_type: ClientConnectionPoolType) -> Iterator[_FakeClient]:
        self.used_pool_type = pool_type
        yield self.client

    async def execute_with_retries(self, func: Callable[[], Awaitable[T]]) -> T:
        return await func()

    def run_coroutine_threadsafe(self, coro: Awaitable[str]) -> str:
        return asyncio.run(coro)


def _make_rest_client(
    responses: list[_SessionTraceExportPollResponse],
) -> tuple[RestClient, _FakeHolder]:
    client = _FakeClient(responses)
    holder = _FakeHolder(client)
    rest_client = RestClient(holder)  # type: ignore[arg-type]
    return rest_client, holder


def test_export_session_trace_polls_until_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tinker.lib.public_interfaces.rest_client._TRACE_EXPORT_POLL_INTERVAL_SECONDS", 0.0
    )
    rest_client, holder = _make_rest_client(
        [
            _SessionTraceExportPollResponse(status="pending"),
            _SessionTraceExportPollResponse(status="pending"),
            _SessionTraceExportPollResponse(
                status="ready", url="https://download.example.test/session.pftrace"
            ),
        ]
    )

    result = rest_client._export_session_trace_submit("session-id")

    assert result == "https://download.example.test/session.pftrace"
    assert holder.used_pool_type == ClientConnectionPoolType.TRAIN
    assert len(holder.client.requests) == 3
    assert holder.client.requests[0] == "/api/v1/sessions/session-id/trace_export"


def test_export_session_trace_raises_on_failure() -> None:
    rest_client, _ = _make_rest_client(
        [_SessionTraceExportPollResponse(status="failed", error="no events found")]
    )

    with pytest.raises(RuntimeError, match="no events found"):
        rest_client._export_session_trace_submit("session-id")
