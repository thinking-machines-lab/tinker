"""Tests for ServiceClient.copy_weights."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from tinker import types
from tinker.lib.public_interfaces.service_client import ServiceClient


class _Future:
    """Stand-in for the holder's future, awaitable or blocking like the real one."""

    def __init__(self, coro: Any) -> None:
        self._coro = coro

    def result(self) -> Any:
        return asyncio.new_event_loop().run_until_complete(self._coro)

    def __await__(self) -> Any:
        return self._coro.__await__()


def _client_with_stub_transport(seq_ids: list[int]) -> tuple[Any, Any]:
    """Mocked at the holder, not over HTTP: the TRAIN pool can use the pyqwest
    transport, which respx cannot intercept."""
    holder = MagicMock()
    holder.get_session_id.return_value = "test-session-id"
    holder.get_training_client_id.side_effect = seq_ids
    holder.aclient.return_value.__enter__.return_value.models.copy_weights = AsyncMock(
        return_value=types.CopyWeightsResponse(tinker_path="tinker://dest:train:0/weights/copy")
    )

    async def _execute_with_retries(fn: Any) -> Any:
        return await fn()

    holder.execute_with_retries = _execute_with_retries
    holder.run_coroutine_threadsafe.side_effect = _Future

    client = ServiceClient.__new__(ServiceClient)
    client._session_holder = holder
    return client, holder.aclient.return_value.__enter__.return_value.models.copy_weights


def test_copy_weights_addresses_the_destination_by_session_and_seq_id() -> None:
    client, copy = _client_with_stub_transport([0, 1])

    # One method, both access patterns: block on the future, or await it.
    result = client.copy_weights(
        "tinker://src/weights/0001",
        ttl_seconds=3600,
        weights_access_token="token-1",
    ).result()

    async def _awaited() -> str:
        return await client.copy_weights("tinker://src/weights/0002")

    asyncio.run(_awaited())

    assert result == "tinker://dest:train:0/weights/copy"
    first = copy.await_args_list[0].kwargs["request"]
    assert first.session_id == "test-session-id"
    assert first.source_path == "tinker://src/weights/0001"
    assert first.ttl_seconds == 3600
    assert first.weights_access_token == "token-1"

    # Both entry points allocate from the same counter.
    assert [c.kwargs["request"].model_seq_id for c in copy.await_args_list] == [0, 1]
