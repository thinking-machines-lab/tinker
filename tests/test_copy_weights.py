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
        return asyncio.run(self._coro)

    def __await__(self) -> Any:
        return self._coro.__await__()


def _client_with_stub_transport() -> tuple[Any, Any]:
    """Mocked at the holder, not over HTTP: the TRAIN pool can use the pyqwest
    transport, which respx cannot intercept."""
    holder = MagicMock()
    holder.get_session_id.side_effect = AssertionError("Copies must not use the SDK session")
    holder.get_training_client_id.side_effect = AssertionError(
        "Copies must not allocate SDK run slots"
    )
    holder.aclient.return_value.__enter__.return_value.models.copy_weights = AsyncMock(
        return_value=types.CopyWeightsResponse(tinker_path="tinker://dest:train:0/weights/copy")
    )

    async def _execute_with_retries(fn: Any) -> Any:
        return await fn()

    holder.execute_with_retries = _execute_with_retries
    holder.run_coroutine_threadsafe.side_effect = _Future

    client = ServiceClient(api_key="tml-test-key", project_id="proj-configured")
    client._rest_holder = holder
    return client, holder.aclient.return_value.__enter__.return_value.models.copy_weights


def test_copy_weights_addresses_the_destination_by_configured_project() -> None:
    """Both future access patterns must copy into a fresh session in the configured project."""
    client, copy = _client_with_stub_transport()

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
    assert first.session_id is None
    assert first.source_path == "tinker://src/weights/0001"
    assert first.ttl_seconds == 3600
    assert first.weights_access_token is None
    assert first.destination_project_id == "proj-configured"
    assert (
        copy.await_args_list[0].kwargs["extra_headers_secret"]["X-Tinker-Weights-Access-Token"]
        == "token-1"
    )
    assert "X-Tinker-Weights-Access-Token" not in copy.await_args_list[0].kwargs["extra_headers"]
    assert copy.await_args_list[1].kwargs["extra_headers_secret"] is None

    assert all(c.kwargs["request"].model_seq_id is None for c in copy.await_args_list)
