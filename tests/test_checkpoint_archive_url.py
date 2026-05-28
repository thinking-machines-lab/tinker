from __future__ import annotations

import asyncio
import contextlib
import datetime
from collections.abc import Awaitable, Callable, Iterator
from typing import TypeVar

import httpx
import pytest

from tinker._client import AsyncTinker
from tinker._exceptions import APIConnectionError
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.public_interfaces.rest_client import RestClient
from tinker.types import CheckpointArchiveUrlResponse

T = TypeVar("T")


class _FakeWeights:
    def __init__(self, response: CheckpointArchiveUrlResponse) -> None:
        self.attempts = 0
        self.response = response

    async def get_checkpoint_archive_url(
        self,
        *,
        model_id: str,
        checkpoint_id: str,
    ) -> CheckpointArchiveUrlResponse:
        self.attempts += 1
        if self.attempts == 1:
            raise APIConnectionError(httpx.Request("GET", "https://api.example.test/archive"))
        return self.response


class _FakeClient:
    def __init__(self, weights: _FakeWeights) -> None:
        self.weights = weights


class _FakeHolder:
    def __init__(self, weights: _FakeWeights) -> None:
        self.weights = weights
        self.used_pool_type: ClientConnectionPoolType | None = None
        self.execute_with_retries_called = False

    @contextlib.contextmanager
    def aclient(self, pool_type: ClientConnectionPoolType) -> Iterator[_FakeClient]:
        self.used_pool_type = pool_type
        yield _FakeClient(self.weights)

    async def execute_with_retries(self, func: Callable[[], Awaitable[T]]) -> T:
        self.execute_with_retries_called = True
        try:
            return await func()
        except APIConnectionError:
            return await func()

    def run_coroutine_threadsafe(
        self, coro: Awaitable[CheckpointArchiveUrlResponse]
    ) -> CheckpointArchiveUrlResponse:
        return asyncio.run(coro)


def test_get_checkpoint_archive_url_uses_holder_retries() -> None:
    response = CheckpointArchiveUrlResponse(
        url="https://download.example.test/archive.tar",
        expires=datetime.datetime(2026, 5, 27, tzinfo=datetime.UTC),
    )
    weights = _FakeWeights(response)
    holder = _FakeHolder(weights)
    rest_client = RestClient(holder)  # type: ignore[arg-type]

    result = rest_client._get_checkpoint_archive_url_submit(
        "run-id",
        "sampler_weights/final",
    )

    assert result is response
    assert holder.execute_with_retries_called
    assert holder.used_pool_type == ClientConnectionPoolType.CHECKPOINT_ARCHIVE_URL
    assert weights.attempts == 2


@pytest.mark.asyncio
async def test_get_checkpoint_archive_url_accepts_current_backend_redirect_response() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        accept_values = {value.strip() for value in request.headers["accept"].split(",")}
        assert accept_values == {"application/json"}
        return httpx.Response(
            302,
            headers={
                "Location": "https://download.example.test/archive.tar",
                "Expires": "Wed, 27 May 2026 20:00:00 GMT",
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        async_tinker = AsyncTinker(
            base_url="https://api.example.test",
            api_key="tml-test-api-key",
            http_client=http_client,
        )

        result = await async_tinker.weights.get_checkpoint_archive_url(
            model_id="run-id",
            checkpoint_id="sampler_weights/final",
        )
    finally:
        await http_client.aclose()

    assert result.url == "https://download.example.test/archive.tar"
    assert result.expires == datetime.datetime(2026, 5, 27, 20, tzinfo=datetime.UTC)
    assert len(captured) == 1
    assert "redirect" not in captured[0].url.params


@pytest.mark.asyncio
async def test_get_checkpoint_archive_url_accepts_future_backend_json_response() -> None:
    captured: list[httpx.Request] = []
    expires = datetime.datetime(2026, 5, 27, 20, tzinfo=datetime.UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "url": "https://download.example.test/archive.tar",
                "expires": expires.isoformat(),
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        async_tinker = AsyncTinker(
            base_url="https://api.example.test",
            api_key="tml-test-api-key",
            http_client=http_client,
        )

        result = await async_tinker.weights.get_checkpoint_archive_url(
            model_id="run-id",
            checkpoint_id="sampler_weights/final",
        )
    finally:
        await http_client.aclose()

    assert result.url == "https://download.example.test/archive.tar"
    assert result.expires == expires
    assert len(captured) == 1
    assert "redirect" not in captured[0].url.params
