from __future__ import annotations

import asyncio
import contextlib
import datetime
from collections.abc import Awaitable, Callable, Iterator
from typing import TypeVar

import httpx

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
    assert holder.used_pool_type == ClientConnectionPoolType.TRAIN
    assert weights.attempts == 2
