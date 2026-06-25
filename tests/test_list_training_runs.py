from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Iterator
from typing import Any, TypeVar

from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.public_interfaces.rest_client import RestClient
from tinker.types import TrainingRunsResponse
from tinker.types.cursor import Cursor

T = TypeVar("T")

_EMPTY = TrainingRunsResponse(training_runs=[], cursor=Cursor(offset=0, limit=20, total_count=0))


class _FakeClient:
    def __init__(self, captured: dict[str, Any]) -> None:
        self._captured = captured

    async def get(
        self, path: str, *, options: dict[str, Any], cast_to: Any
    ) -> TrainingRunsResponse:
        self._captured["path"] = path
        self._captured["params"] = options["params"]
        return _EMPTY


class _FakeHolder:
    def __init__(self) -> None:
        self.captured: dict[str, Any] = {}

    @contextlib.contextmanager
    def aclient(self, pool_type: ClientConnectionPoolType) -> Iterator[_FakeClient]:
        yield _FakeClient(self.captured)

    async def execute_with_retries(self, func: Callable[[], Awaitable[T]]) -> T:
        return await func()

    def run_coroutine_threadsafe(self, coro: Awaitable[T]) -> T:
        return asyncio.run(coro)


def test_list_training_runs_forwards_project_id() -> None:
    holder = _FakeHolder()
    rest_client = RestClient(holder)  # type: ignore[arg-type]

    rest_client._list_training_runs_submit(limit=5, offset=10, project_id="proj-123")

    assert holder.captured["path"] == "/api/v1/training_runs"
    params = holder.captured["params"]
    assert params["project_id"] == "proj-123"
    assert params["limit"] == 5
    assert params["offset"] == 10


def test_list_training_runs_omits_project_id_when_unset() -> None:
    holder = _FakeHolder()
    rest_client = RestClient(holder)  # type: ignore[arg-type]

    rest_client._list_training_runs_submit()

    assert "project_id" not in holder.captured["params"]
