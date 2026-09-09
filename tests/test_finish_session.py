from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Awaitable, Callable, Iterator
from typing import TYPE_CHECKING, Any, Coroutine, TypeVar, cast

import pytest

from tinker._types import NoneType
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.public_interfaces.service_client import ServiceClient

if TYPE_CHECKING:
    from tinker.lib.internal_client_holder import InternalClientHolder

T = TypeVar("T")


class _Future:
    """Stand-in for the holder's future, awaitable or blocking like the real one."""

    def __init__(self, coro: Coroutine[object, object, None]) -> None:
        self._coro = coro

    def result(self) -> None:
        return asyncio.new_event_loop().run_until_complete(self._coro)

    def __await__(self) -> Any:
        return self._coro.__await__()


class _FakeClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, object]]] = []

    async def post(
        self,
        path: str,
        *,
        body: dict[str, object],
        cast_to: type[None],
    ) -> None:
        assert cast_to is NoneType
        self.requests.append((path, body))


class _FakeHolder:
    def __init__(self) -> None:
        self.client = _FakeClient()
        self.used_pool_type: ClientConnectionPoolType | None = None
        self.close_calls: int = 0

    def get_session_id(self) -> str:
        return "session-1"

    @contextlib.contextmanager
    def aclient(self, pool_type: ClientConnectionPoolType) -> Iterator[_FakeClient]:
        self.used_pool_type = pool_type
        yield self.client

    async def execute_with_retries(self, func: Callable[[], Awaitable[T]]) -> T:
        return await func()

    def run_coroutine_threadsafe(
        self,
        coro: Coroutine[object, object, None],
    ) -> _Future:
        return _Future(coro)

    def get_telemetry(self) -> None:
        return None

    async def aclose(self) -> None:
        self.close_calls += 1


def _service_client(session_holder: _FakeHolder | None = None) -> ServiceClient:
    service_client = ServiceClient.__new__(ServiceClient)
    service_client._session_holder = cast("InternalClientHolder | None", session_holder)
    service_client._rest_holder = None
    service_client._lifecycle_lock = threading.Lock()
    service_client._closed = False
    return service_client


def test_close_finishes_session_and_closes_holder() -> None:
    holder = _FakeHolder()
    service_client = _service_client(holder)

    # One method, both access patterns: block on the future, or await it.
    service_client.close("success", detail="training complete").result()

    assert holder.used_pool_type == ClientConnectionPoolType.TRAIN
    assert holder.client.requests == [
        (
            "/api/v1/sessions/session-1/finish",
            {"reason": {"type": "success"}, "detail": "training complete"},
        )
    ]
    assert holder.close_calls == 1
    assert service_client._closed

    async_holder = _FakeHolder()
    async_client = _service_client(async_holder)

    async def _awaited() -> None:
        await async_client.close("errored", detail="boom")

    asyncio.run(_awaited())

    assert async_holder.client.requests == [
        ("/api/v1/sessions/session-1/finish", {"reason": {"type": "errored"}, "detail": "boom"})
    ]
    assert async_holder.close_calls == 1


def test_close_without_session_does_not_create_one() -> None:
    service_client = _service_client()

    service_client.close("success").result()

    assert service_client._session_holder is None
    assert service_client._closed


def test_close_is_idempotent() -> None:
    holder = _FakeHolder()
    service_client = _service_client(holder)

    service_client.close("success").result()
    service_client.close("errored", detail="should not post again").result()

    assert holder.client.requests == [
        ("/api/v1/sessions/session-1/finish", {"reason": {"type": "success"}, "detail": None})
    ]
    assert holder.close_calls == 1


def test_close_rejects_lazy_holder_creation() -> None:
    service_client = _service_client()

    service_client.close("success").result()

    with pytest.raises(RuntimeError, match="ServiceClient is closed"):
        _ = service_client.holder
    with pytest.raises(RuntimeError, match="ServiceClient is closed"):
        service_client.create_rest_client()
