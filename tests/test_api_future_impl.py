from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Generator
from contextlib import contextmanager
from typing import Any, cast

import httpx
import orjson
from pyqwest.httpx import AsyncPyqwestTransport

from tinker import types
from tinker._client import AsyncTinker
from tinker.lib import api_future_impl
from tinker.lib.api_future_impl import _UNCOMPUTED, _APIFuture
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.internal_client_holder import BytesSemaphore
from tinker.lib.retry_handler import RetryConfig, RetryHandler


class _ResponseBody:
    def __init__(self, chunks: list[bytes], *, stall: bool = False) -> None:
        self._chunks = chunks
        self._stall = stall
        self._never_finishes = asyncio.Event()
        self.started = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.started.set()
        for chunk in self._chunks:
            yield chunk
        if self._stall:
            await self._never_finishes.wait()


class _PyqwestResponse:
    def __init__(self, body: _ResponseBody) -> None:
        self.status = 200
        self.headers = {"content-type": "application/json"}
        self.trailers: dict[str, str] = {}
        self.body = body
        self.content = body
        self.closed = asyncio.Event()

    async def aclose(self) -> None:
        self.closed.set()


class _StallFirstPyqwestTransport:
    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.stalled_response = _PyqwestResponse(_ResponseBody([b'{"ok":'], stall=True))
        self.closed_before_second_request = False

    async def execute(self, request: Any) -> _PyqwestResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return self.stalled_response

        self.closed_before_second_request = self.stalled_response.closed.is_set()
        return _PyqwestResponse(_ResponseBody([b'{"ok":true}']))


class _Holder:
    def __init__(self, client: AsyncTinker) -> None:
        self._client = client
        self._inflight_response_bytes_semaphore = BytesSemaphore(1024 * 1024)

    @contextmanager
    def aclient(
        self,
        client_pool_type: ClientConnectionPoolType,  # noqa: ARG002
    ) -> Generator[AsyncTinker, None, None]:
        yield self._client

    def _should_pause_on_billing(self, status_code: int, detail: str) -> bool:  # noqa: ARG002
        return False

    def get_telemetry(self) -> None:
        return None


def _make_future(holder: _Holder, request_id: str) -> _APIFuture[dict[str, bool]]:
    # Bypass __init__, which starts the polling coroutine on the holder's
    # background event loop. This test drives _result_async directly.
    future = cast(Any, object.__new__(_APIFuture))
    future.model_cls = dict
    future.holder = holder
    future.untyped_future = types.UntypedAPIFuture(request_id=request_id)
    future.request_type = "Forward"
    future.request_start_time = time.time()
    future.request_future_start_time = time.time()
    future.request_queue_roundtrip_time = 0.0
    future._cached_result = _UNCOMPUTED
    future._queue_state_observer = None
    return cast(_APIFuture[dict[str, bool]], future)


async def test_stalled_pyqwest_body_repolls_same_future_without_rerunning_operation(
    monkeypatch: Any,
) -> None:
    native_transport = _StallFirstPyqwestTransport()
    pyqwest_transport = AsyncPyqwestTransport(transport=cast(Any, native_transport))
    http_client = httpx.AsyncClient(transport=pyqwest_transport)
    client = AsyncTinker(
        base_url="http://test",
        api_key="tml-test-api-key",
        http_client=http_client,
        _client_config=types.ClientConfigResponse(use_pyqwest_transport=False),
    )
    holder = _Holder(client)
    request_id = "future-stalled-body"
    original_operation_calls = 0

    monkeypatch.setattr(api_future_impl, "_RETRIEVE_FUTURE_TIMEOUT_SECONDS", 0.05)

    async def original_operation() -> dict[str, bool]:
        nonlocal original_operation_calls
        original_operation_calls += 1
        return await _make_future(holder, request_id)._result_async()

    outer_retry_handler: RetryHandler[dict[str, bool]] = RetryHandler(
        RetryConfig(
            retry_delay_base=0,
            retry_delay_max=0,
            jitter_factor=0,
            enable_stuck_detection=False,
        )
    )

    try:
        result = await asyncio.wait_for(outer_retry_handler.execute(original_operation), timeout=3)
    finally:
        await http_client.aclose()

    assert result == {"ok": True}
    assert original_operation_calls == 1
    assert len(native_transport.requests) == 2
    assert native_transport.stalled_response.body.started.is_set()
    assert native_transport.closed_before_second_request

    request_bodies = [
        orjson.loads(cast(bytes, request.content)) for request in native_transport.requests
    ]
    assert [body["request_id"] for body in request_bodies] == [request_id, request_id]
    assert all(
        str(request.url).endswith("/api/v1/retrieve_future")
        for request in native_transport.requests
    )
