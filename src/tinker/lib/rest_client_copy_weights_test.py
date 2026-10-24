"""Protect the copy_weights cutover contract and safe retries over HTTP."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

import httpx
import pytest

from tinker import PermissionDeniedError, types
from tinker._base_client import make_request_options
from tinker._client import AsyncTinker
from tinker._version import __version__ as tinker_sdk_version
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.internal_client_holder import InternalClientHolder
from tinker.lib.public_interfaces.rest_client import RestClient
from tinker.lib.public_interfaces.service_client import ServiceClient


class _TransportHolder(InternalClientHolder):
    """Use real scheduling and retries with an in-memory HTTP transport."""

    def __init__(self, client: AsyncTinker, loop: asyncio.AbstractEventLoop) -> None:
        self.client = client
        self._loop = loop
        self._telemetry = None
        self._session_id = "test-session"
        self._training_client_counter = 0
        self._training_client_lock = threading.Lock()
        self.attempts = 0

    @contextmanager
    def aclient(self, client_pool_type: ClientConnectionPoolType) -> Iterator[AsyncTinker]:
        assert client_pool_type == ClientConnectionPoolType.TRAIN
        self.attempts += 1
        yield self.client


_Transport = tuple[RestClient, _TransportHolder, list[httpx.Request], list[int | str]]


@pytest.fixture
def transport() -> Iterator[_Transport]:
    requests: list[httpx.Request] = []
    failures: list[int | str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["X-Api-Key"] == "tml-test-key"
        if request.method == "GET":
            assert request.url.path == "/api/v1/sessions/test-session"
            return httpx.Response(
                200,
                json={
                    "training_run_ids": [],
                    "sampler_ids": [],
                },
                headers={"X-Private-Response": "private-response-value"},
            )
        assert request.method == "POST"
        assert request.url.path == "/api/v1/copy_weights"
        assert request.headers["x-sdk-method"] == "copy_weights"
        assert request.headers["x-sdk-version"] == tinker_sdk_version
        assert uuid.UUID(request.headers["X-Idempotency-Key"]).version == 4
        if failures:
            failure = failures.pop(0)
            if failure == "timeout":
                raise httpx.ReadTimeout("Copy response lost", request=request)
            if failure == "connection":
                raise httpx.ConnectError("Connection lost", request=request)
            return httpx.Response(int(failure), json={"message": "Copy unavailable"})
        return httpx.Response(200, json={"tinker_path": "tinker://dest/weights/copy"})

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    client = AsyncTinker(
        api_key="tml-test-key",
        base_url="https://api.example.test",
        http_client=http,
        max_retries=1,
        _strict_response_validation=True,
    )
    holder = _TransportHolder(client, loop)
    try:
        yield RestClient(holder), holder, requests, failures
    finally:
        asyncio.run_coroutine_threadsafe(http.aclose(), loop).result()
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()


@pytest.mark.parametrize("access", ["result", "await", "result_async", "async_method"])
def test_copy_preserves_service_client_call_and_result_contract(
    transport: _Transport,
    access: Literal["result", "await", "result_async", "async_method"],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Existing keyword calls and both future access patterns must survive cutover."""
    rest, _, requests, _ = transport
    caplog.set_level(logging.DEBUG, logger="tinker")

    def submit():
        return rest.copy_weights(
            path="tinker://src/weights/one",
            destination_project_id="proj-dest",
            ttl_seconds=3600,
            weights_access_token="source-access-token",
        )

    async def copy() -> str:
        if access == "async_method":
            return await rest.copy_weights_async(
                path="tinker://src/weights/one",
                destination_project_id="proj-dest",
                ttl_seconds=3600,
                weights_access_token="source-access-token",
            )
        future = submit()
        if access == "result_async":
            return await future.result_async(timeout=5)
        return await future

    result = submit().result(timeout=5) if access == "result" else asyncio.run(copy())

    assert result == "tinker://dest/weights/copy"
    assert len(requests) == 1
    assert requests[0].headers["X-Tinker-Weights-Access-Token"] == "source-access-token"
    assert "source-access-token" not in caplog.text
    assert json.loads(requests[0].content) == {
        "source_path": "tinker://src/weights/one",
        "destination_project_id": "proj-dest",
        "ttl_seconds": 3600,
    }


@pytest.mark.parametrize("failure", [503, "timeout", "connection"])
@pytest.mark.parametrize("use_service", [False, True])
def test_copy_reuses_key_and_source_token_across_both_retry_layers(
    transport: _Transport,
    failure: int | str,
    use_service: bool,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A lost response must not duplicate copies or expose source credentials across retries."""
    rest, holder, requests, failures = transport
    caplog.set_level(logging.DEBUG, logger="tinker")
    failures.extend([failure] * (holder.client.max_retries + 1))
    if use_service:
        service = ServiceClient(api_key="tml-test-key", project_id="proj-service")
        monkeypatch.setattr(service, "_get_rest_holder", lambda: holder)
        future = service.copy_weights(
            "tinker://src/weights/one", weights_access_token="source-access-token"
        )
    else:
        future = rest.copy_weights(
            "tinker://src/weights/one", "proj-dest", weights_access_token="source-access-token"
        )
    result = future.result(timeout=15)

    assert result == "tinker://dest/weights/copy"
    copies = [request for request in requests if request.method == "POST"]
    assert holder.attempts == 2
    assert len(copies) == holder.client.max_retries + 2
    assert len({request.headers["X-Idempotency-Key"] for request in copies}) == 1
    assert all(
        request.headers["X-Tinker-Weights-Access-Token"] == "source-access-token"
        for request in copies
    )
    assert all(request.content == copies[0].content for request in copies)
    assert "source-access-token" not in caplog.text
    assert "X-Tinker-Weights-Access-Token" not in caplog.text
    assert copies[0].headers["X-Idempotency-Key"] in caplog.text
    if use_service:
        assert json.loads(copies[0].content)["destination_project_id"] == "proj-service"
        assert holder.get_training_client_id() == 0


@pytest.mark.parametrize("project_id", [None, "proj-service"])
def test_service_copy_uses_the_configured_project_without_reusing_the_session(
    transport: _Transport,
    monkeypatch: pytest.MonkeyPatch,
    project_id: str | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Copies must use the configured project and create new sessions without allocating SDK run slots."""
    _, holder, requests, _ = transport
    monkeypatch.setenv("TINKER_PROJECT_ID", "proj-env-configured")
    service = ServiceClient(api_key="tml-test-key", project_id=project_id)
    monkeypatch.setattr(service, "_get_rest_holder", lambda: holder)
    monkeypatch.setenv("TINKER_PROJECT_ID", "proj-env-changed")

    def fail_if_session_created() -> None:
        raise AssertionError("Copying must not create or access the SDK session")

    monkeypatch.setattr(service, "_get_session_holder", fail_if_session_created)
    caplog.set_level(logging.DEBUG, logger="tinker")

    result = service.copy_weights(
        path="tinker://src/weights/one",
        ttl_seconds=3600,
        weights_access_token="source-access-token",
    ).result(timeout=5)

    async def copy() -> str:
        return await service.copy_weights(path="tinker://src/weights/two")

    assert result == asyncio.run(copy()) == "tinker://dest/weights/copy"
    copies = [request for request in requests if request.method == "POST"]
    assert [json.loads(request.content) for request in copies] == [
        {
            "destination_project_id": project_id or "proj-env-configured",
            "source_path": "tinker://src/weights/one",
            "ttl_seconds": 3600,
        },
        {
            "destination_project_id": project_id or "proj-env-configured",
            "source_path": "tinker://src/weights/two",
        },
    ]
    assert len(requests) == 2
    assert holder.get_training_client_id() == 0
    assert copies[0].headers["X-Tinker-Weights-Access-Token"] == "source-access-token"
    assert "X-Tinker-Weights-Access-Token" not in copies[1].headers
    assert len({request.headers["X-Idempotency-Key"] for request in copies}) == 2
    assert "source-access-token" not in caplog.text


@pytest.mark.parametrize("extra_headers", [None, {"X-Public-Request": "public-request-value"}])
def test_ordinary_request_and_response_headers_remain_logged(
    transport: _Transport,
    caplog: pytest.LogCaptureFixture,
    extra_headers: dict[str, str] | None,
) -> None:
    """Separating secrets must preserve existing logging for ordinary headers and responses."""
    _, holder, requests, _ = transport
    caplog.set_level(logging.DEBUG, logger="tinker")

    async def get_session() -> types.GetSessionResponse:
        return await holder.client.get(
            "/api/v1/sessions/test-session",
            cast_to=types.GetSessionResponse,
            options=make_request_options(extra_headers=extra_headers),
        )

    result = asyncio.run_coroutine_threadsafe(get_session(), holder.get_loop()).result(timeout=5)
    assert result.training_run_ids == []
    assert len(requests) == 1
    assert requests[0].method == "GET"
    if extra_headers is not None:
        assert requests[0].headers["X-Public-Request"] == "public-request-value"
        assert "public-request-value" in caplog.text
    messages = [record.getMessage() for record in caplog.records]
    request_messages = [message for message in messages if message.startswith("Request options:")]
    assert request_messages
    assert any(
        message.startswith("HTTP Response:") and "private-response-value" in message
        for message in messages
    )
    assert "tml-test-key" not in caplog.text


@pytest.mark.parametrize(
    "extra_headers_secret",
    [
        None,
        {"X-Secret-Request": "secret-request-value", "x-public-request": "secret-override-value"},
    ],
)
def test_secret_post_headers_bypass_options_logging(
    transport: _Transport,
    caplog: pytest.LogCaptureFixture,
    extra_headers_secret: dict[str, str] | None,
) -> None:
    """Arbitrary secrets must reach HTTP and override ordinary headers without entering logs."""
    _, holder, requests, _ = transport
    caplog.set_level(logging.DEBUG, logger="tinker")
    headers = {
        "x-sdk-method": "copy_weights",
        "x-sdk-version": tinker_sdk_version,
        "X-Idempotency-Key": str(uuid.uuid4()),
        "X-Public-Request": "public-request-value",
    }

    async def copy() -> types.CopyWeightsResponse:
        return await holder.client.models.copy_weights(
            request=types.CopyWeightsRequest(
                source_path="tinker://src/weights/one", destination_project_id="proj-dest"
            ),
            extra_headers=headers,
            extra_headers_secret=extra_headers_secret,
        )

    result = asyncio.run_coroutine_threadsafe(copy(), holder.get_loop()).result(timeout=5)
    assert result.tinker_path == "tinker://dest/weights/copy"
    assert len(requests) == 1
    if extra_headers_secret is not None:
        assert requests[0].headers["X-Secret-Request"] == "secret-request-value"
        assert requests[0].headers["X-Public-Request"] == "secret-override-value"
    else:
        assert "X-Secret-Request" not in requests[0].headers
        assert requests[0].headers["X-Public-Request"] == "public-request-value"
    assert headers["X-Public-Request"] == "public-request-value"
    assert "public-request-value" in caplog.text
    for value in (
        "secret-request-value",
        "secret-override-value",
        "x-secret-request",
        "extra_headers_secret",
    ):
        assert value not in caplog.text.lower()


def test_identical_copies_remain_independent(transport: _Transport) -> None:
    """A key belongs to one operation even when the source and destination match."""
    rest, _, requests, _ = transport
    futures = [rest.copy_weights("tinker://src/weights/one", "proj-dest") for _ in range(2)]
    assert [future.result(timeout=5) for future in futures] == ["tinker://dest/weights/copy"] * 2
    assert len({request.headers["X-Idempotency-Key"] for request in requests}) == 2
    assert all("X-Tinker-Weights-Access-Token" not in request.headers for request in requests)
    assert all("ttl_seconds" not in json.loads(request.content) for request in requests)


@pytest.mark.parametrize("project_id", [None, "proj-client"])
def test_copy_destination_is_independent_of_service_project(
    transport: _Transport, monkeypatch: pytest.MonkeyPatch, project_id: str | None
) -> None:
    """A session's project must not silently route a REST copy to the wrong project."""
    _, holder, requests, _ = transport
    monkeypatch.setenv("TINKER_PROJECT_ID", "proj-env")
    service = ServiceClient(api_key="tml-test-key", project_id=project_id)
    monkeypatch.setattr(service, "_get_rest_holder", lambda: holder)
    monkeypatch.setenv("TINKER_PROJECT_ID", "proj-changed")
    rest = RestClient(service._get_rest_holder())

    rest.copy_weights("tinker://src/weights/one").result(timeout=5)
    rest.copy_weights("tinker://src/weights/one", "proj-explicit").result(timeout=5)

    assert [json.loads(request.content)["destination_project_id"] for request in requests] == [
        "proj-changed",
        "proj-explicit",
    ]

    monkeypatch.delenv("TINKER_PROJECT_ID")
    with pytest.raises(ValueError, match="Provide destination_project_id or set TINKER_PROJECT_ID"):
        rest.copy_weights("tinker://src/weights/one")
    assert len(requests) == 2


def test_rest_copy_defaults_to_environment(
    transport: _Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A standalone REST client follows the SDK environment default."""
    rest, _, requests, _ = transport
    monkeypatch.setenv("TINKER_PROJECT_ID", "proj-env")
    rest.copy_weights("tinker://src/weights/one", destination_project_id=None).result(timeout=5)
    assert json.loads(requests[0].content)["destination_project_id"] == "proj-env"


@pytest.mark.parametrize("project_env", [None, ""])
def test_missing_destination_fails_before_sending_copy(
    transport: _Transport, monkeypatch: pytest.MonkeyPatch, project_env: str | None
) -> None:
    """Fail locally if the REST endpoint's required project cannot be resolved."""
    rest, _, requests, _ = transport
    if project_env is None:
        monkeypatch.delenv("TINKER_PROJECT_ID", raising=False)
    else:
        monkeypatch.setenv("TINKER_PROJECT_ID", project_env)
    with pytest.raises(ValueError, match="Provide destination_project_id or set TINKER_PROJECT_ID"):
        rest.copy_weights("tinker://src/weights/one")
    assert requests == []


def test_copy_permission_error_propagates_without_retry(transport: _Transport) -> None:
    """Permission failures must remain visible and must not be retried."""
    rest, holder, requests, failures = transport
    failures.append(403)
    with pytest.raises(PermissionDeniedError):
        rest.copy_weights("tinker://src/weights/one", "proj-dest").result(timeout=5)
    assert len(requests) == holder.attempts == 1
