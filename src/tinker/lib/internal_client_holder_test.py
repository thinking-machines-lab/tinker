"""Tests for InternalClientHolder helpers."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tinker._exceptions import APIStatusError
from tinker.lib._auth_token_provider import ApiKeyAuthProvider, AuthTokenProvider
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.internal_client_holder import ClientConnectionPool, InternalClientHolder
from tinker.types.client_config_response import ClientConfigResponse as _ClientConfigResponse
from tinker.types.client_dynamic_config_response import (
    ClientDynamicConfigResponse as _ClientDynamicConfigResponse,
)


class _MockHolder:
    """Minimal stand-in for testing the client config fetch/refresh helpers."""

    def __init__(self, response: _ClientConfigResponse | Exception) -> None:
        service = MagicMock()
        if isinstance(response, Exception):
            service.client_config = AsyncMock(side_effect=response)
        else:
            service.client_config = AsyncMock(return_value=response)
        client = MagicMock()
        client.service = service
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=client)
        cm.__exit__ = MagicMock(return_value=None)
        self._cm = cm

        self._constructor_kwargs: dict[str, Any] = {}
        self._default_auth = MagicMock(spec=AuthTokenProvider)
        self._loop = asyncio.get_event_loop()
        self._client_dynamic_config = _ClientDynamicConfigResponse()

    def get_loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    def aclient(self, pool_type: ClientConnectionPoolType) -> Any:
        return self._cm

    async def execute_with_retries(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return await func(*args, **kwargs)

    # Bind the real methods so the pool/client they use is our mock client.
    _create_client_connection_pool = InternalClientHolder._create_client_connection_pool
    _fetch_client_config = InternalClientHolder._fetch_client_config
    _fetch_client_dynamic_config = InternalClientHolder._fetch_client_dynamic_config
    _fetch_initial_client_dynamic_config = InternalClientHolder._fetch_initial_client_dynamic_config
    _refresh_client_dynamic_config_once = InternalClientHolder._refresh_client_dynamic_config_once


def _patch_pool(monkeypatch: pytest.MonkeyPatch, holder: _MockHolder) -> None:
    monkeypatch.setattr(ClientConnectionPool, "aclient", lambda self: holder._cm)


# ---------------------------------------------------------------------------
# _fetch_client_config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_client_config_returns_flags_from_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder = _MockHolder(_ClientConfigResponse(pjwt_auth_enabled=True))
    _patch_pool(monkeypatch, holder)
    result = await InternalClientHolder._fetch_client_config(holder, holder._default_auth)  # type: ignore[arg-type]
    assert result.pjwt_auth_enabled is True


@pytest.mark.asyncio
async def test_fetch_client_config_returns_defaults_when_server_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder = _MockHolder(_ClientConfigResponse(pjwt_auth_enabled=False))
    _patch_pool(monkeypatch, holder)
    result = await InternalClientHolder._fetch_client_config(holder, holder._default_auth)  # type: ignore[arg-type]
    assert result.pjwt_auth_enabled is False


@pytest.mark.asyncio
async def test_fetch_client_config_raises_on_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder = _MockHolder(Exception("connection refused"))
    _patch_pool(monkeypatch, holder)
    with pytest.raises(Exception, match="connection refused"):
        await InternalClientHolder._fetch_client_config(holder, holder._default_auth)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_fetch_client_config_passes_sdk_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tinker._version import __version__ as tinker_sdk_version

    holder = _MockHolder(_ClientConfigResponse(pjwt_auth_enabled=False))
    _patch_pool(monkeypatch, holder)
    await InternalClientHolder._fetch_client_config(holder, holder._default_auth)  # type: ignore[arg-type]

    call_kwargs = holder._cm.__enter__.return_value.service.client_config.call_args
    assert call_kwargs.kwargs["request"].sdk_version == tinker_sdk_version


# ---------------------------------------------------------------------------
# _fetch_initial_client_dynamic_config
# ---------------------------------------------------------------------------


def _set_dynamic_config_response(
    holder: _MockHolder, response: _ClientDynamicConfigResponse | Exception
) -> None:
    service = holder._cm.__enter__.return_value.service
    if isinstance(response, Exception):
        service.client_dynamic_config = AsyncMock(side_effect=response)
    else:
        service.client_dynamic_config = AsyncMock(return_value=response)


@pytest.mark.asyncio
async def test_fetch_initial_client_dynamic_config_returns_flags_from_server() -> None:
    expected = _ClientDynamicConfigResponse(refresh_interval_sec=60)
    holder = _MockHolder(_ClientConfigResponse())
    _set_dynamic_config_response(holder, expected)
    result = await holder._fetch_initial_client_dynamic_config()
    assert result == expected


@pytest.mark.asyncio
async def test_fetch_initial_client_dynamic_config_falls_back_to_defaults_on_404() -> None:
    response = httpx.Response(
        404, request=httpx.Request("POST", "https://example.com/api/v1/client/dynamic_config")
    )
    holder = _MockHolder(_ClientConfigResponse())
    _set_dynamic_config_response(holder, APIStatusError("not found", response=response, body=None))
    result = await holder._fetch_initial_client_dynamic_config()
    assert result == _ClientDynamicConfigResponse()


@pytest.mark.asyncio
async def test_fetch_initial_client_dynamic_config_raises_on_network_error() -> None:
    holder = _MockHolder(_ClientConfigResponse())
    _set_dynamic_config_response(holder, Exception("connection refused"))
    with pytest.raises(Exception, match="connection refused"):
        await holder._fetch_initial_client_dynamic_config()


# ---------------------------------------------------------------------------
# Pickle round-trip: ambient TINKER_API_KEY must travel with the pickle
# ---------------------------------------------------------------------------


def _make_holder(
    api_key: str | None = None,
    dynamic_config_response: _ClientDynamicConfigResponse | None = None,
    **holder_kwargs: Any,
) -> InternalClientHolder:
    """Build a primary InternalClientHolder with server calls stubbed out."""
    with (
        patch.object(
            InternalClientHolder,
            "_fetch_client_config",
            new_callable=AsyncMock,
            # pjwt_auth_enabled=False → plain API-key auth path
            return_value=_ClientConfigResponse(pjwt_auth_enabled=False),
        ),
        patch.object(
            InternalClientHolder,
            "_fetch_client_dynamic_config",
            new_callable=AsyncMock,
            return_value=dynamic_config_response or _ClientDynamicConfigResponse(),
        ),
        patch.object(
            InternalClientHolder,
            "_create_session",
            new_callable=AsyncMock,
            return_value="sess-pickle-test",
        ),
        patch.object(
            InternalClientHolder,
            "_start_heartbeat",
            new_callable=AsyncMock,
        ),
        patch.object(
            InternalClientHolder,
            "_start_client_dynamic_config_refresh",
            new_callable=AsyncMock,
        ),
    ):
        holder = InternalClientHolder(api_key=api_key, **holder_kwargs)
        holder._session_heartbeat_task = MagicMock()
        holder._client_dynamic_config_refresh_task = MagicMock()
        return holder


def test_sampling_client_pickle_roundtrip_without_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pickle a SamplingClient created with ambient TINKER_API_KEY, then
    unpickle it in an environment without the env var (simulating a worker
    process). The credential must travel inside the pickle payload."""
    import pickle

    from tinker.lib.public_interfaces.sampling_client import SamplingClient

    monkeypatch.setenv("TINKER_API_KEY", "tml-key-from-env")
    holder = _make_holder(api_key=None)
    client = SamplingClient(holder, sampling_session_id="samp-1")

    payload = pickle.dumps(client)

    # Simulate the worker: no TINKER_API_KEY available.
    monkeypatch.delenv("TINKER_API_KEY")

    with (
        patch.object(
            InternalClientHolder,
            "_start_heartbeat",
            new_callable=AsyncMock,
        ),
        patch.object(
            InternalClientHolder,
            "_start_client_dynamic_config_refresh",
            new_callable=AsyncMock,
        ),
    ):
        restored = pickle.loads(payload)

    assert isinstance(restored.holder._default_auth, ApiKeyAuthProvider)
    assert restored.holder._default_auth._token == "tml-key-from-env"


# ---------------------------------------------------------------------------
# Dynamic client config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_client_dynamic_config_swaps_in_new_flags() -> None:
    new_config = _ClientDynamicConfigResponse(refresh_interval_sec=60)
    holder = _MockHolder(_ClientConfigResponse())
    _set_dynamic_config_response(holder, new_config)
    await holder._refresh_client_dynamic_config_once()
    assert holder._client_dynamic_config == new_config


@pytest.mark.asyncio
async def test_refresh_client_dynamic_config_keeps_last_known_good_on_failure() -> None:
    last_known_good = _ClientDynamicConfigResponse(refresh_interval_sec=42)
    holder = _MockHolder(_ClientConfigResponse())
    _set_dynamic_config_response(holder, Exception("connection refused"))
    holder._client_dynamic_config = last_known_good
    await holder._refresh_client_dynamic_config_once()
    assert holder._client_dynamic_config == last_known_good


@pytest.mark.asyncio
async def test_refresh_client_dynamic_config_passes_sdk_version() -> None:
    from tinker._version import __version__ as tinker_sdk_version

    holder = _MockHolder(_ClientConfigResponse())
    _set_dynamic_config_response(holder, _ClientDynamicConfigResponse())
    await holder._refresh_client_dynamic_config_once()

    call_kwargs = holder._cm.__enter__.return_value.service.client_dynamic_config.call_args
    assert call_kwargs.kwargs["request"].sdk_version == tinker_sdk_version


def test_holder_fetches_dynamic_config_in_constructor() -> None:
    fetched = _ClientDynamicConfigResponse(refresh_interval_sec=123)
    holder = _make_holder(api_key="tml-test-key", dynamic_config_response=fetched)
    assert holder._client_dynamic_config == fetched


def test_holder_seeds_dynamic_config_from_kwargs() -> None:
    seed = _ClientDynamicConfigResponse(refresh_interval_sec=77)
    holder = _make_holder(api_key="tml-test-key", _client_dynamic_config=seed.model_dump())
    assert holder._client_dynamic_config == seed


def test_shadow_kwargs_carry_dynamic_config_snapshot() -> None:
    seed = _ClientDynamicConfigResponse(refresh_interval_sec=77)
    holder = _make_holder(api_key="tml-test-key", _client_dynamic_config=seed.model_dump())
    assert holder.shadow_kwargs["_client_dynamic_config"] == seed.model_dump()


def test_rest_support_redirect_pool_disables_pyqwest_when_enabled_by_config() -> None:
    holder = _make_holder(api_key="tml-test-key")

    redirect_pool = holder._get_client_connection_pool(
        ClientConnectionPoolType.REST_SUPPORT_REDIRECT
    )
    train_pool = holder._get_client_connection_pool(ClientConnectionPoolType.TRAIN)
    sample_pool = holder._get_client_connection_pool(ClientConnectionPoolType.SAMPLE)

    assert redirect_pool._constructor_kwargs["_client_config"].use_pyqwest_transport is False
    assert train_pool._constructor_kwargs["_client_config"].use_pyqwest_transport is True
    assert sample_pool._constructor_kwargs["_client_config"].use_pyqwest_transport is True
