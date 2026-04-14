"""Tests for InternalClientHolder helpers."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tinker.lib._auth_token_provider import AuthTokenProvider
from tinker.lib.internal_client_holder import ClientConnectionPool, InternalClientHolder
from tinker.types.client_config_response import ClientConfigResponse as _ClientConfigResponse


class _MockHolder:
    """Minimal stand-in for testing _fetch_client_config."""

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

    def get_loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    async def execute_with_retries(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return await func(*args, **kwargs)

    # Bind the real method so the pool it creates uses our mock client
    _fetch_client_config = InternalClientHolder._fetch_client_config


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
