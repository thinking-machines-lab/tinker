"""Tests for JWT authentication helpers."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from tinker._exceptions import TinkerError
from tinker.lib._auth_token_provider import (
    ApiKeyAuthProvider,
    CredentialCmdAuthProvider,
    resolve_auth_provider,
)
from tinker.lib._jwt_auth import (
    JwtAuthProvider,
    _jwt_expiry,
    _seconds_until_expiry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jwt(exp: float) -> str:
    """Build a minimal fake JWT with a given exp claim."""
    header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
    payload_bytes = json.dumps({"exp": exp, "sub": "test"}).encode()
    payload = base64.urlsafe_b64encode(payload_bytes).rstrip(b"=").decode()
    return f"{header}.{payload}.fakesig"


class _MockAuthResponse:
    def __init__(self, jwt: str) -> None:
        self.jwt = jwt


class _MockHolder:
    """Minimal mock providing aclient() for testing JwtAuthProvider."""

    def __init__(self, response_jwt: str, *, fail: bool = False) -> None:
        service = MagicMock()
        if fail:
            service.auth_token = AsyncMock(side_effect=Exception("network error"))
        else:
            service.auth_token = AsyncMock(return_value=_MockAuthResponse(response_jwt))
        client = MagicMock()
        client.service = service
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=client)
        cm.__exit__ = MagicMock(return_value=None)
        self._cm = cm

    def aclient(self):
        return self._cm


# ---------------------------------------------------------------------------
# _jwt_expiry
# ---------------------------------------------------------------------------


def test_jwt_expiry_parses_valid():
    exp = time.time() + 3600
    assert abs(_jwt_expiry(_make_jwt(exp)) - exp) < 1


def test_jwt_expiry_raises_on_invalid():
    with pytest.raises(Exception):
        _jwt_expiry("not.a.jwt")


def test_jwt_expiry_raises_on_missing_exp():
    header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(b'{"sub":"x"}').rstrip(b"=").decode()
    with pytest.raises(Exception):
        _jwt_expiry(f"{header}.{payload}.sig")


# ---------------------------------------------------------------------------
# AuthTokenProvider hierarchy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_key_provider_resolves_key():
    auth = ApiKeyAuthProvider(api_key="tml-test-key")
    assert await auth.get_token() == "tml-test-key"


@pytest.mark.asyncio
async def test_credential_cmd_provider_runs_command():
    auth = CredentialCmdAuthProvider("echo test-credential")
    assert await auth.get_token() == "test-credential"


@pytest.mark.asyncio
async def test_resolve_auth_provider_fallback_to_cmd(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.setenv("TINKER_CREDENTIAL_CMD", "echo fallback-cred")
    auth = resolve_auth_provider(api_key=None, enforce_cmd=False)
    assert isinstance(auth, CredentialCmdAuthProvider)
    assert await auth.get_token() == "fallback-cred"


def test_credential_cmd_provider_raises_with_empty_cmd():
    with pytest.raises(TinkerError, match="dynamic credentials"):
        CredentialCmdAuthProvider("")


# ---------------------------------------------------------------------------
# JwtAuthProvider.init
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_init_fetches_jwt_and_stores_it():
    exp = time.time() + 7200
    jwt = _make_jwt(exp)
    holder = _MockHolder(jwt)
    provider = JwtAuthProvider(holder.aclient)

    await provider.init()

    assert await provider.get_token() == jwt
    holder._cm.__enter__.return_value.service.auth_token.assert_called_once()


@pytest.mark.asyncio
async def test_init_raises_on_fetch_failure():
    holder = _MockHolder("some-jwt", fail=True)
    provider = JwtAuthProvider(holder.aclient)

    with pytest.raises(Exception, match="network error"):
        await provider.init()


# ---------------------------------------------------------------------------
# JwtAuthProvider._fetch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_returns_and_stores_token():
    exp = time.time() + 7200
    jwt = _make_jwt(exp)
    holder = _MockHolder(jwt)
    provider = JwtAuthProvider(holder.aclient)

    result = await provider._fetch()

    assert result == jwt
    assert await provider.get_token() == jwt


# ---------------------------------------------------------------------------
# _seconds_until_expiry
# ---------------------------------------------------------------------------


def test_seconds_until_expiry_returns_remaining_for_valid_jwt():
    exp = time.time() + 3600
    assert abs(_seconds_until_expiry(_make_jwt(exp)) - 3600) < 1


def test_seconds_until_expiry_returns_zero_for_unparseable_jwt():
    assert _seconds_until_expiry("not.a.jwt") == 0.0


def test_seconds_until_expiry_returns_negative_for_expired_jwt():
    exp = time.time() - 60
    assert _seconds_until_expiry(_make_jwt(exp)) < 0


# ---------------------------------------------------------------------------
# JwtAuthProvider.get_token on-demand refresh
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_token_returns_cached_when_fresh():
    """Cached token with comfortable runway is returned without refetching."""
    fresh_jwt = _make_jwt(time.time() + 7200)
    holder = _MockHolder("should-not-be-fetched")
    provider = JwtAuthProvider(holder.aclient, seed_token=fresh_jwt)

    assert await provider.get_token() == fresh_jwt
    holder._cm.__enter__.return_value.service.auth_token.assert_not_called()


@pytest.mark.asyncio
async def test_get_token_refreshes_when_near_expiry():
    """If cached token has <= _REFRESH_ON_DEMAND_SECS left, fetch a new one."""
    near_expiry_jwt = _make_jwt(time.time() + 30)  # 30s left, under threshold
    refreshed_jwt = _make_jwt(time.time() + 7200)
    holder = _MockHolder(refreshed_jwt)
    provider = JwtAuthProvider(holder.aclient, seed_token=near_expiry_jwt)

    assert await provider.get_token() == refreshed_jwt
    holder._cm.__enter__.return_value.service.auth_token.assert_called_once()


@pytest.mark.asyncio
async def test_get_token_refreshes_when_already_expired():
    """An expired cached token must trigger refresh, not be served as-is."""
    expired_jwt = _make_jwt(time.time() - 30)
    refreshed_jwt = _make_jwt(time.time() + 7200)
    holder = _MockHolder(refreshed_jwt)
    provider = JwtAuthProvider(holder.aclient, seed_token=expired_jwt)

    assert await provider.get_token() == refreshed_jwt
    holder._cm.__enter__.return_value.service.auth_token.assert_called_once()


@pytest.mark.asyncio
async def test_get_token_refreshes_when_cached_token_is_unparseable():
    """A garbled cached token (e.g. corrupt seed) is treated as expired."""
    refreshed_jwt = _make_jwt(time.time() + 7200)
    holder = _MockHolder(refreshed_jwt)
    provider = JwtAuthProvider(holder.aclient, seed_token="not.a.jwt")

    assert await provider.get_token() == refreshed_jwt
    holder._cm.__enter__.return_value.service.auth_token.assert_called_once()


@pytest.mark.asyncio
async def test_get_token_concurrent_refresh_only_fires_once():
    """Many concurrent get_token() calls share a single in-flight refresh."""
    near_expiry_jwt = _make_jwt(time.time() + 30)
    refreshed_jwt = _make_jwt(time.time() + 7200)

    fetch_started = asyncio.Event()
    fetch_release = asyncio.Event()
    fetch_count = 0

    async def slow_auth_token():
        nonlocal fetch_count
        fetch_count += 1
        fetch_started.set()
        await fetch_release.wait()
        return _MockAuthResponse(refreshed_jwt)

    service = MagicMock()
    service.auth_token = slow_auth_token
    client = MagicMock()
    client.service = service
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=client)
    cm.__exit__ = MagicMock(return_value=None)

    provider = JwtAuthProvider(lambda: cm, seed_token=near_expiry_jwt)

    tasks = [asyncio.create_task(provider.get_token()) for _ in range(5)]

    # Wait for the first task to enter the fetch, then let the others queue
    # up at the lock before releasing the in-flight fetch.
    await fetch_started.wait()
    await asyncio.sleep(0)
    fetch_release.set()
    results = await asyncio.gather(*tasks)

    assert fetch_count == 1
    assert all(r == refreshed_jwt for r in results)


@pytest.mark.asyncio
async def test_get_token_returns_stale_token_when_refresh_fails(
    caplog: pytest.LogCaptureFixture,
):
    """If on-demand refresh fails, return cached token + log a warning.

    Better than raising — the request will surface its own error if the
    token really is rejected, and other in-flight requests sharing this
    provider can still make progress on transient refresh failures.
    """
    near_expiry_jwt = _make_jwt(time.time() + 30)
    holder = _MockHolder("unused", fail=True)
    provider = JwtAuthProvider(holder.aclient, seed_token=near_expiry_jwt)

    with caplog.at_level("WARNING"):
        result = await provider.get_token()

    assert result == near_expiry_jwt
    assert "On-demand JWT refresh failed" in caplog.text


@pytest.mark.asyncio
async def test_get_token_returns_none_when_no_token_and_refresh_fails():
    """No cached token + failed refresh => return None (no header sent)."""
    holder = _MockHolder("unused", fail=True)
    provider = JwtAuthProvider(holder.aclient, seed_token=None)

    assert await provider.get_token() is None
