"""Tests for JWT authentication helpers."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from tinker._client import AsyncTinker
from tinker._exceptions import APIConnectionError, AuthenticationError, TinkerError
from tinker.lib import _jwt_auth
from tinker.lib._auth_token_provider import (
    ApiKeyAuthProvider,
    AuthTokenProvider,
    CredentialCmdAuthProvider,
    resolve_auth_provider,
)
from tinker.lib._jwt_auth import (
    JwtAuthProvider,
    _jwt_expiry,
    _refresh_delay,
    _seconds_until_expiry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jwt_with_claims(claims: dict[str, Any]) -> str:
    """Build a minimal fake JWT with the given claims."""
    header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
    payload_bytes = json.dumps(claims).encode()
    payload = base64.urlsafe_b64encode(payload_bytes).rstrip(b"=").decode()
    return f"{header}.{payload}.fakesig"


def _make_jwt(exp: float) -> str:
    """Build a minimal fake one-hour JWT with a given exp claim."""
    return _jwt_with_claims({"iat": exp - 3600, "exp": exp, "sub": "test"})


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


def _provider_with_token(aclient_fn: Callable[[], Any], token: str) -> JwtAuthProvider:
    provider = JwtAuthProvider(aclient_fn)
    provider._token = token
    return provider


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
@pytest.mark.parametrize(
    "provider_for",
    [ApiKeyAuthProvider, lambda jwt: CredentialCmdAuthProvider(f"echo {jwt}")],
)
async def test_provider_refuses_an_expired_or_malformed_jwt(
    provider_for: Callable[[str], AuthTokenProvider],
):
    fresh = _make_jwt(time.time() + 60)
    assert await provider_for(fresh).get_token() == fresh
    with pytest.raises(TinkerError, match="expired"):
        await provider_for(_make_jwt(time.time() - 60)).get_token()
    with pytest.raises(ValueError):
        await provider_for("eyJnot-a-jwt").get_token()


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
# JwtAuthProvider first fetch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_get_token_fetches_and_starts_the_refresh_loop():
    jwt = _make_jwt(time.time() + 7200)
    holder = _MockHolder(jwt)
    provider = JwtAuthProvider(holder.aclient)
    assert provider._refresh_task is None

    assert await provider.get_token() == jwt
    assert await provider.get_token() == jwt
    holder._cm.__enter__.return_value.service.auth_token.assert_called_once()
    assert provider._refresh_task is not None
    await provider.close()


@pytest.mark.asyncio
async def test_get_token_raises_when_first_fetch_fails():
    holder = _MockHolder("some-jwt", fail=True)
    provider = JwtAuthProvider(holder.aclient)

    with pytest.raises(Exception, match="network error"):
        await provider.get_token()
    assert provider._refresh_task is None


_EXCHANGE_REQUEST = httpx.Request("POST", "https://api.example.test/api/v1/auth/token")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        AuthenticationError(
            "bad key", response=httpx.Response(401, request=_EXCHANGE_REQUEST), body=None
        ),
        APIConnectionError(request=_EXCHANGE_REQUEST),
    ],
)
async def test_auth_provider_failure_reaches_caller_unchanged(error: Exception):
    """A failed exchange inside the auth flow is surfaced as is, not retried and
    wrapped as a connection error of the request it was authenticating."""

    class _FailingAuth(AuthTokenProvider):
        calls = 0

        async def get_token(self) -> str | None:
            self.calls += 1
            raise error

    auth = _FailingAuth()
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    )
    try:
        client = AsyncTinker(
            base_url="https://api.example.test", _auth=auth, http_client=http_client
        )
        with pytest.raises(type(error)) as raised:
            await client.get("/api/v1/healthz", cast_to=object)
    finally:
        await http_client.aclose()
    assert raised.value is error
    assert auth.calls == 1


# ---------------------------------------------------------------------------
# JwtAuthProvider replay after a rejected token
# ---------------------------------------------------------------------------

_INVALID_TOKEN = {"WWW-Authenticate": 'Bearer error="invalid_token"'}


def _api(accepted: str, rejection_headers: dict[str, str], seen: list[str]) -> httpx.AsyncClient:
    """An API that accepts one token and rejects every other with a 401."""

    def handle(request: httpx.Request) -> httpx.Response:
        token = request.headers["X-API-Key"]
        seen.append(token)
        if token == accepted:
            return httpx.Response(200, json={})
        return httpx.Response(401, headers=rejection_headers, json={"detail": "Invalid JWT"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


@pytest.mark.asyncio
@pytest.mark.parametrize("new_token_accepted", [True, False])
async def test_invalid_token_401_replaces_the_token_and_replays_once(new_token_accepted: bool):
    old, new = _make_jwt(time.time() + 3000), _make_jwt(time.time() + 3600)
    holder = _MockHolder(new)
    provider = _provider_with_token(holder.aclient, old)
    seen: list[str] = []
    http_client = _api(new if new_token_accepted else "", _INVALID_TOKEN, seen)
    try:
        client = AsyncTinker(
            base_url="https://api.example.test", _auth=provider, http_client=http_client
        )
        if new_token_accepted:
            await client.get("/api/v1/healthz", cast_to=object)
        else:
            with pytest.raises(AuthenticationError):
                await client.get("/api/v1/healthz", cast_to=object)
    finally:
        await http_client.aclose()
    assert seen == [old, new]
    holder._cm.__enter__.return_value.service.auth_token.assert_called_once()


@pytest.mark.asyncio
async def test_unmarked_401_is_not_replayed():
    """A 401 that a new token would not fix (e.g. a disabled credential type)."""
    old = _make_jwt(time.time() + 3000)
    holder = _MockHolder(_make_jwt(time.time() + 3600))
    provider = _provider_with_token(holder.aclient, old)
    seen: list[str] = []
    http_client = _api("", {}, seen)
    try:
        client = AsyncTinker(
            base_url="https://api.example.test", _auth=provider, http_client=http_client
        )
        with pytest.raises(AuthenticationError):
            await client.get("/api/v1/healthz", cast_to=object)
    finally:
        await http_client.aclose()
    assert seen == [old]
    holder._cm.__enter__.return_value.service.auth_token.assert_not_called()


@pytest.mark.asyncio
async def test_concurrent_invalid_token_401s_share_one_exchange():
    n = 5
    old, new = _make_jwt(time.time() + 3000), _make_jwt(time.time() + 3600)
    seen: list[str] = []
    all_rejected = asyncio.Event()
    exchanges = 0

    async def exchange_after_all_rejections():
        nonlocal exchanges
        exchanges += 1
        await all_rejected.wait()
        return _MockAuthResponse(new)

    def handle(request: httpx.Request) -> httpx.Response:
        token = request.headers["X-API-Key"]
        seen.append(token)
        if token == new:
            return httpx.Response(200, json={})
        if seen.count(old) == n:
            all_rejected.set()
        return httpx.Response(401, headers=_INVALID_TOKEN, json={"detail": "Invalid JWT"})

    holder = _MockHolder(new)
    holder._cm.__enter__.return_value.service.auth_token = exchange_after_all_rejections
    provider = _provider_with_token(holder.aclient, old)
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        client = AsyncTinker(
            base_url="https://api.example.test", _auth=provider, http_client=http_client
        )
        await asyncio.gather(*(client.get("/api/v1/healthz", cast_to=object) for _ in range(n)))
    finally:
        await http_client.aclose()
    assert exchanges == 1
    assert seen.count(old) == n and seen.count(new) == n


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
# Refresh timing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lifetime", [300, 600, 3600])
def test_refresh_timing_scales_with_lifetime(lifetime: int):
    now = time.time()
    jwt = _jwt_with_claims({"iat": now, "exp": now + lifetime})
    assert 0.45 * lifetime <= _refresh_delay(jwt) <= 0.55 * lifetime


# ---------------------------------------------------------------------------
# Background refresh after failures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "stops"),
    [
        (
            AuthenticationError(
                "revoked", response=httpx.Response(401, request=_EXCHANGE_REQUEST), body=None
            ),
            True,
        ),
        (Exception("network error"), False),
    ],
)
async def test_refresh_loop_stops_before_expiry_only_when_rejected(
    monkeypatch: pytest.MonkeyPatch, error: Exception, stops: bool
):
    """A rejected credential (401) stops the loop before the token expires. Any other
    failure, such as an outage, keeps it retrying."""
    now = [1_000_000.0]
    exp = now[0] + 3600

    async def fake_sleep(secs: float) -> None:
        if now[0] > exp:
            raise asyncio.CancelledError  # stands in for close() on a loop still retrying
        now[0] += secs

    monkeypatch.setattr(_jwt_auth, "time", SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    holder = _MockHolder("unused")
    holder._cm.__enter__.return_value.service.auth_token.side_effect = error
    token = _make_jwt(exp)
    provider = _provider_with_token(holder.aclient, token)
    provider._refresh_task = asyncio.create_task(provider._refresh_loop(token))
    await provider._refresh_task

    assert (now[0] < exp) == stops
    # A stopped loop clears its task, so the next successful get_token() starts a new one.
    assert (provider._refresh_task is None) == stops


@pytest.mark.asyncio
async def test_get_token_after_close_starts_no_refresh_loop():
    """Shutdown sends the last telemetry batch after close(). If the loop had stopped,
    a refresh then must not start a new loop, because nothing would cancel it."""
    refreshed_jwt = _make_jwt(time.time() + 7200)
    holder = _MockHolder(refreshed_jwt)
    provider = _provider_with_token(holder.aclient, _make_jwt(time.time() + 30))
    await provider.close()

    assert await provider.get_token() == refreshed_jwt
    assert provider._refresh_task is None


# ---------------------------------------------------------------------------
# JwtAuthProvider.get_token on-demand refresh
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_token_returns_cached_when_fresh():
    """Cached token with comfortable runway is returned without refetching."""
    fresh_jwt = _make_jwt(time.time() + 7200)
    holder = _MockHolder("should-not-be-fetched")
    provider = _provider_with_token(holder.aclient, fresh_jwt)

    assert await provider.get_token() == fresh_jwt
    holder._cm.__enter__.return_value.service.auth_token.assert_not_called()


@pytest.mark.asyncio
async def test_get_token_refreshes_when_near_expiry():
    """If cached token has <= _REFRESH_ON_DEMAND_SECS left, fetch a new one."""
    near_expiry_jwt = _make_jwt(time.time() + 30)  # 30s left, under threshold
    refreshed_jwt = _make_jwt(time.time() + 7200)
    holder = _MockHolder(refreshed_jwt)
    provider = _provider_with_token(holder.aclient, near_expiry_jwt)

    assert await provider.get_token() == refreshed_jwt
    holder._cm.__enter__.return_value.service.auth_token.assert_called_once()
    await provider.close()


@pytest.mark.asyncio
async def test_get_token_refreshes_when_already_expired():
    """An expired cached token must trigger refresh, not be served as-is."""
    expired_jwt = _make_jwt(time.time() - 30)
    refreshed_jwt = _make_jwt(time.time() + 7200)
    holder = _MockHolder(refreshed_jwt)
    provider = _provider_with_token(holder.aclient, expired_jwt)

    assert await provider.get_token() == refreshed_jwt
    holder._cm.__enter__.return_value.service.auth_token.assert_called_once()
    await provider.close()


@pytest.mark.asyncio
async def test_get_token_refreshes_when_cached_token_is_unparseable():
    """A garbled cached token is treated as expired."""
    refreshed_jwt = _make_jwt(time.time() + 7200)
    holder = _MockHolder(refreshed_jwt)
    provider = _provider_with_token(holder.aclient, "not.a.jwt")

    assert await provider.get_token() == refreshed_jwt
    holder._cm.__enter__.return_value.service.auth_token.assert_called_once()
    await provider.close()


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

    provider = _provider_with_token(lambda: cm, near_expiry_jwt)

    tasks = [asyncio.create_task(provider.get_token()) for _ in range(5)]

    # Wait for the first task to enter the fetch, then let the others queue
    # up at the lock before releasing the in-flight fetch.
    await fetch_started.wait()
    await asyncio.sleep(0)
    fetch_release.set()
    results = await asyncio.gather(*tasks)

    assert fetch_count == 1
    assert all(r == refreshed_jwt for r in results)
    await provider.close()


@pytest.mark.asyncio
async def test_get_token_raises_when_refresh_is_rejected():
    """A 401 from the exchange means the credential itself is bad: surface it
    rather than serve the cached token until it expires."""
    holder = _MockHolder("unused")
    rejected = AuthenticationError(
        "credential revoked", response=httpx.Response(401, request=_EXCHANGE_REQUEST), body=None
    )
    holder._cm.__enter__.return_value.service.auth_token.side_effect = rejected
    provider = _provider_with_token(holder.aclient, _make_jwt(time.time() + 30))

    with pytest.raises(AuthenticationError, match="credential revoked"):
        await provider.get_token()


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
    provider = _provider_with_token(holder.aclient, near_expiry_jwt)

    with caplog.at_level("WARNING"):
        result = await provider.get_token()

    assert result == near_expiry_jwt
    assert "On-demand JWT refresh failed" in caplog.text


@pytest.mark.asyncio
async def test_get_token_raises_when_refresh_fails_and_cached_token_expired():
    """An expired cached token is never sent, even when the refresh fails."""
    holder = _MockHolder("unused", fail=True)
    provider = _provider_with_token(holder.aclient, _make_jwt(time.time() - 30))

    with pytest.raises(Exception, match="network error"):
        await provider.get_token()
    holder._cm.__enter__.return_value.service.auth_token.assert_called_once()
