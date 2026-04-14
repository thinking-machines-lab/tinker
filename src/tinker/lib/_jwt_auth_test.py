"""Tests for JWT authentication helpers."""

from __future__ import annotations

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
