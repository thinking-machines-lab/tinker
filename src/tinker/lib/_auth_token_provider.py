"""Authentication credential management for the Tinker SDK.

Provides composable credential providers that plug into httpx's async auth flow:
- AuthTokenProvider: abstract base (httpx.Auth) — subclasses implement get_token()
- ApiKeyAuthProvider: resolves from api_key arg or TINKER_API_KEY env var
- CredentialCmdAuthProvider: runs a command on every call for fresh credentials
- resolve_auth_provider(): factory that picks the right provider
"""

from __future__ import annotations

import abc
import asyncio
import os
from collections.abc import AsyncGenerator

import httpx

from tinker._exceptions import TinkerError


class AuthTokenProvider(httpx.Auth):
    """Abstract base auth provider. Subclasses implement get_token()."""

    @abc.abstractmethod
    async def get_token(self) -> str | None: ...

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        token = await self.get_token()
        if token:
            request.headers["X-API-Key"] = token
        yield request


class ApiKeyAuthProvider(AuthTokenProvider):
    """Resolves api_key from constructor arg or TINKER_API_KEY env var."""

    def __init__(self, api_key: str | None = None) -> None:
        resolved = api_key or os.environ.get("TINKER_API_KEY")
        if not resolved:
            raise TinkerError(
                "The api_key client option must be set either by passing api_key to the client"
                " or by setting the TINKER_API_KEY environment variable"
            )
        if not resolved.startswith("tml-") and not resolved.startswith("eyJ"):
            raise TinkerError("The api_key must start with the 'tml-' prefix")
        self._token = resolved

    async def get_token(self) -> str | None:
        return self._token


class CredentialCmdAuthProvider(AuthTokenProvider):
    """Runs TINKER_CREDENTIAL_CMD on every get_token() call.

    Always produces a fresh credential (e.g. short-lived bearer tokens).
    Uses async subprocess to avoid blocking the event loop.
    """

    def __init__(self, cmd: str) -> None:
        if not cmd:
            raise TinkerError(
                "Your organization requires dynamic credentials — set TINKER_CREDENTIAL_CMD"
                " to a command that prints a valid credential."
            )
        self._cmd = cmd

    async def get_token(self) -> str | None:
        proc = await asyncio.create_subprocess_shell(
            self._cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        credential = stdout.decode().strip()
        if not credential:
            raise TinkerError("TINKER_CREDENTIAL_CMD returned an empty credential.")
        return credential


def resolve_auth_provider(api_key: str | None, enforce_cmd: bool) -> AuthTokenProvider:
    """Construct the appropriate auth provider based on available credentials.

    - enforce_cmd=True: uses TINKER_CREDENTIAL_CMD, unless the api_key is
      already a JWT (dynamic credential) — in which case it's used directly.
    - enforce_cmd=False: tries api_key first, falls back to TINKER_CREDENTIAL_CMD
    """
    credential_cmd = os.environ.get("TINKER_CREDENTIAL_CMD", "")

    # A JWT passed as api_key is already a dynamic credential — use it
    # directly even when credential_cmd is enforced.
    resolved = api_key or os.environ.get("TINKER_API_KEY", "")
    if resolved and resolved.startswith("eyJ"):
        return ApiKeyAuthProvider(api_key=resolved)

    if enforce_cmd:
        return CredentialCmdAuthProvider(credential_cmd)

    try:
        return ApiKeyAuthProvider(api_key=api_key)
    except TinkerError:
        if credential_cmd:
            return CredentialCmdAuthProvider(credential_cmd)
        raise
