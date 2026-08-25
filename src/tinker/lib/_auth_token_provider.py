"""Authentication credential management for the Tinker SDK.

Provides composable credential providers that plug into httpx's async auth flow:
- AuthTokenProvider: abstract base (httpx.Auth) — subclasses implement get_token()
- ApiKeyAuthProvider: a static API key; create_or_env() resolves the api_key arg
  or the TINKER_API_KEY env var, create_from_stored() the stored default
  credential (~/.tinker/credentials.json, see `tinker auth login`)
- CredentialCmdAuthProvider: runs a command on every call for fresh credentials
- resolve_auth_provider(): factory that picks the right provider, raising if no
  credential source is available
"""

from __future__ import annotations

import abc
import asyncio
import os
from collections.abc import AsyncGenerator

import httpx

from tinker._exceptions import TinkerError
from tinker.lib import credentials

MISSING_API_KEY_MESSAGE = (
    "The api_key client option must be set by passing api_key to the client,"
    " by setting the TINKER_API_KEY environment variable,"
    " or by storing a key with `tinker auth login`"
)


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
    """A static API key."""

    def __init__(self, api_key: str) -> None:
        if not api_key.startswith("tml-") and not api_key.startswith("eyJ"):
            raise TinkerError("The api_key must start with the 'tml-' prefix")
        self._token = api_key

    @property
    def api_key(self) -> str:
        return self._token

    @staticmethod
    def create_or_env(api_key: str | None = None) -> ApiKeyAuthProvider | None:
        """Create a provider from the passed in api_key arg, or parsing TINKER_API_KEY
        env variable if no arg is passed in. Return None if no arg is passed in and no
        TINKER_API_KEY is set."""
        resolved = api_key or os.environ.get("TINKER_API_KEY")
        return ApiKeyAuthProvider(resolved) if resolved else None

    @staticmethod
    def create_from_stored(
        path: str | os.PathLike[str] | None = None,
    ) -> ApiKeyAuthProvider | None:
        """Provider for the stored default credential (`tinker auth login`),
        or None if no default key is stored."""
        store = credentials.JsonCredentialStore(
            path if path is not None else credentials.default_credentials_path()
        )
        record = store.get_default_key()
        return ApiKeyAuthProvider(record.key) if record is not None else None

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
    - enforce_cmd=False: tries api_key first, falls back to TINKER_CREDENTIAL_CMD,
      then to the stored default credential (`tinker auth login`)

    Raises TinkerError if no credential source is available.
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
        provider: AuthTokenProvider | None = ApiKeyAuthProvider.create_or_env(api_key)
    except TinkerError:
        # An explicitly-set key that fails prefix validation falls back to the
        # credential command; without one, surface the validation error rather
        # than silently using the stored default.
        if credential_cmd:
            return CredentialCmdAuthProvider(credential_cmd)
        raise
    if provider is None and credential_cmd:
        provider = CredentialCmdAuthProvider(credential_cmd)
    if provider is None:
        # The stored default must not shadow TINKER_CREDENTIAL_CMD, so it is
        # only consulted when no credential command is configured.
        provider = ApiKeyAuthProvider.create_from_stored()
    if provider is None:
        raise TinkerError(MISSING_API_KEY_MESSAGE)
    return provider
