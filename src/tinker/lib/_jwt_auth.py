"""JWT authentication for Tinker SDK.

Internal to the SDK; not part of the public API.

When the server sets pjwt_auth_enabled, the SDK exchanges the caller's
credential for a short-lived JWT minted by the Tinker server.  The JWT is
cached and refreshed in the background before it expires.  As a safety
net, get_token() also refreshes on demand if the cached token is near or
past expiry — so a delayed/failed background refresh cannot leave callers
sending a stale JWT (which the server rejects with 401 Invalid JWT, and
401 is not retried by the request layer).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import time
from collections.abc import Callable
from contextlib import AbstractContextManager

from tinker.lib._auth_token_provider import AuthTokenProvider

logger = logging.getLogger(__name__)

_REFRESH_BEFORE_EXPIRY_SECS = 300  # background loop refreshes 5 min before expiry
_REFRESH_ON_DEMAND_SECS = 60  # get_token() refreshes if <= this many seconds left
_RETRY_DELAY_SECS = 60  # backoff after a failed refresh


def jwt_claims(jwt: str) -> dict:
    """Decode a JWT's payload claims without verifying the signature.

    The SDK only ever decodes JWTs it received from the Tinker server over
    TLS, so signature verification is not needed here.
    """
    try:
        payload = jwt.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception as e:
        raise ValueError(f"Failed to parse JWT claims: {e}") from e


def _jwt_expiry(jwt: str) -> float:
    """Return the exp claim of a JWT as a Unix timestamp."""
    try:
        return float(jwt_claims(jwt)["exp"])
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"Failed to parse JWT expiry: {e}") from e


def _seconds_until_expiry(jwt: str) -> float:
    """Seconds until the JWT expires; 0 if expiry can't be parsed."""
    try:
        return _jwt_expiry(jwt) - time.time()
    except ValueError:
        return 0.0


class JwtAuthProvider(AuthTokenProvider):
    """AuthTokenProvider that exchanges a credential for a short-lived JWT.

    The first get_token() fetches the JWT and starts a background task that
    proactively refreshes it before it expires.  get_token() also refreshes on
    demand if the cached token is near or past expiry, so a stuck or delayed
    background refresh cannot leak a stale JWT into a request.
    """

    def __init__(self, aclient_fn: Callable[[], AbstractContextManager]) -> None:
        self._token: str = ""
        self._aclient_fn = aclient_fn
        self._refresh_lock = asyncio.Lock()
        self._refresh_task: asyncio.Task[None] | None = None

    async def get_token(self) -> str | None:
        # Fast path: cached token has comfortable runway.
        if self._token and _seconds_until_expiry(self._token) > _REFRESH_ON_DEMAND_SECS:
            return self._token

        async with self._refresh_lock:
            # Re-check after acquiring the lock — another caller may have
            # just refreshed the token while we were waiting.
            if self._token and _seconds_until_expiry(self._token) > _REFRESH_ON_DEMAND_SECS:
                return self._token
            try:
                token = await self._fetch()
            except Exception as e:
                if not self._token:
                    # No token to fall back to: the request cannot be sent.
                    raise
                # If the refresh fails, fall back to whatever we have.
                # The background loop keeps trying; if the server is
                # genuinely down the request will surface the error.
                logger.warning("On-demand JWT refresh failed: %s", e)
                return self._token
            if self._refresh_task is None:
                self._refresh_task = asyncio.create_task(self._refresh_loop(token))
            return token

    async def close(self) -> None:
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._refresh_task

    async def _fetch(self) -> str:
        """Exchange the current credential for a JWT via /api/v1/auth/token."""
        with self._aclient_fn() as client:
            response = await client.service.auth_token()
        jwt: str = response.jwt
        self._token = jwt
        return jwt

    async def _refresh_loop(self, token: str) -> None:
        while True:
            try:
                delay = max(
                    0.0,
                    _jwt_expiry(token) - time.time() - _REFRESH_BEFORE_EXPIRY_SECS,
                )
            except ValueError:
                logger.debug("Failed to parse JWT expiry, retrying in %ds", _RETRY_DELAY_SECS)
                delay = float(_RETRY_DELAY_SECS)
            try:
                await asyncio.sleep(delay)
                # Coordinate with on-demand refreshes in get_token() so we
                # don't fire two concurrent /auth/token requests.
                async with self._refresh_lock:
                    token = await self._fetch()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.debug("JWT refresh failed, retrying in %ds: %s", _RETRY_DELAY_SECS, e)
                # Explicit backoff: without the old max(60, ...) floor on
                # `delay`, a stale token would otherwise compute delay=0
                # next iteration and tight-loop on persistent failures.
                try:
                    await asyncio.sleep(_RETRY_DELAY_SECS)
                except asyncio.CancelledError:
                    return
