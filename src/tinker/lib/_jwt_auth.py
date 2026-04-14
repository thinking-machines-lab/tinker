"""JWT authentication for Tinker SDK.

Internal to the SDK; not part of the public API.

When the server sets pjwt_auth_enabled, the SDK exchanges the caller's
credential for a short-lived JWT minted by the Tinker server.  The JWT is
cached and refreshed in the background before it expires, so callers always
send a valid token without any per-request overhead.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections.abc import Callable
from contextlib import AbstractContextManager

from tinker.lib._auth_token_provider import AuthTokenProvider

logger = logging.getLogger(__name__)

_REFRESH_BEFORE_EXPIRY_SECS = 300  # refresh 5 min before expiry
_RETRY_DELAY_SECS = 60


def _jwt_expiry(jwt: str) -> float:
    """Return the exp claim of a JWT as a Unix timestamp."""
    try:
        payload = jwt.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return float(json.loads(base64.urlsafe_b64decode(payload))["exp"])
    except Exception as e:
        raise ValueError(f"Failed to parse JWT expiry: {e}") from e


class JwtAuthProvider(AuthTokenProvider):
    """AuthTokenProvider that exchanges a credential for a short-lived JWT.

    After init(), get_token() returns the current JWT.  A background task
    refreshes the JWT before it expires.
    """

    def __init__(
        self,
        aclient_fn: Callable[[], AbstractContextManager],
        seed_token: str | None = None,
    ) -> None:
        self._token = seed_token or ""
        self._aclient_fn = aclient_fn

    async def get_token(self) -> str | None:
        return self._token

    async def init(self) -> None:
        """Fetch a JWT (unless seeded) then start the background refresh loop.

        When seed_token was provided, skips the initial fetch and starts
        refreshing from the seed — useful for shadow holders that already
        have a valid JWT from the primary holder.
        """
        token = self._token if self._token else await self._fetch()
        self._refresh_task = asyncio.create_task(self._refresh_loop(token))

    async def _fetch(self) -> str:
        """Exchange the current credential for a JWT via /api/v1/auth/token."""
        with self._aclient_fn() as client:
            response = await client.service.auth_token()
        self._token = response.jwt
        return response.jwt

    async def _refresh_loop(self, token: str) -> None:
        while True:
            try:
                delay = max(
                    _RETRY_DELAY_SECS,
                    _jwt_expiry(token) - time.time() - _REFRESH_BEFORE_EXPIRY_SECS,
                )
            except ValueError:
                logger.debug("Failed to parse JWT expiry, retrying in %ds", _RETRY_DELAY_SECS)
                delay = _RETRY_DELAY_SECS
            try:
                await asyncio.sleep(delay)
                token = await self._fetch()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.debug("JWT refresh failed, retrying in %ds: %s", _RETRY_DELAY_SECS, e)
