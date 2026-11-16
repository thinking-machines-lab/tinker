"""JWT authentication for Tinker SDK.

Internal to the SDK; not part of the public API.

When the server sets pjwt_auth_enabled, the SDK exchanges the caller's
credential for a short-lived JWT minted by the Tinker server.  The JWT is
cached and refreshed in the background before it expires.  As a safety
net, get_token() also refreshes on demand if the cached token is near or
past expiry — so a delayed/failed background refresh cannot leave callers
sending a stale JWT.  If the server rejects a JWT before it expires (a 401
marked invalid_token), the auth flow replaces the JWT and replays the request
once.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
import time
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import AbstractContextManager
from typing import Any

import httpx

from tinker._exceptions import AuthenticationError, _AuthFlowError
from tinker.lib._auth_token_provider import AuthTokenProvider
from tinker.lib.sdk_method import context_without_sdk_method

logger = logging.getLogger(__name__)

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


def _lifetime(jwt: str) -> float:
    """The exp - iat of a JWT, in seconds."""
    try:
        claims = jwt_claims(jwt)
        lifetime = float(claims["exp"]) - float(claims["iat"])
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"Failed to parse JWT lifetime: {e}") from e
    if lifetime <= 0:
        raise ValueError(f"JWT lifetime is not positive: {lifetime}")
    return lifetime


def _refresh_delay(jwt: str) -> float:
    """Seconds after receipt to refresh a JWT: half its lifetime with +/-10% jitter.

    Both claims come from the server, so local clock skew cannot shorten the delay.
    """
    try:
        return _lifetime(jwt) * random.uniform(0.45, 0.55)
    except ValueError:
        logger.debug("Failed to parse JWT lifetime, retrying in %ds", _RETRY_DELAY_SECS)
        return _RETRY_DELAY_SECS


def _is_invalid_token(response: httpx.Response) -> bool:
    """Whether the server rejected the request's token itself (RFC 6750)."""
    return response.status_code == 401 and 'error="invalid_token"' in response.headers.get(
        "WWW-Authenticate", ""
    )


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
        # Started on the event loop, cancelled by close().
        self._background_tasks: list[asyncio.Task[Any]] = []
        self._refresh_loop_running = False
        self._closed = False

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        try:
            token = await self.get_token()
        except Exception as e:
            raise _AuthFlowError(e) from e
        request.headers["X-API-Key"] = token
        response = yield request
        if not _is_invalid_token(response):
            return
        # Release the connection before the exchange, which can take a while.
        await response.aread()
        try:
            token = await self._replace(token)
        except Exception as e:
            raise _AuthFlowError(e) from e
        request.headers["X-API-Key"] = token
        yield request

    async def get_token(self) -> str:
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
            except AuthenticationError:
                # The server rejected the credential itself, so the cached token
                # cannot be renewed: report why now.
                raise
            except Exception as e:
                if _seconds_until_expiry(self._token) <= 0:
                    # No live token to fall back to: the request cannot be sent.
                    raise
                # If the refresh fails, fall back to the cached token until it
                # expires; if the server rejects it, the request surfaces the error.
                logger.warning("On-demand JWT refresh failed: %s", e)
                return self._token
            if not self._refresh_loop_running and not self._closed:
                self._refresh_loop_running = True
                await self._run_in_background(self._refresh_loop(token))
            return token

    async def _run_in_background(self, coroutine: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        """A task for `coroutine` on the running loop, cancelled by `close()`."""
        task = asyncio.create_task(coroutine, context=context_without_sdk_method())
        self._background_tasks.append(task)
        return task

    async def close(self) -> None:
        # get_token() still serves tokens after close, because shutdown sends the
        # last telemetry batch. It does not start a new refresh loop.
        self._closed = True
        for task in self._background_tasks:
            task.cancel()
        await asyncio.gather(*self._background_tasks, return_exceptions=True)

    async def _replace(self, rejected: str) -> str:
        """A token other than `rejected`. Exchanges for one only if no other request
        has already, so concurrent rejections of one token cost one exchange."""
        if self._token != rejected:
            return self._token
        async with self._refresh_lock:
            if self._token != rejected:
                return self._token
            return await self._fetch()

    async def _fetch(self) -> str:
        """Exchange the current credential for a JWT via /api/v1/auth/token."""
        with self._aclient_fn() as client:
            response = await client.service.auth_token()
        jwt: str = response.jwt
        self._token = jwt
        return jwt

    async def _refresh_loop(self, token: str) -> None:
        delay = _refresh_delay(token)
        failing = False
        while True:
            try:
                await asyncio.sleep(delay)
                # Coordinate with on-demand refreshes in get_token() so we
                # don't fire two concurrent /auth/token requests.
                async with self._refresh_lock:
                    token = await self._fetch()
                delay = _refresh_delay(token)
                failing = False
            except asyncio.CancelledError:
                return
            except Exception as e:
                if isinstance(e, AuthenticationError) and (
                    _seconds_until_expiry(self._token) - _RETRY_DELAY_SECS
                    <= _REFRESH_ON_DEMAND_SECS
                ):
                    # Retry a rejected credential only until get_token() takes over near
                    # expiry; its next successful refresh starts a new loop.
                    logger.warning("JWT refresh rejected, stopping background refresh: %s", e)
                    self._refresh_loop_running = False
                    return
                log = logger.debug if failing else logger.warning
                log("JWT refresh failed, retrying in %ds: %s", _RETRY_DELAY_SECS, e)
                delay = _RETRY_DELAY_SECS
                failing = True
