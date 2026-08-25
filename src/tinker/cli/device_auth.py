"""The OAuth 2.0 device authorization grant (RFC 8628) against WorkOS AuthKit.

`tinker auth login` uses this to get a WorkOS access token without a
browser redirect back to the CLI: request a device code, show the user a
short user code plus a verification URL, and poll WorkOS until they
approve the login.

The module owns the flow, not the wire: WorkOS is only reached through the
WorkOsClient dependency (`workos_api`). Callers display the returned
DeviceAuthorization however they like and get back an access token.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlsplit

from .workos_api import DeviceCodeGrantError, WorkOsApiError, WorkOsClient

# Used when WorkOS omits `interval` from the authorization response.
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
# RFC 8628 section 3.5: a `slow_down` error means back off by 5 seconds.
SLOW_DOWN_SECONDS = 5.0

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
# Errors that end the flow. Anything else unrecognized is also terminal, but
# only these have a message worth showing instead of WorkOS's own.
_TERMINAL_ERRORS = {
    "access_denied": "The login request was denied.",
    "expired_token": "The login code expired before it was approved.",
}


class DeviceAuthError(Exception):
    """The device authorization flow failed, expired, or was denied."""


@dataclass(frozen=True)
class DeviceAuthorization:
    """A pending device authorization: what to show the user, and how to poll.

    `device_code` is the secret the CLI polls with and must never be shown;
    `user_code` is the short code the user confirms in the browser.
    """

    device_code: str
    user_code: str
    verification_uri: str
    # `verification_uri` with the user code prefilled, so a browser opened on
    # it needs no typing. Falls back to `verification_uri` when WorkOS omits it.
    verification_uri_complete: str
    expires_in_seconds: float
    interval_seconds: float


class DeviceAuthClient:
    """Runs the device authorization grant over a WorkOsClient.

    `sleep` and `monotonic` are injectable so polling can be tested without
    real time passing.
    """

    def __init__(
        self,
        workos_client: WorkOsClient,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._workos = workos_client
        self._sleep = sleep
        self._monotonic = monotonic

    def authorize(self) -> DeviceAuthorization:
        """Ask WorkOS for a device code and the URL the user should visit."""
        try:
            body = self._workos.request_device_authorization()
        except WorkOsApiError as e:
            raise DeviceAuthError(str(e)) from e

        verification_uri = _checked_verification_url(body.verification_uri)
        complete = (body.verification_uri_complete or "").strip()
        return DeviceAuthorization(
            device_code=body.device_code,
            user_code=body.user_code,
            verification_uri=verification_uri,
            verification_uri_complete=(
                _checked_verification_url(complete) if complete else verification_uri
            ),
            expires_in_seconds=body.expires_in,
            interval_seconds=(
                body.interval
                if body.interval and body.interval > 0
                else DEFAULT_POLL_INTERVAL_SECONDS
            ),
        )

    def poll_for_access_token(self, authorization: DeviceAuthorization) -> str:
        """Poll until the user approves the login, returning an access token.

        Blocks for up to the authorization's lifetime, raising DeviceAuthError
        if it expires, is denied, or WorkOS reports any other error.
        """
        deadline = self._monotonic() + authorization.expires_in_seconds
        interval = authorization.interval_seconds
        while True:
            try:
                result = self._workos.authenticate_with_device_code(authorization.device_code)
            except WorkOsApiError as e:
                raise DeviceAuthError(str(e)) from e
            if not isinstance(result, DeviceCodeGrantError):
                return result

            if result.error == "slow_down":
                interval += SLOW_DOWN_SECONDS
            elif result.error != "authorization_pending":
                raise DeviceAuthError(
                    _TERMINAL_ERRORS.get(result.error)
                    or result.description
                    or result.error
                    or "WorkOS rejected the login"
                )

            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise DeviceAuthError("Timed out waiting for the login to be approved.")
            self._sleep(min(interval, remaining))


def _checked_verification_url(url: str) -> str:
    """`url`, if it is safe to open in a browser.

    A verification URL comes from the network, so refuse anything that isn't
    HTTPS (or plain HTTP on loopback, which local WorkOS mocks use).
    """
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "https" or (parsed.scheme == "http" and host in _LOOPBACK_HOSTS):
        return url
    raise DeviceAuthError("WorkOS returned an insecure verification URL")
