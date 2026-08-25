"""Typed client for the WorkOS AuthKit API calls `tinker auth login` makes.

WorkOsClient wraps the two `user_management` endpoints of the OAuth 2.0
device authorization grant (RFC 8628): requesting a device code, and
exchanging it for an access token. It owns the wire protocol — endpoint
paths, form encoding, response validation, and error payload parsing — and
nothing else: what to do with the responses (polling, backoff, deadlines)
lives in `device_auth`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import httpx
from pydantic import BaseModel, ValidationError

DEFAULT_WORKOS_API_URL = "https://api.workos.com"
DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"


class WorkOsApiError(Exception):
    """A WorkOS API call failed outright: network error, rejected request,
    or a response that doesn't fit the wire protocol."""


class DeviceAuthorizationResponse(BaseModel):
    """The RFC 8628 device authorization response, as WorkOS sends it."""

    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: Optional[str] = None
    expires_in: float
    interval: Optional[float] = None


@dataclass(frozen=True)
class DeviceCodeGrantError:
    """An OAuth error from the token endpoint (RFC 8628 section 3.5).

    Returned rather than raised: errors like `authorization_pending` and
    `slow_down` are how the endpoint says "keep polling", so interpreting
    the code is the caller's job.
    """

    # The OAuth error code, empty if WorkOS sent an unparseable body.
    error: str
    # WorkOS's human-readable explanation, empty if it did not send one.
    description: str


class _AccessTokenResponse(BaseModel):
    access_token: str


class _ErrorResponse(BaseModel):
    error: Optional[str] = None
    error_description: Optional[str] = None
    message: Optional[str] = None


class WorkOsClient:
    """Client for one WorkOS deployment, on behalf of one OAuth client id."""

    def __init__(
        self,
        client_id: str,
        http_client: httpx.Client,
        *,
        api_url: str = DEFAULT_WORKOS_API_URL,
    ) -> None:
        self._client_id = client_id
        self._http = http_client
        self._api_url = api_url.rstrip("/")

    def request_device_authorization(self) -> DeviceAuthorizationResponse:
        """Ask WorkOS to start a device authorization grant."""
        response = self._post("/user_management/authorize/device", {"client_id": self._client_id})
        if not response.is_success:
            raise WorkOsApiError(
                _error_message(response, "WorkOS rejected the device authorization request")
            )
        try:
            body = DeviceAuthorizationResponse.model_validate(response.json())
        except (ValueError, ValidationError) as e:
            raise WorkOsApiError("WorkOS returned an invalid device authorization") from e
        if not body.device_code.strip() or not body.user_code.strip() or body.expires_in <= 0:
            raise WorkOsApiError("WorkOS returned an invalid device authorization")
        return body

    def authenticate_with_device_code(self, device_code: str) -> Union[str, DeviceCodeGrantError]:
        """Try once to exchange a device code for an access token.

        Returns the access token if the user has approved the login, or the
        OAuth error WorkOS sent instead (`authorization_pending` until then).
        Raises WorkOsApiError for failures outside the protocol: network
        errors and malformed success responses.
        """
        response = self._post(
            "/user_management/authenticate",
            {
                "client_id": self._client_id,
                "device_code": device_code,
                "grant_type": DEVICE_GRANT_TYPE,
            },
        )
        if not response.is_success:
            body = _parse_error(response)
            return DeviceCodeGrantError(
                error=body.error or "",
                description=body.error_description or body.message or "",
            )
        try:
            return _AccessTokenResponse.model_validate(response.json()).access_token
        except (ValueError, ValidationError) as e:
            raise WorkOsApiError("WorkOS returned an invalid access token") from e

    def _post(self, path: str, data: dict[str, str]) -> httpx.Response:
        try:
            return self._http.post(f"{self._api_url}{path}", data=data)
        except httpx.HTTPError as e:
            raise WorkOsApiError(f"Could not reach WorkOS: {e}") from e


def _parse_error(response: httpx.Response) -> _ErrorResponse:
    """The error payload of a failed WorkOS response, empty if unparseable."""
    try:
        return _ErrorResponse.model_validate(response.json())
    except (ValueError, ValidationError):
        return _ErrorResponse()


def _error_message(response: httpx.Response, fallback: str) -> str:
    body = _parse_error(response)
    return body.error_description or body.message or body.error or fallback
