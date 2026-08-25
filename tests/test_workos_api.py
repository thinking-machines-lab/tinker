"""Tests for the typed WorkOS API client used by `tinker auth login`."""

from __future__ import annotations

from typing import Callable

import httpx
import pytest

from tinker.cli.workos_api import (
    DeviceAuthorizationResponse,
    DeviceCodeGrantError,
    WorkOsApiError,
    WorkOsClient,
)

API_URL = "https://workos.test"

AUTHORIZATION_BODY = {
    "device_code": "device-secret",
    "user_code": "WXYZ-1234",
    "verification_uri": "https://auth.test/device",
    "verification_uri_complete": "https://auth.test/device?user_code=WXYZ-1234",
    "expires_in": 300,
    "interval": 5,
}


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> WorkOsClient:
    return WorkOsClient(
        "client_123",
        httpx.Client(transport=httpx.MockTransport(handler)),
        api_url=API_URL,
    )


class TestRequestDeviceAuthorization:
    # Catches the request drifting off the WorkOS wire protocol (wrong endpoint,
    # wrong form encoding, missing client_id) or a response field being dropped
    # or mismapped on its way into the typed response.
    def test_requests_a_device_code_and_parses_it(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=AUTHORIZATION_BODY)

        body = _client(handler).request_device_authorization()

        assert requests[0].url == httpx.URL(f"{API_URL}/user_management/authorize/device")
        assert requests[0].content == b"client_id=client_123"
        assert body == DeviceAuthorizationResponse.model_validate(AUTHORIZATION_BODY)

    # Catches WorkOS's own explanation (e.g. a misconfigured client id) being
    # swallowed and replaced with an unactionable generic error.
    def test_surfaces_workos_error(self) -> None:
        response = httpx.Response(400, json={"error_description": "Unknown client"})
        with pytest.raises(WorkOsApiError, match="Unknown client"):
            _client(lambda _: response).request_device_authorization()

    # Catches a malformed 200 response escaping as a raw pydantic
    # ValidationError instead of the WorkOsApiError callers handle. A body
    # that parses but is unusable (blank codes, zero lifetime) is just as
    # invalid as one that doesn't.
    @pytest.mark.parametrize(
        "body",
        [
            {"device_code": "x"},
            {**AUTHORIZATION_BODY, "device_code": "  "},
            {**AUTHORIZATION_BODY, "user_code": ""},
            {**AUTHORIZATION_BODY, "expires_in": 0},
        ],
    )
    def test_invalid_body_raises(self, body: dict[str, object]) -> None:
        with pytest.raises(WorkOsApiError, match="invalid device authorization"):
            _client(lambda _: httpx.Response(200, json=body)).request_device_authorization()

    # Catches network failures escaping as raw httpx exceptions instead of the
    # WorkOsApiError callers handle.
    def test_connection_failure_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route", request=request)

        with pytest.raises(WorkOsApiError, match="Could not reach WorkOS"):
            _client(handler).request_device_authorization()


class TestAuthenticateWithDeviceCode:
    # Catches the token request drifting off the RFC 8628 wire format, or the
    # access token being lost on the way out.
    def test_exchanges_the_device_code_for_a_token(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"access_token": "wos-token"})

        token = _client(handler).authenticate_with_device_code("device-secret")

        assert token == "wos-token"
        assert requests[0].url == httpx.URL(f"{API_URL}/user_management/authenticate")
        assert dict(httpx.QueryParams(requests[0].content.decode())) == {
            "client_id": "client_123",
            "device_code": "device-secret",
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }

    # Catches OAuth errors being raised instead of returned: the caller must
    # see `authorization_pending` (and its explanation, when there is one) to
    # keep the polling loop alive.
    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            (
                {"error": "authorization_pending"},
                DeviceCodeGrantError(error="authorization_pending", description=""),
            ),
            (
                {"error": "invalid_grant", "message": "bad grant"},
                DeviceCodeGrantError(error="invalid_grant", description="bad grant"),
            ),
            ("not json", DeviceCodeGrantError(error="", description="")),
        ],
    )
    def test_returns_the_oauth_error(self, body: object, expected: DeviceCodeGrantError) -> None:
        response = (
            httpx.Response(400, json=body)
            if isinstance(body, dict)
            else httpx.Response(400, text=str(body))
        )
        assert _client(lambda _: response).authenticate_with_device_code("dc") == expected

    # Catches a malformed success response escaping as a raw pydantic
    # ValidationError instead of the WorkOsApiError callers handle.
    def test_invalid_success_body_raises(self) -> None:
        with pytest.raises(WorkOsApiError, match="invalid access token"):
            _client(lambda _: httpx.Response(200, json={"nope": 1})).authenticate_with_device_code(
                "dc"
            )

    # Catches network failures escaping as raw httpx exceptions instead of the
    # WorkOsApiError callers handle.
    def test_connection_failure_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route", request=request)

        with pytest.raises(WorkOsApiError, match="Could not reach WorkOS"):
            _client(handler).authenticate_with_device_code("dc")
