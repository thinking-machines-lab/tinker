"""Tests for the device authorization flow used by `tinker auth login`.

The WorkOS wire protocol is tested in test_workos_api.py; these tests drive
DeviceAuthClient's flow logic through a scripted WorkOsClient double.
"""

from __future__ import annotations

from typing import Optional, Union

import httpx
import pytest

from tinker.cli.device_auth import (
    DEFAULT_POLL_INTERVAL_SECONDS,
    SLOW_DOWN_SECONDS,
    DeviceAuthClient,
    DeviceAuthError,
    DeviceAuthorization,
)
from tinker.cli.workos_api import (
    DeviceAuthorizationResponse,
    DeviceCodeGrantError,
    WorkOsApiError,
    WorkOsClient,
)

AUTHORIZATION_RESPONSE = DeviceAuthorizationResponse(
    device_code="device-secret",
    user_code="WXYZ-1234",
    verification_uri="https://auth.test/device",
    verification_uri_complete="https://auth.test/device?user_code=WXYZ-1234",
    expires_in=300,
    interval=5,
)

PENDING = DeviceCodeGrantError(error="authorization_pending", description="")

PollResult = Union[str, DeviceCodeGrantError, WorkOsApiError]


class FakeClock:
    """A monotonic clock that only advances when something sleeps on it."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def monotonic(self) -> float:
        return self.now


class FakeWorkOs(WorkOsClient):
    """A WorkOsClient that replays canned responses instead of calling out."""

    def __init__(
        self,
        authorization: Union[DeviceAuthorizationResponse, WorkOsApiError] = AUTHORIZATION_RESPONSE,
        poll_results: tuple[PollResult, ...] = (),
    ) -> None:
        super().__init__("client_123", httpx.Client(transport=httpx.MockTransport(_no_requests)))
        self._authorization = authorization
        self._poll_results = list(poll_results)
        self.polled_device_codes: list[str] = []

    def request_device_authorization(self) -> DeviceAuthorizationResponse:
        if isinstance(self._authorization, WorkOsApiError):
            raise self._authorization
        return self._authorization

    def authenticate_with_device_code(self, device_code: str) -> Union[str, DeviceCodeGrantError]:
        self.polled_device_codes.append(device_code)
        result = self._poll_results.pop(0)
        if isinstance(result, WorkOsApiError):
            raise result
        return result


def _no_requests(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"the fake client must not make HTTP requests: {request.url}")


def _client(workos: WorkOsClient, clock: Optional[FakeClock] = None) -> DeviceAuthClient:
    clock = clock or FakeClock()
    return DeviceAuthClient(workos, sleep=clock.sleep, monotonic=clock.monotonic)


def _authorization(**overrides: object) -> DeviceAuthorization:
    fields: dict[str, object] = {
        "device_code": "device-secret",
        "user_code": "WXYZ-1234",
        "verification_uri": "https://auth.test/device",
        "verification_uri_complete": "https://auth.test/device?user_code=WXYZ-1234",
        "expires_in_seconds": 300.0,
        "interval_seconds": 5.0,
    }
    fields.update(overrides)
    return DeviceAuthorization(**fields)  # type: ignore[arg-type]


class TestAuthorize:
    # Catches a WorkOS response field being dropped or mismapped on its way
    # into DeviceAuthorization (e.g. expires_in landing in the interval).
    def test_builds_the_authorization(self) -> None:
        assert _client(FakeWorkOs()).authorize() == _authorization()

    # Catches a crash or a nonsense value (interval 0, empty URL to open) when
    # WorkOS legally omits the optional fields of the RFC 8628 response.
    def test_defaults_missing_optional_fields(self) -> None:
        response = AUTHORIZATION_RESPONSE.model_copy(
            update={"interval": None, "verification_uri_complete": ""}
        )

        authorization = _client(FakeWorkOs(authorization=response)).authorize()

        assert authorization.interval_seconds == DEFAULT_POLL_INTERVAL_SECONDS
        assert authorization.verification_uri_complete == "https://auth.test/device"

    # Catches the security check being lost: a compromised or spoofed server
    # must not be able to make the CLI open a plain-HTTP login page.
    def test_rejects_insecure_verification_url(self) -> None:
        response = AUTHORIZATION_RESPONSE.model_copy(
            update={"verification_uri": "http://evil.test/device"}
        )
        with pytest.raises(DeviceAuthError, match="insecure verification URL"):
            _client(FakeWorkOs(authorization=response)).authorize()

    # Catches the HTTPS check being over-tightened so that local WorkOS mocks
    # (plain HTTP on loopback) can no longer be used for development.
    def test_allows_loopback_verification_url(self) -> None:
        response = AUTHORIZATION_RESPONSE.model_copy(
            update={
                "verification_uri": "http://localhost:9000/device",
                "verification_uri_complete": "http://localhost:9000/device?user_code=WXYZ-1234",
            }
        )
        authorization = _client(FakeWorkOs(authorization=response)).authorize()
        assert authorization.verification_uri == "http://localhost:9000/device"

    # Catches WorkOS API failures escaping as raw WorkOsApiError instead of
    # the DeviceAuthError callers handle — with the reason kept.
    def test_workos_failures_become_device_auth_errors(self) -> None:
        workos = FakeWorkOs(authorization=WorkOsApiError("Could not reach WorkOS: no route"))
        with pytest.raises(DeviceAuthError, match="Could not reach WorkOS"):
            _client(workos).authorize()


class TestPollForAccessToken:
    # Catches `authorization_pending` being treated as fatal (login would
    # never complete), polling faster than the server-requested interval, or
    # polling with anything but the secret device code.
    def test_polls_until_authorized(self) -> None:
        clock = FakeClock()
        workos = FakeWorkOs(poll_results=(PENDING, PENDING, "wos-token"))

        token = _client(workos, clock).poll_for_access_token(_authorization())

        assert token == "wos-token"
        assert clock.slept == [5.0, 5.0]
        assert workos.polled_device_codes == ["device-secret"] * 3

    # Catches `slow_down` being treated as fatal or ignored: RFC 8628 requires
    # backing off by 5 seconds, and the backoff must stick for later polls.
    def test_slow_down_backs_off(self) -> None:
        clock = FakeClock()
        workos = FakeWorkOs(
            poll_results=(
                DeviceCodeGrantError(error="slow_down", description=""),
                PENDING,
                "wos-token",
            )
        )

        assert _client(workos, clock).poll_for_access_token(_authorization()) == "wos-token"
        assert clock.slept == [5.0 + SLOW_DOWN_SECONDS, 5.0 + SLOW_DOWN_SECONDS]

    # Catches a denied or expired login being retried anyway (`slept == []`
    # proves polling stopped) or reported without a human-readable reason.
    @pytest.mark.parametrize(
        ("error", "message"),
        [("access_denied", "was denied"), ("expired_token", "expired")],
    )
    def test_terminal_errors_stop_polling(self, error: str, message: str) -> None:
        clock = FakeClock()
        workos = FakeWorkOs(poll_results=(DeviceCodeGrantError(error=error, description=""),))

        with pytest.raises(DeviceAuthError, match=message):
            _client(workos, clock).poll_for_access_token(_authorization())
        assert clock.slept == []

    # Catches errors outside the known set being mistaken for "keep polling",
    # which would spin until the code expires instead of failing immediately —
    # and WorkOS's own explanation being swallowed on the way out.
    def test_unknown_error_stops_polling(self) -> None:
        workos = FakeWorkOs(
            poll_results=(DeviceCodeGrantError(error="invalid_grant", description="bad grant"),)
        )
        with pytest.raises(DeviceAuthError, match="bad grant"):
            _client(workos).poll_for_access_token(_authorization())

    # Catches the deadline being ignored: a login nobody approves must fail
    # once the device code expires, not poll forever.
    def test_times_out_when_the_code_expires(self) -> None:
        clock = FakeClock()
        workos = FakeWorkOs(poll_results=(PENDING,) * 4)

        with pytest.raises(DeviceAuthError, match="Timed out"):
            _client(workos, clock).poll_for_access_token(_authorization(expires_in_seconds=12.0))

        # Never sleeps past the deadline: 5 + 5 + the 2 seconds that remain.
        assert clock.slept == [5.0, 5.0, 2.0]

    # Catches WorkOS API failures mid-poll escaping as raw WorkOsApiError
    # instead of the DeviceAuthError callers handle.
    def test_workos_failures_become_device_auth_errors(self) -> None:
        workos = FakeWorkOs(poll_results=(WorkOsApiError("Could not reach WorkOS: no route"),))
        with pytest.raises(DeviceAuthError, match="Could not reach WorkOS"):
            _client(workos).poll_for_access_token(_authorization())
