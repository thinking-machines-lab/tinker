"""Tests for `tinker auth login`'s browser flow: device auth -> minted key."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

import httpx
import pytest
from click.testing import CliRunner

from tinker.cli import login as login_module
from tinker.cli.auth_api import AuthApiError
from tinker.cli.commands.auth import cli as auth_cli
from tinker.cli.device_auth import DeviceAuthError
from tinker.cli.exceptions import TinkerCliError
from tinker.cli.login import device_login
from tinker.lib.credentials import (
    ApiKeyDetails,
    ApiKeyOrgDetails,
    ApiKeyUserDetails,
    GeneratedKey,
    JsonCredentialStore,
)

BASE_URL = "https://tinker.test/services/tinker-prod"
WORKOS_URL = "https://workos.test"

AUTH_CONFIG_PATH = "/api/unauthed/auth-config"
AUTHORIZE_PATH = "/user_management/authorize/device"
AUTHENTICATE_PATH = "/user_management/authenticate"
API_KEY_PATH = "/api/v1/auth/apikey"

PREFILLED_URL = "https://auth.test/device?user_code=WXYZ-1234"
AUTHORIZATION_BODY = {
    "device_code": "device-secret",
    "user_code": "WXYZ-1234",
    "verification_uri": "https://auth.test/device",
    "verification_uri_complete": PREFILLED_URL,
    "expires_in": 300,
    "interval": 5,
}
API_KEY_BODY = {
    "api_key": "tml-minted-key",
    "id": 42,
    "details": {
        "org_details": {"name": "Acme"},
        "user_details": {"email": "user@acme.test"},
    },
}


class FakeServer:
    """The Tinker API and WorkOS, served over an httpx MockTransport."""

    def __init__(self, **overrides: httpx.Response) -> None:
        self.responses: dict[str, httpx.Response] = {
            AUTH_CONFIG_PATH: httpx.Response(200, json={"clientId": "client_123"}),
            AUTHORIZE_PATH: httpx.Response(200, json=AUTHORIZATION_BODY),
            AUTHENTICATE_PATH: httpx.Response(200, json={"access_token": "wos-token"}),
            API_KEY_PATH: httpx.Response(200, json=API_KEY_BODY),
            **overrides,
        }
        self.requests: dict[str, httpx.Request] = {}

    def _handle(self, request: httpx.Request) -> httpx.Response:
        for path, response in self.responses.items():
            if request.url.path.endswith(path):
                self.requests[path] = request
                return response
        raise AssertionError(f"unexpected request: {request.url}")

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._handle))


@pytest.fixture(autouse=True)
def store_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / ".tinker" / "credentials.json"
    monkeypatch.setattr("tinker.lib.credentials.default_credentials_path", lambda: path)
    return path


def _login(
    server: FakeServer,
    store_path: Path,
    *,
    echoed: Optional[list[str]] = None,
    browser_opens: bool = True,
) -> GeneratedKey:
    lines = echoed if echoed is not None else []
    return device_login(
        lines.append,
        base_url=BASE_URL,
        workos_api_url=WORKOS_URL,
        credential_store=JsonCredentialStore(store_path),
        http_client=server.client(),
        open_browser=lambda _: browser_opens,
    )


@pytest.fixture(autouse=True)
def machine_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the hostname so key names don't depend on the test machine."""
    monkeypatch.setattr(login_module.socket, "gethostname", lambda: "laptop.local")


class TestDeviceLogin:
    # Catches regressions in the stored file: the SDK reads this exact JSON
    # shape via JsonCredentialStore, so a renamed field, a lost `default`
    # pointer, or a mangled key value would break every subsequent SDK call.
    # The exact match also proves the WorkOS access token never reaches disk.
    def test_stores_minted_key_as_the_default_credential(self, store_path: Path) -> None:
        record = _login(FakeServer(), store_path)

        # The key is stored under the API key's own public id.
        assert json.loads(store_path.read_text()) == {
            "version": 1,
            "default": "42",
            "keys": {
                "42": {
                    "type": "generated",
                    "key": "tml-minted-key",
                    "name": "tinker-cli-laptop",
                    "details": {
                        "org_details": {"name": "Acme"},
                        "user_details": {"email": "user@acme.test"},
                    },
                }
            },
        }
        assert record == GeneratedKey(
            key="tml-minted-key",
            name="tinker-cli-laptop",
            details=ApiKeyDetails(
                org_details=ApiKeyOrgDetails(name="Acme"),
                user_details=ApiKeyUserDetails(email="user@acme.test"),
            ),
        )

    # Catches the mint request body drifting from what the API expects, and
    # keys showing up in the console with unrecognizable names.
    def test_mints_a_key_named_after_the_machine(self, store_path: Path) -> None:
        server = FakeServer()
        _login(server, store_path)

        request = server.requests[API_KEY_PATH]
        assert json.loads(request.content) == {
            "name": "tinker-cli-laptop",
            "note": "Generated by Tinker CLI",
        }

    # Catches the WorkOS token not reaching the mint request as a bearer
    # credential — the one authentication the endpoint accepts.
    def test_mints_the_key_with_the_workos_access_token(self, store_path: Path) -> None:
        server = FakeServer()
        _login(server, store_path)

        assert server.requests[API_KEY_PATH].headers["Authorization"] == "Bearer wos-token"

    # Catches the two halves coming unglued: the client id fetched from the
    # Tinker API (at the configured base URL) must be the one sent to WorkOS.
    def test_authorizes_with_the_client_id_from_the_tinker_api(self, store_path: Path) -> None:
        server = FakeServer()
        _login(server, store_path)

        assert server.requests[AUTH_CONFIG_PATH].url == httpx.URL(f"{BASE_URL}{AUTH_CONFIG_PATH}")
        assert server.requests[AUTHORIZE_PATH].content == b"client_id=client_123"

    # Catches the terminal output losing what the user needs to complete the
    # login (the code and the URL), the browser being pointed at anything but
    # the prefilled URL, and — critically — the secret device code being shown.
    def test_shows_the_code_and_url_and_opens_the_prefilled_url(self, store_path: Path) -> None:
        echoed: list[str] = []
        opened: list[str] = []

        def record_open(url: str) -> bool:
            opened.append(url)
            return True

        device_login(
            echoed.append,
            base_url=BASE_URL,
            workos_api_url=WORKOS_URL,
            credential_store=JsonCredentialStore(store_path),
            http_client=FakeServer().client(),
            open_browser=record_open,
        )

        assert opened == [PREFILLED_URL]
        assert "Confirmation code: WXYZ-1234" in echoed
        assert f"Open this URL to log in: {PREFILLED_URL}" in echoed
        assert any("Opening it in your default browser" in line for line in echoed)
        # The device code is a secret and must never be shown.
        assert "device-secret" not in "\n".join(echoed)

    # Catches a headless login claiming "opening your browser" when nothing
    # was opened — over SSH there is no browser, and the printed URL is the
    # whole login.
    def test_prints_the_url_without_claiming_a_browser_opened(self, store_path: Path) -> None:
        echoed: list[str] = []
        _login(FakeServer(), store_path, echoed=echoed, browser_opens=False)

        assert f"Open this URL to log in: {PREFILLED_URL}" in echoed
        assert "browser" not in "\n".join(echoed)

    # Catches the login flow ignoring TINKER_BASE_URL: it must mint the key
    # from the same deployment the SDK will later talk to.
    def test_base_url_comes_from_the_environment(
        self, store_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TINKER_BASE_URL", "https://staging.test/")
        server = FakeServer()
        device_login(
            lambda _: None,
            workos_api_url=WORKOS_URL,
            credential_store=JsonCredentialStore(store_path),
            http_client=server.client(),
            open_browser=lambda _: False,
        )

        assert server.requests[AUTH_CONFIG_PATH].url.host == "staging.test"
        assert server.requests[API_KEY_PATH].url == httpx.URL(f"https://staging.test{API_KEY_PATH}")

    # Catches TINKER_WORKOS_API_URL being ignored, which would break logging
    # in against non-prod WorkOS deployments.
    def test_workos_api_url_comes_from_the_environment(
        self, store_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TINKER_WORKOS_API_URL", "https://workos-staging.test/")
        server = FakeServer()
        device_login(
            lambda _: None,
            base_url=BASE_URL,
            credential_store=JsonCredentialStore(store_path),
            http_client=server.client(),
            open_browser=lambda _: False,
        )

        assert server.requests[AUTHORIZE_PATH].url.host == "workos-staging.test"

    # Catches the API's own explanation for a refused mint being swallowed,
    # and a failed login leaving a half-written credentials file behind.
    def test_surfaces_the_api_error_when_minting_fails(self, store_path: Path) -> None:
        server = FakeServer(
            **{
                API_KEY_PATH: httpx.Response(
                    403, json={"detail": "Your organization does not allow API key creation."}
                )
            }
        )
        with pytest.raises(AuthApiError, match="does not allow API key creation"):
            _login(server, store_path)
        assert not store_path.exists()

    # Catches a denied login being papered over — nothing may be stored, and
    # the user must be told the denial rather than a generic failure.
    def test_stores_nothing_when_the_login_is_denied(self, store_path: Path) -> None:
        server = FakeServer(
            **{AUTHENTICATE_PATH: httpx.Response(400, json={"error": "access_denied"})}
        )
        with pytest.raises(DeviceAuthError, match="was denied"):
            _login(server, store_path)
        assert not store_path.exists()


class TestLoginCommand:
    """The command wiring around device_login."""

    # Catches the success summary losing who the user logged in as and which
    # key was stored — their only confirmation the right account got the key.
    def test_reports_who_logged_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        record = GeneratedKey(
            key="tml-minted-key",
            name="tinker-cli-laptop",
            details=ApiKeyDetails(
                org_details=ApiKeyOrgDetails(name="Acme"),
                user_details=ApiKeyUserDetails(email="user@acme.test"),
            ),
        )
        monkeypatch.setattr(login_module, "device_login", lambda echo, **_: record)
        result = CliRunner().invoke(auth_cli, ["login"])

        assert result.exit_code == 0, result.output
        assert "Logged in as user@acme.test (Acme)" in result.output
        assert "tinker-cli-laptop" in result.output

    # Catches login failures escaping as raw tracebacks instead of the
    # TinkerCliError the CLI's central handler formats — with the reason kept.
    def test_failure_becomes_a_cli_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail(echo: object, **_: object) -> GeneratedKey:
            raise DeviceAuthError("The login code expired before it was approved.")

        monkeypatch.setattr(login_module, "device_login", fail)
        result = CliRunner().invoke(auth_cli, ["login"])

        assert result.exit_code != 0
        assert isinstance(result.exception, TinkerCliError)
        assert "expired" in (result.exception.details or "")


class TestOpenUrl:
    """Opening the browser is best-effort and must never block the login."""

    @staticmethod
    def _record_open(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], threading.Event]:
        opened: list[str] = []
        done = threading.Event()

        def fake_open(url: str) -> bool:
            opened.append(url)
            done.set()
            return True

        monkeypatch.setattr(login_module.webbrowser, "open", fake_open)
        return opened, done

    # Catches webbrowser's headless-Linux fallback: without a graphical
    # session it launches a console browser (lynx) that takes over the very
    # terminal the login is running in.
    def test_no_browser_without_a_graphical_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(login_module.sys, "platform", "linux")
        for name in ("BROWSER", "DISPLAY", "WAYLAND_DISPLAY"):
            monkeypatch.delenv(name, raising=False)
        opened, _ = self._record_open(monkeypatch)

        assert login_module.open_url("https://auth.test/device") is False
        assert opened == []

    # Catches the headless guard being over-tightened so that Linux desktops
    # (which do have a display) stop getting a browser opened for them.
    def test_opens_the_url_with_a_display(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(login_module.sys, "platform", "linux")
        monkeypatch.setenv("DISPLAY", ":0")
        opened, done = self._record_open(monkeypatch)

        assert login_module.open_url("https://auth.test/device") is True
        assert done.wait(timeout=5)
        assert opened == ["https://auth.test/device"]

    # Catches open_url calling webbrowser synchronously: a $BROWSER command
    # runs in the foreground, which would stall the login (it still has to
    # poll WorkOS) for as long as that browser stays open.
    def test_returns_without_waiting_for_a_blocking_browser(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(login_module.sys, "platform", "darwin")
        release = threading.Event()
        monkeypatch.setattr(
            login_module.webbrowser, "open", lambda url: bool(release.wait(timeout=10))
        )
        try:
            assert login_module.open_url("https://auth.test/device") is True
        finally:
            release.set()

    # Catches a browser that fails to start (webbrowser.Error/OSError on the
    # background thread) spewing a thread-crash traceback into the terminal
    # in the middle of a login that can proceed fine with the printed URL.
    def test_browser_errors_are_not_fatal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(login_module.sys, "platform", "darwin")
        raised = threading.Event()

        def boom(url: str) -> bool:
            raised.set()
            raise login_module.webbrowser.Error("no browser")

        monkeypatch.setattr(login_module.webbrowser, "open", boom)
        assert login_module.open_url("https://auth.test/device") is True
        assert raised.wait(timeout=5)
