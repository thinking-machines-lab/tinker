"""Tests for `tinker auth login --api-key` (manual key entry) and logout.

The browser flow, which `tinker auth login` runs by default, is covered in
test_cli_login.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner

from tinker._exceptions import AuthenticationError
from tinker.cli import auth_api as auth_api_module
from tinker.cli.auth_api import AuthApiError, TinkerAuthApi
from tinker.cli.commands.auth import cli as auth_cli
from tinker.cli.exceptions import TinkerCliError
from tinker.lib.credentials import (
    ApiKeyDetails,
    ApiKeyOrgDetails,
    ApiKeyUserDetails,
    GeneratedKey,
    JsonCredentialStore,
    ManualKey,
)


def _authentication_error() -> AuthenticationError:
    request = httpx.Request("GET", "https://tinker.test/api/v1/get_server_capabilities")
    response = httpx.Response(401, request=request)
    return AuthenticationError("invalid credential", response=response, body=None)


@pytest.fixture
def store_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / ".tinker" / "credentials.json"
    monkeypatch.setattr("tinker.lib.credentials.default_credentials_path", lambda: path)
    return path


def test_login_api_key_stores_key_and_sets_default(store_path: Path) -> None:
    result = CliRunner().invoke(auth_cli, ["login", "--api-key"], input="tml-secret\n")

    assert result.exit_code == 0, result.output
    assert json.loads(store_path.read_text()) == {
        "version": 1,
        "default": "manual",
        "keys": {
            "manual": {"type": "manual", "key": "tml-secret", "name": "Manually added api key"}
        },
    }


@pytest.mark.parametrize("login_args", [["login"], ["login", "--api-key"]])
def test_login_again_requires_logout_first(store_path: Path, login_args: list[str]) -> None:
    store = JsonCredentialStore(store_path)
    existing_key = ManualKey(key="tml-old", name="Existing key")
    store.add_key("existing", existing_key)
    store.set_default("existing")

    result = CliRunner().invoke(auth_cli, login_args)

    assert isinstance(result.exception, TinkerCliError)
    assert result.exception.message == "Already logged in"
    assert result.exception.details == "Run 'tinker auth logout' before logging in again."
    assert store.get_default_key() == existing_key


def test_login_key_value_is_not_echoed(store_path: Path) -> None:
    result = CliRunner().invoke(auth_cli, ["login", "--api-key"], input="tml-secret\n")
    assert result.exit_code == 0, result.output
    assert "tml-secret" not in result.output


def test_login_whitespace_key_errors(store_path: Path) -> None:
    result = CliRunner().invoke(auth_cli, ["login", "--api-key"], input=" \n")
    assert isinstance(result.exception, TinkerCliError)
    assert not store_path.exists()


class FakeAuthApi:
    """Stands in for TinkerAuthApi; records the keys deleted on the server."""

    deleted: list[str] = []
    error: AuthApiError | None = None

    def __init__(self, http_client: httpx.Client) -> None:
        pass

    def delete_self_api_key(self, api_key: str) -> None:
        if type(self).error is not None:
            raise type(self).error
        type(self).deleted.append(api_key)


@pytest.fixture
def fake_auth_api(monkeypatch: pytest.MonkeyPatch) -> type[FakeAuthApi]:
    FakeAuthApi.deleted = []
    FakeAuthApi.error = None
    monkeypatch.setattr(auth_api_module, "TinkerAuthApi", FakeAuthApi)
    return FakeAuthApi


def _generated_key(key: str, name: str) -> GeneratedKey:
    return GeneratedKey(
        key=key,
        name=name,
        details=ApiKeyDetails(
            org_details=ApiKeyOrgDetails(name="Acme"),
            user_details=ApiKeyUserDetails(email="user@acme.test"),
        ),
    )


def _store_two_manual_keys(store_path: Path) -> JsonCredentialStore:
    """A store with a default manual key ('manual') plus another key ('other')."""
    store = JsonCredentialStore(store_path)
    store.add_key("manual", ManualKey(key="tml-secret", name="Manually added api key"))
    store.add_key("other", ManualKey(key="tml-other", name="Another key"))
    store.set_default("manual")
    return store


def _store_generated_key(store_path: Path) -> JsonCredentialStore:
    """A store whose default is a browser-minted key."""
    store = JsonCredentialStore(store_path)
    store.add_key("generated", _generated_key("tml-secret", "cli-login-key"))
    store.set_default("generated")
    return store


def test_logout_generated_key_is_deleted_on_the_server(
    store_path: Path, fake_auth_api: type[FakeAuthApi]
) -> None:
    store = _store_generated_key(store_path)

    result = CliRunner().invoke(auth_cli, ["logout"])

    assert result.exit_code == 0, result.output
    assert fake_auth_api.deleted == ["tml-secret"]
    assert store.get_default_key() is None
    assert "Removed credential 'cli-login-key'" in result.output


def test_logout_manual_key_is_only_removed_locally(
    store_path: Path, fake_auth_api: type[FakeAuthApi]
) -> None:
    store = _store_two_manual_keys(store_path)

    result = CliRunner().invoke(auth_cli, ["logout"])

    assert result.exit_code == 0, result.output
    assert fake_auth_api.deleted == []
    # Only the default key is removed; other stored keys survive a logout.
    assert store.get_key("manual") is None
    assert store.get_key("other") is not None
    assert store.get_default_key() is None
    assert "Removed credential 'Manually added api key'" in result.output
    assert "still active" in result.output


def test_logout_without_a_default_credential_errors(
    store_path: Path, fake_auth_api: type[FakeAuthApi]
) -> None:
    result = CliRunner().invoke(auth_cli, ["logout"])

    assert isinstance(result.exception, TinkerCliError)
    assert fake_auth_api.deleted == []


# Catches the failure mode that would strand a live key: if the server delete
# fails, the credential must stay stored so the user can retry instead of
# losing their only handle on the key.
def test_logout_keeps_the_credential_when_the_server_delete_fails(
    store_path: Path, fake_auth_api: type[FakeAuthApi]
) -> None:
    store = _store_generated_key(store_path)
    fake_auth_api.error = AuthApiError("Service is not available (HTTP 503 from ...)")

    result = CliRunner().invoke(auth_cli, ["logout"])

    assert isinstance(result.exception, TinkerCliError)
    assert store.get_default_key() == _generated_key("tml-secret", "cli-login-key")


# Catches the wire protocol drifting from the server handler, which
# authenticates DELETE /api/v1/auth/apikey/me solely via the X-API-Key header
# (a bearer Authorization header is rejected).
def test_delete_self_api_key_sends_the_key_in_the_x_api_key_header() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(204)

    client = httpx.Client(transport=httpx.MockTransport(handle))
    TinkerAuthApi(client, base_url="https://tinker.test").delete_self_api_key("tml-secret")

    (request,) = requests
    assert request.method == "DELETE"
    assert request.url == httpx.URL("https://tinker.test/api/v1/auth/apikey/me")
    assert request.headers["X-API-Key"] == "tml-secret"
    assert "Authorization" not in request.headers


def test_delete_self_api_key_surfaces_the_api_error() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Unable to validate credential"})

    client = httpx.Client(transport=httpx.MockTransport(handle))
    api = TinkerAuthApi(client, base_url="https://tinker.test")
    with pytest.raises(AuthApiError, match="Unable to validate credential"):
        api.delete_self_api_key("tml-secret")


def test_status_reports_credentials_and_accessibility(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tinker.auth.tinker_has_credentials", lambda: True)
    monkeypatch.setattr("tinker.auth.raise_if_tinker_not_accessible", lambda: None)

    result = CliRunner().invoke(auth_cli, ["status"])

    assert result.exit_code == 0, result.output
    assert result.output == "Credentials available: yes\nTinker accessible: yes\n"


def test_status_fails_when_credentials_are_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tinker.auth.tinker_has_credentials", lambda: False)

    result = CliRunner().invoke(auth_cli, ["status"])

    assert isinstance(result.exception, TinkerCliError)
    assert "Credentials available: no" in result.output


def test_status_fails_when_credentials_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def _reject() -> None:
        raise _authentication_error()

    monkeypatch.setattr("tinker.auth.tinker_has_credentials", lambda: True)
    monkeypatch.setattr("tinker.auth.raise_if_tinker_not_accessible", _reject)

    result = CliRunner().invoke(auth_cli, ["status"])

    assert isinstance(result.exception, TinkerCliError)
    assert "Credentials available: yes" in result.output
    assert "Tinker accessible: no" in result.output
