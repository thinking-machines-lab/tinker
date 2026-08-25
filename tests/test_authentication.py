"""Tests for the public Tinker authentication helpers."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from respx import MockRouter

import tinker
import tinker.auth
from tinker._exceptions import APIConnectionError, AuthenticationError, BillingError, TinkerError
from tinker.lib.credentials import JsonCredentialStore, ManualKey

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


@pytest.fixture(autouse=True)
def isolated_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "credentials.json"
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.delenv("TINKER_CREDENTIAL_CMD", raising=False)
    monkeypatch.setattr("tinker.lib.credentials.default_credentials_path", lambda: path)
    return path


def test_get_tinker_token_returns_none_when_unconfigured() -> None:
    assert tinker.auth.get_tinker_token() is None


def test_get_tinker_token_finds_environment_key_with_credential_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TINKER_API_KEY", "tml-environment")
    monkeypatch.setenv("TINKER_CREDENTIAL_CMD", "command-that-does-not-exist")
    assert tinker.auth.get_tinker_token() == "tml-environment"


def test_get_tinker_token_finds_stored_default(isolated_credentials: Path) -> None:
    store = JsonCredentialStore(isolated_credentials)
    store.add_key("test", ManualKey(key="tml-stored", name="test"))
    store.set_default("test")

    assert tinker.auth.get_tinker_token() == "tml-stored"


def test_get_tinker_token_finds_stored_default_with_credential_command(
    isolated_credentials: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonCredentialStore(isolated_credentials)
    store.add_key("test", ManualKey(key="tml-stored", name="test"))
    store.set_default("test")
    monkeypatch.setenv("TINKER_CREDENTIAL_CMD", "command-that-does-not-exist")

    assert tinker.auth.get_tinker_token() == "tml-stored"


def test_tinker_has_credentials_returns_false_when_unconfigured() -> None:
    assert tinker.auth.tinker_has_credentials() is False


def test_tinker_has_credentials_finds_environment_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    assert tinker.auth.tinker_has_credentials() is True


def test_tinker_has_credentials_does_not_run_credential_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TINKER_CREDENTIAL_CMD", "command-that-does-not-exist")
    assert tinker.auth.tinker_has_credentials() is True


def test_tinker_has_credentials_finds_stored_default(isolated_credentials: Path) -> None:
    store = JsonCredentialStore(isolated_credentials)
    store.add_key("test", ManualKey(key="tml-test", name="test"))
    store.set_default("test")

    assert tinker.auth.tinker_has_credentials() is True


def _authentication_error() -> AuthenticationError:
    request = httpx.Request("GET", "https://tinker.test/api/v1/get_server_capabilities")
    response = httpx.Response(401, request=request)
    return AuthenticationError("invalid credential", response=response, body=None)


def _connection_error() -> APIConnectionError:
    request = httpx.Request("GET", "https://tinker.test/api/v1/get_server_capabilities")
    return APIConnectionError(request)


def _patch_service_client(monkeypatch: pytest.MonkeyPatch, client: MagicMock) -> None:
    monkeypatch.setattr(
        "tinker.lib.public_interfaces.ServiceClient", MagicMock(return_value=client)
    )


def test_raise_if_tinker_not_accessible_makes_one_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    _patch_service_client(monkeypatch, client)

    tinker.auth.raise_if_tinker_not_accessible()

    client._check_accessible.assert_called_once_with()


def test_raise_if_tinker_not_accessible_raises_for_rejected_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client._check_accessible.side_effect = _authentication_error()
    _patch_service_client(monkeypatch, client)

    with pytest.raises(AuthenticationError):
        tinker.auth.raise_if_tinker_not_accessible()


def test_raise_if_tinker_not_accessible_raises_connection_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client._check_accessible.side_effect = _connection_error()
    _patch_service_client(monkeypatch, client)

    with pytest.raises(APIConnectionError):
        tinker.auth.raise_if_tinker_not_accessible()


def test_raise_if_tinker_not_accessible_wraps_non_tinker_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client._check_accessible.side_effect = OSError("credentials file unreadable")
    _patch_service_client(monkeypatch, client)

    with pytest.raises(TinkerError) as excinfo:
        tinker.auth.raise_if_tinker_not_accessible()

    assert isinstance(excinfo.value.__cause__, OSError)


@pytest.mark.respx(base_url=base_url)
def test_raise_if_tinker_not_accessible_surfaces_billing_error_from_one_request(
    respx_mock: MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 402 (billing not set up) is raised, not waited out."""
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    monkeypatch.setenv("TINKER_BASE_URL", base_url)
    respx_mock.post("/api/v1/client/config").mock(
        return_value=httpx.Response(
            200,
            # Keep the pause window short so a regression that reinstates the
            # billing retry loop fails the call count below instead of hanging.
            json={"use_pyqwest_transport": False, "billing_exception_max_pause_duration_sec": 1},
        )
    )
    respx_mock.post("/api/v1/client/dynamic_config").mock(return_value=httpx.Response(200, json={}))
    capabilities = respx_mock.get("/api/v1/get_server_capabilities").mock(
        return_value=httpx.Response(402, json={"detail": "Access is blocked due to billing status"})
    )

    with pytest.raises(BillingError) as excinfo:
        tinker.auth.raise_if_tinker_not_accessible()

    assert excinfo.value.status_code == 402
    assert capabilities.call_count == 1


def test_authentication_helpers_are_not_exported_at_package_root() -> None:
    assert not hasattr(tinker, "get_tinker_token")
    assert not hasattr(tinker, "tinker_has_credentials")
    assert not hasattr(tinker, "raise_if_tinker_not_accessible")
