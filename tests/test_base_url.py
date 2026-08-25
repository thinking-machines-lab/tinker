"""Tests for the shared Tinker base-URL resolution (tinker.lib.base_url).

The SDK client and the CLI's login flow both resolve the base URL through
resolve_base_url, so `tinker auth login` mints keys from the same deployment
the SDK will later talk to.
"""

from __future__ import annotations

import httpx
import pytest

from tinker._client import AsyncTinker
from tinker.cli.auth_api import TinkerAuthApi
from tinker.lib.base_url import DEFAULT_BASE_URL, resolve_base_url


class TestResolveBaseUrl:
    # Catches the production URL being changed accidentally — spelled out as a
    # literal here so editing the constant forces editing this test too.
    def test_defaults_to_prod(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TINKER_BASE_URL", raising=False)
        assert resolve_base_url() == "https://tinker.thinkingmachines.dev/services/tinker-prod"

    # Catches TINKER_BASE_URL being ignored, and a trailing slash in the env
    # var producing double-slash request paths.
    def test_env_var_overrides_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TINKER_BASE_URL", "https://staging.test/api/")
        assert resolve_base_url() == "https://staging.test/api"

    # Catches the precedence inverting: an explicit argument (e.g. base_url
    # passed to the client) must beat the environment.
    def test_argument_overrides_the_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TINKER_BASE_URL", "https://staging.test")
        assert resolve_base_url("https://other.test/") == "https://other.test"

    # Catches empty values (e.g. `TINKER_BASE_URL=` in a shell profile) being
    # taken literally, which would send requests to a blank base URL.
    def test_empty_values_count_as_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TINKER_BASE_URL", "")
        assert resolve_base_url("") == DEFAULT_BASE_URL


# Catches the two consumers drifting apart again — the bug this helper exists
# to prevent: the login flow minting a key from one deployment while the SDK
# sends requests to another.
def test_sdk_client_and_cli_resolve_identically(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TINKER_BASE_URL", "https://staging.test/api")
    sdk = AsyncTinker(api_key="tml-test")
    cli = TinkerAuthApi(httpx.Client())
    # The SDK's base client re-adds a trailing slash for URL joining.
    assert str(sdk.base_url).rstrip("/") == cli.base_url == "https://staging.test/api"
