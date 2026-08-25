"""Tests for the credential store (~/.tinker/credentials.json) and the SDK's
stored-credential fallback in resolve_auth_provider."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
from pydantic import ValidationError

from tinker._exceptions import TinkerError
from tinker.lib._auth_token_provider import (
    ApiKeyAuthProvider,
    CredentialCmdAuthProvider,
    resolve_auth_provider,
)
from tinker.lib.credentials import (
    ApiKeyDetails,
    ApiKeyOrgDetails,
    ApiKeyUserDetails,
    GeneratedKey,
    JsonCredentialStore,
    ManualKey,
)


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / ".tinker" / "credentials.json"


@pytest.fixture
def store(store_path: Path) -> JsonCredentialStore:
    return JsonCredentialStore(store_path)


def _manual(name: str = "work", key: str = "tml-secret") -> ManualKey:
    return ManualKey(key=key, name=name)


class TestJsonCredentialStore:
    def test_add_and_get_key(self, store: JsonCredentialStore) -> None:
        store.add_key("work", _manual())
        assert store.get_key("work") == _manual()

    def test_get_missing_key_returns_none(self, store: JsonCredentialStore) -> None:
        assert store.get_key("nope") is None

    def test_add_key_replaces_existing(self, store: JsonCredentialStore) -> None:
        store.add_key("work", _manual(key="tml-old"))
        store.add_key("work", _manual(key="tml-new"))
        record = store.get_key("work")
        assert record is not None and record.key == "tml-new"

    def test_delete_key(self, store: JsonCredentialStore) -> None:
        store.add_key("work", _manual())
        store.delete_key("work")
        assert store.get_key("work") is None

    def test_delete_missing_key_raises(self, store: JsonCredentialStore) -> None:
        store.add_key("work", _manual())
        with pytest.raises(KeyError):
            store.delete_key("nope")

    def test_delete_default_key_clears_default(self, store: JsonCredentialStore) -> None:
        store.add_key("work", _manual())
        store.set_default("work")
        store.delete_key("work")
        assert store.get_default_key() is None

    def test_delete_other_key_keeps_default(self, store: JsonCredentialStore) -> None:
        store.add_key("work", _manual())
        store.add_key("other", _manual(name="other", key="tml-other"))
        store.set_default("work")
        store.delete_key("other")
        assert store.get_default_key() == _manual()

    def test_set_default_and_get_default_key(self, store: JsonCredentialStore) -> None:
        store.add_key("work", _manual())
        store.add_key("other", _manual(name="other", key="tml-other"))
        store.set_default("other")
        record = store.get_default_key()
        assert record is not None and record.key == "tml-other"

    def test_set_default_missing_key_raises(self, store: JsonCredentialStore) -> None:
        with pytest.raises(KeyError):
            store.set_default("nope")

    def test_set_default_none_clears(self, store: JsonCredentialStore) -> None:
        store.add_key("work", _manual())
        store.set_default("work")
        store.set_default(None)
        assert store.get_default_key() is None

    def test_get_default_key_none_when_unset(self, store: JsonCredentialStore) -> None:
        store.add_key("work", _manual())
        assert store.get_default_key() is None

    def test_get_default_key_id(self, store: JsonCredentialStore) -> None:
        store.add_key("work", _manual())
        assert store.get_default_key_id() is None
        store.set_default("work")
        assert store.get_default_key_id() == "work"

    def test_generated_key_round_trips(self, store: JsonCredentialStore) -> None:
        record = GeneratedKey(
            key="tml-gen",
            name="gen",
            details=ApiKeyDetails(
                org_details=ApiKeyOrgDetails(name="Acme"),
                user_details=ApiKeyUserDetails(email="user@acme.test"),
            ),
        )
        store.add_key("gen", record)
        assert store.get_key("gen") == record

    def test_on_disk_format_matches_spec(
        self, store: JsonCredentialStore, store_path: Path
    ) -> None:
        """The file is the raw StoredCredentialsV1 shape: a version tag, the
        default key id, and a key-id -> tagged-record mapping."""
        store.add_key("work", _manual())
        store.set_default("work")
        assert json.loads(store_path.read_text()) == {
            "version": 1,
            "default": "work",
            "keys": {"work": {"type": "manual", "key": "tml-secret", "name": "work"}},
        }

    def test_reads_generated_key_from_disk(
        self, store: JsonCredentialStore, store_path: Path
    ) -> None:
        store_path.parent.mkdir(parents=True)
        store_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "default": "gen",
                    "keys": {
                        "gen": {
                            "type": "generated",
                            "key": "tml-gen",
                            "name": "gen",
                            "details": {
                                "org_details": {"name": "Acme"},
                                "user_details": {"email": "user@acme.test"},
                            },
                        }
                    },
                }
            )
        )
        record = store.get_default_key()
        assert isinstance(record, GeneratedKey)
        assert record.details.org_details.name == "Acme"
        assert record.details.user_details.email == "user@acme.test"

    def test_corrupt_file_raises(self, store: JsonCredentialStore, store_path: Path) -> None:
        store_path.parent.mkdir(parents=True)
        store_path.write_text('{"version": 999}')
        with pytest.raises(ValidationError):
            store.get_key("work")

    @pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
    def test_file_and_dir_permissions(self, store: JsonCredentialStore, store_path: Path) -> None:
        store.add_key("work", _manual())
        assert stat.S_IMODE(store_path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(store_path.stat().st_mode) == 0o600


class TestCreateFromStored:
    """ApiKeyAuthProvider.create_from_stored(): a provider for the stored
    default key, or None if the store has nothing to offer."""

    def test_missing_file_returns_none(self, store_path: Path) -> None:
        assert ApiKeyAuthProvider.create_from_stored(store_path) is None

    def test_no_default_returns_none(self, store: JsonCredentialStore, store_path: Path) -> None:
        store.add_key("work", _manual())
        assert ApiKeyAuthProvider.create_from_stored(store_path) is None

    async def test_resolves_default_key_value(
        self, store: JsonCredentialStore, store_path: Path
    ) -> None:
        store.add_key("work", _manual())
        store.set_default("work")
        provider = ApiKeyAuthProvider.create_from_stored(store_path)
        assert provider is not None
        assert await provider.get_token() == "tml-secret"

    def test_dangling_default_returns_none(self, store_path: Path) -> None:
        store_path.parent.mkdir(parents=True)
        store_path.write_text(json.dumps({"version": 1, "default": "gone", "keys": {}}))
        assert ApiKeyAuthProvider.create_from_stored(store_path) is None


class TestStoredCredentialFallback:
    """resolve_auth_provider precedence: api_key arg > TINKER_API_KEY >
    TINKER_CREDENTIAL_CMD > stored default credential."""

    @pytest.fixture(autouse=True)
    def _isolated_credentials(
        self,
        monkeypatch: pytest.MonkeyPatch,
        store: JsonCredentialStore,
        store_path: Path,
    ) -> None:
        monkeypatch.delenv("TINKER_API_KEY", raising=False)
        monkeypatch.delenv("TINKER_CREDENTIAL_CMD", raising=False)
        monkeypatch.setattr("tinker.lib.credentials.default_credentials_path", lambda: store_path)
        store.add_key("work", _manual(key="tml-stored"))
        store.set_default("work")

    async def test_stored_default_used_when_env_unset(self) -> None:
        auth = resolve_auth_provider(api_key=None, enforce_cmd=False)
        assert isinstance(auth, ApiKeyAuthProvider)
        assert await auth.get_token() == "tml-stored"

    async def test_api_key_arg_wins_over_store(self) -> None:
        auth = resolve_auth_provider(api_key="tml-arg", enforce_cmd=False)
        assert await auth.get_token() == "tml-arg"

    async def test_env_var_wins_over_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TINKER_API_KEY", "tml-env")
        auth = resolve_auth_provider(api_key=None, enforce_cmd=False)
        assert await auth.get_token() == "tml-env"

    async def test_credential_cmd_wins_over_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TINKER_CREDENTIAL_CMD", "echo cmd-cred")
        auth = resolve_auth_provider(api_key=None, enforce_cmd=False)
        assert isinstance(auth, CredentialCmdAuthProvider)
        assert await auth.get_token() == "cmd-cred"

    async def test_invalid_key_falls_back_to_credential_cmd(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TINKER_API_KEY", "not-a-tml-key")
        monkeypatch.setenv("TINKER_CREDENTIAL_CMD", "echo cmd-cred")
        auth = resolve_auth_provider(api_key=None, enforce_cmd=False)
        assert isinstance(auth, CredentialCmdAuthProvider)
        assert await auth.get_token() == "cmd-cred"

    def test_invalid_key_without_cmd_raises_and_ignores_store(self) -> None:
        with pytest.raises(TinkerError, match="tml-"):
            resolve_auth_provider(api_key="not-a-tml-key", enforce_cmd=False)

    def test_api_key_provider_ignores_store(self) -> None:
        assert ApiKeyAuthProvider.create_or_env() is None

    def test_raises_when_store_has_no_default(self, store: JsonCredentialStore) -> None:
        store.set_default(None)
        with pytest.raises(TinkerError, match="tinker auth login"):
            resolve_auth_provider(api_key=None, enforce_cmd=False)
