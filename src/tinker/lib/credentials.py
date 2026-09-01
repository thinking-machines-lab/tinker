"""Credential storage for the Tinker SDK and CLI.

Credentials live in a single JSON file (by default ~/.tinker/credentials.json)
holding named API keys plus the name of the default key. `CredentialStore` is
the abstract CRUD interface; `JsonCredentialStore` is the file-backed
implementation shared by `tinker auth login` and the SDK's credential
resolution (see _auth_token_provider.ApiKeyAuthProvider.create_from_stored).
"""

from __future__ import annotations

import abc
import contextlib
import os
import tempfile
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class ApiKeyOrgDetails(BaseModel):
    """The organization a key was minted under."""

    name: str


class ApiKeyUserDetails(BaseModel):
    """The user a key was minted for."""

    email: str


class ApiKeyDetails(BaseModel):
    """Who a minted key belongs to.

    Mirrors the ApiKeyDetails the API key creation endpoint returns, so the
    login flow can store the response as-is.
    """

    org_details: ApiKeyOrgDetails
    user_details: ApiKeyUserDetails


class ManualKey(BaseModel):
    """An API key the user pasted in via `tinker auth login --api-key`.

    `note` and `details` are optional only so credentials written before
    manual keys were verified against the server remain readable.
    """

    type: Literal["manual"] = "manual"
    key: str
    name: str
    note: str | None = None
    details: ApiKeyDetails | None = None


class GeneratedKey(BaseModel):
    """An API key minted through the browser login flow (`tinker auth login`)."""

    type: Literal["generated"] = "generated"
    key: str
    name: str
    details: ApiKeyDetails


KeyRecordV1 = Annotated[Union[ManualKey, GeneratedKey], Field(discriminator="type")]


class StoredCredentialsV1(BaseModel):
    """The raw on-disk shape of the credentials file."""

    version: Literal[1] = 1
    default: str | None = None
    keys: dict[str, KeyRecordV1] = Field(default_factory=dict)


def default_credentials_path() -> Path:
    """The standard location of the credentials file."""
    return Path("~/.tinker/credentials.json").expanduser()


class CredentialStore(abc.ABC):
    """CRUD interface over named API keys with an optional default key."""

    @abc.abstractmethod
    def add_key(self, key_id: str, record: KeyRecordV1) -> None:
        """Store record under key_id, replacing any existing record."""

    @abc.abstractmethod
    def delete_key(self, key_id: str) -> None:
        """Remove the key stored under key_id, clearing the default if it
        pointed at this key. Raises KeyError if key_id is not stored."""

    @abc.abstractmethod
    def set_default(self, key_id: str | None) -> None:
        """Mark key_id as the default key (None clears the default).
        Raises KeyError if key_id is not stored."""

    @abc.abstractmethod
    def get_key(self, key_id: str) -> KeyRecordV1 | None:
        """The record stored under key_id, or None."""

    @abc.abstractmethod
    def get_default_key(self) -> KeyRecordV1 | None:
        """The default key's record, or None if no default is set."""

    @abc.abstractmethod
    def get_default_key_id(self) -> str | None:
        """The id the default key is stored under, or None if no default is
        set or the default points at a key that is no longer stored."""


class JsonCredentialStore(CredentialStore):
    """CredentialStore backed by a StoredCredentialsV1 JSON file.

    The parent directory is created with mode 0700 and the file is written
    with mode 0600, since it holds live API keys. Writes go through a temp
    file + rename so the file is never observable partially written or with
    permissive modes.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)

    def add_key(self, key_id: str, record: KeyRecordV1) -> None:
        credentials = self._load()
        credentials.keys[key_id] = record
        self._save(credentials)

    def delete_key(self, key_id: str) -> None:
        credentials = self._load()
        if key_id not in credentials.keys:
            raise KeyError(key_id)
        del credentials.keys[key_id]
        if credentials.default == key_id:
            credentials.default = None
        self._save(credentials)

    def set_default(self, key_id: str | None) -> None:
        credentials = self._load()
        if key_id is not None and key_id not in credentials.keys:
            raise KeyError(key_id)
        credentials.default = key_id
        self._save(credentials)

    def get_key(self, key_id: str) -> KeyRecordV1 | None:
        return self._load().keys.get(key_id)

    def get_default_key(self) -> KeyRecordV1 | None:
        credentials = self._load()
        if credentials.default is None:
            return None
        return credentials.keys.get(credentials.default)

    def get_default_key_id(self) -> str | None:
        credentials = self._load()
        if credentials.default not in credentials.keys:
            return None
        return credentials.default

    def _load(self) -> StoredCredentialsV1:
        try:
            raw = self._path.read_text()
        except FileNotFoundError:
            return StoredCredentialsV1()
        return StoredCredentialsV1.model_validate_json(raw)

    def _save(self, credentials: StoredCredentialsV1) -> None:
        # The file holds live API keys, so restrict both the directory
        # (owner-only rwx) and the file (owner-only rw) to the owner.
        dir_mode = 0o700
        file_mode = 0o600
        directory = self._path.parent
        if not directory.is_dir():
            directory.mkdir(mode=dir_mode, parents=True, exist_ok=True)
            # mkdir's mode is masked by the umask; enforce it exactly.
            directory.chmod(dir_mode)
        # Write to a temp file and atomically rename it over the destination:
        # readers never see a partially written or world-readable file, and a
        # crash mid-write leaves the previous contents intact.
        fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f".{self._path.name}.")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(credentials.model_dump_json(indent=2, exclude_none=True))
            os.chmod(tmp_name, file_mode)
            os.replace(tmp_name, self._path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
