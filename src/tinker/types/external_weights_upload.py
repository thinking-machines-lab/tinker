from __future__ import annotations

import datetime
import os
from collections.abc import Mapping
from dataclasses import dataclass

import httpx

from .._models import BaseModel, StrictBase
from .lora_config import LoraConfig

__all__ = ["ExternalWeightsUploadStartResult", "ExternalWeightsUploadFinishResult", "UploadURL"]


@dataclass(frozen=True)
class UploadURL:
    """Where to upload one file of an external weights upload."""

    url: str
    """Signed URL to PUT the file to"""

    headers: Mapping[str, str]
    """Headers the PUT must send"""

    def upload(self, path: str | os.PathLike[str]) -> None:
        """Upload the file at `path`, streaming it in an HTTP PUT that sends `headers`."""
        if not self.url.startswith(("https://", "http://")):
            raise ValueError(f"Cannot upload to {self.url}; write the object directly")
        with open(path, "rb") as f:
            httpx.put(
                self.url, content=f, headers=dict(self.headers), timeout=600.0
            ).raise_for_status()


@dataclass(frozen=True)
class ExternalWeightsUploadStartResult:
    """An upload started by `RestClient.external_weights_upload_start`."""

    upload_id: str
    """Identifies the upload when finishing it"""

    urls: Mapping[str, UploadURL]
    """Where to upload each file, keyed by file name; upload each with `UploadURL.upload`"""

    expires: datetime.datetime
    """When the signed URLs expire"""


@dataclass(frozen=True)
class ExternalWeightsUploadFinishResult:
    """The checkpoint `RestClient.external_weights_upload_finish` made from an upload."""

    tinker_path: str
    """Tinker path of the external weights checkpoint holding the uploaded weights"""


class ExternalWeightsUploadStartRequest(StrictBase):
    """Transport: HTTP/JSON request body of /api/v1/external_weights_upload_start, from the SDK
    to the API server, which also stores it with the upload; never stored by the SDK."""

    base_model: str
    lora_config: LoraConfig
    destination_project_id: str | None = None
    supports_urls_v2: bool = True


class ExternalWeightsUploadURL(BaseModel):
    """Transport: part of ExternalWeightsUploadStartResponse; never stored."""

    url: str
    headers: dict[str, str]


class ExternalWeightsUploadStartResponse(BaseModel):
    """Transport: HTTP/JSON response of /api/v1/external_weights_upload_start, from the API
    server to the SDK; never stored."""

    upload_id: str
    # Servers send `urls_v2` to SDKs that support it, and older servers only `urls`.
    urls: dict[str, str] | None = None
    urls_v2: dict[str, ExternalWeightsUploadURL] | None = None
    expires: datetime.datetime


class ExternalWeightsUploadFinishRequest(StrictBase):
    """Transport: HTTP/JSON request body of /api/v1/external_weights_upload_finish, from the SDK
    to the API server; never stored."""

    upload_id: str


class ExternalWeightsUploadFinishResponse(BaseModel):
    """Transport: HTTP/JSON response of /api/v1/external_weights_upload_finish, from the API
    server to the SDK; never stored."""

    tinker_path: str
