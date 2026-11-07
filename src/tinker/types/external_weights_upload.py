from __future__ import annotations

import datetime
from collections.abc import Mapping
from dataclasses import dataclass

from .._models import BaseModel, StrictBase
from .lora_config import LoraConfig

__all__ = ["ExternalWeightsUploadStartResult", "ExternalWeightsUploadFinishResult"]


@dataclass(frozen=True)
class ExternalWeightsUploadStartResult:
    """An upload started by `RestClient.external_weights_upload_start`."""

    upload_id: str
    """Identifies the upload when finishing it"""

    urls: Mapping[str, str]
    """Signed URL per file to upload, keyed by file name; upload each file with an HTTP PUT"""

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


class ExternalWeightsUploadStartResponse(BaseModel):
    """Transport: HTTP/JSON response of /api/v1/external_weights_upload_start, from the API
    server to the SDK; never stored."""

    upload_id: str
    urls: dict[str, str]
    expires: datetime.datetime


class ExternalWeightsUploadFinishRequest(StrictBase):
    """Transport: HTTP/JSON request body of /api/v1/external_weights_upload_finish, from the SDK
    to the API server; never stored."""

    upload_id: str


class ExternalWeightsUploadFinishResponse(BaseModel):
    """Transport: HTTP/JSON response of /api/v1/external_weights_upload_finish, from the API
    server to the SDK; never stored."""

    tinker_path: str
