# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Required, TypedDict

from .model_id import ModelID
from .request_id import RequestID

__all__ = ["FutureRetrieveParams"]


class FutureRetrieveParams(TypedDict, total=False):
    request_id: Required[RequestID]

    model_id: ModelID
