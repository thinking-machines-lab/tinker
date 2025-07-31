# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

from .model_id import ModelID

__all__ = ["ModelGetInfoParams"]


class ModelGetInfoParams(TypedDict, total=False):
    model_id: Required[ModelID]

    type: Literal["get_info"] = "get_info"
