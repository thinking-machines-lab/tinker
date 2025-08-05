# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

from .lora_config_param import LoraConfigParam

__all__ = ["ModelCreateParams"]


class ModelCreateParams(TypedDict, total=False):
    base_model: Required[str]

    lora_config: LoraConfigParam

    type: Literal["create_model"] = "create_model"
