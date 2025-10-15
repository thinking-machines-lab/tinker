from __future__ import annotations

from typing import Any, Optional

from typing_extensions import Literal, Required, TypedDict

from .lora_config_param import LoraConfigParam

__all__ = ["ModelCreateParams"]


class ModelCreateParams(TypedDict, total=False):
    base_model: Required[str]

    lora_config: LoraConfigParam

    """Optional metadata about this model/training run, set by the end-user"""
    user_metadata: Optional[dict[str, Any]]

    type: Literal["create_model"] = "create_model"
