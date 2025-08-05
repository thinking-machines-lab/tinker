# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from .._models import StrictBase
from .lora_config import LoraConfig

__all__ = ["CreateModelRequest"]


class CreateModelRequest(StrictBase):
    base_model: str

    lora_config: Optional[LoraConfig] = None

    type: Literal["create_model"] = "create_model"
