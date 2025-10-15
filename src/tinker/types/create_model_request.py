from typing import Any, Optional

from typing_extensions import Literal

from .._models import StrictBase
from .lora_config import LoraConfig

__all__ = ["CreateModelRequest"]


class CreateModelRequest(StrictBase):
    base_model: str

    """Optional metadata about this model/training run, set by the end-user"""
    user_metadata: Optional[dict[str, Any]] = None

    lora_config: Optional[LoraConfig] = None

    type: Literal["create_model"] = "create_model"
