from typing import Any, Optional

from typing_extensions import Literal

from .._compat import PYDANTIC_V2, ConfigDict
from .._models import StrictBase
from .lora_config import LoraConfig

__all__ = ["CreateModelRequest"]


class CreateModelRequest(StrictBase):
    session_id: str

    model_seq_id: int

    base_model: str

    """Optional metadata about this model/training run, set by the end-user"""
    user_metadata: Optional[dict[str, Any]] = None

    lora_config: Optional[LoraConfig] = None

    type: Literal["create_model"] = "create_model"

    if PYDANTIC_V2:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
