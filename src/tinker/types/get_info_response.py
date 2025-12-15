from typing import Literal, Optional

from .._compat import PYDANTIC_V2, ConfigDict
from .._models import BaseModel
from .model_id import ModelID

__all__ = ["GetInfoResponse", "ModelData"]


class ModelData(BaseModel):
    """Metadata about a model's architecture and configuration."""

    arch: Optional[str] = None
    """The model architecture identifier."""

    model_name: Optional[str] = None
    """The human-readable model name."""

    tokenizer_id: Optional[str] = None
    """The identifier of the tokenizer used by this model."""


class GetInfoResponse(BaseModel):
    """Response containing information about a training client's model."""

    type: Optional[Literal["get_info"]] = None
    """Response type identifier."""

    model_data: ModelData
    """Detailed metadata about the model."""

    model_id: ModelID
    """Unique identifier for the model."""

    is_lora: Optional[bool] = None
    """Whether this is a LoRA fine-tuned model."""

    lora_rank: Optional[int] = None
    """The rank of the LoRA adaptation, if applicable."""

    model_name: Optional[str] = None
    """The name of the model."""

    if PYDANTIC_V2:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
