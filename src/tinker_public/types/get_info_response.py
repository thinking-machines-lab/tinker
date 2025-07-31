# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional, Literal

from .._compat import PYDANTIC_V2, ConfigDict
from .._models import BaseModel
from .model_id import ModelID

__all__ = ["GetInfoResponse", "ModelData"]


class ModelData(BaseModel):
    arch: Optional[str] = None

    model_name: Optional[str] = None

class GetInfoResponse(BaseModel):
    type: Optional[Literal["get_info"]] = None

    model_data: ModelData

    model_id: ModelID

    is_lora: Optional[bool] = None

    lora_rank: Optional[int] = None

    model_name: Optional[str] = None

    if PYDANTIC_V2:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
