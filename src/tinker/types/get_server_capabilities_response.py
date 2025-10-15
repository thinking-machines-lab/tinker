from typing import List, Optional

from .._models import BaseModel

__all__ = ["GetServerCapabilitiesResponse", "SupportedModel"]


class SupportedModel(BaseModel):
    model_name: Optional[str] = None


class GetServerCapabilitiesResponse(BaseModel):
    supported_models: List[SupportedModel]
