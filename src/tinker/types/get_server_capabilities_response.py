from typing import List, Optional

from .._models import BaseModel

__all__ = ["GetServerCapabilitiesResponse", "SupportedModel"]


class SupportedModel(BaseModel):
    """Information about a model supported by the server."""

    model_name: Optional[str] = None
    """The name of the supported model."""


class GetServerCapabilitiesResponse(BaseModel):
    """Response containing the server's supported models and capabilities."""

    supported_models: List[SupportedModel]
    """List of models available on the server."""
