from typing import Optional

from typing_extensions import Literal

from .._models import BaseModel

__all__: list[str] = ["SaveWeightsForSamplerResponse"]


class SaveWeightsForSamplerResponseInternal(BaseModel):
    path: str | None = None
    """A tinker URI for model weights for sampling at a specific step"""
    sampling_session_id: str | None = None
    """The generated sampling session ID"""

    type: Optional[Literal["save_weights_for_sampler"]] = None


class SaveWeightsForSamplerResponse(BaseModel):
    path: str
    """A tinker URI for model weights for sampling at a specific step"""

    type: Optional[Literal["save_weights_for_sampler"]] = None
