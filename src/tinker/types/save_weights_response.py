from typing import Optional
from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["SaveWeightsResponse"]


class SaveWeightsResponse(BaseModel):
    path: str
    """A tinker URI for model weights at a specific step"""

    type: Optional[Literal["save_weights"]] = None
