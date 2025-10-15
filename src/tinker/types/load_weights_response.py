from typing import Optional
from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["LoadWeightsResponse"]


class LoadWeightsResponse(BaseModel):
    path: Optional[str] = None
    """A tinker URI for model weights at a specific step"""

    type: Optional[Literal["load_weights"]] = None
