# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from .._compat import PYDANTIC_V2, ConfigDict
from .._models import StrictBase
from .model_id import ModelID

__all__ = ["LoadWeightsRequest"]


class LoadWeightsRequest(StrictBase):
    model_id: ModelID

    path: str
    """A tinker URI for model weights at a specific step"""

    type: Optional[Literal["load_weights"]] = None
