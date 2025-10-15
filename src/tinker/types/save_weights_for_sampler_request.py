from typing import Optional
from typing_extensions import Literal

from .._compat import PYDANTIC_V2, ConfigDict
from .._models import StrictBase
from .model_id import ModelID

__all__ = ["SaveWeightsForSamplerRequest"]


class SaveWeightsForSamplerRequest(StrictBase):
    model_id: ModelID

    path: Optional[str] = None
    """A file/directory name for the weights"""

    type: Optional[Literal["save_weights_for_sampler"]] = None
