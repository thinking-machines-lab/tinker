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

    seq_id: Optional[int] = None

    type: Literal["load_weights"] = "load_weights"

    if PYDANTIC_V2:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
