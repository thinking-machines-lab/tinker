from typing import Optional
from typing_extensions import Literal

from .._compat import PYDANTIC_V2, ConfigDict
from .._models import StrictBase
from .model_id import ModelID

__all__ = ["SaveWeightsRequest"]


class SaveWeightsRequest(StrictBase):
    model_id: ModelID

    path: Optional[str] = None
    """A file/directory name for the weights"""

    seq_id: Optional[int] = None

    type: Literal["save_weights"] = "save_weights"

    if PYDANTIC_V2:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
