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

    sampling_session_seq_id: Optional[int] = None

    seq_id: Optional[int] = None

    type: Literal["save_weights_for_sampler"] = "save_weights_for_sampler"

    if PYDANTIC_V2:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
