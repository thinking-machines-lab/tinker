from typing import Optional

from typing_extensions import Literal

from .._compat import PYDANTIC_V2, ConfigDict
from .._models import StrictBase
from .model_id import ModelID

__all__ = ["SaveWeightsExternalRequest"]


class SaveWeightsExternalRequest(StrictBase):
    model_id: ModelID

    path: str
    """A file/directory name for the external weights (required)"""

    seq_id: Optional[int] = None

    ttl_seconds: Optional[int] = None
    """TTL in seconds for this checkpoint (None = never expires)"""

    type: Literal["save_weights_external"] = "save_weights_external"

    if PYDANTIC_V2:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
