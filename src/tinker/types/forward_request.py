from typing import Optional

from .._compat import PYDANTIC_V2, ConfigDict
from .._models import StrictBase
from .forward_backward_input import ForwardBackwardInput
from .model_id import ModelID

__all__ = ["ForwardRequest"]


class ForwardRequest(StrictBase):
    forward_input: ForwardBackwardInput

    model_id: ModelID

    seq_id: Optional[int] = None

    if PYDANTIC_V2:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
