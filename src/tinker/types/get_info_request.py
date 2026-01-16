from typing_extensions import Literal

from .._compat import PYDANTIC_V2, ConfigDict
from .._models import StrictBase
from .model_id import ModelID

__all__ = ["GetInfoRequest"]


class GetInfoRequest(StrictBase):
    model_id: ModelID

    type: Literal["get_info"] = "get_info"

    if PYDANTIC_V2:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
