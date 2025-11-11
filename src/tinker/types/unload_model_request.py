from typing import Optional
from typing_extensions import Literal

from .._compat import PYDANTIC_V2, ConfigDict
from .._models import StrictBase
from .model_id import ModelID

__all__ = ["UnloadModelRequest"]


class UnloadModelRequest(StrictBase):
    model_id: ModelID

    type: Literal["unload_model"] = "unload_model"

    if PYDANTIC_V2:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
