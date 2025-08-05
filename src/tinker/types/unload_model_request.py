# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from .._compat import PYDANTIC_V2, ConfigDict
from .._models import StrictBase
from .model_id import ModelID

__all__ = ["UnloadModelRequest"]


class UnloadModelRequest(StrictBase):
    model_id: ModelID

    type: Optional[Literal["unload_model"]] = None
