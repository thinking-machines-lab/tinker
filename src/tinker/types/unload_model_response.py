from typing import Optional

from typing_extensions import Literal

from .._models import BaseModel
from .model_id import ModelID

__all__ = ["UnloadModelResponse"]


class UnloadModelResponse(BaseModel):
    model_id: ModelID

    type: Optional[Literal["unload_model"]] = None
