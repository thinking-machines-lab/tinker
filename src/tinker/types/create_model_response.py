from typing_extensions import Literal

from .._models import BaseModel
from .model_id import ModelID

__all__ = ["CreateModelResponse"]


class CreateModelResponse(BaseModel):
    model_id: ModelID

    type: Literal["create_model"] = "create_model"
