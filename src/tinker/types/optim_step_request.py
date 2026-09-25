from typing import Any, Optional

from pydantic import AliasChoices, Field, SerializerFunctionWrapHandler, model_serializer
from typing_extensions import Literal

from .._compat import PYDANTIC_V2, ConfigDict
from .._models import StrictBase
from .model_id import ModelID
from .optimizer import AdamParams, OptimParams

__all__ = ["OptimStepRequest", "AdamParams"]


class OptimStepRequest(StrictBase):
    optim_params: OptimParams = Field(
        validation_alias=AliasChoices("optim_params", "adam_params", "optimizer_params")
    )

    model_id: ModelID

    seq_id: Optional[int] = None

    type: Literal["optim_step"] = "optim_step"

    if PYDANTIC_V2:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())

    @model_serializer(mode="wrap")
    def _serialize_wire(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if "optim_params" in data:
            params = data.pop("optim_params")
            if isinstance(self.optim_params, AdamParams):
                data["adam_params"] = params
            else:
                data["optimizer_params"] = params
        return data
