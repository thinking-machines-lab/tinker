from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

from .model_id import ModelID

__all__ = ["ModelUnloadParams"]


class ModelUnloadParams(TypedDict, total=False):
    model_id: Required[ModelID]

    type: Literal["unload_model"] = "unload_model"
