from __future__ import annotations

from typing_extensions import Required, TypedDict

from .model_id import ModelID
from .forward_backward_input_param import ForwardBackwardInputParam

__all__ = ["TrainingForwardBackwardParams"]


class TrainingForwardBackwardParams(TypedDict, total=False):
    forward_backward_input: Required[ForwardBackwardInputParam]

    model_id: Required[ModelID]
