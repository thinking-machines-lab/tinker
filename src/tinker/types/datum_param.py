from __future__ import annotations

from typing_extensions import Required, TypedDict

from .model_input_param import ModelInputParam
from .loss_fn_inputs_param import LossFnInputsParam

__all__ = ["DatumParam"]


class DatumParam(TypedDict, total=False):
    loss_fn_inputs: Required[LossFnInputsParam]
    """Dictionary mapping field names to tensor data"""

    model_input: Required[ModelInputParam]
