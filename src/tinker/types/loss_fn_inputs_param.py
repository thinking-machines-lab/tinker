from __future__ import annotations

from typing import Dict
from typing_extensions import TypeAlias

from .tensor_data_param import TensorDataParam

__all__ = ["LossFnInputsParam"]

LossFnInputsParam: TypeAlias = Dict[str, TensorDataParam]
