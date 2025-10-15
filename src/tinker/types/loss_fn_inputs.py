from typing import Dict
from typing_extensions import TypeAlias

from .tensor_data import TensorData

__all__ = ["LossFnInputs"]

LossFnInputs: TypeAlias = Dict[str, TensorData]
