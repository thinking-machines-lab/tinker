# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Dict
from typing_extensions import TypeAlias

from .tensor_data import TensorData

__all__ = ["LossFnOutput"]

LossFnOutput: TypeAlias = Dict[str, TensorData]
