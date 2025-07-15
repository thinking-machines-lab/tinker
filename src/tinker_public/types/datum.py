# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Any, Dict, TYPE_CHECKING
from pydantic import model_validator

from .._models import StrictBase
from .model_input import ModelInput
from .loss_fn_inputs import LossFnInputs
from .tensor_data import TensorData
from .tensor_dtype import TensorDtype

try:
    import torch
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

import numpy as np

if TYPE_CHECKING:
    import torch

__all__ = ["Datum"]


class Datum(StrictBase):
    loss_fn_inputs: LossFnInputs
    """Dictionary mapping field names to tensor data"""

    model_input: ModelInput

    @model_validator(mode='before')
    @classmethod
    def convert_tensors(cls, data: Any) -> Any:
        """Convert torch.Tensor and numpy arrays to TensorData in loss_fn_inputs during construction."""
        if isinstance(data, dict) and 'loss_fn_inputs' in data:
            loss_fn_inputs = data['loss_fn_inputs']
            if isinstance(loss_fn_inputs, dict):
                converted_inputs = {}
                for key, value in loss_fn_inputs.items():
                    converted_inputs[key] = cls._maybe_convert_array(value)
                data = dict(data)  # Make a copy
                data['loss_fn_inputs'] = converted_inputs
        return data

    @classmethod
    def _maybe_convert_array(cls, value: Any) -> Any:
        """Convert torch.Tensor or numpy array to TensorData if needed."""
        if _HAVE_TORCH and isinstance(value, torch.Tensor):
            return TensorData(
                data=value.flatten().tolist(),
                dtype=_convert_torch_type(value.dtype),
                shape=[float(dim) for dim in value.shape]
            )
        elif isinstance(value, np.ndarray):
            return TensorData(
                data=value.flatten().tolist(),
                dtype=_convert_numpy_type(value.dtype),
                shape=[float(dim) for dim in value.shape]
            )
        else:
            return value


def _convert_torch_type(dtype: 'torch.dtype') -> TensorDtype:
    """Convert torch dtype to TensorDtype."""
    if dtype.is_floating_point:
        return "float32"
    else:
        return "int64"


def _convert_numpy_type(dtype: np.dtype) -> TensorDtype:
    """Convert numpy dtype to TensorDtype."""
    if dtype.kind == "f":
        return "float32"
    elif dtype.kind == "i":
        return "int64"
    else:
        raise ValueError(f"Unsupported numpy dtype: {dtype}")