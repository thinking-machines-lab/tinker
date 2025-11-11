from typing import Any, TYPE_CHECKING

from pydantic import model_validator

from .._models import StrictBase
from .loss_fn_inputs import LossFnInputs
from .model_input import ModelInput
from .tensor_data import TensorData

try:
    import torch  # type: ignore[import-not-found]

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

    @model_validator(mode="before")
    @classmethod
    def convert_tensors(cls, data: Any) -> Any:
        """Convert torch.Tensor and numpy arrays to TensorData in loss_fn_inputs during construction."""
        if isinstance(data, dict) and "loss_fn_inputs" in data:
            loss_fn_inputs = data["loss_fn_inputs"]
            if isinstance(loss_fn_inputs, dict):
                converted_inputs = {}
                for key, value in loss_fn_inputs.items():
                    converted_inputs[key] = cls._maybe_convert_array(key, value)
                data = dict(data)  # Make a copy
                data["loss_fn_inputs"] = converted_inputs
        return data

    @classmethod
    def _maybe_convert_array(cls, key: str, value: Any) -> Any:
        """Convert torch.Tensor, numpy array, or 1-D list to TensorData if needed."""
        if _HAVE_TORCH and isinstance(value, torch.Tensor):
            return TensorData.from_torch(value)
        elif isinstance(value, np.ndarray):
            return TensorData.from_numpy(value)
        elif isinstance(value, list):
            # assume it's 1d and infer the dtype from the key
            return TensorData(data=value, dtype=_key_to_type[key], shape=[len(value)])
        else:
            return value


_key_to_type = {
    "target_tokens": "int64",
    "weights": "float32",
    "advantages": "float32",
    "logprobs": "float32",
    "clip_low_threshold": "float32",
    "clip_high_threshold": "float32",
}
