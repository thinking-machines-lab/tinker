from typing import TYPE_CHECKING, Any

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
    import torch  # noqa: TC004

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
        """Convert torch.Tensor, numpy array, or numeric lists to TensorData if needed."""
        if _HAVE_TORCH and isinstance(value, torch.Tensor):
            # Auto-sparsify 2-D target_tokens and weights to reduce wire payload
            if key in _sparse_eligible_keys and value.ndim == 2:
                return TensorData.from_torch_sparse(value)
            return TensorData.from_torch(value)
        elif isinstance(value, np.ndarray):
            return TensorData.from_numpy(value)
        elif isinstance(value, list):
            try:
                array = np.asarray(value)
            except ValueError as exc:
                if any(isinstance(item, list) for item in value):
                    raise ValueError(
                        f"{key} must be a rectangular numeric array; ragged nested lists are not supported"
                    ) from exc
                raise
            if array.dtype.kind in ("f", "i", "u"):
                if _key_to_type[key] == "int64":
                    array = array.astype(np.int64)
                else:
                    array = array.astype(np.float32)
                return TensorData.from_numpy(array)
            if any(isinstance(item, list) for item in value):
                raise ValueError(
                    f"{key} must be a rectangular numeric array; ragged nested lists are not supported"
                )
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

_sparse_eligible_keys = {"target_tokens", "weights"}
