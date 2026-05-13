from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Union

import numpy as np

from .loss_fn_inputs import LossFnInputs
from .tensor_data import TensorData

try:
    import torch  # type: ignore[import-not-found]

    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

if TYPE_CHECKING:
    import torch  # noqa: TC004

from ._pydantic_types.model_input import ModelInput

__all__ = ["Datum"]

# Field-name → wire dtype for raw Python lists with no inferable numpy dtype.
_KEY_TO_TYPE = {
    "target_tokens": "int64",
    "weights": "float32",
    "advantages": "float32",
    "logprobs": "float32",
    "clip_low_threshold": "float32",
    "clip_high_threshold": "float32",
}

_SPARSE_ELIGIBLE_KEYS = {"target_tokens", "weights"}


@dataclass(frozen=True)
class Datum:
    model_input: ModelInput
    loss_fn_inputs: LossFnInputs = field(default_factory=dict)

    def __post_init__(self) -> None:
        coerced: Dict[str, TensorData] = {}
        for key, value in self.loss_fn_inputs.items():
            coerced[key] = _maybe_convert_array(key, value)
        object.__setattr__(self, "loss_fn_inputs", coerced)


def _maybe_convert_array(
    key: str, value: Union[TensorData, "torch.Tensor", np.ndarray, list]
) -> TensorData:
    if isinstance(value, TensorData):
        return value
    if _HAVE_TORCH and isinstance(value, torch.Tensor):
        if key in _SPARSE_ELIGIBLE_KEYS and value.ndim == 2:
            return TensorData.from_torch_sparse(value)
        return TensorData.from_torch(value)
    if isinstance(value, np.ndarray):
        return TensorData.from_numpy(value)
    if isinstance(value, list):
        try:
            array = np.asarray(value)
        except ValueError as exc:
            if any(isinstance(item, list) for item in value):
                raise ValueError(
                    f"{key} must be a rectangular numeric array; ragged nested lists are not supported"
                ) from exc
            raise
        if array.dtype.kind in ("f", "i", "u"):
            target_dtype = _KEY_TO_TYPE[key]
            if target_dtype == "int64":
                array = array.astype(np.int64)
            else:
                array = array.astype(np.float32)
            return TensorData.from_numpy(array)
        if any(isinstance(item, list) for item in value):
            raise ValueError(
                f"{key} must be a rectangular numeric array; ragged nested lists are not supported"
            )
        return TensorData(data=value, dtype=_KEY_TO_TYPE[key], shape=[len(value)])
    raise TypeError(f"Unsupported loss_fn_inputs value for {key!r}: {type(value).__name__}")
