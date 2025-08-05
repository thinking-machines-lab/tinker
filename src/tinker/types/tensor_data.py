# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Union, Optional, TYPE_CHECKING

from .._models import StrictBase
from .tensor_dtype import TensorDtype

try:
    import torch
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

import numpy as np

if TYPE_CHECKING:
    import torch

__all__ = ["TensorData"]


class TensorData(StrictBase):
    data: Union[List[int], List[float]]
    """Flattened tensor data as array of numbers."""

    dtype: TensorDtype

    shape: Optional[List[int]] = None
    """Optional.

    The shape of the tensor (see PyTorch tensor.shape). The shape of a
    one-dimensional list of length N is `(N,)`. Can usually be inferred if not
    provided, and is generally inferred as a 1D tensor.
    """

    def to_numpy(self) -> np.ndarray:
        """Convert TensorData to numpy array."""
        # Convert dtype
        numpy_dtype = _convert_tensor_dtype_to_numpy(self.dtype)

        # Create numpy array from data
        arr = np.array(self.data, dtype=numpy_dtype)

        # Reshape if shape is provided
        if self.shape is not None:
            arr = arr.reshape(self.shape)

        return arr

    def to_torch(self) -> 'torch.Tensor':
        """Convert TensorData to torch tensor."""
        if not _HAVE_TORCH:
            raise ImportError("PyTorch is not installed. Cannot convert to torch tensor.")

        # Convert dtype
        torch_dtype = _convert_tensor_dtype_to_torch(self.dtype)

        # Create torch tensor from data
        tensor = torch.tensor(self.data, dtype=torch_dtype)

        # Reshape if shape is provided
        if self.shape is not None:
            tensor = tensor.reshape(self.shape)

        return tensor

    def tolist(self) -> list:
        return self.to_numpy().tolist()

def _convert_tensor_dtype_to_numpy(dtype: TensorDtype) -> np.dtype:
    """Convert TensorDtype to numpy dtype."""
    if dtype == "float32":
        return np.float32
    elif dtype == "int64":
        return np.int64
    else:
        raise ValueError(f"Unsupported TensorDtype: {dtype}")


def _convert_tensor_dtype_to_torch(dtype: TensorDtype) -> 'torch.dtype':
    """Convert TensorDtype to torch dtype."""
    if dtype == "float32":
        return torch.float32
    elif dtype == "int64":
        return torch.int64
    else:
        raise ValueError(f"Unsupported TensorDtype: {dtype}")
