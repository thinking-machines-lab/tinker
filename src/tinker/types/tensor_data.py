from typing import TYPE_CHECKING, Any, List, Optional, Union

from .._models import StrictBase
from .tensor_dtype import TensorDtype

try:
    import torch  # type: ignore[import-not-found]

    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import torch  # noqa: TC004

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

    sparse_crow_indices: Optional[List[int]] = None
    """Optional CSR compressed row pointers. When set, this tensor is sparse CSR:
    - data contains only the non-zero values (flattened)
    - sparse_crow_indices contains the row pointers (length = nrows + 1)
    - sparse_col_indices contains the column indices (length = nnz)
    - shape is required and specifies the dense shape
    """

    sparse_col_indices: Optional[List[int]] = None
    """Optional CSR column indices. Must be set together with sparse_crow_indices."""

    @classmethod
    def from_numpy(cls, array: npt.NDArray[Any]) -> "TensorData":
        return cls(
            data=array.flatten().tolist(),
            dtype=_convert_numpy_dtype_to_tensor(array.dtype),
            shape=list(array.shape),
        )

    @classmethod
    def from_torch(cls, tensor: "torch.Tensor") -> "TensorData":
        return cls(
            data=tensor.flatten().tolist(),
            dtype=_convert_torch_dtype_to_tensor(tensor.dtype),
            shape=list(tensor.shape),
        )

    @classmethod
    def from_torch_sparse(cls, tensor: "torch.Tensor") -> "TensorData":
        """Create a sparse CSR TensorData from a dense 2-D torch tensor.

        Automatically detects sparsity and encodes as CSR when it saves space.
        Falls back to dense if the tensor is 1-D or mostly non-zero.
        """
        if not _HAVE_TORCH:
            raise ImportError("PyTorch is not installed.")

        if tensor.ndim != 2:
            return cls.from_torch(tensor)

        # Only use sparse if it actually saves space
        # Dense: nrows * ncols values
        # CSR: (nrows + 1) crow_indices + nnz col_indices + nnz values
        nnz = tensor.count_nonzero().item()
        dense_size = tensor.shape[0] * tensor.shape[1]
        csr_size = (tensor.shape[0] + 1) + 2 * nnz
        if csr_size >= dense_size:
            return cls.from_torch(tensor)

        sparse_csr = tensor.to_sparse_csr()
        return cls(
            data=sparse_csr.values().tolist(),
            dtype=_convert_torch_dtype_to_tensor(tensor.dtype),
            shape=list(tensor.shape),
            sparse_crow_indices=sparse_csr.crow_indices().tolist(),
            sparse_col_indices=sparse_csr.col_indices().tolist(),
        )

    def to_numpy(self) -> npt.NDArray[Any]:
        """Convert TensorData to numpy array."""
        if self.sparse_crow_indices is not None:
            return self.to_torch().numpy()
        numpy_dtype = _convert_tensor_dtype_to_numpy(self.dtype)
        arr = np.array(self.data, dtype=numpy_dtype)
        if self.shape is not None:
            arr = arr.reshape(self.shape)
        return arr

    def to_torch(self) -> "torch.Tensor":
        """Convert TensorData to torch tensor."""
        if not _HAVE_TORCH:
            raise ImportError("PyTorch is not installed. Cannot convert to torch tensor.")

        torch_dtype = _convert_tensor_dtype_to_torch(self.dtype)

        if self.sparse_crow_indices is not None:
            assert self.sparse_col_indices is not None, (
                "sparse_col_indices required with sparse_crow_indices"
            )
            assert self.shape is not None, "shape is required for sparse tensors"
            crow = torch.tensor(self.sparse_crow_indices, dtype=torch.int64)
            col = torch.tensor(self.sparse_col_indices, dtype=torch.int64)
            values = torch.tensor(self.data, dtype=torch_dtype)
            return torch.sparse_csr_tensor(crow, col, values, self.shape).to_dense()

        tensor = torch.tensor(self.data, dtype=torch_dtype)
        if self.shape is not None:
            tensor = tensor.reshape(self.shape)
        return tensor

    def tolist(self) -> List[Any]:
        return self.to_numpy().tolist()


def _convert_tensor_dtype_to_numpy(dtype: TensorDtype) -> npt.DTypeLike:
    """Convert TensorDtype to numpy dtype-like."""
    if dtype == "float32":
        return np.float32
    elif dtype == "int64":
        return np.int64
    else:
        raise ValueError(f"Unsupported TensorDtype: {dtype}")


def _convert_tensor_dtype_to_torch(dtype: TensorDtype) -> "torch.dtype":
    """Convert TensorDtype to torch dtype."""
    if not _HAVE_TORCH:
        raise ImportError("PyTorch is not installed. Cannot convert to torch dtype.")

    if dtype == "float32":
        return torch.float32
    elif dtype == "int64":
        return torch.int64
    else:
        raise ValueError(f"Unsupported TensorDtype: {dtype}")


def _convert_numpy_dtype_to_tensor(dtype: np.dtype[Any]) -> TensorDtype:
    """Convert numpy dtype to TensorDtype."""
    if dtype.kind == "f":
        return "float32"
    elif dtype.kind == "i":
        return "int64"
    else:
        raise ValueError(f"Unsupported numpy dtype: {dtype}")


def _convert_torch_dtype_to_tensor(dtype: "torch.dtype") -> TensorDtype:
    """Convert torch dtype to TensorDtype."""
    # torch.dtype objects have .is_floating_point
    if getattr(dtype, "is_floating_point", False):
        return "float32"
    else:
        return "int64"
