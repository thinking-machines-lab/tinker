from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Union

import numpy as np
import numpy.typing as npt

from .tensor_dtype import TensorDtype

try:
    import torch  # type: ignore[import-not-found]

    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

if TYPE_CHECKING:
    import torch  # noqa: TC004

__all__ = ["TensorData"]


@dataclass(frozen=True, eq=False, init=False)
class TensorData:
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

    _numpy: np.ndarray = field(repr=False)

    def __init__(
        self,
        data: Union[List[int], List[float], np.ndarray, None] = None,
        dtype: Optional[TensorDtype] = None,
        shape: Optional[List[int]] = None,
        sparse_crow_indices: Optional[List[int]] = None,
        sparse_col_indices: Optional[List[int]] = None,
    ) -> None:
        if dtype is None:
            raise TypeError("TensorData requires `dtype`")
        np_dtype = _convert_tensor_dtype_to_numpy(dtype)
        if isinstance(data, np.ndarray):
            if data.dtype != np_dtype:
                arr = data.astype(np_dtype)
            elif not data.flags.writeable:
                # torch.from_numpy warns on non-writable buffers (np.frombuffer
                # over immutable bytes, mmap_mode='r', Arrow-backed arrays, etc.).
                arr = data.copy()
            else:
                arr = data
        elif data is None:
            arr = np.empty(0, dtype=np_dtype)
        else:
            arr = np.asarray(data, dtype=np_dtype)
        if sparse_crow_indices is None and shape is not None and list(arr.shape) != shape:
            arr = arr.reshape(shape)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "sparse_crow_indices", sparse_crow_indices)
        object.__setattr__(self, "sparse_col_indices", sparse_col_indices)
        object.__setattr__(self, "_numpy", arr)

    @property
    def data(self) -> Union[List[int], List[float]]:
        """Flattened tensor data as array of numbers."""
        return self._numpy.flatten().tolist()

    @classmethod
    def from_numpy(cls, array: npt.NDArray[Any]) -> TensorData:
        return cls(
            data=array,
            dtype=_convert_numpy_dtype_to_tensor(array.dtype),
            shape=list(array.shape),
        )

    @classmethod
    def from_torch(cls, tensor: torch.Tensor) -> TensorData:
        if tensor.dtype == torch.bfloat16:
            # Public TensorDtype is float32 | int64; widen bfloat16 to float32.
            arr = tensor.float().contiguous().numpy()
        else:
            arr = tensor.contiguous().numpy()
        return cls.from_numpy(arr)

    @classmethod
    def from_torch_sparse(cls, tensor: torch.Tensor) -> TensorData:
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
            data=sparse_csr.values().numpy(),
            dtype=_convert_torch_dtype_to_tensor(tensor.dtype),
            shape=list(tensor.shape),
            sparse_crow_indices=sparse_csr.crow_indices().tolist(),
            sparse_col_indices=sparse_csr.col_indices().tolist(),
        )

    def to_numpy(self) -> npt.NDArray[Any]:
        """Convert TensorData to numpy array."""
        if self.sparse_crow_indices is not None:
            return self.to_torch().numpy()
        return self._numpy

    def to_torch(self) -> torch.Tensor:
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
            values = torch.from_numpy(self._numpy).to(torch_dtype)
            return torch.sparse_csr_tensor(crow, col, values, self.shape).to_dense()

        t = torch.from_numpy(self._numpy)
        return t.to(torch_dtype) if t.dtype != torch_dtype else t

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


def _convert_tensor_dtype_to_torch(dtype: TensorDtype) -> torch.dtype:
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


def _convert_torch_dtype_to_tensor(dtype: torch.dtype) -> TensorDtype:
    """Convert torch dtype to TensorDtype."""
    # torch.dtype objects have .is_floating_point
    if getattr(dtype, "is_floating_point", False):
        return "float32"
    else:
        return "int64"
