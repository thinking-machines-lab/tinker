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
    - data contains only the listed values (flattened); every other entry takes
      the `pad_value` given when converting to or from a dense tensor
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
    def from_torch_sparse(cls, tensor: torch.Tensor, pad_value: int = 0) -> TensorData:
        """Create a sparse CSR TensorData from a dense 2-D torch tensor.

        Entries equal to `pad_value` are left out; the rest are stored as CSR
        values. Automatically detects sparsity and encodes as CSR when it saves
        space. Falls back to dense if the tensor is 1-D or mostly non-pad.

        `pad_value` must be an integer: the tensor is shifted by it so torch's
        zero-based CSR conversion can be reused, and an integer shift is exact
        for `int64` tensors. For `float32` tensors with a non-zero pad it may
        round values whose magnitude is far below `pad_value`.
        """
        if not _HAVE_TORCH:
            raise ImportError("PyTorch is not installed.")

        pad_value = _check_pad_value(pad_value)

        if tensor.ndim != 2:
            return cls.from_torch(tensor)

        # torch's CSR layout treats 0 as the implicit value, so encode the
        # offset from pad_value and add it back to the stored values.
        shifted = tensor - pad_value if pad_value != 0 else tensor

        # Only use sparse if it actually saves space
        # Dense: nrows * ncols values
        # CSR: (nrows + 1) crow_indices + nnz col_indices + nnz values
        nnz = shifted.count_nonzero().item()
        dense_size = tensor.shape[0] * tensor.shape[1]
        csr_size = (tensor.shape[0] + 1) + 2 * nnz
        if csr_size >= dense_size:
            return cls.from_torch(tensor)

        sparse_csr = shifted.to_sparse_csr()
        values = sparse_csr.values()
        if pad_value != 0:
            values = values + pad_value
        return cls(
            data=values.numpy(),
            dtype=_convert_torch_dtype_to_tensor(tensor.dtype),
            shape=list(tensor.shape),
            sparse_crow_indices=sparse_csr.crow_indices().tolist(),
            sparse_col_indices=sparse_csr.col_indices().tolist(),
        )

    def to_numpy(self, pad_value: int = 0) -> npt.NDArray[Any]:
        """Convert TensorData to numpy array.

        A sparse CSR tensor is densified with `pad_value` in every unlisted entry.
        """
        if self.sparse_crow_indices is not None:
            return self.to_torch(pad_value).numpy()
        return self._numpy

    def to_torch(self, pad_value: int = 0) -> torch.Tensor:
        """Convert TensorData to torch tensor.

        A sparse CSR tensor is densified with `pad_value` in every unlisted
        entry. `pad_value` must be an integer so the shift around torch's
        zero-based densification is exact for `int64` tensors.
        """
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
            pad_value = _check_pad_value(pad_value)
            if pad_value == 0:
                return torch.sparse_csr_tensor(crow, col, values, self.shape).to_dense()
            # torch densifies unlisted entries as 0: shift the listed values
            # down by pad_value, densify, then shift everything back up.
            shifted = torch.sparse_csr_tensor(crow, col, values - pad_value, self.shape)
            return shifted.to_dense() + pad_value

        t = torch.from_numpy(self._numpy)
        return t.to(torch_dtype) if t.dtype != torch_dtype else t

    def tolist(self, pad_value: int = 0) -> List[Any]:
        return self.to_numpy(pad_value).tolist()


def _check_pad_value(pad_value: Any) -> int:
    """Validate `pad_value`: an integer (numpy integers accepted, bool rejected)."""
    if isinstance(pad_value, bool) or not isinstance(pad_value, (int, np.integer)):
        raise TypeError(
            f"pad_value must be an integer, got {type(pad_value).__name__}: {pad_value!r}"
        )
    return int(pad_value)


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
