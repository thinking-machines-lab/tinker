# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union, Iterable, Optional
from typing_extensions import Required, TypedDict

from .tensor_dtype import TensorDtype

__all__ = ["TensorDataParam"]


class TensorDataParam(TypedDict, total=False):
    data: Required[Union[Iterable[int], Iterable[float]]]
    """Flattened tensor data as array of numbers."""

    dtype: Required[TensorDtype]

    shape: Optional[Iterable[int]]
    """Optional.

    The shape of the tensor (see PyTorch tensor.shape). The shape of a
    one-dimensional list of length N is `(N,)`. Can usually be inferred if not
    provided, and is generally inferred as a 1D tensor.
    """
