from __future__ import annotations

import warnings

import numpy as np
import torch

from tinker.types.tensor_data import TensorData


def test_init_copies_non_writable_numpy() -> None:
    # np.frombuffer over an immutable bytes buffer returns a read-only array.
    # Without an explicit copy, this flag propagates through TensorData and
    # triggers a UserWarning from torch.from_numpy in to_torch().
    arr = np.frombuffer(np.arange(8, dtype=np.float32).tobytes(), dtype=np.float32)
    assert not arr.flags.writeable

    td = TensorData(data=arr, dtype="float32", shape=[8])
    assert td._numpy.flags.writeable

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        t = td.to_torch()
    assert torch.equal(t, torch.arange(8, dtype=torch.float32))


def test_init_preserves_writable_numpy_without_copy() -> None:
    arr = np.arange(8, dtype=np.float32)
    td = TensorData(data=arr, dtype="float32", shape=[8])
    # Writable arrays of matching dtype should be stored by reference.
    assert td._numpy is arr
