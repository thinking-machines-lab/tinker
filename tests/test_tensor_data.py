from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from tinker.types.tensor_data import TensorData

# torch's sparse CSR support is in beta and warns on every conversion; the SDK
# test config promotes warnings to errors.
_ignore_torch_sparse_beta_warnings = pytest.mark.filterwarnings("ignore:Sparse:UserWarning")


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


@_ignore_torch_sparse_beta_warnings
def test_from_torch_sparse_with_pad_value_keeps_token_id_zero() -> None:
    # Pad with -1 so a top-k row containing token id 0 is preserved exactly.
    dense = torch.full((5, 3), -1, dtype=torch.int64)
    dense[4] = torch.tensor([0, 7, 9])
    td = TensorData.from_torch_sparse(dense, pad_value=-1)
    assert td.sparse_crow_indices == [0, 0, 0, 0, 0, 3]
    assert td.sparse_col_indices == [0, 1, 2]
    # Listed values are the true values, not offsets from the pad.
    assert td.data == [0, 7, 9]
    # The pad is not part of the TensorData: densifying needs it again.
    assert torch.equal(td.to_torch(pad_value=-1), dense)
    assert np.array_equal(td.to_numpy(pad_value=-1), dense.numpy())
    assert td.to_torch()[0].tolist() == [0, 0, 0]


@_ignore_torch_sparse_beta_warnings
def test_from_torch_sparse_with_pad_value_float32() -> None:
    dense = torch.full((4, 4), 2.0, dtype=torch.float32)
    dense[0, 0] = 0.0
    dense[3, 2] = -1.5
    td = TensorData.from_torch_sparse(dense, pad_value=2)
    assert td.data == [0.0, -1.5]
    torch.testing.assert_close(td.to_torch(pad_value=2), dense)


def test_from_torch_sparse_falls_back_to_dense_when_mostly_non_pad() -> None:
    # All zeros, but nothing equals the pad: sparsity must be judged against
    # pad_value, not against 0, or this would encode as an empty CSR.
    dense = torch.zeros(3, 4, dtype=torch.int64)
    td = TensorData.from_torch_sparse(dense, pad_value=-1)
    assert td.sparse_crow_indices is None
    assert torch.equal(td.to_torch(), dense)


@_ignore_torch_sparse_beta_warnings
def test_hand_built_sparse_tensor_data_densifies_with_the_given_pad() -> None:
    td = TensorData(
        data=[4, 6],
        dtype="int64",
        shape=[2, 2],
        sparse_crow_indices=[0, 1, 2],
        sparse_col_indices=[1, 0],
    )
    assert td.to_torch(pad_value=np.int64(9)).tolist() == [[9, 4], [6, 9]]
    assert td.tolist(pad_value=-1) == [[-1, 4], [6, -1]]
    assert td.tolist() == [[0, 4], [6, 0]]


@pytest.mark.parametrize("pad_value", [0.5, True, "0", np.float32(1)])
def test_pad_value_must_be_an_integer(pad_value: object) -> None:
    dense = torch.zeros(4, 4, dtype=torch.int64)
    with pytest.raises(TypeError, match="integer"):
        TensorData.from_torch_sparse(dense, pad_value=pad_value)  # type: ignore[arg-type]
    td = TensorData(
        data=[1], dtype="int64", shape=[1, 4], sparse_crow_indices=[0, 1], sparse_col_indices=[0]
    )
    with pytest.raises(TypeError, match="integer"):
        td.to_torch(pad_value=pad_value)  # type: ignore[arg-type]
