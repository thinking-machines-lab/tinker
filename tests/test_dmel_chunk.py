"""Tests for ``DmelChunk.length`` as public SDK behavior."""

from __future__ import annotations

import pytest

from tinker import types

from .test_tensor_container import encode_tensor_container_header


def test_length_reads_shape0() -> None:
    """DmelChunk.length delegates to the shape reader at axis 0.

    Exhaustive header/varint edge cases live in ``test_tensor_container.py``;
    here we only pin the wiring from the public type to the reader.
    """
    dmel = encode_tensor_container_header((130, 80))
    assert types.DmelChunk(dmel=dmel).length == 130


def test_length_matches_actual_dmel_serializer() -> None:
    """Ground-truth check against the real ``TensorContainer`` serializer.

    Skips in the public SDK env (no torch / tml.tensor_utils); runs in-monorepo
    where the rust ext is built.
    """
    torch = pytest.importorskip("torch")
    tensor_utils = pytest.importorskip("tml.tensor_utils")
    TensorContainer = tensor_utils.TensorContainer

    for num_tokens in (1, 3, 130, 5000):
        tensor = torch.randint(0, 256, (num_tokens, 80), dtype=torch.uint8)
        dmel = TensorContainer.to_fast_containers(tensor)
        assert types.DmelChunk(dmel=dmel).length == num_tokens

    # Constant tensor takes the compact CONSTANT_TENSOR_MARKER path, which
    # stores no payload — length must still recover shape[0].
    constant = torch.full((37, 80), 3, dtype=torch.uint8)
    dmel = TensorContainer.to_fast_containers(constant)
    assert types.DmelChunk(dmel=dmel).length == 37
