"""Tests for lightweight TensorContainer header readers.

These run in the public SDK env, which does not require ``torch`` or
``tml.tensor_utils``. The bytes mirror the header format owned by
``third_party/tml-renderers/tml-tensor-utils/src/serialization.rs``.
"""

from __future__ import annotations

import pytest

from tinker.types._tensor_container import tensor_container_dim, tensor_container_shape

_CONSTANT_TENSOR_MARKER = 254


def _varint(value: int) -> bytes:
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def encode_tensor_container_header(
    shape: tuple[int, ...],
    *,
    dtype_byte: int = 4,  # uint8
    constant: bool = False,
) -> bytes:
    """Mirror ``serialization.rs`` far enough to exercise header readers."""
    out = bytearray()
    if constant:
        out.append(_CONSTANT_TENSOR_MARKER)
    out.append(dtype_byte)
    out += _varint(len(shape))
    for dim in shape:
        out += _varint(dim)
    out += b"\x00\x00\x00\x00"
    return bytes(out)


@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 80),
        (127, 80),
        (128, 80),
        (16_383, 80),
        (16_384, 80),
        (1_000_000, 80),
        (7, 80, 4),
    ],
)
def test_shape_decodes_header(shape: tuple[int, ...], constant: bool) -> None:
    encoded = encode_tensor_container_header(shape, constant=constant)

    assert tensor_container_shape(encoded) == shape
    for dim, expected in enumerate(shape):
        assert tensor_container_dim(encoded, dim) == expected


@pytest.mark.parametrize(
    ("encoded", "match"),
    [
        (b"", "empty"),
        (b"\x04", "Truncated"),
        (b"\x04\x02", "Truncated"),
        (b"\x04\x02\x80", "Truncated"),
        (b"\xfe", "Truncated"),
        (b"\x08\x02\x01\x50", "Unknown TensorContainer dtype byte"),
        (b"\x04" + b"\x80" * 10, "varint is too long"),
    ],
)
def test_shape_rejects_invalid_headers(encoded: bytes, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        tensor_container_shape(encoded)


@pytest.mark.parametrize("dim", [2, 3])
def test_dim_rejects_missing_dim(dim: int) -> None:
    with pytest.raises(ValueError, match=f"no dimension {dim}"):
        tensor_container_dim(encode_tensor_container_header((7, 80)), dim)
