"""TensorContainer header readers used by public SDK types."""

from __future__ import annotations

_CONSTANT_TENSOR_MARKER = 254
_MIN_DTYPE_BYTE = 0
_MAX_DTYPE_BYTE = 7


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise ValueError("Truncated TensorContainer header")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte & 0x80 == 0:
            return value, offset
        shift += 7
        if shift > 63:
            raise ValueError("TensorContainer varint is too long")


def tensor_container_shape(data: bytes) -> tuple[int, ...]:
    """Read a TensorContainer shape from bytes without deserializing payload."""
    if not data:
        raise ValueError("TensorContainer bytes are empty")

    offset = 0
    if data[offset] == _CONSTANT_TENSOR_MARKER:
        offset += 1
        if offset >= len(data):
            raise ValueError("Truncated TensorContainer header")

    dtype_byte = data[offset]
    if not (_MIN_DTYPE_BYTE <= dtype_byte <= _MAX_DTYPE_BYTE):
        raise ValueError(f"Unknown TensorContainer dtype byte: {dtype_byte}")
    offset += 1

    ndim, offset = _read_varint(data, offset)
    shape: list[int] = []
    for _ in range(ndim):
        dim, offset = _read_varint(data, offset)
        shape.append(dim)
    return tuple(shape)


def tensor_container_dim(data: bytes, dim: int) -> int:
    """Read one dimension from a TensorContainer shape."""
    shape = tensor_container_shape(data)
    try:
        return shape[dim]
    except IndexError as exc:
        raise ValueError(f"TensorContainer tensor has no dimension {dim}") from exc
