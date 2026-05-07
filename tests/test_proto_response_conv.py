"""SDK-side tests for proto response deserialization.

The bulk per-datum decode is property-tested with hypothesis (varying dtype ×
leading dim × trailing shape × n_datums); edge cases that exercise specific
code paths (empty record, num_datums-only record, dispatch) stay hand-crafted.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn

from tinker.proto import tinker_public_pb2 as public_pb
from tinker.proto.response_conv import (
    PROTO_SUPPORTED_TYPES,
    deserialize_forward_backward_output,
    deserialize_proto_response,
)
from tinker.types.forward_backward_output import ForwardBackwardOutput


def _build_fwdbwd_proto(
    type_tag: str,
    *,
    metrics: dict[str, float] | None = None,
    fields: dict[str, tuple[np.ndarray, list[int], int]] | None = None,
) -> bytes:
    """Build a proto ForwardBackwardOutput fixture.

    fields: name -> (concatenated data, per-datum byte sizes, dtype enum)
    """
    msg = public_pb.ForwardBackwardOutput()
    msg.loss_fn_output_type = type_tag
    for k, v in (metrics or {}).items():
        msg.metrics[k] = v
    if fields:
        record = msg.loss_fn_outputs.add()
        record.type_tag = type_tag
        # Per-field offsets encode the datum boundaries; num_datums carries the
        # same count at the record level so the SDK can recover it when every
        # field is stripped by a server-side filter (drop_fwdbwd_logprobs).
        record.num_datums = len(next(iter(fields.values()))[1])
        for fname, (arr, sizes, dtype_enum) in fields.items():
            bt = record.fields[fname]
            bt.data = arr.tobytes()
            offsets = np.cumsum([0] + sizes, dtype=np.int64)
            bt.offsets = offsets.tobytes()
            bt.dtype = dtype_enum
    return msg.SerializeToString()


def test_forward_backward_output_in_proto_supported_types() -> None:
    assert ForwardBackwardOutput in PROTO_SUPPORTED_TYPES


# ---------------------------------------------------------------------------
# Hypothesis strategies for per-datum BatchedTensor reads
# ---------------------------------------------------------------------------


_FINITE_FLOAT32 = st.floats(allow_nan=False, allow_infinity=False, allow_subnormal=False, width=32)


@st.composite
def _datum_field(
    draw: DrawFn,
) -> tuple[int, list[int], list[np.ndarray], str]:
    """Yield (dtype_enum, trailing_shape, per_datum_arrays, public_dtype_str).

    public_dtype_str is the TensorDtype label the SDK assigns after collapsing
    int32→int64 and bfloat16→float32 on the public wire.
    """
    dtype_choice = draw(st.sampled_from(["float32", "int64", "int32", "bfloat16"]))
    np_dtype: np.dtype
    dtype_enum: int
    public_dtype: str
    if dtype_choice == "float32":
        np_dtype = np.dtype(np.float32)
        dtype_enum = public_pb.DTYPE_FLOAT32
        public_dtype = "float32"
    elif dtype_choice == "int64":
        np_dtype = np.dtype(np.int64)
        dtype_enum = public_pb.DTYPE_INT64
        public_dtype = "int64"
    elif dtype_choice == "int32":
        np_dtype = np.dtype(np.int32)
        dtype_enum = public_pb.DTYPE_INT32
        public_dtype = "int64"  # int32 collapses to int64 on the public wire
    else:  # bfloat16: stored as uint16 bytes
        np_dtype = np.dtype(np.uint16)
        dtype_enum = public_pb.DTYPE_BFLOAT16
        public_dtype = "float32"  # bfloat16 widens to float32 on the public wire

    trailing_len = draw(st.integers(min_value=0, max_value=1))
    trailing = draw(
        st.lists(
            st.integers(min_value=1, max_value=3),
            min_size=trailing_len,
            max_size=trailing_len,
        )
    )
    n_datums = draw(st.integers(min_value=1, max_value=4))
    leading_dims = draw(
        st.lists(
            st.integers(min_value=0, max_value=5),
            min_size=n_datums,
            max_size=n_datums,
        )
    )
    arrays: list[np.ndarray] = []
    for leading in leading_dims:
        n = leading
        for d in trailing:
            n *= d
        if dtype_choice in ("float32",):
            data = draw(st.lists(_FINITE_FLOAT32, min_size=n, max_size=n))
        elif dtype_choice in ("int64", "int32"):
            data = draw(
                st.lists(
                    st.integers(min_value=-(2**31), max_value=2**31 - 1),
                    min_size=n,
                    max_size=n,
                )
            )
        else:  # bfloat16: random uint16 bytes
            data = draw(
                st.lists(
                    st.integers(min_value=0, max_value=2**16 - 1),
                    min_size=n,
                    max_size=n,
                )
            )
        arrays.append(np.array(data, dtype=np_dtype).reshape((leading, *trailing)))
    return dtype_enum, trailing, arrays, public_dtype


@given(_datum_field())
def test_deserialize_recovers_per_datum_shape_and_dtype(
    field: tuple[int, list[int], list[np.ndarray], str],
) -> None:
    """``deserialize_forward_backward_output`` reconstructs N TensorData per
    datum with matching dtype, shape, and values across the dtype × shape
    space."""
    dtype_enum, trailing, arrays, public_dtype = field
    msg = public_pb.ForwardBackwardOutput()
    msg.loss_fn_output_type = "TorchLossReturn"
    record = msg.loss_fn_outputs.add()
    record.type_tag = "TorchLossReturn"
    record.num_datums = len(arrays)
    bt = record.fields["logprobs"]
    bt.data = np.concatenate(arrays, axis=0).tobytes() if arrays else b""
    sizes = [a.size * a.itemsize for a in arrays]
    bt.offsets = np.cumsum([0] + sizes, dtype=np.int64).tobytes()
    bt.dtype = dtype_enum
    bt.trailing_shape.extend(trailing)

    result = deserialize_forward_backward_output(msg.SerializeToString())
    assert len(result.loss_fn_outputs) == len(arrays)
    for orig, datum in zip(arrays, result.loss_fn_outputs, strict=True):
        td = datum["logprobs"]
        assert td.dtype == public_dtype
        assert td.shape == list(orig.shape)
        # Compare values; for bfloat16 the SDK widens uint16 bytes to float32
        # via the upper-16-bits trick — bit-exact for finite bf16 values.
        if dtype_enum == public_pb.DTYPE_BFLOAT16:
            expected = (orig.astype(np.uint32) << 16).view(np.float32).reshape(orig.shape)
            np.testing.assert_array_equal(np.array(td.data).reshape(td.shape), expected)
        else:
            np.testing.assert_array_equal(np.array(td.data).reshape(td.shape), orig)


# ---------------------------------------------------------------------------
# Hand-crafted edge cases
# ---------------------------------------------------------------------------


def test_deserialize_empty_loss_fn_outputs() -> None:
    proto_bytes = _build_fwdbwd_proto("", metrics={"loss:sum": 0.0})
    result = deserialize_forward_backward_output(proto_bytes)
    assert result.loss_fn_outputs == []
    # Fallback class name when no records are present (matches the JSON shape).
    assert result.loss_fn_output_type == "ArrayRecord"
    assert result.metrics == {"loss:sum": 0.0}


def test_deserialize_proto_response_dispatches_to_fwd_bwd() -> None:
    proto_bytes = _build_fwdbwd_proto("TorchLossReturn", metrics={"k": 1.0})
    result = deserialize_proto_response(proto_bytes, ForwardBackwardOutput)
    assert isinstance(result, ForwardBackwardOutput)
    assert result.metrics == {"k": 1.0}


def test_deserialize_num_datums_authoritative_when_all_fields_stripped() -> None:
    # Server-side drop_fwdbwd_logprobs can strip every field on a single-field
    # ArrayRecord (e.g. TorchLossReturn); ArrayRecord.num_datums preserves the
    # datum count so the SDK still emits N empty dicts (matching the JSON path).
    msg = public_pb.ForwardBackwardOutput()
    msg.loss_fn_output_type = "TorchLossReturn"
    record = msg.loss_fn_outputs.add()
    record.type_tag = "TorchLossReturn"
    record.num_datums = 3  # fields stripped upstream but count survives

    result = deserialize_forward_backward_output(msg.SerializeToString())
    assert len(result.loss_fn_outputs) == 3
    for datum in result.loss_fn_outputs:
        assert datum == {}
