"""Conversion helpers for proto responses to SDK types.

Deserializes proto wire format into SDK types (SampleResponse,
ForwardBackwardOutput, etc.). SampleResponse's proto path populates numpy
arrays directly. ForwardBackwardOutput's proto path reconstructs the same
Pydantic shape the JSON path produces (wire-speed win, no public API
change).
"""

from __future__ import annotations

from typing import cast

import numpy as np

from tinker.proto import tinker_public_pb2 as public_pb
from tinker.types.forward_backward_output import ForwardBackwardOutput
from tinker.types.sample_response import SampleResponse
from tinker.types.sampled_sequence import SampledSequence
from tinker.types.stop_reason import StopReason
from tinker.types.tensor_data import TensorData
from tinker.types.tensor_dtype import TensorDtype
from tinker.types.topk_prompt_logprobs import TopkPromptLogprobs

# Set of model classes that support proto deserialization.
# Used by api_future_impl to decide whether to send Accept: application/x-protobuf.
PROTO_SUPPORTED_TYPES: set[type] = {SampleResponse, ForwardBackwardOutput}

# Proto enum -> SDK string mapping
_STOP_REASON_TO_STR: dict[int, StopReason] = {
    public_pb.STOP_REASON_STOP: "stop",
    public_pb.STOP_REASON_LENGTH: "length",
}

# Proto DType enum -> numpy dtype for frombuffer reads. bfloat16 arrives as
# uint16 bytes and is cast to float32 before being placed in TensorData (which
# only supports float32 | int64 on the public wire).
_PROTO_DTYPE_TO_NUMPY: dict[int, np.dtype] = {
    public_pb.DTYPE_FLOAT32: np.dtype(np.float32),
    public_pb.DTYPE_INT64: np.dtype(np.int64),
    public_pb.DTYPE_INT32: np.dtype(np.int32),
    public_pb.DTYPE_BFLOAT16: np.dtype(np.uint16),
}

# Proto DType -> public TensorDtype. Integer widths (int32, int64) collapse to
# "int64" and bfloat16 collapses to "float32" to maintain compatibility with
# the JSON response shape (TensorDtype is Literal["float32", "int64"]).
_PROTO_DTYPE_TO_TENSOR_DTYPE: dict[int, TensorDtype] = {
    public_pb.DTYPE_FLOAT32: "float32",
    public_pb.DTYPE_BFLOAT16: "float32",
    public_pb.DTYPE_INT64: "int64",
    public_pb.DTYPE_INT32: "int64",
}


def deserialize_sample_response(proto_bytes: bytes) -> SampleResponse:
    """Deserialize proto bytes into a SampleResponse.

    Populates numpy arrays directly — no list conversion until the user
    accesses legacy list properties (tokens, logprobs, prompt_logprobs, etc.).
    """
    proto = public_pb.SampleResponse()
    proto.ParseFromString(proto_bytes)

    sequences = []
    for seq in proto.sequences:
        stop_reason = _STOP_REASON_TO_STR.get(seq.stop_reason)
        if stop_reason is None:
            raise ValueError(
                f"Unknown stop_reason enum value {seq.stop_reason} in proto SampleResponse"
            )
        tokens_np = np.frombuffer(seq.tokens, dtype=np.int32).copy()
        logprobs_np = (
            np.frombuffer(seq.logprobs, dtype=np.float32).copy() if seq.logprobs else None
        )
        sequences.append(
            SampledSequence(
                stop_reason=stop_reason,
                tokens_np=tokens_np,
                logprobs_np=logprobs_np,
            )
        )

    prompt_logprobs_np: np.ndarray | None = None
    if proto.prompt_logprobs:
        prompt_logprobs_np = np.frombuffer(proto.prompt_logprobs, dtype=np.float32).copy()

    topk_prompt_logprobs_np: TopkPromptLogprobs | None = None
    if proto.HasField("topk_prompt_logprobs"):
        topk = proto.topk_prompt_logprobs
        n, k = topk.prompt_length, topk.k
        if n > 0 and k > 0:
            topk_prompt_logprobs_np = TopkPromptLogprobs(
                token_ids=np.ndarray((n, k), dtype=np.int32, buffer=topk.token_ids).copy(),
                logprobs=np.ndarray((n, k), dtype=np.float32, buffer=topk.logprobs).copy(),
            )

    return SampleResponse(
        sequences=sequences,
        prompt_logprobs_np=prompt_logprobs_np,
        topk_prompt_logprobs_np=topk_prompt_logprobs_np,
    )


def _decode_batched_tensor_to_per_datum_lists(
    bt: public_pb.BatchedTensor,
) -> list[list[int] | list[float]]:
    """Slice a BatchedTensor into per-datum flat lists.

    Handles bfloat16 by reinterpreting the uint16 bytes as the upper half of
    float32 — lossless widening to maintain compatibility with the JSON
    response shape.
    """
    np_dtype = _PROTO_DTYPE_TO_NUMPY.get(bt.dtype)
    if np_dtype is None:
        raise ValueError(f"Unsupported proto DType value: {bt.dtype}")
    offsets = np.frombuffer(bt.offsets, dtype=np.int64)
    buf = np.frombuffer(bt.data, dtype=np_dtype)
    if bt.dtype == public_pb.DTYPE_BFLOAT16:
        # bfloat16 is the upper 16 bits of a float32. Shift into place and view.
        buf = (buf.astype(np.uint32) << 16).view(np.float32)

    itemsize = np_dtype.itemsize
    per_datum: list[list[int] | list[float]] = []
    for i in range(len(offsets) - 1):
        start = int(offsets[i]) // itemsize
        end = int(offsets[i + 1]) // itemsize
        per_datum.append(buf[start:end].tolist())
    return per_datum


def deserialize_forward_backward_output(proto_bytes: bytes) -> ForwardBackwardOutput:
    """Deserialize proto bytes into a ForwardBackwardOutput (Pydantic).

    Produces the same shape the JSON path emits so the public SDK API is
    wire-format-agnostic. Field order within each LossFnOutput follows the
    proto map iteration order; callers look up fields by name regardless.

    Supports any number of ArrayRecord chunks (v1 emits exactly one batched
    record; future chunked emission is transparent to this path).
    """
    proto = public_pb.ForwardBackwardOutput()
    proto.ParseFromString(proto_bytes)

    metrics = {k: float(v) for k, v in proto.metrics.items()}

    if not proto.loss_fn_outputs:
        return ForwardBackwardOutput(
            loss_fn_output_type=proto.loss_fn_output_type or "ArrayRecord",
            loss_fn_outputs=[],
            metrics=metrics,
        )

    # Bare class name — maintains compatibility with the JSON response shape.
    class_name = proto.loss_fn_output_type.rsplit(":", 1)[-1]

    # Collect per-datum lists + per-field metadata across every chunk. Field
    # names and shape/dtype are invariant across chunks (enforced by the
    # writer — same ArrayRecord subclass for all datums in one response).
    per_field_datum_lists: dict[str, list[list[int] | list[float]]] = {}
    per_field_meta: dict[str, tuple[TensorDtype, list[int]]] = {}
    for record in proto.loss_fn_outputs:
        for name, bt in record.fields.items():
            tensor_dtype = _PROTO_DTYPE_TO_TENSOR_DTYPE.get(bt.dtype)
            if tensor_dtype is None:
                raise ValueError(f"Unsupported proto DType on field {name}: {bt.dtype}")
            chunk_datums = _decode_batched_tensor_to_per_datum_lists(bt)
            per_field_datum_lists.setdefault(name, []).extend(chunk_datums)
            per_field_meta.setdefault(name, (tensor_dtype, list(bt.trailing_shape)))

    # Prefer ArrayRecord.num_datums (authoritative and survives server-side
    # filters like drop_fwdbwd_logprobs that may strip every field). Fall back
    # to the first-field count for encoded messages predating num_datums.
    num_datums = sum(r.num_datums for r in proto.loss_fn_outputs)
    if num_datums == 0 and per_field_datum_lists:
        num_datums = len(next(iter(per_field_datum_lists.values())))

    loss_fn_outputs: list[dict[str, TensorData]] = []
    for i in range(num_datums):
        datum: dict[str, TensorData] = {}
        for name, per_datum in per_field_datum_lists.items():
            tensor_dtype, trailing_shape = per_field_meta[name]
            data = per_datum[i]
            # Leading dim = total elements / product(trailing_shape).
            trailing_elems = 1
            for s in trailing_shape:
                trailing_elems *= s
            leading = len(data) // trailing_elems if trailing_elems else 0
            shape = [leading, *trailing_shape]
            datum[name] = TensorData(
                data=cast("list[int] | list[float]", data),
                dtype=tensor_dtype,
                shape=shape,
            )
        loss_fn_outputs.append(datum)

    return ForwardBackwardOutput(
        loss_fn_output_type=class_name,
        loss_fn_outputs=loss_fn_outputs,
        metrics=metrics,
    )


def deserialize_proto_response(proto_bytes: bytes, model_cls: type) -> object:
    """Deserialize a proto response based on the expected model class.

    Dispatches to the appropriate deserializer based on model_cls.
    Raises ValueError for unsupported types.
    """
    if model_cls is SampleResponse:
        return deserialize_sample_response(proto_bytes)
    if model_cls is ForwardBackwardOutput:
        return deserialize_forward_backward_output(proto_bytes)
    raise ValueError(f"Proto deserialization not supported for {model_cls}")
