"""Conversion helpers for SDK requests → proto wire format.

Mirror of ``response_conv.py`` for the send path. Today only
``ForwardBackwardRequest`` is wired up; new request types follow the same
pattern (Pydantic chunk / tensor → public_pb messages, then SerializeToString
in the resource layer).
"""

from __future__ import annotations

import numpy as np

from tinker.proto import tinker_public_pb2 as public_pb
from tinker.types.encoded_text_chunk import EncodedTextChunk
from tinker.types.forward_backward_request import ForwardBackwardRequest
from tinker.types.image_chunk import ImageChunk
from tinker.types.model_input_chunk import ModelInputChunk
from tinker.types.tensor_data import TensorData

# Public TensorDtype → proto DType. Public wire collapses to {float32, int64};
# bfloat16/int32 are not exposed to SDK users, so we don't need to encode them.
_TENSOR_DTYPE_TO_PROTO: dict[str, public_pb.DType.ValueType] = {
    "float32": public_pb.DTYPE_FLOAT32,
    "int64": public_pb.DTYPE_INT64,
}

_TENSOR_DTYPE_TO_NUMPY: dict[str, np.dtype] = {
    "float32": np.dtype(np.float32),
    "int64": np.dtype(np.int64),
}


def _tensor_data_to_proto(td: TensorData) -> public_pb.Tensor:
    """Encode a TensorData as a public proto Tensor.

    Dense path: contiguous bytes in the declared dtype. Sparse CSR path:
    values bytes (declared dtype) + crow/col indices as int64 bytes,
    matching the server-side codec.
    """
    msg = public_pb.Tensor()
    proto_dtype = _TENSOR_DTYPE_TO_PROTO.get(td.dtype)
    np_dtype = _TENSOR_DTYPE_TO_NUMPY.get(td.dtype)
    if proto_dtype is None or np_dtype is None:
        raise ValueError(f"Unsupported TensorData dtype for proto write: {td.dtype}")
    msg.dtype = proto_dtype

    if td.shape is not None:
        msg.shape.extend(td.shape)

    arr = td._numpy if td._numpy.dtype == np_dtype else td._numpy.astype(np_dtype)
    if td.sparse_crow_indices is not None:
        if td.sparse_col_indices is None:
            raise ValueError(
                "sparse_col_indices required with sparse_crow_indices"
            )
        msg.sparse_csr.values = arr.tobytes()
        msg.sparse_csr.crow_indices = np.asarray(
            td.sparse_crow_indices, dtype=np.int64
        ).tobytes()
        msg.sparse_csr.col_indices = np.asarray(
            td.sparse_col_indices, dtype=np.int64
        ).tobytes()
    else:
        msg.dense = arr.tobytes()

    return msg


def _write_chunk(msg: public_pb.Chunk, chunk: ModelInputChunk) -> None:
    """Encode a ModelInputChunk into ``msg`` (a Chunk oneof) in place.

    EncodedTextChunk: tokens packed as int32 bytes. ImageChunk: raw bytes
    pass through; the server uploads + computes width/height/tokens.
    """
    if isinstance(chunk, EncodedTextChunk):
        msg.encoded_text.tokens = np.asarray(chunk.tokens, dtype=np.int32).tobytes()
        return
    if isinstance(chunk, ImageChunk):
        msg.image.data = chunk.data
        msg.image.format = chunk.format
        if chunk.expected_tokens is not None:
            msg.image.expected_tokens = chunk.expected_tokens
        return
    raise ValueError(f"Unsupported model input chunk type: {type(chunk).__name__}")


def forward_backward_request_to_proto(
    request: ForwardBackwardRequest,
) -> public_pb.ForwardBackwardRequest:
    """Encode a ``ForwardBackwardRequest`` as the public proto."""
    msg = public_pb.ForwardBackwardRequest()
    msg.model_id = request.model_id
    # seq_id is non-optional on the public proto. The SDK always sets it
    # (training_client uses ``request_id + 1`` so seq_id ≥ 1); a caller-
    # supplied None decodes server-side as 0 and trips the seq_id range check.
    msg.seq_id = request.seq_id if request.seq_id is not None else 0

    msg.loss_fn = request.forward_backward_input.loss_fn
    if request.forward_backward_input.loss_fn_config is not None:
        for k, v in request.forward_backward_input.loss_fn_config.items():
            msg.loss_fn_config[k] = float(v)

    for datum in request.forward_backward_input.data:
        datum_msg = msg.data.add()
        for chunk in datum.model_input.chunks:
            _write_chunk(datum_msg.model_input.add(), chunk)
        for name, td in datum.loss_fn_inputs.items():
            datum_msg.loss_fn_inputs[name].CopyFrom(_tensor_data_to_proto(td))

    return msg
