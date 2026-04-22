"""Conversion helpers for proto responses to SDK types.

Deserializes proto wire format into SDK types (SampleResponse, etc.).
Proto path populates numpy arrays directly — no list conversion.
"""

from __future__ import annotations

import numpy as np

from tinker.proto import tinker_public_pb2 as public_pb
from tinker.types.sample_response import SampleResponse
from tinker.types.sampled_sequence import SampledSequence
from tinker.types.stop_reason import StopReason
from tinker.types.topk_prompt_logprobs import TopkPromptLogprobs

# Set of model classes that support proto deserialization.
# Used by api_future_impl to decide whether to send Accept: application/x-protobuf.
PROTO_SUPPORTED_TYPES: set[type] = {SampleResponse}

# Proto enum -> SDK string mapping
_STOP_REASON_TO_STR: dict[int, StopReason] = {
    public_pb.STOP_REASON_STOP: "stop",
    public_pb.STOP_REASON_LENGTH: "length",
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


def deserialize_proto_response(proto_bytes: bytes, model_cls: type) -> object:
    """Deserialize a proto response based on the expected model class.

    Dispatches to the appropriate deserializer based on model_cls.
    Raises ValueError for unsupported types.
    """
    if model_cls is SampleResponse:
        return deserialize_sample_response(proto_bytes)
    raise ValueError(f"Proto deserialization not supported for {model_cls}")
