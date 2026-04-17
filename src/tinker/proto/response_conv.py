"""Conversion helpers for proto responses to Pydantic models.

Deserializes proto wire format into SDK Pydantic types (SampleResponse, etc.).
"""

from __future__ import annotations

import numpy as np

from tinker.proto import tinker_public_pb2 as public_pb
from tinker.types.sample_response import SampleResponse
from tinker.types.sampled_sequence import SampledSequence
from tinker.types.stop_reason import StopReason

# Set of model classes that support proto deserialization.
# Used by api_future_impl to decide whether to send Accept: application/x-protobuf.
PROTO_SUPPORTED_TYPES: set[type] = {SampleResponse}

MASK_LOGPROB = -99999.0

# Proto enum -> SDK string mapping
_STOP_REASON_TO_STR: dict[int, StopReason] = {
    public_pb.STOP_REASON_STOP: "stop",
    public_pb.STOP_REASON_LENGTH: "length",
}


def deserialize_sample_response(proto_bytes: bytes) -> SampleResponse:
    """Deserialize proto bytes into a Pydantic SampleResponse."""
    proto = public_pb.SampleResponse()
    proto.ParseFromString(proto_bytes)

    sequences = []
    for seq in proto.sequences:
        stop_reason = _STOP_REASON_TO_STR.get(seq.stop_reason)
        if stop_reason is None:
            raise ValueError(
                f"Unknown stop_reason enum value {seq.stop_reason} in proto SampleResponse"
            )
        tokens = np.frombuffer(seq.tokens, dtype=np.int32).tolist()
        logprobs = np.frombuffer(seq.logprobs, dtype=np.float32).tolist() if seq.logprobs else None
        sequences.append(
            SampledSequence.model_construct(
                stop_reason=stop_reason,
                tokens=tokens,
                logprobs=logprobs,
            )
        )

    prompt_logprobs: list[float | None] | None = None
    if proto.prompt_logprobs:
        arr = np.frombuffer(proto.prompt_logprobs, dtype=np.float32)
        prompt_logprobs_list: list[float | None] = arr.tolist()
        for i in np.flatnonzero(np.isnan(arr)):
            prompt_logprobs_list[i] = None
        prompt_logprobs = prompt_logprobs_list

    topk_prompt_logprobs: list[list[tuple[int, float]] | None] | None = None
    if proto.HasField("topk_prompt_logprobs"):
        topk_prompt_logprobs = _topk_from_proto(proto.topk_prompt_logprobs)

    return SampleResponse.model_construct(
        sequences=sequences,
        prompt_logprobs=prompt_logprobs,
        topk_prompt_logprobs=topk_prompt_logprobs,
    )


def _topk_from_proto(
    topk: public_pb.TopkPromptLogprobs,
) -> list[list[tuple[int, float]] | None]:
    """Convert dense N×K TopkPromptLogprobs to Python list format."""
    n = topk.prompt_length
    k = topk.k

    if n == 0 or k == 0:
        return []

    token_ids = np.ndarray((n, k), dtype=np.int32, buffer=topk.token_ids)
    logprobs = np.ndarray((n, k), dtype=np.float32, buffer=topk.logprobs)

    # Single flat zip (faster than 32K per-row zips), then slice per row
    tid_flat = token_ids.ravel().tolist()
    lp_flat = logprobs.ravel().tolist()
    all_tuples = list(zip(tid_flat, lp_flat))

    mask_lp = MASK_LOGPROB
    result: list[list[tuple[int, float]] | None] = []
    for i in range(n):
        start = i * k
        # First-element sentinel check: if first entry is sentinel, whole row is None
        if tid_flat[start] == 0 and lp_flat[start] == mask_lp:
            result.append(None)
        else:
            end = start + k
            while end > start and tid_flat[end - 1] == 0 and lp_flat[end - 1] == mask_lp:
                end -= 1
            result.append(all_tuples[start:end])
    return result


def deserialize_proto_response(proto_bytes: bytes, model_cls: type) -> object:
    """Deserialize a proto response based on the expected model class.

    Dispatches to the appropriate deserializer based on model_cls.
    Raises ValueError for unsupported types.
    """
    if model_cls is SampleResponse:
        return deserialize_sample_response(proto_bytes)
    raise ValueError(f"Proto deserialization not supported for {model_cls}")
