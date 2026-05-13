"""Conversion between public dataclass types and internal Pydantic mirrors.

The public SDK API is `@dataclass(frozen=True)`; Pydantic mirrors live in
`tinker.types._pydantic_types/` and are used only at the JSON wire boundary:

- **Read path** (server → SDK, JSON): ``deserialize_json_response`` looks up
  the public type, deserializes via Pydantic ``model_validate``, then runs the
  registered converter to produce the public dataclass.
- **Write path** (SDK → server, JSON): callers convert the public request to
  Pydantic via ``to_pydantic_request`` so that ``model_dump(mode="json")``
  emits the legacy wire shape.

Non-migrated types fall through to ``model_validate`` directly on the read
path; no converter needed.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

import pydantic

from tinker.types._pydantic_types.datum import Datum as _PydanticDatum
from tinker.types._pydantic_types.forward_backward_input import (
    ForwardBackwardInput as _PydanticForwardBackwardInput,
)
from tinker.types._pydantic_types.forward_backward_output import (
    ForwardBackwardOutput as _PydanticForwardBackwardOutput,
)
from tinker.types._pydantic_types.forward_backward_request import (
    ForwardBackwardRequest as _PydanticForwardBackwardRequest,
)
from tinker.types._pydantic_types.forward_request import (
    ForwardRequest as _PydanticForwardRequest,
)
from tinker.types._pydantic_types.sample_response import SampleResponse as _PydanticSampleResponse
from tinker.types._pydantic_types.tensor_data import TensorData as _PydanticTensorData
from tinker.types.datum import Datum
from tinker.types.forward_backward_input import ForwardBackwardInput
from tinker.types.forward_backward_output import ForwardBackwardOutput
from tinker.types.forward_backward_request import ForwardBackwardRequest
from tinker.types.forward_request import ForwardRequest
from tinker.types.sample_response import SampleResponse
from tinker.types.sampled_sequence import SampledSequence
from tinker.types.tensor_data import TensorData

# Registry: public dataclass type -> (pydantic_cls, read_converter_fn)
_CONVERTERS: dict[type, tuple[type, Callable[[Any], Any]]] = {}


# ─── SampleResponse (read-only, already migrated) ───────────────────────────


def _convert_sample_response(pydantic_obj: _PydanticSampleResponse) -> SampleResponse:
    return SampleResponse(
        sequences=[
            SampledSequence(
                stop_reason=s.stop_reason,
                _tokens_list=s.tokens,
                _logprobs_list=s.logprobs,
            )
            for s in pydantic_obj.sequences
        ],
        _prompt_logprobs_list=pydantic_obj.prompt_logprobs,
        _topk_prompt_logprobs_list=pydantic_obj.topk_prompt_logprobs,
    )


_CONVERTERS[SampleResponse] = (_PydanticSampleResponse, _convert_sample_response)


# ─── TensorData (bidirectional) ─────────────────────────────────────────────


def _from_pydantic_tensor_data(p: _PydanticTensorData) -> TensorData:
    return TensorData(
        data=p.data,
        dtype=p.dtype,
        shape=p.shape,
        sparse_crow_indices=p.sparse_crow_indices,
        sparse_col_indices=p.sparse_col_indices,
    )


def _to_pydantic_tensor_data(td: TensorData) -> _PydanticTensorData:
    return _PydanticTensorData(
        data=td.data,
        dtype=td.dtype,
        shape=td.shape,
        sparse_crow_indices=td.sparse_crow_indices,
        sparse_col_indices=td.sparse_col_indices,
    )


# ─── ForwardBackwardOutput (read path) ──────────────────────────────────────


def _convert_forward_backward_output(
    p: _PydanticForwardBackwardOutput,
) -> ForwardBackwardOutput:
    loss_fn_outputs = [
        {k: _from_pydantic_tensor_data(v) for k, v in datum.items()} for datum in p.loss_fn_outputs
    ]
    return ForwardBackwardOutput(
        loss_fn_output_type=p.loss_fn_output_type,
        loss_fn_outputs=loss_fn_outputs,
        metrics=dict(p.metrics),
    )


_CONVERTERS[ForwardBackwardOutput] = (
    _PydanticForwardBackwardOutput,
    _convert_forward_backward_output,
)


# ─── Request side (write path: dataclass → Pydantic) ────────────────────────
#
# Used by ``resources/training.py`` to dump the request body as JSON via the
# Pydantic mirror.


def _to_pydantic_datum(d: Datum) -> _PydanticDatum:
    return _PydanticDatum.model_construct(
        model_input=d.model_input,
        loss_fn_inputs={k: _to_pydantic_tensor_data(v) for k, v in d.loss_fn_inputs.items()},
    )


def to_pydantic_input(fbi: ForwardBackwardInput) -> _PydanticForwardBackwardInput:
    """Convert a public ``ForwardBackwardInput`` to its Pydantic mirror.

    Public for cross-package compat tests that need ``model_dump(mode="json")``
    on just the input (without wrapping in a request).
    """
    return _PydanticForwardBackwardInput.model_construct(
        data=[_to_pydantic_datum(d) for d in fbi.data],
        loss_fn=fbi.loss_fn,
        loss_fn_config=fbi.loss_fn_config,
    )


def to_pydantic_request(
    request: ForwardBackwardRequest | ForwardRequest,
) -> _PydanticForwardBackwardRequest | _PydanticForwardRequest:
    """Convert a public request dataclass to its Pydantic mirror so callers
    can use ``model_dump(mode="json")`` for the JSON wire path."""
    if isinstance(request, ForwardBackwardRequest):
        return _PydanticForwardBackwardRequest.model_construct(
            forward_backward_input=to_pydantic_input(request.forward_backward_input),
            model_id=request.model_id,
            seq_id=request.seq_id,
        )
    return _PydanticForwardRequest.model_construct(
        forward_input=to_pydantic_input(request.forward_input),
        model_id=request.model_id,
        seq_id=request.seq_id,
    )


def deserialize_json_response(result_dict: dict[str, Any], model_cls: type) -> Any:
    """Deserialize a JSON response dict into the expected type.

    Migrated types are deserialized via their Pydantic counterpart, then
    converted to the public dataclass. Non-migrated types use Pydantic directly.
    """
    entry = _CONVERTERS.get(model_cls)
    if entry is not None:
        pydantic_cls, converter = entry
        pydantic_obj = pydantic_cls.model_validate(result_dict)
        return converter(pydantic_obj)
    if inspect.isclass(model_cls) and issubclass(model_cls, pydantic.BaseModel):
        return model_cls.model_validate(result_dict)
    return result_dict
