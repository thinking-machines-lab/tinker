"""Provenance spans on the SDK: types, with_provenance, wire conversion, rejections."""

from __future__ import annotations

import pytest

from tinker import types as tinker_types
from tinker.proto.request_conv import forward_backward_request_to_proto
from tinker.types import PromptProvenanceSpan, SampledProvenanceSpan


def _datum(*, model_input_spans=None, loss_fn_input_spans=None) -> tinker_types.Datum:
    return tinker_types.Datum(
        model_input=tinker_types.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": tinker_types.TensorData(data=[2, 3, 4], dtype="int64", shape=[3]),
            "weights": tinker_types.TensorData(data=[1.0, 1.0, 1.0], dtype="float32", shape=[3]),
        },
        model_input_spans=model_input_spans,
        loss_fn_input_spans=loss_fn_input_spans,
    )


def test_span_type_validation():
    with pytest.raises(ValueError, match="length must be >= 1"):
        PromptProvenanceSpan(sequence_id="req:0", length=0)
    with pytest.raises(ValueError, match="sequence_id must be non-empty"):
        PromptProvenanceSpan(
            sequence_id="",
            length=1,
        )
    with pytest.raises(ValueError, match="offset must be >= 0"):
        PromptProvenanceSpan(
            sequence_id="req:0",
            offset=-1,
            length=1,
        )
    with pytest.raises(ValueError, match="length must be >= 1"):
        SampledProvenanceSpan(
            sequence_id="req:0",
            length=0,
        )
    with pytest.raises(ValueError, match="offset must be >= 0"):
        SampledProvenanceSpan(
            sequence_id="req:0",
            offset=-1,
            length=1,
        )
    # Offsets are local to the arm's type; 0 (the default) is the
    # generation start.
    assert (
        SampledProvenanceSpan(
            sequence_id="req:0",
            length=1,
        ).offset
        == 0
    )


def test_datum_rejects_foreign_span_types():
    with pytest.raises(TypeError, match="must be PromptProvenanceSpan or SampledProvenanceSpan"):
        _datum(loss_fn_input_spans=[("prompt", 3)])


def test_with_provenance_requires_both_lists():
    base = _datum()
    with pytest.raises(TypeError):
        base.with_provenance(  # type: ignore[call-arg]
            model_input=[
                PromptProvenanceSpan(
                    sequence_id="req:0",
                    length=3,
                )
            ]
        )
    with pytest.raises(TypeError):
        base.with_provenance(  # type: ignore[call-arg]
            loss_fn_inputs=[
                SampledProvenanceSpan(
                    sequence_id="req:0",
                    length=3,
                )
            ]
        )


def test_with_provenance_validates_tiling_at_call_site():
    base = _datum()
    loss_spans = [
        PromptProvenanceSpan(
            sequence_id="req:0",
            offset=1,
            length=1,
        ),
        SampledProvenanceSpan(
            sequence_id="req:0",
            length=2,
        ),
    ]
    with pytest.raises(ValueError, match="cover 2 positions but the field has 3"):
        base.with_provenance(
            model_input=[
                PromptProvenanceSpan(
                    sequence_id="req:0",
                    length=2,
                )
            ],
            loss_fn_inputs=loss_spans,
        )
    with pytest.raises(ValueError, match="cover 5 positions but the field has 3"):
        base.with_provenance(
            model_input=[
                PromptProvenanceSpan(
                    sequence_id="req:0",
                    length=3,
                )
            ],
            loss_fn_inputs=[
                PromptProvenanceSpan(
                    sequence_id="req:0",
                    offset=1,
                    length=2,
                ),
                SampledProvenanceSpan(
                    sequence_id="req:0",
                    length=3,
                ),
            ],
        )


def test_with_provenance_returns_annotated_copy():
    base = _datum()
    datum = base.with_provenance(
        model_input=[
            PromptProvenanceSpan(
                sequence_id="req:0",
                length=3,
            )
        ],
        loss_fn_inputs=[
            PromptProvenanceSpan(
                sequence_id="req:0",
                offset=1,
                length=1,
            ),
            SampledProvenanceSpan(
                sequence_id="req:0",
                length=2,
            ),
        ],
    )
    assert base.model_input_spans is None and base.loss_fn_input_spans is None
    assert datum.model_input_spans is not None and len(datum.model_input_spans) == 1
    assert datum.loss_fn_input_spans is not None and len(datum.loss_fn_input_spans) == 2


def test_request_conv_writes_both_lists():
    datum = _datum().with_provenance(
        model_input=[
            PromptProvenanceSpan(
                sequence_id="req:0",
                length=3,
            )
        ],
        loss_fn_inputs=[
            PromptProvenanceSpan(
                sequence_id="req:0",
                offset=1,
                length=1,
            ),
            SampledProvenanceSpan(
                sequence_id="req:0",
                length=2,
            ),
        ],
    )
    request = tinker_types.ForwardBackwardRequest(
        model_id="model",
        seq_id=1,
        forward_backward_input=tinker_types.ForwardBackwardInput(
            data=[datum], loss_fn="cross_entropy"
        ),
    )
    msg = forward_backward_request_to_proto(request)
    input_spans = msg.data[0].model_input_spans
    assert len(input_spans) == 1
    assert input_spans[0].WhichOneof("span") == "prompt_tokens"
    assert input_spans[0].prompt_tokens.length == 3
    assert input_spans[0].prompt_tokens.sequence_id == "req:0"
    assert input_spans[0].prompt_tokens.offset == 0

    loss_spans = msg.data[0].loss_fn_input_spans
    assert len(loss_spans) == 2
    assert loss_spans[0].WhichOneof("span") == "prompt_tokens"
    assert loss_spans[0].prompt_tokens.offset == 1
    assert loss_spans[1].WhichOneof("span") == "sampled_tokens"
    assert loss_spans[1].sampled_tokens.length == 2
    assert loss_spans[1].sampled_tokens.offset == 0


def test_request_conv_omits_unset_spans():
    request = tinker_types.ForwardBackwardRequest(
        model_id="model",
        seq_id=1,
        forward_backward_input=tinker_types.ForwardBackwardInput(
            data=[_datum()], loss_fn="cross_entropy"
        ),
    )
    msg = forward_backward_request_to_proto(request)
    assert len(msg.data[0].model_input_spans) == 0
    assert len(msg.data[0].loss_fn_input_spans) == 0
