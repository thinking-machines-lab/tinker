"""Tests for the dataclass→Pydantic write-path conversion in tinker.lib._pydantic_conv."""

from __future__ import annotations

from tinker import types
from tinker.lib._pydantic_conv import to_pydantic_request


def _make_request() -> types.ForwardBackwardRequest:
    return types.ForwardBackwardRequest(
        forward_backward_input=types.ForwardBackwardInput(
            data=[
                types.Datum(
                    model_input=types.ModelInput.from_ints([1, 2, 3]),
                    loss_fn_inputs={
                        "target_tokens": [4, 5, 6],
                        "weights": [1.0, 1.0, 1.0],
                    },
                )
            ],
            loss_fn="cross_entropy",
            loss_fn_config={"clip_low": 0.8},
        ),
        model_id="m-test",
        seq_id=42,
    )


def test_to_pydantic_request_round_trips_envelope() -> None:
    """``to_pydantic_request(...).model_dump(mode="json")`` preserves the
    envelope fields the server expects (model_id, seq_id, loss_fn,
    loss_fn_config) and the per-datum loss_fn_inputs payload."""
    request = _make_request()
    dumped = to_pydantic_request(request).model_dump(mode="json")

    assert dumped["model_id"] == "m-test"
    assert dumped["seq_id"] == 42
    fbi = dumped["forward_backward_input"]
    assert fbi["loss_fn"] == "cross_entropy"
    assert fbi["loss_fn_config"] == {"clip_low": 0.8}
    assert len(fbi["data"]) == 1
    datum = fbi["data"][0]
    assert datum["loss_fn_inputs"]["target_tokens"]["data"] == [4, 5, 6]
    assert datum["loss_fn_inputs"]["weights"]["data"] == [1.0, 1.0, 1.0]


def test_to_pydantic_request_for_forward_request() -> None:
    """``ForwardRequest`` (non-backward) routes through the same converter and
    produces a ``forward_input`` envelope (not ``forward_backward_input``)."""
    fwd_req = types.ForwardRequest(
        forward_input=types.ForwardBackwardInput(
            data=[
                types.Datum(
                    model_input=types.ModelInput.from_ints([1]),
                    loss_fn_inputs={"target_tokens": [2], "weights": [1.0]},
                )
            ],
            loss_fn="cross_entropy",
        ),
        model_id="m-test",
        seq_id=1,
    )
    dumped = to_pydantic_request(fwd_req).model_dump(mode="json")
    assert "forward_input" in dumped
    assert "forward_backward_input" not in dumped
    assert dumped["forward_input"]["loss_fn"] == "cross_entropy"
