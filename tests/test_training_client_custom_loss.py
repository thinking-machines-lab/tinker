from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import math
import threading
from concurrent.futures import Future
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from tinker import types
from tinker.lib.internal_client_holder import InternalClientHolder
from tinker.lib.public_interfaces.api_future import AwaitableConcurrentFuture
from tinker.lib.public_interfaces.service_client import ServiceClient
from tinker.lib.public_interfaces.training_client import TrainingClient


class _DummyHolder:
    def run_coroutine_threadsafe(self, coro):
        future: Future = Future()
        future.set_result(coro)
        return future

    def get_telemetry(self):
        return None

    def get_session_id(self):
        return "session"


class _CustomLossHolder:
    """Holder for create_custom_loss: a session, client config, its own event loop,
    the real custom-loss registry and _create_custom_loss, and a client that, like the
    real one, only works on that loop and records the request it was sent."""

    _register_custom_loss = InternalClientHolder._register_custom_loss
    _create_custom_loss = InternalClientHolder._create_custom_loss

    def __init__(self, session_id: str, server_side: bool):
        self.config = types.ClientConfigResponse(custom_loss_server_side=server_side)
        self.requests: list[types.CreateCustomLossRequest] = []
        self._session_id = session_id
        self._custom_losses = {}
        self._custom_loss_counter = 0
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

    def get_client_config(self):
        return self.config

    def get_session_id(self):
        return self._session_id

    def get_telemetry(self):
        return None

    def run_coroutine_threadsafe(self, coro):
        return AwaitableConcurrentFuture(asyncio.run_coroutine_threadsafe(coro, self.loop))

    async def execute_with_retries(self, send):
        return await send()

    @contextlib.contextmanager
    def aclient(self, pool_type):
        del pool_type
        assert asyncio.get_running_loop() is self.loop, "client used off the holder's loop"
        holder = self

        class _Service:
            async def create_custom_loss(self, *, request):
                holder.requests.append(request)
                return types.UntypedAPIFuture(
                    request_id=f"{request.session_id}:loss:{request.custom_loss_seq_id}:0"
                )

        yield SimpleNamespace(service=_Service())


def _service_client(session_id: str = "session", server_side: bool = False) -> ServiceClient:
    client = object.__new__(ServiceClient)
    client._lifecycle_lock = threading.Lock()
    client._closed = False
    client._session_holder = _CustomLossHolder(session_id, server_side)  # type: ignore[assignment]
    return client


class _FakeTrainingClient(TrainingClient):
    def __init__(self):
        self.holder = _DummyHolder()
        self.forward_calls: list[
            tuple[list[types.Datum], types.LossFnType, dict[str, float] | None]
        ] = []
        self.backward_calls: list[
            tuple[list[types.Datum], types.LossFnType, dict[str, float] | None]
        ] = []

    async def forward_async(
        self,
        data: list[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None = None,
    ):
        self.forward_calls.append((data, loss_fn, loss_fn_config))
        result = types.ForwardBackwardOutput(
            metrics={},
            loss_fn_output_type="target_token_logprobs",
            loss_fn_outputs=[
                {
                    "logprobs": types.TensorData(
                        data=[-3.0, -2.0, -1.0, 0.0],
                        dtype="float32",
                        shape=[2, 2],
                    ),
                }
            ],
        )
        future: Future = Future()
        future.set_result(result)
        return AwaitableConcurrentFuture(future)

    async def forward_backward_async(
        self,
        data: list[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None = None,
    ):
        self.backward_calls.append((data, loss_fn, loss_fn_config))
        result = types.ForwardBackwardOutput(
            metrics={"base:sum": 1.0, "loss:sum": 12.0},
            loss_fn_output_type="target_token_logprobs",
            loss_fn_outputs=[],
        )
        future: Future = Future()
        future.set_result(result)
        return AwaitableConcurrentFuture(future)


@pytest.mark.asyncio
async def test_forward_backward_custom_supports_2d_cross_entropy_targets():
    client = _FakeTrainingClient()
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            "target_tokens": [[101, 102], [201, 202]],
        },
    )

    assert datum.loss_fn_inputs["target_tokens"].shape == [2, 2]

    def custom_loss(
        data: list[types.Datum], logprobs_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del data
        logprobs = logprobs_list[0]
        assert logprobs.shape == (2, 2)
        probs = torch.softmax(logprobs[1], dim=-1)
        target_distribution = torch.tensor([0.0, 1.0], dtype=torch.float32)
        loss = torch.sum((probs - target_distribution) ** 2)
        return loss, {"selected_prob:mean": float(probs[1].detach())}

    result_future = await client.forward_backward_custom_async(
        [datum],
        custom_loss,
    )
    result = await result_future.result_async()

    assert client.forward_calls[0][1] == "cross_entropy"
    forward_datum = client.forward_calls[0][0][0]
    assert forward_datum.loss_fn_inputs["weights"].shape == [2, 2]

    assert client.backward_calls[0][1] == "cross_entropy"
    backward_datum = client.backward_calls[0][0][0]
    assert backward_datum.loss_fn_inputs["target_tokens"].shape == [2, 2]
    assert backward_datum.loss_fn_inputs["weights"].shape == [2, 2]
    assert "weights" not in datum.loss_fn_inputs
    assert result.metrics["selected_prob:mean"] > 0.0
    expected_loss = torch.sum(
        (torch.softmax(torch.tensor([-1.0, 0.0]), dim=-1) - torch.tensor([0.0, 1.0])) ** 2
    )
    assert result.metrics["loss:sum"] == pytest.approx(expected_loss.item())
    assert result.metrics["surrogate_loss:sum"] == 12.0


@pytest.mark.asyncio
async def test_forward_backward_custom_preserves_1d_cross_entropy_targets():
    client = _FakeTrainingClient()
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={"target_tokens": [101, 102]},
    )

    async def forward_async_1d(
        data: list[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None = None,
    ):
        client.forward_calls.append((data, loss_fn, loss_fn_config))
        result = types.ForwardBackwardOutput(
            metrics={},
            loss_fn_output_type="target_token_logprobs",
            loss_fn_outputs=[
                {
                    "logprobs": types.TensorData(
                        data=[-3.0, -1.0],
                        dtype="float32",
                        shape=[2],
                    ),
                }
            ],
        )
        future: Future = Future()
        future.set_result(result)
        return AwaitableConcurrentFuture(future)

    client.forward_async = forward_async_1d

    def custom_loss(
        data: list[types.Datum], logprobs_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del data
        logprobs = logprobs_list[0]
        assert logprobs.shape == (2,)
        loss = -logprobs[-1]
        return loss, {"selected_logprob:last": float(logprobs[-1].detach())}

    result_future = await client.forward_backward_custom_async(
        [datum],
        custom_loss,
    )
    result = await result_future.result_async()

    assert client.forward_calls[0][1] == "cross_entropy"
    forward_datum = client.forward_calls[0][0][0]
    assert forward_datum.loss_fn_inputs["weights"].shape == [2]

    assert client.backward_calls[0][1] == "cross_entropy"
    backward_datum = client.backward_calls[0][0][0]
    assert backward_datum.loss_fn_inputs["target_tokens"].shape == [2]
    assert backward_datum.loss_fn_inputs["weights"].shape == [2]
    torch.testing.assert_close(
        torch.tensor(backward_datum.loss_fn_inputs["weights"].data).reshape(
            backward_datum.loss_fn_inputs["weights"].shape
        ),
        torch.tensor([0.0, 1.0], dtype=torch.float32),
    )
    assert result.metrics["selected_logprob:last"] < 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("with_weights", [False, True])
@pytest.mark.parametrize("extra_key", ["target_distribution", "advantages"])
async def test_forward_backward_custom_filters_extra_loss_fn_inputs(
    with_weights: bool, extra_key: str
):
    client = _FakeTrainingClient()
    target_distribution = types.TensorData(
        data=[0.25, 0.75, 0.8, 0.2], dtype="float32", shape=[2, 2]
    )
    loss_fn_inputs = {
        "target_tokens": types.TensorData(data=[101, 102, 201, 202], dtype="int64", shape=[2, 2]),
        extra_key: target_distribution,
    }
    if with_weights:
        loss_fn_inputs["weights"] = types.TensorData(
            data=[1.0, 2.0, 3.0, 4.0], dtype="float32", shape=[2, 2]
        )
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2]),
        loss_fn_inputs=loss_fn_inputs,
    )

    def custom_loss(
        data: list[types.Datum], logprobs_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        assert data[0] is datum
        assert data[0].loss_fn_inputs[extra_key] is target_distribution
        distribution = data[0].loss_fn_inputs[extra_key].to_torch()
        return -(distribution * logprobs_list[0]).sum(), {}

    result_future = await client.forward_backward_custom_async([datum], custom_loss)
    result = await result_future.result_async()

    assert result.metrics["loss:sum"] == pytest.approx(3.05)
    assert datum.loss_fn_inputs == loss_fn_inputs
    forward_datum = client.forward_calls[0][0][0]
    backward_datum = client.backward_calls[0][0][0]
    for server_datum in (forward_datum, backward_datum):
        assert set(server_datum.loss_fn_inputs) == {"target_tokens", "weights"}
        assert server_datum.loss_fn_inputs["target_tokens"] is loss_fn_inputs["target_tokens"]
    torch.testing.assert_close(
        forward_datum.loss_fn_inputs["weights"].to_torch(),
        loss_fn_inputs["weights"].to_torch() if with_weights else torch.zeros((2, 2)),
    )
    torch.testing.assert_close(
        backward_datum.loss_fn_inputs["weights"].to_torch(),
        target_distribution.to_torch(),
    )


@pytest.mark.asyncio
async def test_forward_backward_custom_rejects_unsupported_loss_type_input():
    client = _FakeTrainingClient()
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={"target_tokens": [101, 102]},
    )

    def custom_loss(
        data: list[types.Datum], logprobs_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del data, logprobs_list
        return torch.tensor(0.0, requires_grad=True), {}

    with (
        pytest.warns(DeprecationWarning, match="loss_type_input"),
        pytest.raises(ValueError, match="Unsupported loss_type_input"),
    ):
        await client.forward_backward_custom_async(
            [datum],
            custom_loss,
            loss_type_input="logits",  # type: ignore[arg-type]
        )


def test_datum_rejects_ragged_nested_target_tokens():
    with pytest.raises(ValueError, match="ragged nested lists are not supported"):
        types.Datum(
            model_input=types.ModelInput.from_ints([1, 2]),
            loss_fn_inputs={"target_tokens": [[101, 102], [201]]},
        )


@pytest.mark.asyncio
async def test_forward_backward_custom_requires_target_tokens():
    client = _FakeTrainingClient()
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            "advantages": [1.0, 1.0],
        },
    )

    def custom_loss(
        data: list[types.Datum], logprobs_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del data, logprobs_list
        return torch.tensor(0.0, requires_grad=True), {}

    with pytest.raises(ValueError, match="target_tokens must be provided"):
        await client.forward_backward_custom_async(
            [datum],
            custom_loss,
        )
    assert not client.forward_calls
    assert not client.backward_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("with_weights", [False, True])
async def test_forward_backward_custom_preserves_provenance_spans(with_weights: bool):
    """Both rebuilds (the weights-less forward datum and the linear-loss
    backward datum) must carry the spans; dropping them silently would strip
    provenance from the surrogate passes."""
    client = _FakeTrainingClient()
    spans = [types.PromptProvenanceSpan(sequence_id="req:0", length=2)]
    loss_fn_inputs = {"target_tokens": [101, 102]}
    if with_weights:
        loss_fn_inputs["weights"] = [1, 1]
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2]),
        loss_fn_inputs=loss_fn_inputs,
        model_input_spans=spans,
        loss_fn_input_spans=spans,
    )

    def custom_loss(
        data: list[types.Datum], logprobs_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del data
        return logprobs_list[0].sum(), {}

    await client.forward_backward_custom_async([datum], custom_loss)

    ((forward_data, _, _),) = client.forward_calls
    ((linear_data, _, _),) = client.backward_calls
    for rebuilt in (*forward_data, *linear_data):
        assert rebuilt.model_input_spans == spans
        assert rebuilt.loss_fn_input_spans == spans


@pytest.mark.asyncio
async def test_forward_backward_custom_runs_a_custom_loss_like_the_callable():
    """A custom loss reads its inputs by name, runs per datum, and backpropagates the
    same gradient as the equivalent callable."""

    def soft_ce(logprobs, target_distribution):
        loss = -(target_distribution * logprobs).sum()
        return loss, {"custom/soft_ce:sum": loss, "custom/max_logprob:mean": logprobs.max()}

    def as_callable(data, logprobs_list):
        target = data[0].loss_fn_inputs["target_distribution"].to_torch()
        loss = -(target * logprobs_list[0]).sum()
        return loss, {"custom/soft_ce:sum": float(loss.detach()), "custom/max_logprob:mean": 0.0}

    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            "target_tokens": [[101, 102], [201, 202]],
            "target_distribution": torch.tensor([[0.25, 0.75], [0.5, 0.5]]),
        },
    )
    ranks = {"logprobs": 2, "target_distribution": 2}
    with pytest.raises(ValueError, match="input_ranks must give a rank for each parameter"):
        _service_client().create_custom_loss(soft_ce, {"logprobs": 2})
    service_client = _service_client()
    handle = await service_client.create_custom_loss(soft_ce, ranks).result_async()
    assert (handle.name, handle.input_names) == ("soft_ce", ("target_distribution",))

    # A session creates each loss once, and only its clients can use the handle.
    with pytest.raises(ValueError, match="already created 'soft_ce'"):
        await service_client.create_custom_loss(soft_ce, ranks).result_async()
    other_session = await _service_client("other").create_custom_loss(soft_ce, ranks).result_async()
    with pytest.raises(ValueError, match="only be used by clients of that session"):
        await _FakeTrainingClient().forward_backward_custom_async([datum], other_session)

    weights = []
    for loss_fn in (handle, as_callable):
        client = _FakeTrainingClient()
        result = await (await client.forward_backward_custom_async([datum], loss_fn)).result_async()
        weights.append(client.backward_calls[0][0][0].loss_fn_inputs["weights"].to_torch())
        if loss_fn is handle:
            assert result.metrics["custom/soft_ce:sum"] == pytest.approx(result.metrics["loss:sum"])
            assert result.metrics["custom/max_logprob:mean"] == 0.0
    torch.testing.assert_close(weights[0], weights[1])

    # An uploaded handle runs on the server: one forward_backward naming the loss,
    # with the datum's inputs intact and no client-side forward pass.
    server_side = _FakeTrainingClient()
    uploaded = dataclasses.replace(handle, loss_id="loss-1")
    await (await server_side.forward_backward_custom_async([datum], uploaded)).result_async()
    assert server_side.forward_calls == []
    assert server_side.backward_calls == [([datum], "custom", {"loss_id": "loss-1"})]

    missing = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={"target_tokens": [[101, 102], [201, 202]]},
    )
    with pytest.raises(ValueError, match="no loss_fn_inputs\\['target_distribution'\\]"):
        await _FakeTrainingClient().forward_backward_custom_async([missing], handle)
    flat = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            "target_tokens": [101, 201],
            "target_distribution": torch.tensor([0.25, 0.5]),
        },
    )
    with pytest.raises(ValueError, match="declared logprobs with rank 2"):
        await _FakeTrainingClient().forward_backward_custom_async([flat], handle)


def _branches(logprobs):
    return (logprobs.sum() if logprobs.sum() > 0 else -logprobs.sum()), {}


def _uses_numpy(logprobs):
    return torch.as_tensor(np.exp(logprobs)).sum(), {}


def _unreduced_metric(logprobs):
    return logprobs.sum(), {"total": logprobs.sum()}


def _float_metric(logprobs):
    return logprobs.sum(), {"custom/n:sum": 1.0}


def _uppercase_input(logprobs, Weights):
    return (logprobs * Weights).sum(), {}


def _no_logprobs(advantages):
    return advantages.sum(), {}


_CAPTURED = torch.tensor([0.5, 2.0])


def _captures_tensor(logprobs):
    return (logprobs * _CAPTURED).sum(), {}


def _calls_item(logprobs):
    return logprobs.sum(), {"custom/total:sum": logprobs.sum().item()}


def _constant_loss(logprobs):
    return 0.0, {}


def _calls_backward(logprobs):
    loss = logprobs.sum()
    loss.backward()
    return loss, {}


@pytest.mark.parametrize(
    ("fn", "error"),
    [
        (_branches, "cannot be traced by torch.fx"),
        (_uses_numpy, "cannot be traced by torch.fx"),
        (_unreduced_metric, "must be named custom/<name>:sum"),
        (_float_metric, "metric 'custom/n:sum' must be a torch.Tensor"),
        (_uppercase_input, "parameter 'Weights' must be a lowercase name"),
        (_no_logprobs, "must take logprobs as its first parameter"),
        (_captures_tensor, "captures a tensor"),
        (_calls_item, "calls item\\(\\), which leaves torch"),
        (_calls_backward, "calls backward"),
        (_constant_loss, "must return its loss as a torch.Tensor"),
    ],
)
def test_create_custom_loss_rejects_what_a_custom_loss_cannot_be(fn, error):
    with pytest.raises(ValueError, match=error):
        _service_client().create_custom_loss(fn, {"logprobs": 1})


@pytest.mark.asyncio
async def test_create_custom_loss_uploads_the_traced_graph_when_the_server_runs_it():
    """With the server-side flag on, the traced graph goes up as plain data and the
    handle carries the server's loss id; with it off, nothing is sent."""

    def shifted_ratio(logprobs, advantages):
        ratio = torch.exp(logprobs[..., 1:] - logprobs[..., :-1].detach()).clamp(max=math.inf)
        loss = -(ratio * advantages[..., 1:]).sum(dim=[0], dtype=torch.float32)
        return loss, {"custom/ratio:mean": ratio.mean()}

    async def create(client: ServiceClient):
        ranks = {"logprobs": 1, "advantages": 1}
        return await client.create_custom_loss(shifted_ratio, ranks).result_async()

    local = _service_client(server_side=False)
    assert (await create(local)).loss_id is None
    assert local.holder.requests == []  # type: ignore[attr-defined]

    client = _service_client(server_side=True)

    class _FakeFuture:
        def __init__(self, model_cls, holder, untyped_future, **kwargs):
            del holder, kwargs
            self.response = model_cls(loss_id=f"loss/{untyped_future.request_id}")

        async def result_async(self):
            return self.response

    with mock.patch("tinker.lib.public_interfaces.service_client._APIFuture", _FakeFuture):
        handle = await create(client)

    assert handle.loss_id == "loss/session:loss:0:0"
    (request,) = client.holder.requests  # type: ignore[attr-defined]
    assert request.name == "shifted_ratio"
    assert [(x.name, x.rank) for x in request.inputs] == [("logprobs", 1), ("advantages", 1)]
    graph = json.loads(request.model_dump_json())["graph"]
    assert graph["format"] == 1
    nodes = {node["name"]: node for node in graph["nodes"]}
    assert nodes["logprobs"] == {
        "op": "placeholder",
        "name": "logprobs",
        "target": "logprobs",
        "args": [],
        "kwargs": {},
    }
    first_slice = next(n for n in graph["nodes"] if n["target"] == "operator.getitem")
    assert first_slice["args"] == [
        {"ref": "logprobs"},
        {"tuple": [{"ellipsis": True}, {"slice": [1, None, None]}]},
    ]
    clamp = next(n for n in graph["nodes"] if n["target"] == "clamp")
    assert clamp["kwargs"] == {"max": {"float": "inf"}}
    total = next(n for n in graph["nodes"] if n["target"] == "sum")
    assert total["kwargs"] == {"dim": [0], "dtype": {"dtype": "float32"}}
    assert nodes["output"]["args"] == [
        {"tuple": [{"ref": "neg"}, {"dict": {"custom/ratio:mean": {"ref": "mean"}}}]}
    ]
