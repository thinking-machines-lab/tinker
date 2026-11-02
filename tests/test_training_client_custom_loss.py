from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import math
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from concurrent.futures import Future
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from tinker import types
from tinker.lib.api_future_impl import _APIFuture
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.internal_client_holder import InternalClientHolder
from tinker.lib.public_interfaces.api_future import AwaitableConcurrentFuture
from tinker.lib.public_interfaces.service_client import ServiceClient
from tinker.lib.public_interfaces.training_client import TrainingClient, _get_custom_loss_loop


class _DummyHolder:
    _client_config = types.ClientConfigResponse()

    def run_coroutine_threadsafe(self, coro):
        return AwaitableConcurrentFuture(
            asyncio.run_coroutine_threadsafe(coro, asyncio.get_running_loop())
        )

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
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

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
        super().__init__(_DummyHolder(), 0, "model")
        self.forward_calls: list[
            tuple[list[types.Datum], types.LossFnType, dict[str, float] | None]
        ] = []
        self.backward_calls: list[
            tuple[list[types.Datum], types.LossFnType, dict[str, float] | None]
        ] = []

    def _run_fwd_bwd(self, requests, loss_fn, loss_fn_config, *, forward_only):
        data = [datum for _, chunk, _ in requests for datum in chunk]
        if forward_only:
            return self.forward(data, loss_fn, loss_fn_config)
        return self.forward_backward(data, loss_fn, loss_fn_config)

    def forward(
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

    def forward_backward(
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

    def forward_1d(
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

    client.forward = forward_1d

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

    future = await client.forward_backward_custom_async([datum], custom_loss)
    await future.result_async()

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


class _PipelinedTrainingHolder(_CustomLossHolder):
    def __init__(self, parallel: bool, chunk_len: int, chunk_bytes: int) -> None:
        super().__init__("session", False)
        self._client_config = types.ClientConfigResponse(
            parallel_fwdbwd_chunks=parallel,
            fwdbwd_max_chunk_len=chunk_len,
            fwdbwd_max_chunk_bytes_count=chunk_bytes,
        )
        self.forward_started: Future[None] = Future()
        self.release_forward: Future[None] = Future()
        self.outputs: dict[str, types.ForwardBackwardOutput | types.OptimStepResponse] = {}

    @contextlib.contextmanager
    def aclient(self, pool_type: ClientConnectionPoolType) -> Iterator[SimpleNamespace]:
        del pool_type
        assert asyncio.get_running_loop() is self.loop
        yield SimpleNamespace(training=self)

    @contextlib.asynccontextmanager
    async def fwdbwd_dispatch_rate_limit(self, estimated_bytes_count: int) -> AsyncIterator[None]:
        del estimated_bytes_count
        yield

    async def execute_with_retries(
        self, send: Callable[..., Awaitable[types.UntypedAPIFuture]], *args: object
    ) -> types.UntypedAPIFuture:
        return await send(*args)

    async def forward_backward(
        self, *, request: types.ForwardBackwardRequest, forward_only: bool
    ) -> types.UntypedAPIFuture:
        request_id = f"{'forward' if forward_only else 'backward'}:{request.seq_id}"
        self.outputs[request_id] = types.ForwardBackwardOutput(
            loss_fn_output_type="target_token_logprobs",
            loss_fn_outputs=[
                {"logprobs": types.TensorData(data=[-3.0, -1.0], dtype="float32", shape=[2])}
                for _ in request.forward_backward_input.data
            ],
            metrics={"loss:sum": 4.0 * len(request.forward_backward_input.data)},
        )
        return types.UntypedAPIFuture(request_id=request_id)

    async def optim_step(self, *, request: types.OptimStepRequest) -> types.UntypedAPIFuture:
        request_id = f"optim:{request.seq_id}"
        self.outputs[request_id] = types.OptimStepResponse()
        return types.UntypedAPIFuture(request_id=request_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    ("chunk_len", "chunk_bytes", "chunks"), [(128, 10000, 1), (2, 10000, 2), (128, 64, 3)]
)
@pytest.mark.parametrize("sync", [False, True])
async def test_custom_backward_precedes_pipelined_optimizer(
    parallel: bool, chunk_len: int, chunk_bytes: int, chunks: int, sync: bool
) -> None:
    """Submission returns before forward finishes and reserves every chunk ahead of the optimizer."""
    holder = _PipelinedTrainingHolder(parallel, chunk_len, chunk_bytes)
    client = TrainingClient(holder, 0, "model")
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2]),
        # Integer forward weights are larger than the float32 gradient weights.
        loss_fn_inputs={
            "target_tokens": [101, 102],
            "weights": types.TensorData(data=[1, 1], dtype="int64", shape=[2]),
        },
    )
    data = [datum] * 3

    def custom_loss(
        data: list[types.Datum], logprobs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del data
        return -torch.stack(logprobs).sum(), {"custom:sum": 7.0}

    async def result_async(
        future: _APIFuture[types.ForwardBackwardOutput | types.OptimStepResponse],
    ) -> types.ForwardBackwardOutput | types.OptimStepResponse:
        if future.request_id.startswith("forward:"):
            if not holder.forward_started.done():
                holder.forward_started.set_result(None)
            await asyncio.wrap_future(holder.release_forward)
        return holder.outputs[future.request_id]

    try:
        with mock.patch.object(_APIFuture, "_result_async", result_async):
            backward = await asyncio.wait_for(
                asyncio.to_thread(client.forward_backward_custom, data, custom_loss)
                if sync
                else client.forward_backward_custom_async(data, custom_loss),
                timeout=5,
            )
            optimizer = client.optim_step(types.AdamParams(learning_rate=1e-4))
            await asyncio.wait_for(asyncio.wrap_future(holder.forward_started), timeout=5)
            assert not holder.release_forward.done()
            holder.release_forward.set_result(None)
            await asyncio.wait_for(backward.result_async(), timeout=5)
            await asyncio.wait_for(optimizer.result_async(), timeout=5)

        assert set(holder.outputs) == (
            {f"forward:{seq_id}" for seq_id in range(1, chunks + 1)}
            | {f"backward:{seq_id}" for seq_id in range(chunks + 1, 2 * chunks + 1)}
            | {f"optim:{2 * chunks + 1}"}
        )
    finally:
        holder.loop.call_soon_threadsafe(holder.loop.stop)
        holder.thread.join(timeout=5)
        holder.loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        ("forward", ValueError),
        ("callback", ValueError),
        ("autograd", RuntimeError),
        ("cancel", asyncio.CancelledError),
    ],
)
async def test_failed_custom_loss_poisons_training_client(
    parallel: bool, failure: str, expected_error: type[BaseException]
) -> None:
    """Missing reserved IDs must fail queued and later requests instead of hanging or sending."""
    holder = _PipelinedTrainingHolder(parallel, 1, 10000)
    # Debug mode catches waking holder-loop waiters from the custom-loss worker.
    holder.loop.call_soon_threadsafe(holder.loop.set_debug, True)
    client = TrainingClient(holder, 0, "model")
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={"target_tokens": [101, 102]},
    )

    error = ValueError("custom operation failed")

    def custom_loss(
        data: list[types.Datum], logprobs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del data, logprobs
        if failure == "autograd":
            return torch.tensor(0.0), {}
        raise error

    async def result_async(
        future: _APIFuture[types.ForwardBackwardOutput | types.OptimStepResponse],
    ) -> types.ForwardBackwardOutput | types.OptimStepResponse:
        if future.request_id.startswith("forward:"):
            if not holder.forward_started.done():
                holder.forward_started.set_result(None)
            await asyncio.wrap_future(holder.release_forward)
        return holder.outputs[future.request_id]

    async def run_custom() -> types.ForwardBackwardOutput:
        operation = await client.forward_backward_custom_async([datum] * 3, custom_loss)
        return await operation.result_async()

    try:
        with mock.patch.object(_APIFuture, "_result_async", result_async):
            submission = asyncio.create_task(run_custom())
            await asyncio.wait_for(asyncio.wrap_future(holder.forward_started), timeout=5)
            optimizer = client.optim_step(types.AdamParams(learning_rate=1e-4))
            queued_forward = client.forward([datum], "cross_entropy")
            if failure == "cancel":
                # Start awaiting the operation before cancelling its waiter.
                await asyncio.sleep(0)
                submission.cancel()
            elif failure == "forward":
                holder.release_forward.set_exception(error)
            else:
                holder.release_forward.set_result(None)
            with pytest.raises(expected_error) as raised:
                await asyncio.wait_for(submission, timeout=5)
            if failure in ("forward", "callback"):
                assert raised.value is error
            later_requests = [
                optimizer,
                queued_forward,
                client.forward([datum], "cross_entropy"),
                client.forward_backward([datum], "cross_entropy"),
                client.optim_step(types.AdamParams(learning_rate=1e-4)),
                client.save_state("checkpoint"),
            ]
            for future in later_requests:
                with pytest.raises(RuntimeError, match="TrainingClient is poisoned"):
                    await asyncio.wait_for(future.result_async(), timeout=5)
            with pytest.raises(RuntimeError, match="TrainingClient is poisoned"):
                await asyncio.wait_for(run_custom(), timeout=5)
        assert set(holder.outputs) == {"forward:1", "forward:2", "forward:3"}
    finally:
        holder.loop.call_soon_threadsafe(holder.loop.stop)
        holder.thread.join(timeout=5)
        holder.loop.close()


@pytest.mark.asyncio
async def test_custom_cancelled_before_start_poisons_training_client() -> None:
    """Cancelling an immediately returned future must not strand its unused reservations."""
    holder = _PipelinedTrainingHolder(False, 128, 10000)
    client = TrainingClient(holder, 0, "model")
    paused: Future[None] = Future()
    release = threading.Event()

    def pause_dispatch() -> None:
        paused.set_result(None)
        assert release.wait(timeout=5)

    def custom_loss(
        data: list[types.Datum], logprobs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del data, logprobs
        raise AssertionError("Cancelled work must not reach the loss function")

    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={"target_tokens": [101, 102]},
    )
    try:
        _get_custom_loss_loop().call_soon_threadsafe(pause_dispatch)
        await asyncio.wait_for(asyncio.wrap_future(paused), timeout=5)
        operation = await client.forward_backward_custom_async([datum], custom_loss)
        waiter = asyncio.create_task(operation.result_async())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        optimizer = client.optim_step(types.AdamParams(learning_rate=1e-4))
        release.set()
        with pytest.raises(RuntimeError, match="TrainingClient is poisoned"):
            await asyncio.wait_for(optimizer.result_async(), timeout=5)
        assert holder.outputs == {}
    finally:
        release.set()
        holder.loop.call_soon_threadsafe(holder.loop.stop)
        holder.thread.join(timeout=5)
        holder.loop.close()


@pytest.mark.asyncio
async def test_custom_loss_shares_worker_and_keeps_holder_loops_responsive() -> None:
    """Forward waits overlap, while loss and autograd leave API/heartbeat loops responsive."""
    holders = [_PipelinedTrainingHolder(False, 128, 10000) for _ in range(2)]
    clients = [TrainingClient(holder, 0, "model") for holder in holders]
    caller_loop = asyncio.get_running_loop()
    worker_loops: set[asyncio.AbstractEventLoop] = set()
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={"target_tokens": [101, 102]},
    )

    def check_responsiveness() -> None:
        worker_loop = asyncio.get_running_loop()
        assert worker_loop is not caller_loop
        worker_loops.add(worker_loop)
        for loop in [caller_loop, *(holder.loop for holder in holders)]:
            assert worker_loop is not loop
            responded = threading.Event()
            loop.call_soon_threadsafe(responded.set)
            assert responded.wait(timeout=5), "Custom loss blocked an API or caller loop"

    def custom_loss(
        data: list[types.Datum], logprobs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del data
        check_responsiveness()
        logprobs[0].register_hook(lambda _: check_responsiveness())
        return -logprobs[0].sum(), {}

    async def result_async(
        future: _APIFuture[types.ForwardBackwardOutput | types.OptimStepResponse],
    ) -> types.ForwardBackwardOutput | types.OptimStepResponse:
        holder = future.holder
        assert isinstance(holder, _PipelinedTrainingHolder)
        assert asyncio.get_running_loop() is holder.loop
        if future.request_id.startswith("forward:"):
            holder.forward_started.set_result(None)
            await asyncio.wrap_future(holder.release_forward)
        return holder.outputs[future.request_id]

    try:
        with mock.patch.object(_APIFuture, "_result_async", result_async):
            operations = [
                await client.forward_backward_custom_async([datum], custom_loss)
                for client in clients
            ]
            # Both forwards must reach polling before either completes.
            await asyncio.wait_for(
                asyncio.gather(*(asyncio.wrap_future(h.forward_started) for h in holders)),
                timeout=5,
            )
            for holder in holders:
                holder.release_forward.set_result(None)
            results = await asyncio.wait_for(
                asyncio.gather(
                    *(operation.result_async() for operation in operations), return_exceptions=True
                ),
                timeout=5,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
                assert result.metrics["loss:sum"] == 4.0
        assert len(worker_loops) == 1
        for holder in holders:
            assert set(holder.outputs) == {"forward:1", "backward:2"}
    finally:
        for holder in holders:
            holder.loop.call_soon_threadsafe(holder.loop.stop)
            holder.thread.join(timeout=5)
            holder.loop.close()
