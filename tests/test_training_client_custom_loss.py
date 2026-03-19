from __future__ import annotations

from concurrent.futures import Future

import pytest
import torch

from tinker import types
from tinker.lib.public_interfaces.api_future import AwaitableConcurrentFuture
from tinker.lib.public_interfaces.training_client import TrainingClient


class _DummyHolder:
    def run_coroutine_threadsafe(self, coro):
        future: Future = Future()
        future.set_result(coro)
        return future

    def get_telemetry(self):
        return None


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
            metrics={"base:sum": 1.0},
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
        loss_type_input="logprobs",
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

    setattr(client, "forward_async", forward_async_1d)

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
        loss_type_input="logprobs",
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

    with pytest.raises(ValueError, match="Unsupported loss_type_input"):
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
async def test_forward_backward_custom_rejects_unexpected_loss_fn_input_keys():
    client = _FakeTrainingClient()
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            "target_tokens": [101, 102],
            "advantages": [1.0, 1.0],
        },
    )

    def custom_loss(
        data: list[types.Datum], logprobs_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del data, logprobs_list
        return torch.tensor(0.0, requires_grad=True), {}

    with pytest.raises(ValueError, match="only supports loss_fn_inputs keys"):
        await client.forward_backward_custom_async(
            [datum],
            custom_loss,
            loss_type_input="logprobs",
        )
