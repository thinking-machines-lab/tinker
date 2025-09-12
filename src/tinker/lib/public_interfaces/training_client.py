"""TrainingClient for Tinker API."""

from __future__ import annotations

import logging
import time
from functools import cache
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Tuple, cast

from tinker.lib.async_tinker_provider import ClientConnectionPoolType
import torch

from tinker import types
from tinker.lib.telemetry import Telemetry, TelemetryProvider, capture_exceptions
from tinker.types import training_optim_step_params

from ..chunked_fwdbwd_helpers import combine_fwd_bwd_output_results
from ..retry_handler import RetryConfig
from ..sync_only import sync_only
from .api_future import (
    APIFuture,
    AwaitableConcurrentFuture,
    _APIFuture,
    _CombinedAPIFuture,
)

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

    from ..internal_client_holder import InternalClientHolder
    from .sampling_client import SamplingClient

# pyright: reportPrivateImportUsage=false

logger = logging.getLogger(__name__)

# FwdBwdChunkSize
MAX_CHUNK_LEN = 128
MAX_CHUNK_NUMBER_COUNT = 500000
MODEL_ID_NOT_SET_ERROR = "model_id must be set before calling forward. Try initializing the TrainingClient with a model_id by either calling create_lora_training_client on the ServiceClient, or initiliazing the TrainingClient with an existing model_id."

CustomLossFnV1 = Callable[[List[types.Datum], List[Any]], Tuple[Any, Dict[str, float]]]


class TrainingClient(TelemetryProvider):
    """Client for training ML models with forward/backward passes and optimization.

    The TrainingClient corresponds to a fine-tuned model that you can train and sample from.
    You typically get one by calling `service_client.create_lora_training_client()`.
    Key methods:
    - forward_backward() - compute gradients for training
    - optim_step() - update model parameters with Adam optimizer
    - save_weights_and_get_sampling_client() - export trained model for inference

    Args:
        holder: Internal client managing HTTP connections and async operations
        model_id: Unique identifier for the model to train. Required for training operations.

    Example:
        >>> training_client = service_client.create_lora_training_client(base_model="Qwen/Qwen2.5-7B")
        >>> fwdbwd_future = training_client.forward_backward(training_data, "cross_entropy")
        >>> optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))
        >>> fwdbwd_result = fwdbwd_future.result()  # Wait for gradients
        >>> optim_result = optim_future.result()    # Wait for parameter update
        >>> sampling_client = training_client.save_weights_and_get_sampling_client("my-model")
    """

    def __init__(self, holder: InternalClientHolder, model_id: types.ModelID | None = None):
        self.holder = holder
        self.model_id = model_id

    def _guaranteed_model_id(self) -> types.ModelID:
        assert self.model_id is not None, MODEL_ID_NOT_SET_ERROR
        return self.model_id

    def _estimate_number_count(self, datum: types.Datum) -> int:
        return datum.model_input.length + sum(len(value.data) for _, value in datum.loss_fn_inputs.items())

    def _chunked_requests_generator(self, data: List[types.Datum]) -> Generator[List[types.Datum], None, None]:
        current_chunk: List[types.Datum] = []
        current_chunk_number_count = 0

        for datum in data:
            estimated_number_count = self._estimate_number_count(datum)
            if (
                (len(current_chunk) > 0 and current_chunk_number_count + estimated_number_count > MAX_CHUNK_NUMBER_COUNT) or
                (len(current_chunk) == MAX_CHUNK_LEN)
            ):
                yield current_chunk
                current_chunk = []
                current_chunk_number_count = 0

            current_chunk.append(datum)
            current_chunk_number_count += estimated_number_count

        if len(current_chunk) > 0:
            yield current_chunk

    def _chunked_requests(self, data: List[types.Datum]) -> List[tuple[int, List[types.Datum]]]:
        return [(self.holder.get_request_id(), chunk) for chunk in self._chunked_requests_generator(data)]

    async def _send_single_forward_request(self, request_id: int, data: List[types.Datum], loss_fn: types.LossFnType):
        with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
            return await client.training.forward(
                model_id=self._guaranteed_model_id(),
                forward_input=_to_fwdbwd_input_params(types.ForwardBackwardInput(data=data, loss_fn=loss_fn)),
                idempotency_key=self.holder.make_idempotency_key(request_id),
            )

    @capture_exceptions(fatal=True)
    def forward(
        self, data: List[types.Datum], loss_fn: types.LossFnType
    ) -> APIFuture[types.ForwardBackwardOutput]:
        requests = self._chunked_requests(data)
        @capture_exceptions(fatal=True)
        async def _forward_async():
            start_time = time.time()
            futures = []
            for request_id, data in requests:
                async with self.holder.take_turn(request_id):
                    untyped_future = await self.holder.execute_with_retries(self._send_single_forward_request, request_id, data, loss_fn)
                api_future = _APIFuture(
                    types.ForwardBackwardOutput,
                    self.holder,
                    untyped_future,
                    request_start_time=start_time,
                    request_type="Forward",
                )
                futures.append(api_future)
            return await _CombinedAPIFuture(futures, combine_fwd_bwd_output_results, self.holder)

        return self.holder.run_coroutine_threadsafe(_forward_async())

    async def forward_async(
        self, data: List[types.Datum], loss_fn: types.LossFnType
    ) -> APIFuture[types.ForwardBackwardOutput]:
        return self.forward(data, loss_fn)

    async def _send_single_forward_backward_request(self, request_id: int, data: List[types.Datum], loss_fn: types.LossFnType):
        with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
            return await client.training.forward_backward(
                model_id=self._guaranteed_model_id(),
                forward_backward_input=_to_fwdbwd_input_params(types.ForwardBackwardInput(data=data, loss_fn=loss_fn)),
                idempotency_key=self.holder.make_idempotency_key(request_id),
            )

    @capture_exceptions(fatal=True)
    def forward_backward(
        self, data: List[types.Datum], loss_fn: types.LossFnType
    ) -> APIFuture[types.ForwardBackwardOutput]:
        requests = self._chunked_requests(data)
        @capture_exceptions(fatal=True)
        async def _forward_backward_async():
            futures = []
            start_time = time.time()

            for request_id, data in requests:
                async with self.holder.take_turn(request_id):
                    untyped_future = await self.holder.execute_with_retries(self._send_single_forward_backward_request, request_id, data, loss_fn)
                api_future = _APIFuture(
                    types.ForwardBackwardOutput,
                    self.holder,
                    untyped_future,
                    request_start_time=start_time,
                    request_type="ForwardBackward",
                )
                futures.append(api_future)

            return await _CombinedAPIFuture(futures, combine_fwd_bwd_output_results, self.holder)

        return self.holder.run_coroutine_threadsafe(_forward_backward_async())

    async def forward_backward_async(
        self, data: List[types.Datum], loss_fn: types.LossFnType
    ) -> APIFuture[types.ForwardBackwardOutput]:
        return self.forward_backward(data, loss_fn)

    @sync_only
    @capture_exceptions(fatal=True)
    def forward_backward_custom(
        self, data: List[types.Datum], loss_fn: CustomLossFnV1
    ) -> APIFuture[types.ForwardBackwardOutput]:
        """Synchronous version of forward_backward_custom_async."""
        return self.holder.run_coroutine_threadsafe(
            self.forward_backward_custom_async(data, loss_fn)
        ).result()

    @capture_exceptions(fatal=True)
    async def forward_backward_custom_async(
        self, data: List[types.Datum], loss_fn: CustomLossFnV1
    ) -> APIFuture[types.ForwardBackwardOutput]:
        import torch

        # First do a forward pass and get logprobs
        forward_future = await self.forward_async(data, "cross_entropy")
        forward_result = await forward_future.result_async()
        logprobs_list: List[torch.Tensor] = []
        for out in forward_result.loss_fn_outputs:
            logprob = torch.tensor(out["logprobs"].data).clone().detach().requires_grad_(True)
            logprobs_list.append(logprob)

        # Now apply user-provided function
        loss, metrics = loss_fn(data, logprobs_list)
        loss.backward()
        grads = []
        for logprob in logprobs_list:
            if logprob.grad is None:
                raise ValueError("No gradient computed for logprob tensor")
            grads.append(logprob.grad)

        linear_loss_data = []
        for datum, grad in zip(data, grads):
            loss_fn_inputs: Any = {
                "target_tokens": datum.loss_fn_inputs["target_tokens"],
                "weights": -grad,  # Pass PyTorch tensor directly (will be converted to TensorData)
            }
            linear_loss_data.append(
                types.Datum(
                    model_input=datum.model_input,
                    loss_fn_inputs=loss_fn_inputs,
                )
            )

        # Do the backward pass with the gradients
        backward_future = await self.forward_backward_async(linear_loss_data, "cross_entropy")

        # We need to slightly modify the future to add the custom metrics, so we use _CombinedAPIFuture
        # to transform the future.
        def add_custom_metrics(
            results: List[types.ForwardBackwardOutput],
        ) -> types.ForwardBackwardOutput:
            result = results[0]  # Single result
            result.metrics.update(metrics)
            return result

        return _CombinedAPIFuture([backward_future], add_custom_metrics, self.holder)

    @capture_exceptions(fatal=True)
    def optim_step(
        self, adam_params: types.AdamParams
    ) -> APIFuture[types.OptimStepResponse]:
        request_id = self.holder.get_request_id()
        @capture_exceptions(fatal=True)
        async def _optim_step_async():
            start_time = time.time()
            async def _send_request():
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.training.optim_step(
                        model_id=self._guaranteed_model_id(), adam_params=_to_adam_params(adam_params), idempotency_key=self.holder.make_idempotency_key(request_id)
                    )
            async with self.holder.take_turn(request_id):
                untyped_future = await self.holder.execute_with_retries(_send_request)
            return await _APIFuture(
                types.OptimStepResponse,
                self.holder,
                untyped_future,
                request_start_time=start_time,
                request_type="OptimStep",
            )

        return self.holder.run_coroutine_threadsafe(_optim_step_async())

    async def optim_step_async(
        self, adam_params: types.AdamParams
    ) -> APIFuture[types.OptimStepResponse]:
        return self.optim_step(adam_params)

    @capture_exceptions(fatal=True)
    def save_state(
        self, name: str
    ) -> APIFuture[types.SaveWeightsResponse]:
        request_id = self.holder.get_request_id()
        @capture_exceptions(fatal=True)
        async def _save_state_async():
            start_time = time.time()
            async def _send_request():
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.weights.save(model_id=self._guaranteed_model_id(), path=name, idempotency_key=self.holder.make_idempotency_key(request_id))
            async with self.holder.take_turn(request_id):
                future = await self.holder.execute_with_retries(_send_request)
            return await _APIFuture(
                types.SaveWeightsResponse,
                self.holder,
                future,
                request_start_time=start_time,
                request_type="SaveWeights",
            )

        return self.holder.run_coroutine_threadsafe(_save_state_async())

    async def save_state_async(self, name: str) -> APIFuture[types.SaveWeightsResponse]:
        return self.save_state(name)

    @capture_exceptions(fatal=True)
    def load_state(
        self, path: str
    ) -> APIFuture[types.LoadWeightsResponse]:
        request_id = self.holder.get_request_id()
        @capture_exceptions(fatal=True)
        async def _load_state_async():
            start_time = time.time()
            async def _send_request():
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.weights.load(model_id=self._guaranteed_model_id(), path=path, idempotency_key=self.holder.make_idempotency_key(request_id))
            async with self.holder.take_turn(request_id):
                future = await self.holder.execute_with_retries(_send_request)
            return await _APIFuture(
                types.LoadWeightsResponse,
                self.holder,
                future,
                request_start_time=start_time,
                request_type="LoadWeights",
            )

        return self.holder.run_coroutine_threadsafe(_load_state_async())

    async def load_state_async(self, path: str) -> APIFuture[types.LoadWeightsResponse]:
        return self.load_state(path)

    @capture_exceptions(fatal=True)
    def save_weights_for_sampler(
        self, name: str
    ) -> APIFuture[types.SaveWeightsForSamplerResponse]:
        request_id = self.holder.get_request_id()
        @capture_exceptions(fatal=True)
        async def _save_weights_for_sampler_async():
            start_time = time.time()
            async def _send_request():
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.weights.save_for_sampler(
                        model_id=self._guaranteed_model_id(), path=name, idempotency_key=self.holder.make_idempotency_key(request_id)
                    )
            async with self.holder.take_turn(request_id):
                future = await self.holder.execute_with_retries(_send_request)
            return await _APIFuture(
                types.SaveWeightsForSamplerResponse,
                self.holder,
                future,
                request_start_time=start_time,
                request_type="SaveWeightsForSampler",
            )

        return self.holder.run_coroutine_threadsafe(_save_weights_for_sampler_async())

    async def save_weights_for_sampler_async(
        self, name: str
    ) -> APIFuture[types.SaveWeightsForSamplerResponse]:
        return self.save_weights_for_sampler(name)

    @capture_exceptions(fatal=True)
    def unload_model(
        self,
    ) -> APIFuture[types.UnloadModelResponse]:
        request_id = self.holder.get_request_id()
        @capture_exceptions(fatal=True)
        async def _unload_model_async():
            start_time = time.time()
            async def _send_request():
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.models.unload(model_id=self._guaranteed_model_id(), idempotency_key=self.holder.make_idempotency_key(request_id))
            async with self.holder.take_turn(request_id):
                future = await self.holder.execute_with_retries(_send_request)
            return await _APIFuture(
                types.UnloadModelResponse,
                self.holder,
                future,
                request_start_time=start_time,
                request_type="UnloadModel",
            )

        return self.holder.run_coroutine_threadsafe(_unload_model_async())

    async def unload_model_async(self) -> APIFuture[types.UnloadModelResponse]:
        return self.unload_model()

    def _get_info_submit(self) -> AwaitableConcurrentFuture[types.GetInfoResponse]:
        request_id = self.holder.get_request_id()
        async def _get_info_async():
            async def _send_request():
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.models.get_info(model_id=self._guaranteed_model_id(), idempotency_key=self.holder.make_idempotency_key(request_id))
            async with self.holder.take_turn(request_id):
                return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_get_info_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def get_info(self) -> types.GetInfoResponse:
        return self._get_info_submit().result()

    @capture_exceptions(fatal=True)
    async def get_info_async(self) -> types.GetInfoResponse:
        return await self._get_info_submit()

    @cache
    @capture_exceptions(fatal=True)
    def get_tokenizer(self) -> PreTrainedTokenizer:
        return _get_tokenizer(self._guaranteed_model_id(), self.holder)

    @capture_exceptions(fatal=True)
    def create_sampling_client(
        self, model_path: str, retry_config: RetryConfig | None = None
    ) -> SamplingClient:
        from .sampling_client import SamplingClient

        return SamplingClient(self.holder, model_path=model_path, retry_config=retry_config)

    @capture_exceptions(fatal=True)
    def save_weights_and_get_sampling_client(
        self, name: str, retry_config: RetryConfig | None = None
    ) -> SamplingClient:
        from .sampling_client import SamplingClient

        path = self.save_weights_for_sampler(name).result().path
        return SamplingClient(self.holder, model_path=path, retry_config=retry_config)

    @capture_exceptions(fatal=True)
    async def save_weights_and_get_sampling_client_async(
        self, name: str, retry_config: RetryConfig | None = None
    ) -> SamplingClient:
        from .sampling_client import SamplingClient

        save_weights_future = await self.save_weights_for_sampler_async(name)
        save_weights_result = await save_weights_future.result_async()
        model_path = save_weights_result.path
        return SamplingClient(self.holder, model_path=model_path, retry_config=retry_config)

    def get_telemetry(self) -> Telemetry | None:
        return self.holder.get_telemetry()


def _to_fwdbwd_input_params(x: types.ForwardBackwardInput) -> types._ForwardBackwardInputParam:
    return cast(types._ForwardBackwardInputParam, x.model_dump())


def _to_adam_params(x: types.AdamParams) -> training_optim_step_params.AdamParams:
    return cast(training_optim_step_params.AdamParams, x.model_dump())


def _get_tokenizer(model_id: types.ModelID, holder: InternalClientHolder) -> PreTrainedTokenizer:
    # call get_info on model_id
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    async def _get_info_async():
        with holder.aclient(ClientConnectionPoolType.TRAIN) as client:
            return await client.models.get_info(model_id=model_id)

    info = holder.run_coroutine_threadsafe(_get_info_async()).result()
    model_name = info.model_data.model_name
    assert model_name is not None, "This shouldn't happen: model_name is None"

    # We generally adhere to the huggingface convention of "<org>/<model>" but
    # in some cases we'll deploy variants using the format
    # "<org>/<model>/<variant>". In that case, we want to load the tokenizer
    # using the huggingface convention.
    if model_name.startswith("meta-llama/Llama-3"):
        # Avoid gating of Llama 3 models:
        tokenizer_id = "baseten/Meta-Llama-3-tokenizer"
    elif model_name.count("/") == 2:
        org, model, _variant = model_name.split("/", 2)
        tokenizer_id = f"{org}/{model}"
    else:
        tokenizer_id = model_name

    return AutoTokenizer.from_pretrained(tokenizer_id, fast=True)
