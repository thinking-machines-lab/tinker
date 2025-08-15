"""TrainingClient for Tinker API."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, List, cast

from tinker import types
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
CHUNK_SIZE = 128  # TODO: pick this less arbitrarily
MODEL_ID_NOT_SET_ERROR = "model_id must be set before calling forward. Try initializing the TrainingClient with a model_id by either calling create_lora_training_client on the ServiceClient, or initiliazing the TrainingClient with an existing model_id."


class TrainingClient:
    def __init__(self, holder: InternalClientHolder, model_id: types.ModelID | None = None):
        self.holder = holder
        self.model_id = model_id

    def _guaranteed_model_id(self) -> types.ModelID:
        assert self.model_id is not None, MODEL_ID_NOT_SET_ERROR
        return self.model_id

    def _forward_submit(
        self, data: List[types.Datum], loss_fn: types.LossFnType
    ) -> AwaitableConcurrentFuture[_CombinedAPIFuture[types.ForwardBackwardOutput]]:
        async def _forward_async():
            start_time = time.time()
            futures = []
            for i in range(0, len(data), CHUNK_SIZE):
                with self.holder.aclient() as client:
                    untyped_future = await client.training.forward(
                        model_id=self._guaranteed_model_id(),
                        forward_input=_to_fwdbwd_input_params(
                            types.ForwardBackwardInput(
                                data=data[i : i + CHUNK_SIZE], loss_fn=loss_fn
                            )
                        ),
                    )
                api_future = _APIFuture(
                    types.ForwardBackwardOutput,
                    self.holder,
                    untyped_future,
                    request_start_time=start_time,
                    request_type="Forward",
                )
                futures.append(api_future)
            return _CombinedAPIFuture(futures, combine_fwd_bwd_output_results, self.holder)

        return self.holder.run_coroutine_threadsafe(_forward_async())

    @sync_only
    def forward(
        self, data: List[types.Datum], loss_fn: types.LossFnType
    ) -> APIFuture[types.ForwardBackwardOutput]:
        return self._forward_submit(data, loss_fn).result()

    async def forward_async(
        self, data: List[types.Datum], loss_fn: types.LossFnType
    ) -> APIFuture[types.ForwardBackwardOutput]:
        return await self._forward_submit(data, loss_fn)

    def _forward_backward_submit(
        self, data: List[types.Datum], loss_fn: types.LossFnType
    ) -> AwaitableConcurrentFuture[_CombinedAPIFuture[types.ForwardBackwardOutput]]:
        async def _forward_backward_async():
            futures = []
            start_time = time.time()

            for i in range(0, len(data), CHUNK_SIZE):
                with self.holder.aclient() as client:
                    untyped_future = await client.training.forward_backward(
                        model_id=self._guaranteed_model_id(),
                        forward_backward_input=_to_fwdbwd_input_params(
                            types.ForwardBackwardInput(
                                data=data[i : i + CHUNK_SIZE], loss_fn=loss_fn
                            )
                        ),
                    )
                api_future = _APIFuture(
                    types.ForwardBackwardOutput,
                    self.holder,
                    untyped_future,
                    request_start_time=start_time,
                    request_type="ForwardBackward",
                )
                futures.append(api_future)

            return _CombinedAPIFuture(futures, combine_fwd_bwd_output_results, self.holder)

        return self.holder.run_coroutine_threadsafe(_forward_backward_async())

    @sync_only
    def forward_backward(
        self, data: List[types.Datum], loss_fn: types.LossFnType
    ) -> APIFuture[types.ForwardBackwardOutput]:
        return self._forward_backward_submit(data, loss_fn).result()

    async def forward_backward_async(
        self, data: List[types.Datum], loss_fn: types.LossFnType
    ) -> APIFuture[types.ForwardBackwardOutput]:
        return await self._forward_backward_submit(data, loss_fn)

    def _optim_step_submit(
        self, adam_params: types.AdamParams
    ) -> AwaitableConcurrentFuture[_APIFuture[types.OptimStepResponse]]:
        async def _optim_step_async():
            start_time = time.time()
            with self.holder.aclient() as client:
                future = await client.training.optim_step(
                    model_id=self._guaranteed_model_id(), adam_params=_to_adam_params(adam_params)
                )
            return _APIFuture(
                types.OptimStepResponse,
                self.holder,
                future,
                request_start_time=start_time,
                request_type="OptimStep",
            )

        return self.holder.run_coroutine_threadsafe(_optim_step_async())

    @sync_only
    def optim_step(self, adam_params: types.AdamParams) -> APIFuture[types.OptimStepResponse]:
        return self._optim_step_submit(adam_params).result()

    async def optim_step_async(
        self, adam_params: types.AdamParams
    ) -> APIFuture[types.OptimStepResponse]:
        return await self._optim_step_submit(adam_params)

    def _save_state_submit(
        self, name: str
    ) -> AwaitableConcurrentFuture[_APIFuture[types.SaveWeightsResponse]]:
        async def _save_state_async():
            start_time = time.time()
            with self.holder.aclient() as client:
                future = await client.weights.save(model_id=self._guaranteed_model_id(), path=name)
            return _APIFuture(
                types.SaveWeightsResponse,
                self.holder,
                future,
                request_start_time=start_time,
                request_type="SaveWeights",
            )

        return self.holder.run_coroutine_threadsafe(_save_state_async())

    @sync_only
    def save_state(self, name: str) -> APIFuture[types.SaveWeightsResponse]:
        return self._save_state_submit(name).result()

    async def save_state_async(self, name: str) -> APIFuture[types.SaveWeightsResponse]:
        return await self._save_state_submit(name)

    def _load_state_submit(
        self, path: str
    ) -> AwaitableConcurrentFuture[_APIFuture[types.LoadWeightsResponse]]:
        async def _load_state_async():
            start_time = time.time()
            with self.holder.aclient() as client:
                future = await client.weights.load(model_id=self._guaranteed_model_id(), path=path)
            return _APIFuture(
                types.LoadWeightsResponse,
                self.holder,
                future,
                request_start_time=start_time,
                request_type="LoadWeights",
            )

        return self.holder.run_coroutine_threadsafe(_load_state_async())

    @sync_only
    def load_state(self, path: str) -> APIFuture[types.LoadWeightsResponse]:
        return self._load_state_submit(path).result()

    async def load_state_async(self, path: str) -> APIFuture[types.LoadWeightsResponse]:
        return await self._load_state_submit(path)

    def _save_weights_for_sampler_submit(
        self, name: str
    ) -> AwaitableConcurrentFuture[_APIFuture[types.SaveWeightsForSamplerResponse]]:
        async def _save_weights_for_sampler_async():
            start_time = time.time()
            with self.holder.aclient() as client:
                future = await client.weights.save_for_sampler(
                    model_id=self._guaranteed_model_id(), path=name
                )
            return _APIFuture(
                types.SaveWeightsForSamplerResponse,
                self.holder,
                future,
                request_start_time=start_time,
                request_type="SaveWeightsForSampler",
            )

        return self.holder.run_coroutine_threadsafe(_save_weights_for_sampler_async())

    @sync_only
    def save_weights_for_sampler(self, name: str) -> APIFuture[types.SaveWeightsForSamplerResponse]:
        return self._save_weights_for_sampler_submit(name).result()

    async def save_weights_for_sampler_async(
        self, name: str
    ) -> APIFuture[types.SaveWeightsForSamplerResponse]:
        return await self._save_weights_for_sampler_submit(name)

    def _unload_model_submit(
        self,
    ) -> AwaitableConcurrentFuture[_APIFuture[types.UnloadModelResponse]]:
        async def _unload_model_async():
            start_time = time.time()
            with self.holder.aclient() as client:
                future = await client.models.unload(model_id=self._guaranteed_model_id())
            return _APIFuture(
                types.UnloadModelResponse,
                self.holder,
                future,
                request_start_time=start_time,
                request_type="UnloadModel",
            )

        return self.holder.run_coroutine_threadsafe(_unload_model_async())

    @sync_only
    def unload_model(self) -> APIFuture[types.UnloadModelResponse]:
        return self._unload_model_submit().result()

    async def unload_model_async(self) -> APIFuture[types.UnloadModelResponse]:
        return await self._unload_model_submit()

    def _get_info_submit(self) -> AwaitableConcurrentFuture[types.GetInfoResponse]:
        async def _get_info_async():
            with self.holder.aclient() as client:
                return await client.models.get_info(model_id=self._guaranteed_model_id())

        return self.holder.run_coroutine_threadsafe(_get_info_async())

    @sync_only
    def get_info(self) -> types.GetInfoResponse:
        return self._get_info_submit().result()

    async def get_info_async(self) -> types.GetInfoResponse:
        return await self._get_info_submit()

    def get_tokenizer(self) -> PreTrainedTokenizer:
        return _get_tokenizer(self._guaranteed_model_id(), self.holder)

    def create_sampling_client(
        self, model_path: str, retry_config: RetryConfig | None = None
    ) -> SamplingClient:
        from .sampling_client import SamplingClient

        return SamplingClient(self.holder, model_path=model_path, retry_config=retry_config)

    def save_weights_and_get_sampling_client(
        self, name: str, retry_config: RetryConfig | None = None
    ) -> SamplingClient:
        from .sampling_client import SamplingClient

        path = self.save_weights_for_sampler(name).result().path
        return SamplingClient(self.holder, model_path=path, retry_config=retry_config)

    async def save_weights_and_get_sampling_client_async(
        self, name: str, retry_config: RetryConfig | None = None
    ) -> SamplingClient:
        from .sampling_client import SamplingClient

        save_weights_future = await self.save_weights_for_sampler_async(name)
        save_weights_result = await save_weights_future.result_async()
        model_path = save_weights_result.path
        return SamplingClient(self.holder, model_path=model_path, retry_config=retry_config)


def _to_fwdbwd_input_params(x: types.ForwardBackwardInput) -> types._ForwardBackwardInputParam:
    return cast(types._ForwardBackwardInputParam, x.model_dump())


def _to_adam_params(x: types.AdamParams) -> training_optim_step_params.AdamParams:
    return cast(training_optim_step_params.AdamParams, x.model_dump())


def _get_tokenizer(model_id: types.ModelID, holder: InternalClientHolder) -> PreTrainedTokenizer:
    # call get_info on model_id
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    async def _get_info_async():
        with holder.aclient() as client:
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
