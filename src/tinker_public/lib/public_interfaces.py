"""
These are the public clients.
We've split up the internal client into these separate parts to make it easier to
create different implementations of the TrainingClient and SamplingClient so the
nontrivial algorithmic code for training and evaluating models can be run on either
the Tinker service or these alternative implementations.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import Future as ConcurrentFuture
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Sequence, Type, TypeVar, cast

import tinker_public
from tinker_public import types
from tinker_public._types import NOT_GIVEN
from tinker_public.types import training_optim_step_params

from .._models import BaseModel
from .retry_handler import RetryConfig, RetryHandler
from .sync_only import sync_only

# pyright: reportPrivateImportUsage=false

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

T = TypeVar("T")

def not_none(x: T | None) -> T:
    assert x is not None
    return x

# Sentinel object to indicate that the function hasn't been called yet
_UNCOMPUTED = object()


# TODO maybe consider binding to BaseModel. But currently there's at one class (OptimStepResponse) that isn't a BaseModel.

MODEL_ID_NOT_SET_ERROR = "model_id must be set before calling forward. Try initializing the TrainingClient with a model_id by either calling create_lora_training_client on the ServiceClient, or initiliazing the TrainingClient with an existing model_id."

class ResolvedFuture(Generic[T], ConcurrentFuture[T]):
    def __init__(self, result: T):
        self._result = result

    def result(self, timeout: float | None = None) -> T:
        # This is typed to not be None, but it might be valid to return None for some T
        return self._result  # type: ignore

class InternalClientHolder:
    def __init__(self, **kwargs: Any):
        self._constructor_kwargs = kwargs
        # So we can use async eventloop for parallel sampling requests
        # in sync code.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()

    @cached_property
    def client(self) -> tinker_public.Tinker:
        return tinker_public.Tinker(**self._constructor_kwargs)

    @cached_property
    def aclient(self) -> tinker_public.AsyncTinker:
        return tinker_public.AsyncTinker(**self._constructor_kwargs)

    def get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._start_background_thread()
        assert self._loop is not None, "Background thread not started"
        return self._loop


    def _start_background_thread(self):
        assert self._thread is None, "Background thread already started"
        self._thread = threading.Thread(target=self._background_thread_func, daemon=True)
        self._thread.start()
        self._started.wait()

    def _background_thread_func(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self.async_client = tinker_public.AsyncTinker(**self._constructor_kwargs)
        self._started.set()
        self._loop.run_forever()

    def close(self):
        if self._thread is not None:
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                logger.warning("Background thread did not join")
            self._thread = None
        if self._loop is not None:
            self._loop.close()
            self._loop = None

    def __del__(self):
        self.close()


def to_lora_config_params(x: types.LoraConfig) -> types._LoraConfigParam:
    return cast(types._LoraConfigParam, x.model_dump())

def to_fwdbwd_input_params(x: types.ForwardBackwardInput) -> types._ForwardBackwardInputParam:
    return cast(types._ForwardBackwardInputParam, x.model_dump())

def to_adam_params(x: types.AdamParams) -> training_optim_step_params.AdamParams:
    return cast(training_optim_step_params.AdamParams, x.model_dump())

class APIFuture(Generic[T]):
    def __init__(
        self,
        model_cls: Type[T],
        holder: InternalClientHolder,
        untyped_future: types.UntypedAPIFuture,
        request_start_time: float,
        request_type: str
    ):
        self.model_cls = model_cls
        self.holder = holder
        self.untyped_future = untyped_future
        self.request_type = request_type
        self._cached_result: Any = _UNCOMPUTED

        # This helps us collect telemetry about how long (1) it takes the
        # client to serialize the request, (2) round-trip time to the server
        # and back, and (3) how long the server takes to process the request.
        # We send this delta in a header to the server when retrieving the promise
        # result.
        self.request_start_time = request_start_time
        self.request_future_start_time = time.time()
        self.request_queue_roundtrip_time = self.request_future_start_time - request_start_time

    async def result_async(self, timeout: float | None = None) -> T:
        """Get the result of this future, with automatic retries for transient errors."""
        if self._cached_result is not _UNCOMPUTED:
            return cast(T, self._cached_result)

        start_time = time.time()
        iteration = -1

        while True:
            iteration += 1

            if timeout is not None and time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout of {timeout} seconds reached while waiting for result of {self.untyped_future.request_id=}")

            # Headers for telemetry
            headers = {
                "X-Tinker-Request-Iteration": str(iteration),
                "X-Tinker-Request-Type": self.request_type,
            }
            if iteration == 0:
                headers["X-Tinker-Create-Promise-Roundtrip-Time"] = str(self.request_queue_roundtrip_time)

            # Function hasn't been called yet, execute it now
            try:
                # Get response with retries for everything except 408s,
                # which we want to handle ourselves
                response = await self.holder.aclient.futures.with_raw_response.retrieve(
                    request_id=self.untyped_future.request_id, timeout=timeout, extra_headers=headers
                )
            except tinker_public.APIStatusError as e:
                # Retry 408s until we time out
                if e.status_code == 408:
                    continue
                raise ValueError(f"Error retrieving result: {e} with status code {e.status_code=} for {self.untyped_future.request_id=} and expected type {self.model_cls=}") from e

            # Function hasn't been called yet, execute it now
            result_dict: Dict[str, Any] = await response.json()  # type: ignore

            if "type" in result_dict and result_dict["type"] == "try_again":
                logger.warning(f"Retrying request {self.untyped_future.request_id=} because of try_again")
                continue

            if "error" in result_dict:
                raise ValueError(f"Error retrieving result: {result_dict} for {self.untyped_future.request_id=} and expected type {self.model_cls=}")

            try:
                # Check if model_cls is a BaseModel subclass before calling model_validate
                if issubclass(self.model_cls, BaseModel):
                    self._cached_result = self.model_cls.model_validate(result_dict)
                else:
                    # For non-BaseModel types, just return the result directly
                    self._cached_result = result_dict
                return cast(T, self._cached_result)
            except Exception as e:
                raise ValueError(f"Error retrieving result: {e} for {self.untyped_future.request_id=} and expected type {self.model_cls=}") from e

    @sync_only
    def result(self, timeout: float | None = None) -> T:
        return asyncio.run_coroutine_threadsafe(self.result_async(timeout), self.holder.get_loop()).result()

class ServiceClient:
    def __init__(self, **kwargs: Any):
        default_headers = _get_default_headers() | kwargs.pop("default_headers", {})
        self.holder = InternalClientHolder(**kwargs, default_headers=default_headers, _strict_response_validation=True)

    def get_server_capabilities(self) -> types.GetServerCapabilitiesResponse:
        return self.holder.client.service.get_server_capabilities()

    async def get_server_capabilities_async(self) -> types.GetServerCapabilitiesResponse:
        return await self.holder.aclient.service.get_server_capabilities()

    @sync_only
    def create_lora_training_client(self, base_model: str, rank: int = 32, seed: int | None = None) -> "TrainingClient":
        start_time = time.time()
        future = self.holder.client.models.create(
            base_model=base_model, lora_config=to_lora_config_params(types.LoraConfig(rank=rank, seed=seed))
        )
        api_future = APIFuture(types.CreateModelResponse, self.holder, future, request_start_time=start_time, request_type="CreateModel")
        model_id = api_future.result().model_id
        logger.info(f"Creating TrainingClient for {model_id=}")
        return self.create_training_client(model_id)

    async def create_lora_training_client_async(self, base_model: str, rank: int = 32, seed: int | None = None) -> "TrainingClient":
        start_time = time.time()
        future = await self.holder.aclient.models.create(
            base_model=base_model, lora_config=to_lora_config_params(types.LoraConfig(rank=rank, seed=seed))
        )
        api_future = APIFuture(types.CreateModelResponse, self.holder, future, request_start_time=start_time, request_type="CreateModel")
        return self.create_training_client((await api_future.result_async()).model_id)

    def create_training_client(self, model_id: types.ModelID | None = None) -> "TrainingClient":
        return TrainingClient(self.holder, model_id=model_id)

    def create_sampling_client(
        self,
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> SamplingClient:
        if model_path is None and base_model is None:
            raise ValueError("Either model_path or base_model must be provided")
        return SamplingClient(
            self.holder,
            model_path=model_path,
            base_model=base_model,
            retry_config=retry_config,
        )


class TrainingClient:
    def __init__(
        self, holder: InternalClientHolder, model_id: types.ModelID | None = None
    ):
        self.holder = holder
        self.model_id = model_id

    def _guaranteed_model_id(self) -> types.ModelID:
        assert self.model_id is not None, MODEL_ID_NOT_SET_ERROR
        return self.model_id

    @sync_only
    def forward(self, data: List[types.Datum], loss_fn: types.LossFnType) -> APIFuture[types.ForwardBackwardOutput]:
        return asyncio.run_coroutine_threadsafe(self.forward_async(data, loss_fn), self.holder.get_loop()).result()

    async def forward_async(self, data: List[types.Datum], loss_fn: types.LossFnType) -> APIFuture[types.ForwardBackwardOutput]:
        start_time = time.time()
        future = await self.holder.aclient.training.forward(model_id=self._guaranteed_model_id(), forward_input=to_fwdbwd_input_params(types.ForwardBackwardInput(data=data, loss_fn=loss_fn)))
        return APIFuture(types.ForwardBackwardOutput, self.holder, future, request_start_time=start_time, request_type="Forward")

    @sync_only
    def forward_backward(self, data: List[types.Datum], loss_fn: types.LossFnType) -> APIFuture[types.ForwardBackwardOutput]:
        return asyncio.run_coroutine_threadsafe(self.forward_backward_async(data, loss_fn), self.holder.get_loop()).result()

    async def forward_backward_async(self, data: List[types.Datum], loss_fn: types.LossFnType) -> APIFuture[types.ForwardBackwardOutput]:
        start_time = time.time()
        future = await self.holder.aclient.training.forward_backward(model_id=self._guaranteed_model_id(), forward_backward_input=to_fwdbwd_input_params(types.ForwardBackwardInput(data=data, loss_fn=loss_fn)))
        return APIFuture(types.ForwardBackwardOutput, self.holder, future, request_start_time=start_time, request_type="ForwardBackward")

    @sync_only
    def optim_step(self, adam_params: types.AdamParams) -> APIFuture[types.OptimStepResponse]:
        return asyncio.run_coroutine_threadsafe(self.optim_step_async(adam_params), self.holder.get_loop()).result()

    async def optim_step_async(self, adam_params: types.AdamParams) -> APIFuture[types.OptimStepResponse]:
        start_time = time.time()
        future = await self.holder.aclient.training.optim_step(
            model_id=self._guaranteed_model_id(), adam_params=to_adam_params(adam_params)
        )
        return APIFuture(types.OptimStepResponse, self.holder, future, request_start_time=start_time, request_type="OptimStep")

    @sync_only
    def save_state(self, name: str) -> APIFuture[types.SaveWeightsResponse]:
        return asyncio.run_coroutine_threadsafe(self.save_state_async(name), self.holder.get_loop()).result()

    async def save_state_async(self, name: str) -> APIFuture[types.SaveWeightsResponse]:
        start_time = time.time()
        future = await self.holder.aclient.weights.save(model_id=self._guaranteed_model_id(), path=name)
        return APIFuture(types.SaveWeightsResponse, self.holder, future, request_start_time=start_time, request_type="SaveWeights")

    @sync_only
    def load_state(self, path: str) -> APIFuture[types.LoadWeightsResponse]:
        return asyncio.run_coroutine_threadsafe(self.load_state_async(path), self.holder.get_loop()).result()

    async def load_state_async(self, path: str) -> APIFuture[types.LoadWeightsResponse]:
        start_time = time.time()
        future = await self.holder.aclient.weights.load(model_id=self._guaranteed_model_id(), path=path)
        return APIFuture(types.LoadWeightsResponse, self.holder, future, request_start_time=start_time, request_type="LoadWeights")

    @sync_only
    def save_weights_for_sampler(self, name: str) -> APIFuture[types.SaveWeightsForSamplerResponse]:
        return asyncio.run_coroutine_threadsafe(self.save_weights_for_sampler_async(name), self.holder.get_loop()).result()

    async def save_weights_for_sampler_async(self, name: str) -> APIFuture[types.SaveWeightsForSamplerResponse]:
        start_time = time.time()
        future = await self.holder.aclient.weights.save_for_sampler(model_id=self._guaranteed_model_id(), path=name)
        return APIFuture(types.SaveWeightsForSamplerResponse, self.holder, future, request_start_time=start_time, request_type="SaveWeightsForSampler")

    @sync_only
    def unload_model(self) -> APIFuture[types.UnloadModelResponse]:
        return asyncio.run_coroutine_threadsafe(self.unload_model_async(), self.holder.get_loop()).result()

    async def unload_model_async(self) -> APIFuture[types.UnloadModelResponse]:
        start_time = time.time()
        future = await self.holder.aclient.models.unload(model_id=self._guaranteed_model_id())
        return APIFuture(types.UnloadModelResponse, self.holder, future, request_start_time=start_time, request_type="UnloadModel")

    def get_info(self) -> types.GetInfoResponse:
        return self.holder.client.models.get_info(model_id=self._guaranteed_model_id())

    async def get_info_async(self) -> types.GetInfoResponse:
        return await self.holder.aclient.models.get_info(model_id=self._guaranteed_model_id())

    def get_tokenizer(self) -> "PreTrainedTokenizer":
        return _get_tokenizer(self._guaranteed_model_id(), self.holder)

    def create_sampling_client(self, model_path: str, retry_config: RetryConfig | None = None) -> SamplingClient:
        return SamplingClient(self.holder, model_path=model_path, retry_config=retry_config)

    def save_weights_and_get_sampling_client(self, name: str, retry_config: RetryConfig | None = None) -> SamplingClient:
        path = self.save_weights_for_sampler(name).result().path
        return SamplingClient(self.holder, model_path=path, retry_config=retry_config)

    async def save_weights_and_get_sampling_client_async(self, name: str, retry_config: RetryConfig | None = None) -> SamplingClient:
        save_weights_future = await self.save_weights_for_sampler_async(name)
        save_weights_result = await save_weights_future.result_async()
        model_path = save_weights_result.path
        return SamplingClient(self.holder, model_path=model_path, retry_config=retry_config)


@lru_cache(maxsize=100)
def get_retry_handler(name: str, retry_config: RetryConfig | None = None) -> RetryHandler:
    retry_config = retry_config or RetryConfig()
    return RetryHandler(config=retry_config, name=name)

class SamplingClient:
    def __init__(
        self,
        holder: InternalClientHolder,
        *,
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: RetryConfig | None = None,
    ):
        if model_path and not model_path.startswith("tinker://"):
            raise ValueError("model_path must start with 'tinker://'")

        self.holder = holder
        self.model_path = model_path
        self.base_model = base_model

        # Create retry handler with the provided configuration
        self.retry_handler = get_retry_handler(model_path or base_model, retry_config=retry_config)

    @sync_only
    def sample(self, prompt: types.ModelInput, num_samples: int, sampling_params: types.SamplingParams,  include_prompt_logprobs: bool = False) -> ConcurrentFuture[types.SampleResponse]:
        coro = self.sample_async(prompt, num_samples, sampling_params, include_prompt_logprobs)
        return asyncio.run_coroutine_threadsafe(coro, self.holder.get_loop())

    async def _sample_async_internal(self, prompt: types.ModelInput, num_samples: int, sampling_params: types.SamplingParams, include_prompt_logprobs: bool, timeout: float) -> types.SampleResponse:
        """Internal method that does the actual API call without retry logic."""
        return await self.holder.aclient.sampling.sample(
            num_samples=num_samples,
            prompt=cast(types._ModelInputParam, prompt.model_dump()),
            sampling_params=cast(types._SamplingParamsParam, sampling_params.model_dump()),
            model_path=self.model_path if self.model_path is not None else NOT_GIVEN,
            prompt_logprobs=include_prompt_logprobs,
            base_model=self.base_model if self.base_model is not None else NOT_GIVEN,
            max_retries=0,
            timeout=timeout,
        )

    async def sample_async(self, prompt: types.ModelInput, num_samples: int, sampling_params: types.SamplingParams,  include_prompt_logprobs: bool = False) -> types.SampleResponse:
        """Execute sample request using the retry handler."""
        timeout = 60.0 + (sampling_params.max_tokens or 1000) / 10.0 # 10 token per second
        # TODO make max_tokens a required field
        return await self.retry_handler.execute(
            self._sample_async_internal,
            request_timeout=timeout, # used by RetryHandler.execute
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=sampling_params,
            include_prompt_logprobs=include_prompt_logprobs,
            timeout=timeout, # used by _sample_async_internal
        )

    def sample_sync_for_debugging(self, prompt: types.ModelInput, num_samples: int, sampling_params: types.SamplingParams,  include_prompt_logprobs: bool = False) -> ConcurrentFuture[types.SampleResponse]:
        # just do it sync without any funny business, for speed comparison purposes
        return ResolvedFuture(self.holder.client.sampling.sample(
            num_samples=num_samples,
            prompt=cast(types._ModelInputParam, prompt.model_dump()),
            sampling_params=cast(types._SamplingParamsParam, sampling_params.model_dump()),
            model_path=self.model_path if self.model_path is not None else NOT_GIVEN,
            prompt_logprobs=include_prompt_logprobs,
            base_model=self.base_model if self.base_model is not None else NOT_GIVEN
        ))

    def compute_logprobs(self, prompt: types.ModelInput) -> ConcurrentFuture[Sequence[float | None]]:
        return asyncio.run_coroutine_threadsafe(self.compute_logprobs_async(prompt), self.holder.get_loop())

    async def compute_logprobs_async(self, prompt: types.ModelInput) -> Sequence[float | None]:
        return await self.retry_handler.execute(self._compute_logprobs_async_internal, request_timeout=60.0, prompt=prompt)

    async def _compute_logprobs_async_internal(self, prompt: types.ModelInput) -> list[float | None]:
        sample_res = await self.holder.aclient.sampling.sample(
            num_samples=1,
            prompt=cast(types._ModelInputParam, prompt.model_dump()),
            sampling_params=types._SamplingParamsParam(max_tokens=1),
            model_path=self.model_path if self.model_path is not None else NOT_GIVEN,
            prompt_logprobs=True,
            base_model=self.base_model if self.base_model is not None else NOT_GIVEN,
            max_retries=0,
        )
        return cast(list[float | None], sample_res.prompt_logprobs)


def _get_tokenizer(
    model_id: types.ModelID, holder: InternalClientHolder
) -> "PreTrainedTokenizer":

    # call get_info on model_id
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    info = holder.client.models.get_info(model_id=model_id)
    model_name = info.model_data.model_name
    assert model_name is not None, "This shouldn't happen: model_name is None"

    
    return AutoTokenizer.from_pretrained(model_name, fast=True)

def _get_default_headers() -> dict[str, str]:
    headers = {}

    if (
        api_key := os.environ.get("TINKER_API_KEY", "")
    ) and "X-API-Key" not in headers:
        headers["X-API-Key"] = api_key

    headers["X-Username"] = os.environ.get("USER", "")

    if (
        client_id := os.environ.get("CLOUDFLARE_ACCESS_CLIENT_ID")
    ) and "CF-Access-Client-Id" not in headers:
        headers["CF-Access-Client-Id"] = client_id
    if (
        client_secret := os.environ.get("CLOUDFLARE_ACCESS_CLIENT_SECRET")
    ) and "CF-Access-Client-Secret" not in headers:
        headers["CF-Access-Client-Secret"] = client_secret
    return headers
