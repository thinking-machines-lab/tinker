"""
These are the public clients.
We've split up the internal client into these separate parts to make it easier to
create different implementations of the TrainingClient and SamplingClient so the
nontrivial algorithmic code for training and evaluating models can be run on either
the Tinker service or these alternative implementations.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from contextlib import contextmanager
import logging
import os
import threading
import time
from concurrent.futures import Future as ConcurrentFuture
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Dict, Generic, List, Sequence, Type, TypeVar, cast

import httpx
import tinker
from tinker import types
from tinker._types import NOT_GIVEN
from tinker.types import training_optim_step_params

from .._models import BaseModel
from .retry_handler import RetryConfig, RetryHandler, RetryableException
from .sync_only import sync_only
from .chunked_fwdbwd_helpers import combine_fwd_bwd_output_results

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

# FwdBwdChunkSize
CHUNK_SIZE = 128  # TODO: pick this less arbitrarily


# TODO maybe consider binding to BaseModel. But currently there's at one class (OptimStepResponse) that isn't a BaseModel.

MODEL_ID_NOT_SET_ERROR = "model_id must be set before calling forward. Try initializing the TrainingClient with a model_id by either calling create_lora_training_client on the ServiceClient, or initiliazing the TrainingClient with an existing model_id."

class ResolvedFuture(Generic[T], ConcurrentFuture[T]):
    def __init__(self, result: T):
        self._result = result

    def result(self, timeout: float | None = None) -> T:
        # This is typed to not be None, but it might be valid to return None for some T
        return self._result  # type: ignore


MAX_REQUESTS_PER_HTTPX_CLIENT = 100


class AwaitableConcurrentFuture(Generic[T]):
    def __init__(self, future: ConcurrentFuture[T]):
        self._future : ConcurrentFuture[T] = future

    def __await__(self):
        return asyncio.wrap_future(self._future).__await__()

    def result(self, timeout: float | None = None) -> T:
        return self._future.result(timeout)

    def future(self) -> ConcurrentFuture[T]:
        return self._future



class InternalClientHolder:
    def __init__(self, **kwargs: Any):
        self._constructor_kwargs = kwargs
        # So we can use async eventloop for parallel sampling requests
        # in sync code.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._clients: list[tinker.AsyncTinker] = []
        self._client_active_refcount: list[int] = []

    @contextmanager
    def aclient(self):
        client_idx = -1
        for i, ref_count in enumerate(self._client_active_refcount):
            if ref_count < MAX_REQUESTS_PER_HTTPX_CLIENT:
                client_idx = i
                break
        if client_idx == -1:
            self._clients.append(tinker.AsyncTinker(**self._constructor_kwargs))
            client_idx = len(self._clients) - 1
            self._client_active_refcount.append(1)

        self._client_active_refcount[client_idx] += 1
        try:
            yield self._clients[client_idx]
        finally:
            self._client_active_refcount[client_idx] -= 1

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._start_background_thread()
        assert self._loop is not None, "Background thread not started"
        return self._loop


    def run_coroutine_threadsafe(self, coro : Coroutine[Any, Any, T]) -> AwaitableConcurrentFuture[T]:
        return AwaitableConcurrentFuture(asyncio.run_coroutine_threadsafe(coro, self._get_loop()))


    def _start_background_thread(self):
        assert self._thread is None, "Background thread already started"
        self._thread = threading.Thread(target=self._background_thread_func, daemon=True)
        self._thread.start()
        self._started.wait()

    def _background_thread_func(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
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


class APIFuture(ABC, Generic[T]):
    @abstractmethod
    async def result_async(self, timeout: float | None = None) -> T:
        raise NotImplementedError

    @abstractmethod
    def result(self, timeout: float | None = None) -> T:
        raise NotImplementedError


class _APIFuture(APIFuture[T]):
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
        self._future = self.holder.run_coroutine_threadsafe(self._result_async())

    async def _result_async(self, timeout: float | None = None) -> T:
        """Get the result of this future, with automatic retries for transient errors."""
        if self._cached_result is not _UNCOMPUTED:
            return cast(T, self._cached_result)

        start_time = time.time()
        iteration = -1
        connection_error_retries = 0

        while True:
            iteration += 1

            if timeout is not None and time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout of {timeout} seconds reached while waiting for result of {self.request_id=}")

            # Headers for telemetry
            headers = {
                "X-Tinker-Request-Iteration": str(iteration),
                "X-Tinker-Request-Type": self.request_type,
            }
            if iteration == 0:
                headers["X-Tinker-Create-Promise-Roundtrip-Time"] = str(self.request_queue_roundtrip_time)

            # Function hasn't been called yet, execute it now
            try:
                with self.holder.aclient() as client:
                    response = await client.futures.with_raw_response.retrieve(
                        request_id=self.request_id, timeout=45, extra_headers=headers, max_retries=0
                    )
            except tinker.APIStatusError as e:
                connection_error_retries = 0
                # Retry 408s until we time out
                if e.status_code == 408:
                    continue
                if e.status_code == 410:
                    raise RetryableException(
                        message=f"Promise expired/broken for request {self.untyped_future.request_id}"
                    ) from e
                if e.status_code in range(500, 600):
                    continue
                raise ValueError(f"Error retrieving result: {e} with status code {e.status_code=} for {self.request_id=} and expected type {self.model_cls=}") from e
            except tinker.APIConnectionError as e:
                # Retry all connection errors with exponential backoff
                await asyncio.sleep(min(2 ** connection_error_retries, 30))
                connection_error_retries += 1
                continue

            # Function hasn't been called yet, execute it now
            result_dict: Dict[str, Any] = await response.json() # type: ignore

            if "type" in result_dict and result_dict["type"] == "try_again":
                logger.warning(f"Retrying request {self.request_id=} because of try_again")
                continue

            if "error" in result_dict:
                raise ValueError(f"Error retrieving result: {result_dict} for {self.request_id=} and expected type {self.model_cls=}")

            try:
                # Check if model_cls is a BaseModel subclass before calling model_validate
                if issubclass(self.model_cls, BaseModel):
                    self._cached_result = self.model_cls.model_validate(result_dict)
                else:
                    # For non-BaseModel types, just return the result directly
                    self._cached_result = result_dict
                return cast(T, self._cached_result)
            except Exception as e:
                raise ValueError(f"Error retrieving result: {e} for {self.request_id=} and expected type {self.model_cls=}") from e

    @property
    def request_id(self) -> str:
        return self.untyped_future.request_id

    @sync_only
    def result(self, timeout: float | None = None) -> T:
        return self._future.result(timeout)

    async def result_async(self, timeout: float | None = None) -> T:
        return await asyncio.wait_for(self._future, timeout)


class _CombinedAPIFuture(APIFuture[T]):
    def __init__(self, futures: List[APIFuture[T]], transform: Callable[[List[T]], T], holder: InternalClientHolder):
        self.futures = futures
        self.transform = transform
        self.holder = holder

    @sync_only
    def result(self, timeout: float | None = None) -> T:
        return self.holder.run_coroutine_threadsafe(self.result_async(timeout)).result()

    async def result_async(self, timeout: float | None = None) -> T:
        results = await asyncio.gather(*[future.result_async(timeout) for future in self.futures])
        return self.transform(results)


class ServiceClient:
    def __init__(self, **kwargs: Any):
        default_headers = _get_default_headers() | kwargs.pop("default_headers", {})
        self.holder = InternalClientHolder(**kwargs, default_headers=default_headers, _strict_response_validation=True)

    def _get_server_capabilities_submit(self) -> AwaitableConcurrentFuture[types.GetServerCapabilitiesResponse]:
        async def _get_server_capabilities_async():
            with self.holder.aclient() as client:
                return await client.service.get_server_capabilities()
        return self.holder.run_coroutine_threadsafe(_get_server_capabilities_async())

    @sync_only
    def get_server_capabilities(self) -> types.GetServerCapabilitiesResponse:
        return self._get_server_capabilities_submit().result()

    async def get_server_capabilities_async(self) -> types.GetServerCapabilitiesResponse:
        return await self._get_server_capabilities_submit()

    def _create_model_submit(self, base_model: str, lora_config: types.LoraConfig) -> AwaitableConcurrentFuture[types.ModelID]:
        async def _create_model_async():
            start_time = time.time()
            with self.holder.aclient() as client:
                future = await client.models.create(base_model=base_model, lora_config=to_lora_config_params(lora_config))
            create_model_response = await _APIFuture(types.CreateModelResponse, self.holder, future, request_start_time=start_time, request_type="CreateModel").result_async()
            return create_model_response.model_id
        return self.holder.run_coroutine_threadsafe(_create_model_async())

    @sync_only
    def create_lora_training_client(self, base_model: str, rank: int = 32, seed: int | None = None) -> "TrainingClient":
        model_id = self._create_model_submit(base_model, types.LoraConfig(rank=rank, seed=seed)).result()
        logger.info(f"Creating TrainingClient for {model_id=}")
        return self.create_training_client(model_id)

    async def create_lora_training_client_async(self, base_model: str, rank: int = 32, seed: int | None = None) -> "TrainingClient":
        model_id = await self._create_model_submit(base_model, types.LoraConfig(rank=rank, seed=seed))
        logger.info(f"Creating TrainingClient for {model_id=}")
        return self.create_training_client(model_id)

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

    def _forward_submit(self, data: List[types.Datum], loss_fn: types.LossFnType) -> AwaitableConcurrentFuture[_CombinedAPIFuture[types.ForwardBackwardOutput]]:
        async def _forward_async():
            start_time = time.time()
            futures = []
            for i in range(0, len(data), CHUNK_SIZE):
                with self.holder.aclient() as client:
                    untyped_future = await client.training.forward(
                        model_id=self._guaranteed_model_id(),
                        forward_input=to_fwdbwd_input_params(
                            types.ForwardBackwardInput(data=data[i:i+CHUNK_SIZE], loss_fn=loss_fn)
                        )
                    )
                api_future = _APIFuture(types.ForwardBackwardOutput, self.holder, untyped_future, request_start_time=start_time, request_type="Forward")
                futures.append(api_future)
            return _CombinedAPIFuture(futures, combine_fwd_bwd_output_results, self.holder)
        return self.holder.run_coroutine_threadsafe(_forward_async())

    @sync_only
    def forward(self, data: List[types.Datum], loss_fn: types.LossFnType) -> APIFuture[types.ForwardBackwardOutput]:
        return self._forward_submit(data, loss_fn).result()

    async def forward_async(self, data: List[types.Datum], loss_fn: types.LossFnType) -> APIFuture[types.ForwardBackwardOutput]:
        return await self._forward_submit(data, loss_fn)

    def _forward_backward_submit(self, data: List[types.Datum], loss_fn: types.LossFnType) -> AwaitableConcurrentFuture[_CombinedAPIFuture[types.ForwardBackwardOutput]]:
        async def _forward_backward_async():
            futures = []
            start_time = time.time()

            for i in range(0, len(data), CHUNK_SIZE):
                with self.holder.aclient() as client:
                    untyped_future = await client.training.forward_backward(
                        model_id=self._guaranteed_model_id(),
                        forward_backward_input=to_fwdbwd_input_params(
                        types.ForwardBackwardInput(data=data[i:i+CHUNK_SIZE], loss_fn=loss_fn)
                    )
                )
                api_future = _APIFuture(types.ForwardBackwardOutput, self.holder, untyped_future, request_start_time=start_time, request_type="ForwardBackward")
                futures.append(api_future)

            return _CombinedAPIFuture(futures, combine_fwd_bwd_output_results, self.holder)
        return self.holder.run_coroutine_threadsafe(_forward_backward_async())

    @sync_only
    def forward_backward(self, data: List[types.Datum], loss_fn: types.LossFnType) -> APIFuture[types.ForwardBackwardOutput]:
        return self._forward_backward_submit(data, loss_fn).result()

    async def forward_backward_async(self, data: List[types.Datum], loss_fn: types.LossFnType) -> APIFuture[types.ForwardBackwardOutput]:
        return await self._forward_backward_submit(data, loss_fn)

    def _optim_step_submit(self, adam_params: types.AdamParams) -> AwaitableConcurrentFuture[_APIFuture[types.OptimStepResponse]]:
        async def _optim_step_async():
            start_time = time.time()
            with self.holder.aclient() as client:
                future = await client.training.optim_step(
                    model_id=self._guaranteed_model_id(), adam_params=to_adam_params(adam_params)
                )
            return _APIFuture(types.OptimStepResponse, self.holder, future, request_start_time=start_time, request_type="OptimStep")
        return self.holder.run_coroutine_threadsafe(_optim_step_async())

    @sync_only
    def optim_step(self, adam_params: types.AdamParams) -> APIFuture[types.OptimStepResponse]:
        return self._optim_step_submit(adam_params).result()

    async def optim_step_async(self, adam_params: types.AdamParams) -> APIFuture[types.OptimStepResponse]:
        return await self._optim_step_submit(adam_params)

    def _save_state_submit(self, name: str) -> AwaitableConcurrentFuture[_APIFuture[types.SaveWeightsResponse]]:
        async def _save_state_async():
            start_time = time.time()
            with self.holder.aclient() as client:
                future = await client.weights.save(model_id=self._guaranteed_model_id(), path=name)
            return _APIFuture(types.SaveWeightsResponse, self.holder, future, request_start_time=start_time, request_type="SaveWeights")
        return self.holder.run_coroutine_threadsafe(_save_state_async())

    @sync_only
    def save_state(self, name: str) -> APIFuture[types.SaveWeightsResponse]:
        return self._save_state_submit(name).result()

    async def save_state_async(self, name: str) -> APIFuture[types.SaveWeightsResponse]:
        return await self._save_state_submit(name)

    def _load_state_submit(self, path: str) -> AwaitableConcurrentFuture[_APIFuture[types.LoadWeightsResponse]]:
        async def _load_state_async():
            start_time = time.time()
            with self.holder.aclient() as client:
                future = await client.weights.load(model_id=self._guaranteed_model_id(), path=path)
            return _APIFuture(types.LoadWeightsResponse, self.holder, future, request_start_time=start_time, request_type="LoadWeights")
        return self.holder.run_coroutine_threadsafe(_load_state_async())

    @sync_only
    def load_state(self, path: str) -> APIFuture[types.LoadWeightsResponse]:
        return self._load_state_submit(path).result()

    async def load_state_async(self, path: str) -> APIFuture[types.LoadWeightsResponse]:
        return await self._load_state_submit(path)

    def _save_weights_for_sampler_submit(self, name: str) -> AwaitableConcurrentFuture[_APIFuture[types.SaveWeightsForSamplerResponse]]:
        async def _save_weights_for_sampler_async():
            start_time = time.time()
            with self.holder.aclient() as client:
                future = await client.weights.save_for_sampler(model_id=self._guaranteed_model_id(), path=name)
            return _APIFuture(types.SaveWeightsForSamplerResponse, self.holder, future, request_start_time=start_time, request_type="SaveWeightsForSampler")
        return self.holder.run_coroutine_threadsafe(_save_weights_for_sampler_async())

    @sync_only
    def save_weights_for_sampler(self, name: str) -> APIFuture[types.SaveWeightsForSamplerResponse]:
        return self._save_weights_for_sampler_submit(name).result()

    async def save_weights_for_sampler_async(self, name: str) -> APIFuture[types.SaveWeightsForSamplerResponse]:
        return await self._save_weights_for_sampler_submit(name)

    def _unload_model_submit(self) -> AwaitableConcurrentFuture[_APIFuture[types.UnloadModelResponse]]:
        async def _unload_model_async():
            start_time = time.time()
            with self.holder.aclient() as client:
                future = await client.models.unload(model_id=self._guaranteed_model_id())
            return _APIFuture(types.UnloadModelResponse, self.holder, future, request_start_time=start_time, request_type="UnloadModel")
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

        self.feature_gates = set(os.environ.get("TINKER_FEATURE_GATES", "async_sampling").split(","))


    def _sample_submit(self, prompt: types.ModelInput, num_samples: int, sampling_params: types.SamplingParams, include_prompt_logprobs: bool) -> AwaitableConcurrentFuture[types.SampleResponse]:
        """Internal method that does the actual API call without retry logic."""
        # This timeout can't be determined based on the sampling_params because it also depends on
        # the overall load of the system. So using a large value here.
        timeout = 30 * 60
        async def _sample_async():
            if "async_sampling" in self.feature_gates:
                async def _asample_with_retries():
                    start_time = time.time()
                    retries = 0
                    while True:
                        try:
                            with self.holder.aclient() as client:
                                return await client.sampling.asample(
                                    num_samples=num_samples,
                                    prompt=cast(types._ModelInputParam, prompt.model_dump()),
                                    sampling_params=cast(types._SamplingParamsParam, sampling_params.model_dump()),
                                    model_path=self.model_path if self.model_path is not None else NOT_GIVEN,
                                    prompt_logprobs=include_prompt_logprobs,
                                    base_model=self.base_model if self.base_model is not None else NOT_GIVEN,
                                    max_retries=0,
                            )
                        except tinker.APITimeoutError as e:
                            # Connect timeouts are safe to retry
                            if time.time() - start_time < timeout and e.__cause__ is not None and isinstance(e.__cause__, httpx.ConnectTimeout):
                                await asyncio.sleep(min(2 ** retries, 30))
                                retries += 1
                                continue
                            raise e
                untyped_future = await _asample_with_retries()
                return await _APIFuture(types.SampleResponse, self.holder, untyped_future, request_start_time=time.time(), request_type="Sample").result_async(timeout=timeout)
            else :
                with self.holder.aclient() as client:
                    return await client.sampling.sample(
                        num_samples=num_samples,
                        prompt=cast(types._ModelInputParam, prompt.model_dump()),
                        sampling_params=cast(types._SamplingParamsParam, sampling_params.model_dump()),
                        model_path=self.model_path if self.model_path is not None else NOT_GIVEN,
                        prompt_logprobs=include_prompt_logprobs,
                        base_model=self.base_model if self.base_model is not None else NOT_GIVEN,
                        max_retries=0,
                        timeout=timeout,
                        )

        # TODO make max_tokens a required field
        return self.holder.run_coroutine_threadsafe(
            self.retry_handler.execute(_sample_async, request_timeout=timeout))

    @sync_only
    def sample(self, prompt: types.ModelInput, num_samples: int, sampling_params: types.SamplingParams,  include_prompt_logprobs: bool = False) -> ConcurrentFuture[types.SampleResponse]:
        return self._sample_submit(prompt, num_samples, sampling_params, include_prompt_logprobs).future()

    async def sample_async(self, prompt: types.ModelInput, num_samples: int, sampling_params: types.SamplingParams,  include_prompt_logprobs: bool = False) -> types.SampleResponse:
        return await self._sample_submit(prompt, num_samples, sampling_params, include_prompt_logprobs)

    def _compute_logprobs_submit(self, prompt: types.ModelInput) -> AwaitableConcurrentFuture[Sequence[float | None]]:
        async def _compute_logprobs_async():
            with self.holder.aclient() as client:
                sample_res = await client.sampling.sample(
                    num_samples=1,
                    prompt=cast(types._ModelInputParam, prompt.model_dump()),
                    sampling_params=types._SamplingParamsParam(max_tokens=1),
                    model_path=self.model_path if self.model_path is not None else NOT_GIVEN,
                    prompt_logprobs=True,
                    base_model=self.base_model if self.base_model is not None else NOT_GIVEN,
                )
                return cast(list[float | None], sample_res.prompt_logprobs)
        return self.holder.run_coroutine_threadsafe(self.retry_handler.execute(_compute_logprobs_async, request_timeout=60.0))

    @sync_only
    def compute_logprobs(self, prompt: types.ModelInput) -> ConcurrentFuture[Sequence[float | None]]:
        return self._compute_logprobs_submit(prompt).future()

    async def compute_logprobs_async(self, prompt: types.ModelInput) -> Sequence[float | None]:
        return await self._compute_logprobs_submit(prompt)


def _get_tokenizer(
    model_id: types.ModelID, holder: InternalClientHolder
) -> "PreTrainedTokenizer":

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
