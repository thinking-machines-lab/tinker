"""SamplingClient for Tinker API."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import Future as ConcurrentFuture
from functools import lru_cache
from typing import TYPE_CHECKING, TypeVar, cast

import tinker
from tinker import types
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.public_interfaces.api_future import AwaitableConcurrentFuture
from tinker.lib.telemetry import Telemetry, capture_exceptions
from tinker.lib.telemetry_provider import TelemetryProvider

from ..api_future_impl import QueueState, QueueStateObserver, _APIFuture
from ..retry_handler import RetryConfig, RetryHandler

if TYPE_CHECKING:
    from ..internal_client_holder import InternalClientHolder

# pyright: reportPrivateImportUsage=false

logger = logging.getLogger(__name__)

U = TypeVar("U")


class SamplingClient(TelemetryProvider, QueueStateObserver):
    """Client for text generation and inference from trained or base models.

    The SamplingClient lets you generate text tokens from either a base model or from weights
    you've saved using a TrainingClient. You typically get one by calling
    `service_client.create_sampling_client()` or `training_client.save_weights_and_get_sampling_client()`.
    Key methods:
    - sample() - generate text completions with customizable parameters
    - compute_logprobs() - get log probabilities for prompt tokens

    Args:
        holder: Internal client managing HTTP connections and async operations
        model_path: Path to saved model weights (starts with 'tinker://')
        base_model: Name of base model to use for inference
        retry_config: Configuration for retrying failed requests

    Example:
        >>> sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen2.5-7B")
        >>> prompt = types.ModelInput.from_ints(tokenizer.encode("The weather today is"))
        >>> params = types.SamplingParams(max_tokens=20, temperature=0.7)
        >>> future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
        >>> result = future.result()
    """

    def __init__(
        self,
        holder: InternalClientHolder,
        *,
        model_path: str | None = None,
        base_model: str | None = None,
        sampling_session_id: str | None = None,
        retry_config: RetryConfig | None = None,
    ):
        if model_path and not model_path.startswith("tinker://"):
            raise ValueError("model_path must start with 'tinker://'")

        self.holder = holder
        self.model_path = model_path
        self.base_model = base_model

        # Create retry handler with the provided configuration
        self.retry_handler = _get_retry_handler(
            model_path or base_model, retry_config=retry_config, telemetry=holder.get_telemetry()
        )

        self.feature_gates = set(
            os.environ.get("TINKER_FEATURE_GATES", "async_sampling").split(",")
        )

        self._last_queue_state_logged: float = 0

        self._sampling_session_id: str = (
            sampling_session_id
            or holder.run_coroutine_threadsafe(
                holder._create_sampling_session(model_path=model_path, base_model=base_model)
            ).result()
        )

        self._request_id_counter: int = 0

    async def _send_asample_request(
        self,
        num_samples: int,
        prompt: types.ModelInput,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool,
        topk_prompt_logprobs: int,
    ):
        try:
            request_id = self._request_id_counter
            self._request_id_counter += 1
            request = types.SampleRequest(
                sampling_session_id=self._sampling_session_id,
                seq_id=request_id,
                num_samples=num_samples,
                prompt=prompt,
                sampling_params=sampling_params,
                prompt_logprobs=include_prompt_logprobs,
                topk_prompt_logprobs=topk_prompt_logprobs,
            )
            with self.holder.aclient(ClientConnectionPoolType.SAMPLE) as client:
                return await client.sampling.asample(
                    request=request,
                    max_retries=0,
                    extra_headers={"X-Tinker-Sampling-Backpressure": "1"},
                )
        except tinker.APIStatusError as e:
            if e.status_code == 429:
                return None
            raise e

    async def _sample_async_impl(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool,
        topk_prompt_logprobs: int = 0,
    ) -> types.SampleResponse:
        async with self.holder._sample_dispatch_semaphore:
            while True:
                if (
                    self.holder._sample_backoff_until is not None
                    and time.time() < self.holder._sample_backoff_until
                ):
                    await asyncio.sleep(1)
                    continue

                untyped_future = await self.holder.execute_with_retries(
                    self._send_asample_request,
                    num_samples,
                    prompt,
                    sampling_params,
                    include_prompt_logprobs,
                    topk_prompt_logprobs,
                )
                if untyped_future is not None:
                    break
                # Handle backoff
                self.holder._sample_backoff_until = time.time() + 1
                continue

        return await _APIFuture(
            types.SampleResponse,
            self.holder,
            untyped_future,
            request_start_time=time.time(),
            request_type="Sample",
            queue_state_observer=self,
        ).result_async()

    @capture_exceptions(fatal=True)
    def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> ConcurrentFuture[types.SampleResponse]:
        """Internal method that does the actual API call without retry logic."""

        async def _sample_async():
            return await self._sample_async_impl(
                prompt,
                num_samples,
                sampling_params,
                include_prompt_logprobs,
                topk_prompt_logprobs,
            )

        @capture_exceptions(fatal=True)
        async def _sample_async_with_retries() -> types.SampleResponse:
            return await self.retry_handler.execute(_sample_async)

        # TODO make max_tokens a required field
        return self.holder.run_coroutine_threadsafe(_sample_async_with_retries()).future()

    async def sample_async(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> types.SampleResponse:
        return await AwaitableConcurrentFuture(
            self.sample(
                prompt,
                num_samples,
                sampling_params,
                include_prompt_logprobs,
                topk_prompt_logprobs,
            )
        )

    @capture_exceptions(fatal=True)
    def compute_logprobs(self, prompt: types.ModelInput) -> ConcurrentFuture[list[float | None]]:
        async def _compute_logprobs_async() -> list[float | None]:
            sample_res = await self._sample_async_impl(
                prompt,
                num_samples=1,
                sampling_params=types.SamplingParams(max_tokens=1),
                include_prompt_logprobs=True,
            )
            return cast(list[float | None], sample_res.prompt_logprobs)

        @capture_exceptions(fatal=True)
        async def _compute_logprobs_async_with_retries() -> list[float | None]:
            return await self.retry_handler.execute(_compute_logprobs_async)

        return self.holder.run_coroutine_threadsafe(_compute_logprobs_async_with_retries()).future()

    async def compute_logprobs_async(self, prompt: types.ModelInput) -> list[float | None]:
        return await AwaitableConcurrentFuture(self.compute_logprobs(prompt))

    def get_telemetry(self) -> Telemetry | None:
        return self.holder.get_telemetry()

    def on_queue_state_change(self, queue_state: QueueState) -> None:
        QUEUE_STATE_LOG_INTERVAL = 60
        if queue_state == QueueState.ACTIVE:
            return
        if time.time() - self._last_queue_state_logged < QUEUE_STATE_LOG_INTERVAL:
            return
        if queue_state == QueueState.PAUSED_RATE_LIMIT:
            reason = "concurrent LoRA rate limit hit"
        elif queue_state == QueueState.PAUSED_CAPACITY:
            reason = "out of capacity"
        else:
            reason = "unknown"
        self._last_queue_state_logged = time.time()

        logger.warning(f"Sampling is paused for {self.model_path}. Reason: {reason}")


@lru_cache(maxsize=100)
def _get_retry_handler(
    name: str, retry_config: RetryConfig | None = None, telemetry: Telemetry | None = None
) -> RetryHandler:
    retry_config = retry_config or RetryConfig()
    return RetryHandler(config=retry_config, name=name, telemetry=telemetry)
