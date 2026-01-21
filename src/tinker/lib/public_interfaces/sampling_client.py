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
from tinker.lib.public_interfaces.api_future import APIFuture, AwaitableConcurrentFuture
from tinker.lib.telemetry import Telemetry, capture_exceptions
from tinker.lib.telemetry_provider import TelemetryProvider

from ..api_future_impl import QueueState, QueueStateObserver, _APIFuture
from ..retry_handler import RetryConfig, RetryHandler

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

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

    Create method parameters:
    - `model_path`: Path to saved model weights (starts with 'tinker://')
    - `base_model`: Name of base model to use for inference (e.g., 'Qwen/Qwen3-8B')
    - `retry_config`: Configuration for retrying failed requests

    Example:
    ```python
    sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3-8B")
    prompt = types.ModelInput.from_ints(tokenizer.encode("The weather today is"))
    params = types.SamplingParams(max_tokens=20, temperature=0.7)
    future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
    result = future.result()
    ```
    """

    def __init__(
        self,
        holder: InternalClientHolder,
        *,
        sampling_session_id: str,
        retry_config: RetryConfig | None = None,
    ):
        self.holder = holder

        # Create retry handler with the provided configuration
        self.retry_handler = _get_retry_handler(
            sampling_session_id, retry_config=retry_config, telemetry=holder.get_telemetry()
        )

        self.feature_gates = set(
            os.environ.get("TINKER_FEATURE_GATES", "async_sampling").split(",")
        )

        self._last_queue_state_logged: float = 0

        self._sampling_session_id: str = sampling_session_id

        self._request_id_counter: int = 0

    @staticmethod
    async def _create_impl(
        holder: InternalClientHolder,
        *,
        model_path: str | None,
        base_model: str | None,
        sampling_session_id: str | None,
        retry_config: RetryConfig | None,
    ) -> SamplingClient:
        if sampling_session_id is None:
            sampling_session_id = await holder._create_sampling_session(
                model_path=model_path, base_model=base_model
            )
        return SamplingClient(
            holder, sampling_session_id=sampling_session_id, retry_config=retry_config
        )

    @staticmethod
    def create(
        holder: InternalClientHolder,
        *,
        model_path: str | None = None,
        base_model: str | None = None,
        sampling_session_id: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> APIFuture[SamplingClient]:
        return holder.run_coroutine_threadsafe(
            SamplingClient._create_impl(
                holder,
                model_path=model_path,
                base_model=base_model,
                sampling_session_id=sampling_session_id,
                retry_config=retry_config,
            )
        )

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
        estimated_bytes_count = self.holder.estimate_bytes_count_in_model_input(prompt)
        async with self.holder.sample_dispatch_rate_limit(estimated_bytes_count):
            while True:
                if (
                    self.holder._sample_backoff_until is not None
                    and time.monotonic() < self.holder._sample_backoff_until
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
                backoff_duration = 1 if estimated_bytes_count <= 128 * 1024 else 5
                self.holder._sample_backoff_until = time.monotonic() + backoff_duration
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
        """Generate text completions from the model.

        Args:
        - `prompt`: The input tokens as ModelInput
        - `num_samples`: Number of independent samples to generate
        - `sampling_params`: Parameters controlling generation (temperature, max_tokens, etc.)
        - `include_prompt_logprobs`: Whether to include log probabilities for prompt tokens
        - `topk_prompt_logprobs`: Number of top token log probabilities to return per position

        Returns:
        - A `Future` containing the `SampleResponse` with generated text

        Example:
        ```python
        prompt = types.ModelInput.from_ints(tokenizer.encode("The weather today is"))
        params = types.SamplingParams(max_tokens=20, temperature=0.7)
        future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
        result = future.result()
        for sample in result.samples:
            print(tokenizer.decode(sample.tokens))
        ```
        """

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
        """Async version of sample."""
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
        """Compute log probabilities for prompt tokens.

        Args:
        - `prompt`: The input tokens as ModelInput

        Returns:
        - A `Future` containing a list of log probabilities for each token in the prompt.
            None values indicate tokens where log probabilities couldn't be computed.

        Example:
        ```python
        prompt = types.ModelInput.from_ints(tokenizer.encode("Hello world"))
        future = sampling_client.compute_logprobs(prompt)
        logprobs = future.result()
        for i, logprob in enumerate(logprobs):
            if logprob is not None:
                print(f"Token {i}: logprob = {logprob:.4f}")
        ```
        """

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
        """Async version of compute_logprobs."""
        return await AwaitableConcurrentFuture(self.compute_logprobs(prompt))

    @capture_exceptions(fatal=True)
    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer for the current model.

        Returns:
        - `PreTrainedTokenizer` compatible with the model
        """

        async def _get_sampler_async():
            async def _send_request():
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.get(
                        f"/api/v1/samplers/{self._sampling_session_id}",
                        cast_to=types.GetSamplerResponse,
                    )

            return await self.holder.execute_with_retries(_send_request)

        sampler_info = self.holder.run_coroutine_threadsafe(_get_sampler_async()).result()
        return _load_tokenizer_from_model_info(sampler_info.base_model)

    def get_telemetry(self) -> Telemetry | None:
        return self.holder.get_telemetry()

    def on_queue_state_change(
        self, queue_state: QueueState, queue_state_reason: str | None
    ) -> None:
        QUEUE_STATE_LOG_INTERVAL = 60
        if queue_state == QueueState.ACTIVE:
            return
        if time.time() - self._last_queue_state_logged < QUEUE_STATE_LOG_INTERVAL:
            return
        if not queue_state_reason:
            if queue_state == QueueState.PAUSED_RATE_LIMIT:
                queue_state_reason = "concurrent sampler weights limit hit"
            elif queue_state == QueueState.PAUSED_CAPACITY:
                queue_state_reason = "Tinker backend is running short on capacity, please wait"
            else:
                queue_state_reason = "unknown"
        self._last_queue_state_logged = time.time()

        logger.warning(
            f"Sampling is paused for sampler {self._sampling_session_id}. Reason: {queue_state_reason}"
        )


@lru_cache(maxsize=100)
def _get_retry_handler(
    name: str, retry_config: RetryConfig | None = None, telemetry: Telemetry | None = None
) -> RetryHandler:
    retry_config = retry_config or RetryConfig()
    return RetryHandler(config=retry_config, name=name, telemetry=telemetry)


@lru_cache(maxsize=32)
def _load_tokenizer_from_model_info(
    model_name: str, tokenizer_id: str | None = None
) -> PreTrainedTokenizer:
    """Load a tokenizer given a model name and optional tokenizer_id.

    This is a shared helper used by both TrainingClient and SamplingClient.

    Args:
        model_name: The model name (e.g., "Qwen/Qwen3-8B")
        tokenizer_id: Optional explicit tokenizer ID. If None, heuristics are applied.

    Returns:
        The loaded PreTrainedTokenizer
    """
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    model_name = model_name.split(":")[0]

    # Use tokenizer_id if provided, otherwise fall back to heuristic logic
    kwargs = {}
    if tokenizer_id is None:
        # We generally adhere to the huggingface convention of "<org>/<model>" but
        # in some cases we'll deploy variants using the format
        # "<org>/<model>/<variant>". In that case, we want to load the tokenizer
        # using the huggingface convention.
        if model_name.startswith("meta-llama/Llama-3"):
            # Avoid gating of Llama 3 models:
            tokenizer_id = "thinkingmachineslabinc/meta-llama-3-instruct-tokenizer"
        elif model_name.count("/") == 2:
            org, model, _variant = model_name.split("/", 2)
            tokenizer_id = f"{org}/{model}"
        else:
            tokenizer_id = model_name

    if tokenizer_id.startswith("TML/"):
        from tml_tokenizers.tinker_tokenizers import get_tinker_tokenizer

        if (tokenizer := get_tinker_tokenizer(tokenizer_id)) is not None:
            return tokenizer

    if tokenizer_id == "moonshotai/Kimi-K2-Thinking":
        kwargs = {
            "trust_remote_code": True,
            "revision": "612681931a8c906ddb349f8ad0f582cb552189cd",
        }

    return AutoTokenizer.from_pretrained(tokenizer_id, fast=True, **kwargs)
