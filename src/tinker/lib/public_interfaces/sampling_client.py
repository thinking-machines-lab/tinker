"""SamplingClient for Tinker API."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Sequence
from concurrent.futures import Future as ConcurrentFuture
from functools import lru_cache
from typing import TYPE_CHECKING, TypeVar, cast

import httpx

import tinker
from tinker import types
from tinker._types import NOT_GIVEN

from ..retry_handler import RetryConfig, RetryHandler
from ..sync_only import sync_only
from .api_future import AwaitableConcurrentFuture, _APIFuture

if TYPE_CHECKING:
    from ..internal_client_holder import InternalClientHolder

# pyright: reportPrivateImportUsage=false

logger = logging.getLogger(__name__)

U = TypeVar("U")


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
        self.retry_handler = _get_retry_handler(model_path or base_model, retry_config=retry_config)

        self.feature_gates = set(
            os.environ.get("TINKER_FEATURE_GATES", "async_sampling").split(",")
        )

    def _sample_submit(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool,
    ) -> AwaitableConcurrentFuture[types.SampleResponse]:
        """Internal method that does the actual API call without retry logic."""
        # This timeout can't be determined based on the sampling_params because it also depends on
        # the overall load of the system. So using a large value here.
        timeout = 30 * 60

        async def _sample_async():
            if "async_sampling" in self.feature_gates:

                async def _asample_with_retries():
                    start_time = time.time()
                    retries = 0
                    async with self.holder._sample_dispatch_semaphore:
                        while True:
                            if (
                                self.holder._sample_backoff_until is not None
                                and time.time() < self.holder._sample_backoff_until
                            ):
                                await asyncio.sleep(1)
                                continue
                            try:
                                with self.holder.aclient() as client:
                                    return await client.sampling.asample(
                                        num_samples=num_samples,
                                        prompt=cast(types._ModelInputParam, prompt.model_dump()),
                                        sampling_params=cast(
                                            types._SamplingParamsParam, sampling_params.model_dump()
                                        ),
                                        model_path=self.model_path
                                        if self.model_path is not None
                                        else NOT_GIVEN,
                                        prompt_logprobs=include_prompt_logprobs,
                                        base_model=self.base_model
                                        if self.base_model is not None
                                        else NOT_GIVEN,
                                        max_retries=0,
                                        extra_headers={"X-Tinker-Sampling-Backpressure": "1"},
                                    )
                            except tinker.APIStatusError as e:
                                if e.status_code == 429:
                                    self.holder._sample_backoff_until = time.time() + 1
                                    continue
                                raise e
                            except tinker.APITimeoutError as e:
                                # Connect timeouts are safe to retry
                                if (
                                    time.time() - start_time < timeout
                                    and e.__cause__ is not None
                                    and isinstance(e.__cause__, httpx.ConnectTimeout)
                                ):
                                    await asyncio.sleep(min(2**retries, 30))
                                    retries += 1
                                    continue
                                raise e

                untyped_future = await _asample_with_retries()
                return await _APIFuture(
                    types.SampleResponse,
                    self.holder,
                    untyped_future,
                    request_start_time=time.time(),
                    request_type="Sample",
                ).result_async(timeout=timeout)
            else:
                with self.holder.aclient() as client:
                    return await client.sampling.sample(
                        num_samples=num_samples,
                        prompt=cast(types._ModelInputParam, prompt.model_dump()),
                        sampling_params=cast(
                            types._SamplingParamsParam, sampling_params.model_dump()
                        ),
                        model_path=self.model_path if self.model_path is not None else NOT_GIVEN,
                        prompt_logprobs=include_prompt_logprobs,
                        base_model=self.base_model if self.base_model is not None else NOT_GIVEN,
                        max_retries=0,
                        timeout=timeout,
                    )

        # TODO make max_tokens a required field
        return self.holder.run_coroutine_threadsafe(
            self.retry_handler.execute(_sample_async, request_timeout=timeout)
        )

    @sync_only
    def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
    ) -> ConcurrentFuture[types.SampleResponse]:
        return self._sample_submit(
            prompt, num_samples, sampling_params, include_prompt_logprobs
        ).future()

    async def sample_async(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
    ) -> types.SampleResponse:
        return await self._sample_submit(
            prompt, num_samples, sampling_params, include_prompt_logprobs
        )

    def _compute_logprobs_submit(
        self, prompt: types.ModelInput
    ) -> AwaitableConcurrentFuture[Sequence[float | None]]:
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

        return self.holder.run_coroutine_threadsafe(
            self.retry_handler.execute(_compute_logprobs_async, request_timeout=60.0)
        )

    @sync_only
    def compute_logprobs(
        self, prompt: types.ModelInput
    ) -> ConcurrentFuture[Sequence[float | None]]:
        return self._compute_logprobs_submit(prompt).future()

    async def compute_logprobs_async(self, prompt: types.ModelInput) -> Sequence[float | None]:
        return await self._compute_logprobs_submit(prompt)


@lru_cache(maxsize=100)
def _get_retry_handler(name: str, retry_config: RetryConfig | None = None) -> RetryHandler:
    retry_config = retry_config or RetryConfig()
    return RetryHandler(config=retry_config, name=name)
