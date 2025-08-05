# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal

import httpx

from ..types import sampling_sample_params, sampling_asample_params
from .._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from .._utils import maybe_transform, async_maybe_transform
from .._compat import cached_property
from .._resource import SyncAPIResource, AsyncAPIResource
from .._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from .._base_client import make_request_options
from ..types.sample_response import SampleResponse
from ..types.model_input_param import ModelInputParam
from ..types.sampling_params_param import SamplingParamsParam
from ..types.shared.untyped_api_future import UntypedAPIFuture

__all__ = ["SamplingResource", "AsyncSamplingResource"]


class SamplingResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> SamplingResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/stainless-sdks/tinker-python#accessing-raw-response-data-eg-headers
        """
        return SamplingResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> SamplingResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/stainless-sdks/tinker-python#with_streaming_response
        """
        return SamplingResourceWithStreamingResponse(self)

    def asample(
        self,
        *,
        num_samples: int = 1,
        prompt: ModelInputParam,
        sampling_params: SamplingParamsParam,
        base_model: str | NotGiven = NOT_GIVEN,
        model_path: str | NotGiven = NOT_GIVEN,
        prompt_logprobs: bool | NotGiven = NOT_GIVEN,
        type: Literal["sample"] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
    ) -> UntypedAPIFuture:
        """
        Generates samples from the model using the specified sampling parameters

        Args:
          num_samples: Number of samples to generate

          base_model: Optional base model name to sample from. Is inferred from model_path, if
              provided. If sampling against a base model, this is required.

          model_path: Optional tinker:// path to your model weights or LoRA weights. If not provided,
              samples against the base model.

          prompt_logprobs: If set to `true`, computes and returns logprobs on the prompt tokens. Defaults
              to false.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds

          idempotency_key: Specify a custom idempotency key for this request
        """
        return self._post(
            "/api/v1/asample",
            body=maybe_transform(
                {
                    "num_samples": num_samples,
                    "prompt": prompt,
                    "sampling_params": sampling_params,
                    "base_model": base_model,
                    "model_path": model_path,
                    "prompt_logprobs": prompt_logprobs,
                    "type": type,
                },
                sampling_asample_params.SamplingAsampleParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                idempotency_key=idempotency_key,
            ),
            cast_to=UntypedAPIFuture,
        )

    def sample(
        self,
        *,
        num_samples: int = 1,
        prompt: ModelInputParam,
        sampling_params: SamplingParamsParam,
        base_model: str | NotGiven = NOT_GIVEN,
        model_path: str | NotGiven = NOT_GIVEN,
        prompt_logprobs: bool | NotGiven = NOT_GIVEN,
        type: Literal["sample"] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
    ) -> SampleResponse:
        """
        Generates samples from the model using the specified sampling parameters

        Args:
          num_samples: Number of samples to generate

          base_model: Optional base model name to sample from. Is inferred from model_path, if
              provided. If sampling against a base model, this is required.

          model_path: Optional tinker:// path to your model weights or LoRA weights. If not provided,
              samples against the base model.

          prompt_logprobs: If set to `true`, computes and returns logprobs on the prompt tokens. Defaults
              to false.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds

          idempotency_key: Specify a custom idempotency key for this request
        """
        return self._post(
            "/api/v1/sample",
            body=maybe_transform(
                {
                    "num_samples": num_samples,
                    "prompt": prompt,
                    "sampling_params": sampling_params,
                    "base_model": base_model,
                    "model_path": model_path,
                    "prompt_logprobs": prompt_logprobs,
                    "type": type,
                },
                sampling_sample_params.SamplingSampleParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                idempotency_key=idempotency_key,
            ),
            cast_to=SampleResponse,
        )


class AsyncSamplingResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncSamplingResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/stainless-sdks/tinker-python#accessing-raw-response-data-eg-headers
        """
        return AsyncSamplingResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncSamplingResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/stainless-sdks/tinker-python#with_streaming_response
        """
        return AsyncSamplingResourceWithStreamingResponse(self)

    async def asample(
        self,
        *,
        num_samples: int = 1,
        prompt: ModelInputParam,
        sampling_params: SamplingParamsParam,
        base_model: str | NotGiven = NOT_GIVEN,
        model_path: str | NotGiven = NOT_GIVEN,
        prompt_logprobs: bool | NotGiven = NOT_GIVEN,
        type: Literal["sample"] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
        max_retries: int | NotGiven = NOT_GIVEN,
    ) -> UntypedAPIFuture:
        """
        Generates samples from the model using the specified sampling parameters

        Args:
          num_samples: Number of samples to generate

          base_model: Optional base model name to sample from. Is inferred from model_path, if
              provided. If sampling against a base model, this is required.

          model_path: Optional tinker:// path to your model weights or LoRA weights. If not provided,
              samples against the base model.

          prompt_logprobs: If set to `true`, computes and returns logprobs on the prompt tokens. Defaults
              to false.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds

          idempotency_key: Specify a custom idempotency key for this request
        """
        options = make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                idempotency_key=idempotency_key,
        )
        if max_retries is not NOT_GIVEN:
            options["max_retries"] = max_retries

        return await self._post(
            "/api/v1/asample",
            body=await async_maybe_transform(
                {
                    "num_samples": num_samples,
                    "prompt": prompt,
                    "sampling_params": sampling_params,
                    "base_model": base_model,
                    "model_path": model_path,
                    "prompt_logprobs": prompt_logprobs,
                    "type": type,
                },
                sampling_asample_params.SamplingAsampleParams,
            ),
            options=options,
            cast_to=UntypedAPIFuture,
        )

    async def sample(
        self,
        *,
        num_samples: int = 1,
        prompt: ModelInputParam,
        sampling_params: SamplingParamsParam,
        base_model: str | NotGiven = NOT_GIVEN,
        model_path: str | NotGiven = NOT_GIVEN,
        prompt_logprobs: bool | NotGiven = NOT_GIVEN,
        type: Literal["sample"] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
        max_retries: int | NotGiven = NOT_GIVEN,
    ) -> SampleResponse:
        """
        Generates samples from the model using the specified sampling parameters

        Args:
          num_samples: Number of samples to generate

          base_model: Optional base model name to sample from. Is inferred from model_path, if
              provided. If sampling against a base model, this is required.

          model_path: Optional tinker:// path to your model weights or LoRA weights. If not provided,
              samples against the base model.

          prompt_logprobs: If set to `true`, computes and returns logprobs on the prompt tokens. Defaults
              to false.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds

          idempotency_key: Specify a custom idempotency key for this request
        """
        options = make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                idempotency_key=idempotency_key,
        )
        if max_retries is not NOT_GIVEN:
            options["max_retries"] = max_retries

        return await self._post(
            "/api/v1/sample",
            body=await async_maybe_transform(
                {
                    "num_samples": num_samples,
                    "prompt": prompt,
                    "sampling_params": sampling_params,
                    "base_model": base_model,
                    "model_path": model_path,
                    "prompt_logprobs": prompt_logprobs,
                    "type": type,
                },
                sampling_sample_params.SamplingSampleParams,
            ),
            options=options,
            cast_to=SampleResponse,
        )


class SamplingResourceWithRawResponse:
    def __init__(self, sampling: SamplingResource) -> None:
        self._sampling = sampling

        self.asample = to_raw_response_wrapper(
            sampling.asample,
        )
        self.sample = to_raw_response_wrapper(
            sampling.sample,
        )


class AsyncSamplingResourceWithRawResponse:
    def __init__(self, sampling: AsyncSamplingResource) -> None:
        self._sampling = sampling

        self.asample = async_to_raw_response_wrapper(
            sampling.asample,
        )
        self.sample = async_to_raw_response_wrapper(
            sampling.sample,
        )


class SamplingResourceWithStreamingResponse:
    def __init__(self, sampling: SamplingResource) -> None:
        self._sampling = sampling

        self.asample = to_streamed_response_wrapper(
            sampling.asample,
        )
        self.sample = to_streamed_response_wrapper(
            sampling.sample,
        )


class AsyncSamplingResourceWithStreamingResponse:
    def __init__(self, sampling: AsyncSamplingResource) -> None:
        self._sampling = sampling

        self.asample = async_to_streamed_response_wrapper(
            sampling.asample,
        )
        self.sample = async_to_streamed_response_wrapper(
            sampling.sample,
        )
