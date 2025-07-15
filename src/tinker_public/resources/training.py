# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal

import httpx

from ..types import (
    ModelID,
    training_forward_params,
    training_optim_step_params,
    training_forward_backward_params,
)
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
from ..types.model_id import ModelID
from ..types.shared.untyped_api_future import UntypedAPIFuture
from ..types.forward_backward_input_param import ForwardBackwardInputParam

__all__ = ["TrainingResource", "AsyncTrainingResource"]


class TrainingResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> TrainingResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/stainless-sdks/tinker-python#accessing-raw-response-data-eg-headers
        """
        return TrainingResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> TrainingResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/stainless-sdks/tinker-python#with_streaming_response
        """
        return TrainingResourceWithStreamingResponse(self)

    def forward(
        self,
        *,
        forward_input: ForwardBackwardInputParam,
        model_id: ModelID,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
    ) -> UntypedAPIFuture:
        """
        Performs a forward pass through the model

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds

          idempotency_key: Specify a custom idempotency key for this request
        """
        return self._post(
            "/api/v1/forward",
            body=maybe_transform(
                {
                    "forward_input": forward_input,
                    "model_id": model_id,
                },
                training_forward_params.TrainingForwardParams,
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

    def forward_backward(
        self,
        *,
        forward_backward_input: ForwardBackwardInputParam,
        model_id: ModelID,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
    ) -> UntypedAPIFuture:
        """
        Performs a forward and backward pass through the model

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds

          idempotency_key: Specify a custom idempotency key for this request
        """
        return self._post(
            "/api/v1/forward_backward",
            body=maybe_transform(
                {
                    "forward_backward_input": forward_backward_input,
                    "model_id": model_id,
                },
                training_forward_backward_params.TrainingForwardBackwardParams,
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

    def optim_step(
        self,
        *,
        adam_params: training_optim_step_params.AdamParams,
        model_id: ModelID,
        type: Literal["optim_step"] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
    ) -> UntypedAPIFuture:
        """
        Performs an optimization step using AdamW optimizer

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds

          idempotency_key: Specify a custom idempotency key for this request
        """
        return self._post(
            "/api/v1/optim_step",
            body=maybe_transform(
                {
                    "adam_params": adam_params,
                    "model_id": model_id,
                    "type": type,
                },
                training_optim_step_params.TrainingOptimStepParams,
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


class AsyncTrainingResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncTrainingResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/stainless-sdks/tinker-python#accessing-raw-response-data-eg-headers
        """
        return AsyncTrainingResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncTrainingResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/stainless-sdks/tinker-python#with_streaming_response
        """
        return AsyncTrainingResourceWithStreamingResponse(self)

    async def forward(
        self,
        *,
        forward_input: ForwardBackwardInputParam,
        model_id: ModelID,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
    ) -> UntypedAPIFuture:
        """
        Performs a forward pass through the model

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds

          idempotency_key: Specify a custom idempotency key for this request
        """
        return await self._post(
            "/api/v1/forward",
            body=await async_maybe_transform(
                {
                    "forward_input": forward_input,
                    "model_id": model_id,
                },
                training_forward_params.TrainingForwardParams,
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

    async def forward_backward(
        self,
        *,
        forward_backward_input: ForwardBackwardInputParam,
        model_id: ModelID,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
    ) -> UntypedAPIFuture:
        """
        Performs a forward and backward pass through the model

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds

          idempotency_key: Specify a custom idempotency key for this request
        """
        return await self._post(
            "/api/v1/forward_backward",
            body=await async_maybe_transform(
                {
                    "forward_backward_input": forward_backward_input,
                    "model_id": model_id,
                },
                training_forward_backward_params.TrainingForwardBackwardParams,
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

    async def optim_step(
        self,
        *,
        adam_params: training_optim_step_params.AdamParams,
        model_id: ModelID,
        type: Literal["optim_step"] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
    ) -> UntypedAPIFuture:
        """
        Performs an optimization step using AdamW optimizer

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds

          idempotency_key: Specify a custom idempotency key for this request
        """
        return await self._post(
            "/api/v1/optim_step",
            body=await async_maybe_transform(
                {
                    "adam_params": adam_params,
                    "model_id": model_id,
                    "type": type,
                },
                training_optim_step_params.TrainingOptimStepParams,
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


class TrainingResourceWithRawResponse:
    def __init__(self, training: TrainingResource) -> None:
        self._training = training

        self.forward = to_raw_response_wrapper(
            training.forward,
        )
        self.forward_backward = to_raw_response_wrapper(
            training.forward_backward,
        )
        self.optim_step = to_raw_response_wrapper(
            training.optim_step,
        )


class AsyncTrainingResourceWithRawResponse:
    def __init__(self, training: AsyncTrainingResource) -> None:
        self._training = training

        self.forward = async_to_raw_response_wrapper(
            training.forward,
        )
        self.forward_backward = async_to_raw_response_wrapper(
            training.forward_backward,
        )
        self.optim_step = async_to_raw_response_wrapper(
            training.optim_step,
        )


class TrainingResourceWithStreamingResponse:
    def __init__(self, training: TrainingResource) -> None:
        self._training = training

        self.forward = to_streamed_response_wrapper(
            training.forward,
        )
        self.forward_backward = to_streamed_response_wrapper(
            training.forward_backward,
        )
        self.optim_step = to_streamed_response_wrapper(
            training.optim_step,
        )


class AsyncTrainingResourceWithStreamingResponse:
    def __init__(self, training: AsyncTrainingResource) -> None:
        self._training = training

        self.forward = async_to_streamed_response_wrapper(
            training.forward,
        )
        self.forward_backward = async_to_streamed_response_wrapper(
            training.forward_backward,
        )
        self.optim_step = async_to_streamed_response_wrapper(
            training.optim_step,
        )
