# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal

import httpx

from ..types import (
    ModelID,
    CheckpointsListResponse,
    weight_load_params,
    weight_save_params,
    weight_save_for_sampler_params,
)
from .._types import NOT_GIVEN, Body, Query, Headers, NotGiven, NoneType
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
from ..types.shared.untyped_api_future import UntypedAPIFuture

__all__ = ["WeightsResource", "AsyncWeightsResource"]


class WeightsResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> WeightsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/stainless-sdks/tinker-python#accessing-raw-response-data-eg-headers
        """
        return WeightsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> WeightsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/stainless-sdks/tinker-python#with_streaming_response
        """
        return WeightsResourceWithStreamingResponse(self)

    def load(
        self,
        *,
        model_id: ModelID,
        path: str,
        type: Literal["load_weights"] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
    ) -> UntypedAPIFuture:
        """
        Loads model weights from disk

        Args:
          path: A tinker URI for model weights at a specific step

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds

          idempotency_key: Specify a custom idempotency key for this request
        """
        return self._post(
            "/api/v1/load_weights",
            body=maybe_transform(
                {
                    "model_id": model_id,
                    "path": path,
                    "type": type,
                },
                weight_load_params.WeightLoadParams,
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

    def save(
        self,
        *,
        model_id: ModelID,
        path: str | NotGiven = NOT_GIVEN,
        type: Literal["save_weights"] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
    ) -> UntypedAPIFuture:
        """
        Saves the current model weights to disk

        Args:
          path: A file/directory name for the weights

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds

          idempotency_key: Specify a custom idempotency key for this request
        """
        return self._post(
            "/api/v1/save_weights",
            body=maybe_transform(
                {
                    "model_id": model_id,
                    "path": path,
                    "type": type,
                },
                weight_save_params.WeightSaveParams,
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

    def save_for_sampler(
        self,
        *,
        model_id: ModelID,
        path: str | NotGiven = NOT_GIVEN,
        type: Literal["save_weights_for_sampler"] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
    ) -> UntypedAPIFuture:
        """
        Saves weights in a format compatible with sampling/inference servers

        Args:
          path: A file/directory name for the weights

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds

          idempotency_key: Specify a custom idempotency key for this request
        """
        return self._post(
            "/api/v1/save_weights_for_sampler",
            body=maybe_transform(
                {
                    "model_id": model_id,
                    "path": path,
                    "type": type,
                },
                weight_save_for_sampler_params.WeightSaveForSamplerParams,
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

    def list(
        self,
        model_id: ModelID,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> CheckpointsListResponse:
        """
        Lists available model checkpoints (both training and sampler)

        Args:
          model_id: The model ID to list checkpoints for

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not model_id:
            raise ValueError(f"Expected a non-empty value for `model_id` but received {model_id!r}")
        return self._get(
            f"/api/v1/models/{model_id}/checkpoints",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            ),
            cast_to=CheckpointsListResponse,
        )

    def delete_checkpoint(
        self,
        *,
        model_id: ModelID,
        checkpoint_id: str,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> None:
        """Delete a checkpoint for the given training run."""
        if not model_id:
            raise ValueError(f"Expected a non-empty value for `model_id` but received {model_id!r}")
        if not checkpoint_id:
            raise ValueError(
                f"Expected a non-empty value for `checkpoint_id` but received {checkpoint_id!r}"
            )

        self._delete(
            f"/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            ),
            cast_to=NoneType,
        )

        return None

class AsyncWeightsResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncWeightsResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/stainless-sdks/tinker-python#accessing-raw-response-data-eg-headers
        """
        return AsyncWeightsResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncWeightsResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/stainless-sdks/tinker-python#with_streaming_response
        """
        return AsyncWeightsResourceWithStreamingResponse(self)

    async def load(
        self,
        *,
        model_id: ModelID,
        path: str,
        type: Literal["load_weights"] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
    ) -> UntypedAPIFuture:
        """
        Loads model weights from disk

        Args:
          path: A tinker URI for model weights at a specific step

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds

          idempotency_key: Specify a custom idempotency key for this request
        """
        return await self._post(
            "/api/v1/load_weights",
            body=await async_maybe_transform(
                {
                    "model_id": model_id,
                    "path": path,
                    "type": type,
                },
                weight_load_params.WeightLoadParams,
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

    async def save(
        self,
        *,
        model_id: ModelID,
        path: str | NotGiven = NOT_GIVEN,
        type: Literal["save_weights"] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
    ) -> UntypedAPIFuture:
        """
        Saves the current model weights to disk

        Args:
          path: A file/directory name for the weights

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds

          idempotency_key: Specify a custom idempotency key for this request
        """
        return await self._post(
            "/api/v1/save_weights",
            body=await async_maybe_transform(
                {
                    "model_id": model_id,
                    "path": path,
                    "type": type,
                },
                weight_save_params.WeightSaveParams,
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

    async def save_for_sampler(
        self,
        *,
        model_id: ModelID,
        path: str | NotGiven = NOT_GIVEN,
        type: Literal["save_weights_for_sampler"] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
    ) -> UntypedAPIFuture:
        """
        Saves weights in a format compatible with sampling/inference servers

        Args:
          path: A file/directory name for the weights

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds

          idempotency_key: Specify a custom idempotency key for this request
        """
        return await self._post(
            "/api/v1/save_weights_for_sampler",
            body=await async_maybe_transform(
                {
                    "model_id": model_id,
                    "path": path,
                    "type": type,
                },
                weight_save_for_sampler_params.WeightSaveForSamplerParams,
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

    async def list(
        self,
        model_id: ModelID,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> CheckpointsListResponse:
        """
        Lists available model checkpoints (both training and sampler)

        Args:
          model_id: The model ID to list checkpoints for

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not model_id:
            raise ValueError(f"Expected a non-empty value for `model_id` but received {model_id!r}")
        return await self._get(
            f"/api/v1/training_runs/{model_id}/checkpoints",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            ),
            cast_to=CheckpointsListResponse,
        )

    async def delete_checkpoint(
        self,
        *,
        model_id: ModelID,
        checkpoint_id: str,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> None:
        """Delete a checkpoint for the given training run."""
        if not model_id:
            raise ValueError(f"Expected a non-empty value for `model_id` but received {model_id!r}")
        if not checkpoint_id:
            raise ValueError(
                f"Expected a non-empty value for `checkpoint_id` but received {checkpoint_id!r}"
            )

        await self._delete(
            f"/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            ),
            cast_to=NoneType,
        )

        return None


class WeightsResourceWithRawResponse:
    def __init__(self, weights: WeightsResource) -> None:
        self._weights = weights

        self.load = to_raw_response_wrapper(
            weights.load,
        )
        self.save = to_raw_response_wrapper(
            weights.save,
        )
        self.save_for_sampler = to_raw_response_wrapper(
            weights.save_for_sampler,
        )
        self.list = to_raw_response_wrapper(
            weights.list,
        )
        self.delete_checkpoint = to_raw_response_wrapper(
            weights.delete_checkpoint,
        )


class AsyncWeightsResourceWithRawResponse:
    def __init__(self, weights: AsyncWeightsResource) -> None:
        self._weights = weights

        self.load = async_to_raw_response_wrapper(
            weights.load,
        )
        self.save = async_to_raw_response_wrapper(
            weights.save,
        )
        self.save_for_sampler = async_to_raw_response_wrapper(
            weights.save_for_sampler,
        )
        self.list = async_to_raw_response_wrapper(
            weights.list,
        )
        self.delete_checkpoint = async_to_raw_response_wrapper(
            weights.delete_checkpoint,
        )


class WeightsResourceWithStreamingResponse:
    def __init__(self, weights: WeightsResource) -> None:
        self._weights = weights

        self.load = to_streamed_response_wrapper(
            weights.load,
        )
        self.save = to_streamed_response_wrapper(
            weights.save,
        )
        self.save_for_sampler = to_streamed_response_wrapper(
            weights.save_for_sampler,
        )
        self.list = to_streamed_response_wrapper(
            weights.list,
        )
        self.delete_checkpoint = to_streamed_response_wrapper(
            weights.delete_checkpoint,
        )


class AsyncWeightsResourceWithStreamingResponse:
    def __init__(self, weights: AsyncWeightsResource) -> None:
        self._weights = weights

        self.load = async_to_streamed_response_wrapper(
            weights.load,
        )
        self.save = async_to_streamed_response_wrapper(
            weights.save,
        )
        self.save_for_sampler = async_to_streamed_response_wrapper(
            weights.save_for_sampler,
        )
        self.list = async_to_streamed_response_wrapper(
            weights.list,
        )
        self.delete_checkpoint = async_to_streamed_response_wrapper(
            weights.delete_checkpoint,
        )
