# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Any, cast

import httpx

from ..types import ModelID, RequestID, future_retrieve_params
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
from ..types.request_id import RequestID
from ..types.future_retrieve_response import FutureRetrieveResponse

__all__ = ["FuturesResource", "AsyncFuturesResource"]


class FuturesResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> FuturesResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/stainless-sdks/tinker-python#accessing-raw-response-data-eg-headers
        """
        return FuturesResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> FuturesResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/stainless-sdks/tinker-python#with_streaming_response
        """
        return FuturesResourceWithStreamingResponse(self)

    def retrieve(
        self,
        *,
        request_id: RequestID,
        model_id: ModelID | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
    ) -> FutureRetrieveResponse:
        """
        Retrieves the result of a future by its ID

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds

          idempotency_key: Specify a custom idempotency key for this request
        """
        return cast(
            FutureRetrieveResponse,
            self._post(
                "/api/v1/retrieve_future",
                body=maybe_transform(
                    {
                        "request_id": request_id,
                        "model_id": model_id,
                    },
                    future_retrieve_params.FutureRetrieveParams,
                ),
                options=make_request_options(
                    extra_headers=extra_headers,
                    extra_query=extra_query,
                    extra_body=extra_body,
                    timeout=timeout,
                    idempotency_key=idempotency_key,
                ),
                cast_to=cast(
                    Any, FutureRetrieveResponse
                ),  # Union types cannot be passed in as arguments in the type system
            ),
        )


class AsyncFuturesResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncFuturesResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/stainless-sdks/tinker-python#accessing-raw-response-data-eg-headers
        """
        return AsyncFuturesResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncFuturesResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/stainless-sdks/tinker-python#with_streaming_response
        """
        return AsyncFuturesResourceWithStreamingResponse(self)

    async def retrieve(
        self,
        *,
        request_id: RequestID,
        model_id: ModelID | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
        max_retries: int | NotGiven = NOT_GIVEN,
    ) -> FutureRetrieveResponse:
        """
        Retrieves the result of a future by its ID

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds

          idempotency_key: Specify a custom idempotency key for this request
        """
        options=make_request_options(
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            idempotency_key=idempotency_key,
        )
        if not isinstance(max_retries, NotGiven):
            options["max_retries"] = max_retries

        return cast(
            FutureRetrieveResponse,
            await self._post(
                "/api/v1/retrieve_future",
                body=await async_maybe_transform(
                    {
                        "request_id": request_id,
                        "model_id": model_id,
                    },
                    future_retrieve_params.FutureRetrieveParams,
                ),
                options=options,
                cast_to=cast(
                    Any, FutureRetrieveResponse
                ),  # Union types cannot be passed in as arguments in the type system
            ),
        )


class FuturesResourceWithRawResponse:
    def __init__(self, futures: FuturesResource) -> None:
        self._futures = futures

        self.retrieve = to_raw_response_wrapper(
            futures.retrieve,
        )


class AsyncFuturesResourceWithRawResponse:
    def __init__(self, futures: AsyncFuturesResource) -> None:
        self._futures = futures

        self.retrieve = async_to_raw_response_wrapper(
            futures.retrieve,
        )


class FuturesResourceWithStreamingResponse:
    def __init__(self, futures: FuturesResource) -> None:
        self._futures = futures

        self.retrieve = to_streamed_response_wrapper(
            futures.retrieve,
        )


class AsyncFuturesResourceWithStreamingResponse:
    def __init__(self, futures: AsyncFuturesResource) -> None:
        self._futures = futures

        self.retrieve = async_to_streamed_response_wrapper(
            futures.retrieve,
        )
