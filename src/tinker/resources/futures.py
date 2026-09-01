from __future__ import annotations

from typing import Any, cast

import httpx

from .._base_client import make_request_options
from .._compat import cached_property, model_dump
from .._resource import AsyncAPIResource
from .._response import async_to_raw_response_wrapper
from .._types import NOT_GIVEN, Headers, NotGiven
from ..types.future_retrieve_request import FutureRetrieveRequest
from ..types.future_retrieve_response import FutureRetrieveResponse
from ..types.futures_retrieve_request import FuturesRetrieveRequest
from ..types.futures_retrieve_response import FuturesRetrieveResponse

__all__ = ["AsyncFuturesResource"]


class AsyncFuturesResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncFuturesResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/stainless-sdks/tinker-python#accessing-raw-response-data-eg-headers
        """
        return AsyncFuturesResourceWithRawResponse(self)

    async def retrieve(
        self,
        *,
        request: FutureRetrieveRequest,
        extra_headers: Headers | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        max_retries: int | NotGiven = NOT_GIVEN,
    ) -> FutureRetrieveResponse:
        """
        Retrieves the result of a future by its ID

        Args:
          request: The future retrieve request containing request_id and optional model_id

          extra_headers: Send extra headers

          timeout: Override the client-level default timeout for this request, in seconds
        """
        options = make_request_options(
            extra_headers=extra_headers,
            timeout=timeout,
        )
        if max_retries is not NOT_GIVEN:
            options["max_retries"] = cast(int, max_retries)

        return cast(
            FutureRetrieveResponse,
            await self._post(
                "/api/v1/retrieve_future",
                body=model_dump(request, exclude_unset=False, exclude_none=True, mode="json"),
                options=options,
                cast_to=cast(
                    Any, FutureRetrieveResponse
                ),  # Union types cannot be passed in as arguments in the type system
            ),
        )

    async def retrieve_multi(
        self,
        *,
        request: FuturesRetrieveRequest,
        extra_headers: Headers | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        max_retries: int | NotGiven = NOT_GIVEN,
    ) -> FuturesRetrieveResponse:
        """Poll a sampling session's completion queue for finished/failed requests.

        Args:
          request: The retrieve-futures request (target session + prev_cursor)

          extra_headers: Send extra headers

          timeout: Override the client-level default timeout for this request, in seconds
        """
        options = make_request_options(
            extra_headers=extra_headers,
            timeout=timeout,
        )
        if max_retries is not NOT_GIVEN:
            options["max_retries"] = cast(int, max_retries)

        return cast(
            FuturesRetrieveResponse,
            await self._post(
                "/api/v1/retrieve_futures",
                body=model_dump(request, exclude_unset=False, exclude_none=True, mode="json"),
                options=options,
                cast_to=cast(Any, FuturesRetrieveResponse),
            ),
        )


class AsyncFuturesResourceWithRawResponse:
    def __init__(self, futures: AsyncFuturesResource) -> None:
        self._futures = futures

        self.retrieve = async_to_raw_response_wrapper(
            futures.retrieve,
        )
