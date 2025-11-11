from __future__ import annotations

from typing import cast

import httpx

from .._base_client import make_request_options
from .._compat import model_dump
from .._resource import AsyncAPIResource
from .._types import NOT_GIVEN, Body, Headers, NotGiven, Query
from ..types.sample_request import SampleRequest
from ..types.shared.untyped_api_future import UntypedAPIFuture

__all__ = ["AsyncSamplingResource"]


class AsyncSamplingResource(AsyncAPIResource):
    async def asample(
        self,
        *,
        request: SampleRequest,
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
          request: The sample request containing prompt, sampling params, and options

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
            options["max_retries"] = cast(int, max_retries)

        return await self._post(
            "/api/v1/asample",
            body=model_dump(request, exclude_unset=True, mode="json"),
            options=options,
            cast_to=UntypedAPIFuture,
        )
