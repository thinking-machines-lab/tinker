from __future__ import annotations

from typing import cast

import httpx

from .._base_client import make_request_options
from .._compat import cached_property, model_dump
from .._resource import AsyncAPIResource
from .._response import async_to_raw_response_wrapper
from .._types import NOT_GIVEN, Body, Headers, NotGiven, Query
from ..types.telemetry_response import TelemetryResponse
from ..types.telemetry_send_request import TelemetrySendRequest

__all__ = ["AsyncTelemetryResource"]


class AsyncTelemetryResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncTelemetryResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/stainless-sdks/tinker-python#accessing-raw-response-data-eg-headers
        """
        return AsyncTelemetryResourceWithRawResponse(self)

    async def send(
        self,
        *,
        request: TelemetrySendRequest,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
        max_retries: int | NotGiven = NOT_GIVEN,
    ) -> TelemetryResponse:
        """
        Accepts batches of SDK telemetry events for analytics and diagnostics

        Args:
          request: The telemetry send request containing events and session info

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
            "/api/v1/telemetry",
            body=model_dump(request, exclude_unset=True, mode="json"),
            options=options,
            cast_to=TelemetryResponse,
        )


class AsyncTelemetryResourceWithRawResponse:
    def __init__(self, telemetry: AsyncTelemetryResource) -> None:
        self._telemetry = telemetry

        self.send = async_to_raw_response_wrapper(
            telemetry.send,
        )
