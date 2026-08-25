from __future__ import annotations

import httpx

from .._base_client import make_request_options
from .._compat import cached_property, model_dump
from .._resource import AsyncAPIResource
from .._response import async_to_raw_response_wrapper
from .._types import NOT_GIVEN, NotGiven
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
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> TelemetryResponse:
        """
        Accepts batches of SDK telemetry events for analytics and diagnostics

        Args:
          request: The telemetry send request containing events and session info

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return await self._post(
            "/api/v1/telemetry",
            body=model_dump(request, exclude_unset=False, exclude_none=True, mode="json"),
            options=make_request_options(timeout=timeout),
            cast_to=TelemetryResponse,
        )


class AsyncTelemetryResourceWithRawResponse:
    def __init__(self, telemetry: AsyncTelemetryResource) -> None:
        self._telemetry = telemetry

        self.send = async_to_raw_response_wrapper(
            telemetry.send,
        )
