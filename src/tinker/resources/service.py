# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import httpx

from .._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from .._compat import cached_property
from .._resource import SyncAPIResource, AsyncAPIResource
from .._response import (
    to_raw_response_wrapper,
    to_streamed_response_wrapper,
    async_to_raw_response_wrapper,
    async_to_streamed_response_wrapper,
)
from .._base_client import make_request_options
from ..types.health_response import HealthResponse
from ..types.get_server_capabilities_response import GetServerCapabilitiesResponse

__all__ = ["ServiceResource", "AsyncServiceResource"]


class ServiceResource(SyncAPIResource):
    @cached_property
    def with_raw_response(self) -> ServiceResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/stainless-sdks/tinker-python#accessing-raw-response-data-eg-headers
        """
        return ServiceResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> ServiceResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/stainless-sdks/tinker-python#with_streaming_response
        """
        return ServiceResourceWithStreamingResponse(self)

    def get_server_capabilities(
        self,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> GetServerCapabilitiesResponse:
        """Retrieves information about supported models and server capabilities"""
        return self._get(
            "/api/v1/get_server_capabilities",
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=GetServerCapabilitiesResponse,
        )

    def health_check(
        self,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> HealthResponse:
        """Checks if the API server is ready"""
        return self._get(
            "/api/v1/healthz",
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=HealthResponse,
        )


class AsyncServiceResource(AsyncAPIResource):
    @cached_property
    def with_raw_response(self) -> AsyncServiceResourceWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/stainless-sdks/tinker-python#accessing-raw-response-data-eg-headers
        """
        return AsyncServiceResourceWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncServiceResourceWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/stainless-sdks/tinker-python#with_streaming_response
        """
        return AsyncServiceResourceWithStreamingResponse(self)

    async def get_server_capabilities(
        self,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> GetServerCapabilitiesResponse:
        """Retrieves information about supported models and server capabilities"""
        return await self._get(
            "/api/v1/get_server_capabilities",
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=GetServerCapabilitiesResponse,
        )

    async def health_check(
        self,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> HealthResponse:
        """Checks if the API server is ready"""
        return await self._get(
            "/api/v1/healthz",
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=HealthResponse,
        )


class ServiceResourceWithRawResponse:
    def __init__(self, service: ServiceResource) -> None:
        self._service = service

        self.get_server_capabilities = to_raw_response_wrapper(
            service.get_server_capabilities,
        )
        self.health_check = to_raw_response_wrapper(
            service.health_check,
        )


class AsyncServiceResourceWithRawResponse:
    def __init__(self, service: AsyncServiceResource) -> None:
        self._service = service

        self.get_server_capabilities = async_to_raw_response_wrapper(
            service.get_server_capabilities,
        )
        self.health_check = async_to_raw_response_wrapper(
            service.health_check,
        )


class ServiceResourceWithStreamingResponse:
    def __init__(self, service: ServiceResource) -> None:
        self._service = service

        self.get_server_capabilities = to_streamed_response_wrapper(
            service.get_server_capabilities,
        )
        self.health_check = to_streamed_response_wrapper(
            service.health_check,
        )


class AsyncServiceResourceWithStreamingResponse:
    def __init__(self, service: AsyncServiceResource) -> None:
        self._service = service

        self.get_server_capabilities = async_to_streamed_response_wrapper(
            service.get_server_capabilities,
        )
        self.health_check = async_to_streamed_response_wrapper(
            service.health_check,
        )
